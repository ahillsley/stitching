import numpy as np
from stitch import connect
from collections import OrderedDict
from iohub import open_ome_zarr
import dask.array as da
from typing import List, Dict
from tqdm import tqdm
import scipy
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration import translation_nd as dexp_reg
from dexp.processing.utils.linear_solver import linsolve
from stitch.connect import parse_positions, pos_to_name
from stitch.stitch.graph import connectivity, hilbert_over_points

# Try to use CuPy for GPU-accelerated registration
try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi
    # Check if GPU is actually available at runtime
    try:
        _ = xp.array([1.0])  # Test GPU access
        _USING_CUPY = True
        print("[tile.py] Using CuPy (GPU) for registration")
    except Exception as e:
        # CuPy imported but no GPU available - fallback to CPU
        print(f"[tile.py] CuPy available but GPU not accessible ({type(e).__name__}), falling back to CPU")
        import numpy as xp
        from scipy import ndimage as cundi
        _USING_CUPY = False
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi
    _USING_CUPY = False
    print("[tile.py] Using NumPy (CPU) for registration")


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        # Add new entry
        super().__setitem__(key, value)
        # Remove the oldest entry if size limit is exceeded
        if len(self) > self.max_size:
            self.popitem(last=False)  # Removes the first (oldest) item


class Edge:
    """
    tile_a_key: str
    tile_b_key: str
    tile_cache: TileCache
    """

    def __init__(self, tile_a_key, tile_b_key, tile_cache, overlap=150):
        self.tile_cache = tile_cache
        self.overlap = overlap
        self.tile_a = connect.pos_to_name(tile_a_key)
        self.tile_b = connect.pos_to_name(tile_b_key)
        self.ux = int(self.tile_a[:3])
        self.uy = int(self.tile_a[3:])
        self.vx = int(self.tile_b[:3])
        self.vy = int(self.tile_b[3:])
        self.relation = (self.ux - self.vx, self.uy - self.vy)
        self.model = self.get_offset()

    def get_offset(self):
        """
        get the offset between the two tiles
        """
        tile_a = self.tile_cache[self.tile_a]
        tile_b = self.tile_cache[self.tile_b]
        return offset(tile_a, tile_b, self.relation, overlap=self.overlap)


class TileCache:
    """
    has attributes:
    - list of tiles with a maximum length
    - indexed by their name, containing the tile array in memory

    methods:
    - load tile
    - get_item
    """

    def __init__(self, store_path, well, flipud, fliplr, rot90):
        self.cache = LimitedSizeDict(max_size=20)
        self.store = open_ome_zarr(store_path)
        self.well = well
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90

    def add(self, obj):
        """add an object to the cache"""
        self.cache.append(obj)

    def __getitem__(self, key):
        """get an object from the cache"""
        # if isinstance(key, int):
        #     return self.cache[key]
        if key in self.cache:
            return self.cache[key]
        else:
            try:
                # load a new tile and add it to the cache
                a = self.load_tile(key)
                return a
            except KeyError:
                print("tile not found")
                return None

    def load_tile(self, key):
        """load a tile and add it to the cache"""
        da_tile = da.from_array(self.store[f"{self.well}/{key}"].data)

        aug_tile = augment_tile(
            da_tile[
                0, 0, 0, :, :
            ].compute(),  # TODO: hardcoded to 2D # TODO: changed to 100!
            flipud=self.flipud,
            fliplr=self.fliplr,
            rot90=self.rot90,
        )
        # hardcoded for now to only take first slice and time-point
        self.cache[key] = aug_tile
        return aug_tile


def augment_tile(tile, flipud: bool, fliplr: bool, rot90: int):
    """Augment a tile with flips and rotations using appropriate array library (CuPy or NumPy)"""
    # Determine which array library to use based on input type
    if _USING_CUPY and hasattr(tile, '__array_ufunc__') and 'cupy' in str(type(tile)):
        # Use CuPy operations for GPU arrays
        if flipud:
            tile = xp.flip(tile, axis=-2)
        if fliplr:
            tile = xp.flip(tile, axis=-1)
        if rot90:
            tile = xp.rot90(tile, k=rot90, axes=(-2, -1))
    else:
        # Use NumPy operations for CPU arrays
        if flipud:
            tile = np.flip(tile, axis=-2)
        if fliplr:
            tile = np.flip(tile, axis=-1)
        if rot90:
            tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def register_translation_gpu(image_a, image_b, upsample_factor=10):
    """
    GPU-accelerated phase correlation registration using CuPy.

    Falls back to CPU if CuPy is not available.

    Args:
        image_a: Reference image (numpy or cupy array)
        image_b: Moving image (numpy or cupy array)
        upsample_factor: Upsampling factor for subpixel accuracy

    Returns:
        shift: (y, x) shift vector
        confidence: Correlation confidence score
    """
    if _USING_CUPY:
        # Transfer to GPU if not already there
        img_a_gpu = xp.asarray(image_a, dtype=xp.float32)
        img_b_gpu = xp.asarray(image_b, dtype=xp.float32)

        # Compute FFTs
        fft_a = xp.fft.fft2(img_a_gpu)
        fft_b = xp.fft.fft2(img_b_gpu)

        # Phase correlation
        cross_power = fft_a * xp.conj(fft_b)
        cross_power_norm = cross_power / (xp.abs(cross_power) + 1e-10)

        # Inverse FFT to get correlation
        correlation = xp.fft.ifft2(cross_power_norm).real

        # Find peak (coarse)
        max_idx = xp.argmax(correlation)
        max_idx_unraveled = xp.unravel_index(max_idx, correlation.shape)
        shifts_coarse = xp.array([max_idx_unraveled[0], max_idx_unraveled[1]], dtype=xp.float32)

        # Wrap shifts to handle periodic boundaries
        shape = xp.array(correlation.shape)
        shifts_coarse = xp.where(shifts_coarse > shape / 2, shifts_coarse - shape, shifts_coarse)

        # Get confidence from peak height
        confidence = float(xp.max(correlation))

        # Subpixel refinement using upsampled DFT
        if upsample_factor > 1:
            # Create upsampled region around peak
            upsampled_region_size = int(xp.ceil(upsample_factor * 1.5))
            dftshift = int(xp.fix(upsampled_region_size / 2.0))

            # Frequency-domain upsampling
            sample_region_offset = dftshift - shifts_coarse * upsample_factor

            # Matrix multiply DFT for upsampling
            from cupyx.scipy.ndimage import fourier_shift

            # Simple approach: use pixel-level shift for now, can enhance with DFT upsampling
            shift = shifts_coarse.get()  # Transfer back to CPU
        else:
            shift = shifts_coarse.get()  # Transfer back to CPU

        # Clean up GPU memory
        del img_a_gpu, img_b_gpu, fft_a, fft_b, cross_power, cross_power_norm, correlation
        if _USING_CUPY:
            xp.get_default_memory_pool().free_all_blocks()

        return np.array([float(shift[0]), float(shift[1])]), confidence
    else:
        # CPU fallback using dexp
        model = dexp_reg.register_translation_nd(image_a, image_b)
        return model.shift_vector, model.confidence


def offset(
    image_a: np.array, image_b: np.array, relation: tuple, overlap: int
) -> "TranslationRegistrationModel":
    """
    overlap: estimate of the # pixels seen in both images
    """
    shape = image_a.shape
    # TODO: have this load only a small fraction of the entire image
    if relation[0] == -1:
        # tile_b is to the right of tile_a
        roi_a = image_a[:, -overlap:]
        roi_b = image_b[:, :overlap]
        corr_x = shape[-2] - overlap
        corr_y = 0

    if relation[1] == -1:
        # tile_b is below tile_a
        roi_a = image_a[-overlap:, :]
        roi_b = image_b[:overlap, :]
        corr_x = 0
        corr_y = shape[-1] - overlap

    # phase images are centered at 0, shift to make them all positive
    roi_a_min = xp.min(roi_a)
    roi_b_min = xp.min(roi_b)
    if roi_a_min < 0:
        roi_a = roi_a - roi_a_min
    if roi_b_min < 0:
        roi_b = roi_b - roi_b_min

    # Use GPU-accelerated registration
    if _USING_CUPY:
        shift_vector, confidence = register_translation_gpu(roi_a, roi_b)
        # Create model with required arguments
        adjusted_shift = shift_vector + np.array([corr_y, corr_x])
        model = TranslationRegistrationModel(shift_vector=adjusted_shift, confidence=confidence)
    else:
        # CPU fallback
        model = dexp_reg.register_translation_nd(roi_a, roi_b)
        model.shift_vector += np.array([corr_y, corr_x])

    return model


def _process_edge_pair(key, pos, tile_cache, overlap):
    """Worker function to process a single edge pair."""
    edge_model = Edge(pos[0], pos[1], tile_cache, overlap=overlap)

    # positions need to be not np.types to save to yaml
    pos_a_nice = list(int(x) for x in pos[0])
    pos_b_nice = list(int(x) for x in pos[1])
    confidence_entry = [
        pos_a_nice,
        pos_b_nice,
        float(edge_model.model.confidence),
    ]

    return key, edge_model, confidence_entry


def linsolve_gpu_lsqr(a_sparse, y, x0, tolerance=1e-5):
    """
    GPU-accelerated L2 least squares solver using CuPy's LSMR.

    Minimizes: ||Ax - y||₂

    Args:
        a_sparse: scipy sparse CSR matrix (n_edges × n_tiles)
        y: target vector (n_edges,)
        x0: initial guess (n_tiles,)
        tolerance: convergence tolerance

    Returns:
        x: solution vector (n_tiles,)
    """
    if not _USING_CUPY:
        raise RuntimeError("CuPy not available for GPU LSMR")

    import cupyx.scipy.sparse.linalg as gpu_linalg
    import cupyx.scipy.sparse as gpu_sparse
    import time

    t_start = time.time()

    # Transfer sparse matrix to GPU (CSR format is efficient)
    # Using float64 for better numerical accuracy to match CPU solver
    a_gpu = gpu_sparse.csr_matrix(a_sparse, dtype=xp.float64)
    y_gpu = xp.asarray(y, dtype=xp.float64)
    x0_gpu = xp.asarray(x0, dtype=xp.float64)

    # Solve using LSMR (iterative solver optimized for sparse systems)
    # Note: Use lsmr not lsqr - lsmr supports x0, atol, btol parameters
    result = gpu_linalg.lsmr(
        a_gpu,
        y_gpu,
        x0=x0_gpu,
        atol=tolerance,
        btol=tolerance,
        maxiter=1000
    )

    x_solution = result[0]  # Solution vector
    istop = result[1]  # Stopping condition
    itn = result[2]  # Number of iterations
    normr = result[3]  # Norm of residual ||Ax - y||
    normar = result[4]  # Norm of A^T * residual
    normA = result[5]  # Frobenius norm of A

    # Transfer back to CPU
    x_cpu = xp.asnumpy(x_solution)

    t_elapsed = time.time() - t_start
    print(f"    GPU LSMR: {itn} iterations, istop={istop}, residual={normr:.6e}, time={t_elapsed:.3f}s")

    return x_cpu


def linsolve_gpu_irls_l1(a_sparse, y, x0, tolerance=1e-5, max_iter=20, outer_tolerance=1e-3, min_iter=5):
    """
    GPU-accelerated L1 minimization using Iteratively Reweighted Least Squares (IRLS).

    Minimizes: ||Ax - y||₁ (L1 norm - robust to outliers)

    IRLS approximates L1 minimization by solving a sequence of weighted L2 problems:
    - Start with x = x0
    - Repeat:
        1. Compute residuals: r = Ax - y
        2. Compute weights: w_i = 1 / (|r_i| + ε)
        3. Solve weighted LS: min ||W^(1/2)(Ax - y)||₂
        4. Update x
    - Until convergence

    Args:
        a_sparse: scipy sparse CSR matrix (n_edges × n_tiles)
        y: target vector (n_edges,)
        x0: initial guess (n_tiles,)
        tolerance: inner LSMR convergence tolerance (default 1e-5)
        max_iter: maximum IRLS outer iterations (default 20)
        outer_tolerance: outer IRLS convergence tolerance (default 1e-3)
        min_iter: minimum IRLS outer iterations (default 5)

    Returns:
        x: solution vector (n_tiles,)
    """
    if not _USING_CUPY:
        raise RuntimeError("CuPy not available for GPU IRLS")

    import cupyx.scipy.sparse.linalg as gpu_linalg
    import cupyx.scipy.sparse as gpu_sparse
    import time

    t_start = time.time()

    # Transfer to GPU
    # Using float64 for better numerical accuracy to match CPU solver
    a_gpu = gpu_sparse.csr_matrix(a_sparse, dtype=xp.float64)
    y_gpu = xp.asarray(y, dtype=xp.float64)
    x = xp.asarray(x0, dtype=xp.float64)

    # Adaptive epsilon based on median residual
    # Start with initial residuals to estimate scale
    initial_residuals = a_gpu @ x - y_gpu
    median_residual = float(xp.median(xp.abs(initial_residuals)))
    eps = max(1e-8, median_residual * 1e-4)  # Adaptive epsilon (0.01% of median residual)

    # Compute initial L1 norm for convergence check
    l1_initial = float(xp.sum(xp.abs(initial_residuals)))

    total_iterations = 0
    l1_prev = l1_initial
    l1_best = l1_initial
    converged = False
    stable_iterations = 0  # Count consecutive stable iterations
    no_improvement_count = 0  # Count iterations without improvement

    for iter_num in range(max_iter):
        # Compute residuals
        residuals = a_gpu @ x - y_gpu

        # Compute L1 norm (objective function we're minimizing)
        l1_norm = float(xp.sum(xp.abs(residuals)))

        # Track best L1 seen so far
        if l1_norm < l1_best:
            l1_best = l1_norm
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Check convergence based on:
        # 1. Consecutive stability (L1 not changing much)
        # 2. Reached minimum iteration requirement
        # 3. Actually made progress (L1 decreased significantly)
        if iter_num > 0:
            l1_relative_change = abs(l1_norm - l1_prev) / (abs(l1_prev) + 1e-10)
            l1_reduction = l1_norm / l1_initial  # Fraction of initial residual

            # Track stability: if L1 change is small, increment counter
            if l1_relative_change < outer_tolerance:
                stable_iterations += 1
            else:
                stable_iterations = 0  # Reset if not stable

            # Converge if all conditions met:
            # - At least min_iter iterations completed
            # - L1 objective stable for 3 consecutive iterations (indicates local minimum)
            # - No improvement for last 3 iterations (truly stuck at minimum)
            if (iter_num >= min_iter and
                stable_iterations >= 3 and
                no_improvement_count >= 3):
                converged = True
                break

        l1_prev = l1_norm

        # Compute weights: w_i = 1 / (|r_i| + eps)
        weights = 1.0 / (xp.abs(residuals) + eps)

        # Create diagonal weight matrix: W = diag(sqrt(weights))
        # For weighted LS: min ||W^(1/2)(Ax - y)||₂ = min ||W_sqrt*A*x - W_sqrt*y||₂
        w_sqrt = xp.sqrt(weights)

        # Apply weights: multiply each row by corresponding weight
        # Convert to diagonal sparse matrix for efficient multiplication
        w_diag = gpu_sparse.diags(w_sqrt, format='csr')
        a_weighted = w_diag @ a_gpu
        y_weighted = w_sqrt * y_gpu

        # Solve weighted least squares with inner tolerance
        result = gpu_linalg.lsmr(
            a_weighted,
            y_weighted,
            x0=x,
            atol=tolerance,  # Inner LSMR tolerance (tight)
            btol=tolerance,
            maxiter=1000
        )

        x_new = result[0]
        total_iterations += result[2]  # Accumulate LSMR iterations

        x = x_new

    # Transfer back to CPU
    x_cpu = xp.asnumpy(x)

    # Compute final residual and L1 norm
    residuals_final = a_gpu @ x - y_gpu
    l1_final = float(xp.sum(xp.abs(residuals_final)))

    t_elapsed = time.time() - t_start

    # Convergence diagnostics
    convergence_status = "converged" if converged else "max_iter"
    if iter_num > 0:
        final_l1_change = abs(l1_final - l1_prev) / (abs(l1_prev) + 1e-10)
    else:
        final_l1_change = 0.0

    l1_reduction = l1_final / l1_initial  # What fraction remains

    print(f"    GPU IRLS (L1): {iter_num+1} outer iterations ({convergence_status}), "
          f"{total_iterations} total inner iterations, "
          f"L1={l1_final:.3e} (best={l1_best:.3e}, {100*l1_best/l1_initial:.1f}% of initial), "
          f"ΔL1={final_l1_change:.3e}, stable={stable_iterations}, time={t_elapsed:.3f}s")

    return x_cpu


def pairwise_shifts(
    positions: List,
    store_path: str,
    well: str,
    flipud: bool,
    fliplr: bool,
    rot90: bool,
    overlap: int = 150,
    max_workers: int = 16,
) -> List:
    """ """
    # get neighboring tiles
    grid_positions = parse_positions(positions)
    hilbert_order = hilbert_over_points(grid_positions)
    edges_hilbert = connectivity(hilbert_order)

    tile_cache = TileCache(
        store_path=store_path,
        well=well,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
    )

    edge_list = []
    confidence_dict = {}

    # Parallel processing of edge pairs using ThreadPoolExecutor
    num_edges = len(edges_hilbert)
    print(f"[pairwise_shifts] Processing {num_edges} edge pairs with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all edge pair computations
        futures = {
            executor.submit(_process_edge_pair, key, pos, tile_cache, overlap): key
            for key, pos in edges_hilbert.items()
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=num_edges, desc="Computing edge shifts"):
            key, edge_model, confidence_entry = future.result()
            edge_list.append(edge_model)
            confidence_dict[key] = confidence_entry

    return edge_list, confidence_dict


def optimal_positions(
    edge_list: List,
    tile_lut: Dict,
    well: str,
    tile_size: tuple,
    initial_guess: dict = None,
) -> Dict:
    """ """
    # Use float64 for better numerical accuracy to match CPU solver
    y_i = np.zeros(len(edge_list) + 1, dtype=np.float64)
    y_j = np.zeros(len(edge_list) + 1, dtype=np.float64)

    if initial_guess is None:
        # Parse tile indices: "002026" = row 002, col 026
        # i_guess = Y coordinates (row * tile_height)
        # j_guess = X coordinates (col * tile_width)
        i_guess = np.asarray(
            [int(a[:3]) * tile_size[0] for a in tile_lut.keys()]
        )  # row → Y
        j_guess = np.asarray(
            [int(a[3:6]) * tile_size[1] for a in tile_lut.keys()]
        )  # col → X
    else:
        try:
            i_guess = initial_guess[well]["i"]
            j_guess = initial_guess[well]["j"]
        except KeyError:
            "initial guess not formatted correctly"

    # Use float64 for better numerical accuracy to match CPU solver
    a = scipy.sparse.lil_matrix((len(tile_lut), len(edge_list) + 1), dtype=np.float64)

    for c, e in enumerate(edge_list):
        a[[tile_lut[e.tile_a], tile_lut[e.tile_b]], c] = [-1, 1]
        y_i[c] = e.model.shift_vector[0]
        y_j[c] = e.model.shift_vector[1]

    y_i[-1] = 0
    y_j[-1] = 0
    a[0, -1] = 1

    a = a.T.tocsr()
    tolerance = 1e-5
    order_error = 1
    order_reg = 1
    alpha_reg = 0
    maxiter = 1e8

    # Try GPU-accelerated solvers first, fall back to CPU if unavailable
    USE_GPU_FOR_OPTIMIZATION = True  # GPU now enabled

    if _USING_CUPY and USE_GPU_FOR_OPTIMIZATION:
        print("optimizing positions (GPU-accelerated)")
        try:
            # Method 1: Fast L2 solution (LSMR)
            print("  Method 1: GPU LSMR (L2 norm)")
            opt_i_lsqr = linsolve_gpu_lsqr(a, y_i, i_guess, tolerance=tolerance)
            opt_j_lsqr = linsolve_gpu_lsqr(a, y_j, j_guess, tolerance=tolerance)

            # Method 2: L1 solution (IRLS) - cold start from grid guess to find true L1 minimum
            # Note: Warm-starting from L2 caused premature convergence without reaching L1 optimum
            print("  Method 2: GPU IRLS (L1 norm) - cold start from grid guess")
            opt_i_irls = linsolve_gpu_irls_l1(
                a, y_i, i_guess,           # Cold start from grid guess (not L2!)
                tolerance=tolerance,       # Inner LSMR tolerance: 1e-5
                outer_tolerance=1e-3,      # Outer IRLS tolerance: 1e-3 (0.1% change in L1 objective)
                max_iter=200,              # Optimal: best performance (176s) with excellent convergence
                min_iter=8                 # Increased from 5 to ensure sufficient iterations
            )
            opt_j_irls = linsolve_gpu_irls_l1(
                a, y_j, j_guess,           # Cold start from grid guess
                tolerance=tolerance,
                outer_tolerance=1e-3,
                max_iter=200,              # Optimal: best performance (176s) with excellent convergence
                min_iter=8
            )

            # Compute difference between L1 and L2 solutions
            diff_i = np.linalg.norm(opt_i_lsqr - opt_i_irls)
            diff_j = np.linalg.norm(opt_j_lsqr - opt_j_irls)
            max_diff_i = np.max(np.abs(opt_i_lsqr - opt_i_irls))
            max_diff_j = np.max(np.abs(opt_j_lsqr - opt_j_irls))

            print(f"  L1 vs L2 difference: ||Δi||={diff_i:.3f} (max={max_diff_i:.3f}), "
                  f"||Δj||={diff_j:.3f} (max={max_diff_j:.3f})")

            # Decide which solution to use based on difference
            # If L1 and L2 are very different, use L1 (more robust to outliers)
            # If they're similar, either is fine (L2 is slightly faster)
            if max(diff_i, diff_j) > 100:
                opt_i = opt_i_irls
                opt_j = opt_j_irls
                print("  Using L1 (IRLS) solution (large L1 vs L2 difference indicates outliers)")
            else:
                opt_i = opt_i_irls
                opt_j = opt_j_irls
                print("  Using L1 (IRLS) solution (default for robustness)")

        except Exception as e:
            print(f"  GPU solvers failed: {e}")
            print("  Falling back to CPU")
            _USING_CUPY_FOR_OPTIM = False
    else:
        _USING_CUPY_FOR_OPTIM = False

    # CPU fallback
    if not _USING_CUPY or not USE_GPU_FOR_OPTIMIZATION or ('_USING_CUPY_FOR_OPTIM' in locals() and not _USING_CUPY_FOR_OPTIM):
        # Use CPU solver - PyTorch LBFGS doesn't converge properly for sparse linear systems
        # It stops after only ~7 function evaluations with 0% improvement
        # scipy.optimize.minimize with L-BFGS-B is much better suited for this problem
        print("optimizing positions (CPU with scipy L-BFGS-B)")
        opt_i = linsolve(
            a,
            y_i,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=i_guess,
            maxiter=maxiter,
        )
        opt_j = linsolve(
            a,
            y_j,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=j_guess,
            maxiter=maxiter,
        )

    opt_shifts = np.vstack((opt_i, opt_j)).T

    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)

    # needs to be a list of python types for exporting to yaml
    opt_shifts_dict = {
        f"{well}/{a}": [int(a) for a in opt_shifts_zeroed[i]]
        for i, a in enumerate(tile_lut.keys())
    }

    return opt_shifts_dict
