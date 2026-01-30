#%%
from stitch.stitch import graph
from stitch import generate
import matplotlib.pyplot as plt
# %%
a = graph.generate_hilbert_curve(64)
# %%
points = generate.circle_points((10, 10), 10)
# %%
b = graph.hilbert_over_points(points)
# %%
