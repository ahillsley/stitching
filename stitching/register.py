import dexp


# Given 2 fovs, use dexp to estimate the shift between them


#%%
import dexp
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration import translation_nd as dexp_reg
import numpy as np

def shift_model(
        image_a: np.array,
        image_b: np.array,
        )->TranslationRegistrationModel:
    
    # TODO: have this load only a small fraction of the entire image
    
    return dexp_reg.register_translation_nd(image_a, image_b)


# %%
import skimage
from generate import example_tiles
img = skimage.data.astronaut()[:,:,0]
tiles = example_tiles(img, 2, 20)
# %%
from dexp.processing.registration import translation_nd as dexp_reg

a = dexp_reg.register_translation_nd(tiles[0], tiles[1])


# %%
