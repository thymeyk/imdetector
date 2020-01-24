import os
import shutil


def mkdir(DIR):
    if os.path.isdir(DIR):
        shutil.rmtree(DIR)
    os.makedirs(DIR)
    return DIR


def extract_nonoverlap_patches(imgarr, patch_size=(6, 6)):
    h, w = imgarr.shape
    nrows, ncols = patch_size
    return (imgarr[:h // nrows * nrows, :w // ncols * ncols]
            .reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows * ncols))
