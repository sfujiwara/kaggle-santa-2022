import numpy as np


def reconfiguration(from_config, to_config):
    diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    return np.sqrt(diffs.sum())


def color(from_idx, to_idx, image, color_scale=3.0):
    return np.abs(image[from_idx[0], from_idx[1]] - image[to_idx[0], to_idx[1]]).sum() * color_scale
