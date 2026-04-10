"""
Config handling for .yaml files
"""

from yacs.config import CfgNode as CN

_C = CN()

# Image Data Configuration
_C.DATASET = CN()
_C.DATASET.TYPE=""
_C.DATASET.CACHE_DIR = ""
_C.DATASET.IMAGE_EXT = ""
_C.DATASET.KEY_POINTS = 0
_C.DATASET.CACHED_IMAGE_SIZE = []
_C.DATASET.PIXEL_SIZE = []
_C.DATASET.DISTANCE = 0

# Training configuration
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 10


# Model Configuration
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'Unet'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

