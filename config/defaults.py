import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

_C = CN()
_C.ROOT =""
_C.ANNO_FILE = ""

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 10

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.HEIGHT = 512
# Size of the image during test
_C.INPUT.WIDTH = 512
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Resnet
# -----------------------------------------------------------------------------
_C.MODEL.MOTIONVAE = CN()
_C.MODEL.MOTIONVAE.LATENTDIM = [32, 64, 128, 256, 512]
_C.MODEL.MOTIONVAE.IN_CHANNELS = 3
_C.MODEL.MOTIONVAE.WINDOW_SIZE = 11
_C.MODEL.MOTIONVAE.SIZE_AVG = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.LRSCHEDULER = "step"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500, )

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.0001

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.DTYPE = "float32"
# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False
