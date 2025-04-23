SEED = 42

CLASS_NAMES = ["grassland_shrubland", "logging", "mining", "plantation"]

#VARIABLES FOR INVARIANCE_CONSTRAINED LEARNING
EPSILON = 0.01
ETA_D = 0.001
ETA_P = 0.001
GAMMA=0.1
N_MH_STEPS=2
M_SAMPLES=5

# threshold / area for post-processing
SCORE_THRESH = 0.5
MIN_AREA = 20000

# OBA config
BACKGROUND_PROB = 0.3 # Probability of using a background from a separate dataset
EXTRACT_FROM_SAME_IMAGE = False # Set to True to only extract objects from and paste onto same image
# ^ (currently only 1 background in folder, functionality is implemented, but dataset is not gathered)
OBA_PROB = 0.5 # Probability of using OBA on a sample during training process
NUM_OBA_OBJECTS = 5 # Number of new augmented objects to try to paste onto image
MAX_EXTRACT_TRIES = 5 # Number of exctracting object tries before moving on to next object


TESTING = False # set to False for training

if TESTING:
    EPOCHS = 2

    # Batch sizes
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_VAL = 1
    BATCH_SIZE_TEST = 1

    # Number of workers
    NUM_WORKERS_TRAIN = 0   # Set to 0 for testing
    NUM_WORKERS_VAL = 0     # Set to 0 for testing
    NUM_WORKERS_TEST = 0    # Set to 0 for testing

    PIN_MEMORY = False  # For systems without cuda, set to false
    PERSISTNAT_WORKERS = False  # For systems without cuda, set to false

    NUM_SAMPLE_INDICIES = 2 # Number of sample indicies (train / val), default: 176, else less for testing
    NUM_EVAL_INDICIES = 2   # Number of eval indicie, default: 118, else less for testing
else:
    EPOCHS = 100

    # Batch sizes
    BATCH_SIZE_TRAIN = 4
    BATCH_SIZE_VAL = 4
    BATCH_SIZE_TEST = 4

    # Number of workers
    NUM_WORKERS_TRAIN = 4
    NUM_WORKERS_VAL = 4
    NUM_WORKERS_TEST = 4

    PIN_MEMORY = True # For systems without cuda, set to false
    PERSISTNAT_WORKERS = True # For systems without cuda, set to false

    NUM_SAMPLE_INDICIES = 176 # Number of sample indicies (train / val)
    NUM_EVAL_INDICIES = 118   # Number of eval indicie
    

# For normalizing 12-band images
MEAN = [
    285.8190561180765,
    327.22091430696577,
    552.9305957826701,
    392.1575148484924,
    914.3138803812591,
    2346.1184507500043,
    2884.4831706095824,
    2886.442429854111,
    3176.7501338557763,
    3156.934442092072,
    1727.1940075511282,
    848.573373995044,
]
STD = [
    216.44975668759372,
    269.8880248304874,
    309.92790753407064,
    397.45655590699,
    400.22078920482215,
    630.3269651264278,
    789.8006920468097,
    810.4773696969773,
    852.9031432100967,
    807.5976198303886,
    631.7808113929271,
    502.66788721341396,
]


