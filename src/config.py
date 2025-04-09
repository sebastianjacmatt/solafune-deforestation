EPOCHS = 2
SEED = 42
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 8

CLASS_NAMES = ["grassland_shrubland", "logging", "mining", "plantation"]

#VARIABLES FOR INVARIANCE_CONSTRAINED LEARNING
INVARIANCE_CONSTRAINED_LEARNING = True
EPSILON = 0.5
ETA_D = 0.01
ETA_P = 0.01
GAMMA=0.1
N_MH_STEPS=10
M_SAMPLES=5

# Number of sample indicies (train / val), default: 176, else less for testing
NUM_SAMPLE_INDICIES = 2

# Number of eval indicie, default: 118, else less for testing
NUM_EVAL_INDICIES = 2

# threshold / area for post-processing
SCORE_THRESH = 0.5
MIN_AREA = 20000

TESTING = True # set to False for training

if TESTING:
    # Batch sizes
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_VAL = 1
    BATCH_SIZE_TEST = 1 

    # Number of workers
    NUM_WORKERS_TRAIN = 0 # set to 0 for testing
    NUM_WORKERS_VAL = 0 # set to 0 for testing
    NUM_WORKERS_TEST = 0 # set to 0 for testing

    PIN_MEMORY = False # for systems without cuda, set to false
    PERSISTNAT_WORKERS = False # for systems without cuda, set to false
else:
    # Batch sizes
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_VAL = 4
    BATCH_SIZE_TEST = 4 

    # Number of workers
    NUM_WORKERS_TRAIN = 4 # set to 0 for testing
    NUM_WORKERS_VAL = 4 # set to 0 for testing
    NUM_WORKERS_TEST = 4 # set to 0 for testing

    PIN_MEMORY = True # for systems without cuda, set to false
    PERSISTNAT_WORKERS = True # for systems without cuda, set to false
    

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


