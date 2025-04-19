"""Configuration settings for the CATHODE system."""

# Initial Settings
TURN_ON_BREAK_COUNT = True
JET_TYPE = 'Tau_Jet'
N_DIMS = 2
NUMBER_TO_PAD_TO = 15

# Directory settings
SIGNAL_DIRECTORY = 'TPrime_TwoTau_250'
BACKGROUND_DIRECTORY = 'Mixed_100MassCut'

# Mass bounds for signal/background regions
LOWER_MASS_BOUND = 0.175
UPPER_MASS_BOUND = 0.3

# Signal injection settings
SIGNAL_INJECTION_RATIO = 0.01

# Test set ratio
TEST_SET_RATIO = 0.7

# Batch sizes for dataset preparation
MAX_SIGNAL_EVENTS = 40700  # For TPrime_TwoTau_250

# Background event limitations
if BACKGROUND_DIRECTORY == "Mixed_100MassCut":
    MAX_BACKGROUND_EVENTS = 158000

# Flow model settings
CATHODE_TITLE = 'user_input'
NUMBER_OF_DIMENSIONS = 14
CATHODE_VERSION_NUMBER = 1
NUMBER_OF_BINS = 24
TAIL_BOUND_VALUE = 6
NUM_HIDDEN = 512
NUM_BLOCKS = 6
NUM_CYCLES = 6
NUM_LAYERS = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.00005
LEARNING_RATE_PRINT = '5_10_5'
EPOCHS = None
NUM_COND_INPUTS = 1
EARLY_STOPPING = True
PATIENCE = 50
LOAD_TRAINED_MODEL = True

# File paths
ROOT_FILE_BASE_DIRECTORY = 'user_input'
PICKLE_FILE_BASE_DIRECTORY = 'user_input'
USER = 'user_input'
CATHODE_BASE_DIRECTORY = 'user_input'
PLOT_DIRECTORY = f'user_input'
PLOT_SAVE_DIRECTORY = f'user_input'

# Other flags
USE_BDT = True
DATA_SEED = 42