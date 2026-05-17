#################################################################
#################################################################
# Configuration options                                         #
#################################################################
#################################################################

# Supported: ICC, GCC, ICPX
# COMPILER = GCC
COMPILER = ICPX
# Supported: MKL, OPENBLAS
# BLAS_LIBRARY = OPENBLAS
BLAS_LIBRARY = MKL

# either 64 or 128
ALTO_MASK_LENGTH = 64

# List of modes and ranks to specialize code for; use 0 to
# disable specialization.
MODES_SPECIALIZED := 0
RANKS_SPECIALIZED := 0
MAX_NUM_MODES = 5

# use ALTERNATIVE_PEXT if the ISA does not support BMI2 instructions
ALTERNATIVE_PEXT = true
THP_PRE_ALLOCATION = false

MEMTRACE = false
DEBUG = false
