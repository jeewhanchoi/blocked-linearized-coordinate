CXX      = icpx

OPENMP   = -fiopenmp
ifeq ($(BLAS_LIBRARY),MKL)
BLASLFLAGS	= -qmkl=parallel
BLASINC		=
BLASLIBS	= -liomp5 -lpthread -lm -ldl
else
BLASLFLAGS	=
BLASINC		=
BLASLIBS	= -lopenblas
endif

ifneq ($(ALTERNATIVE_PEXT),true)
CXXFLAGS += -mbmi2
endif

VERSION  = --version
ifeq ($(DEBUG),true)
CXXFLAGS += $(OPENMP) -O0 -g -march=native -Wall -std=c++17 -D_GLIBCXX_PARALLEL
LIBS	 = -lpthread -lm -ldl $(BLASLIBS)
else
CXXFLAGS += $(OPENMP) -O3 -xHost -g -std=c++17 -D_GLIBCXX_PARALLEL
LIBS	 = -Wl,--no-as-needed -lpthread -lm -ldl $(BLASLIBS)
endif

LFLAGS   = $(OPENMP) $(BLASLFLAGS)
DEFINES  = -DALTO_MASK_LENGTH=$(ALTO_MASK_LENGTH) -DMAX_NUM_MODES=$(MAX_NUM_MODES)
INCLUDES = $(BLASINC)
