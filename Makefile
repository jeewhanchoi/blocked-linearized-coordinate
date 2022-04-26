# CONFIGURE BUILD SYSTEM
TARGET     = cpd$(ALTO_MASK_LENGTH)
BUILD_DIR  = ./build-$(ALTO_MASK_LENGTH)
INC_DIR    = ./include
SRC_DIR    = ./src
MAKE_DIR   = ./mk
Q         ?= @


##########################################
# DO NOT EDIT BELOW
include ./config.mk
include $(MAKE_DIR)/include_$(COMPILER).mk
include $(MAKE_DIR)/include_NVCC.mk

space := $(eval) $(eval)
ifneq ($(strip $(MODES_SPECIALIZED)),0)
  MODES_SPECIALIZED := 0,$(MODES_SPECIALIZED)
endif
ifneq ($(strip $(RANKS_SPECIALIZED)),0)
  RANKS_SPECIALIZED := 0,$(RANKS_SPECIALIZED)
endif
DEFINES += -DALTO_MODES_SPECIALIZED=$(subst $(space),,$(MODES_SPECIALIZED))
DEFINES += -DALTO_RANKS_SPECIALIZED=$(subst $(space),,$(RANKS_SPECIALIZED))
ifeq ($(THP_PRE_ALLOCATION),true)
DEFINES += -DALTO_PRE_ALLOC
endif
ifeq ($(ALTERNATIVE_PEXT),true)
DEFINES += -DALT_PEXT
endif
ifeq ($(MEMTRACE),true)
DEFINES += -DALTO_MEM_TRACE
endif
ifeq ($(DEBUG),true)
DEFINES += -DALTO_DEBUG
endif
ifeq ($(BLAS_LIBRARY),MKL)
DEFINES += -DMKL
endif

INCLUDES += -I$(INC_DIR)

SRC        = $(wildcard $(SRC_DIR)/*.cpp)
ASM        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.s,$(SRC))
OBJ        = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o,$(SRC))
CUOBJ	   = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.cuo,$(wildcard $(SRC_DIR)/*.cu))
CPPFLAGS  := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)


$(TARGET): $(BUILD_DIR) $(OBJ) $(CUOBJ)
	@echo "===>  LINKING  $(TARGET)"
	$(CXX) $(LFLAGS) -o $(TARGET) $(OBJ) $(LIBS) $(CUOBJ) $(LIBS) $(NVCCLIBS)

asm: $(BUILD_DIR) $(ASM)

info:
	@echo $(CXXFLAGS)
	$(CXX) $(VERSION)

$(BUILD_DIR)/%.d: $(SRC_DIR)/%.cpp | build_dir
	$(Q)$(CXX) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "===>  COMPILE  $@"
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVCCINCLUDES) $< -o $@

$(BUILD_DIR)/%.cuo: $(SRC_DIR)/%.cu
	@echo "===>  Device COMPILE  $@"
	$(NVCC) -ccbin=$(CXX) -c $(DEFINES) $(OPTIONS) $(INCLUDES) $(NVCCFLAGS) $(NVCCLFLAGS) $(NVCCINCLUDES) $< -o $@	

$(BUILD_DIR)/%.s:  $(SRC_DIR)/%.c
	@echo "===>  GENERATE ASM  $@"
	$(CXX) -S $(CPPFLAGS) $(CXXFLAGS) $< -o $@

.PHONY: build_dir
build_dir: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

ifeq ($(findstring $(MAKECMDGOALS),clean),)
-include $(OBJ:.o=.d)
endif

.PHONY: clean distclean

clean:
	@echo "===>  CLEAN"
	@rm -rf $(BUILD_DIR)

distclean: clean
	@echo "===>  DIST CLEAN"
	@rm -f $(TARGET)
