# ============================================================================
# Systolic Array Accelerator Makefile
# ============================================================================

# Compiler and flags
GXX_FLAGS := -w -O3 -std=c++17
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
SRC := ./src

# Platform
Platform := xilinx_u55c_gen3x16_xdma_3_202210_1

# Targets
TARGET := sa_test
KERNEL := SystolicArrayKernel

# Default target
.DEFAULT_GOAL := $(TARGET)

# ============================================================================
# Build Rules
# ============================================================================

# Compile sa.cpp
sa.o: $(SRC)/sa.cpp $(SRC)/sa.h
	@echo "Compiling sa.cpp..."
	tapa g++ -- $(GXX_FLAGS) -c $< -o $@

# Compile main.cpp
main.o: $(SRC)/main.cpp $(SRC)/sa.h
	@echo "Compiling main.cpp..."
	tapa g++ -- $(GXX_FLAGS) -c $< -o $@

# Link executable
$(TARGET): sa.o main.o
	@echo "Linking $(TARGET)..."
	tapa g++ -- $(GXX_FLAGS) -o $@ $^ $(LIB)
	@echo "Build complete: $(TARGET)"

# ============================================================================
# Simulation and Testing
# ============================================================================

# Software simulation (CSIM)
swsim: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "Running Software Simulation (CSIM)"
	@echo "========================================"
	./$(TARGET)

# Software simulation for GEMV mode
swsim_gemv: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "Running GEMV Mode (M=1)"
	@echo "========================================"
	./$(TARGET) --gemv

# Small test (reduced dimensions for faster testing)
test_small: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "Running Small Test (M=64, K=512, N=1024)"
	@echo "========================================"
	./$(TARGET) --m=64 --k=512 --n=1024

# ============================================================================
# HLS Synthesis
# ============================================================================

# High-level synthesis
hls: $(SRC)/sa.cpp $(SRC)/sa.h
	@echo ""
	@echo "========================================"
	@echo "Running HLS Synthesis"
	@echo "========================================"
	tapa compile \
		--top $(KERNEL) \
		--platform $(Platform) \
		--clock-period 3.33 \
		-f $(SRC)/sa.cpp \
		-o $(TARGET).xo
	@echo "HLS synthesis complete: $(TARGET).xo"

# ============================================================================
# Hardware Emulation
# ============================================================================

# Hardware emulation
hwemu: $(TARGET).xo $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "Running Hardware Emulation"
	@echo "========================================"
	./$(TARGET) --bitstream=$(TARGET).xo

# ============================================================================
# Performance Analysis
# ============================================================================

# Run with different matrix sizes
perf: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "Performance Testing"
	@echo "========================================"
	@echo ""
	@echo "Test 1: Small (M=64, K=256, N=512)"
	./$(TARGET) --m=64 --k=256 --n=512
	@echo ""
	@echo "Test 2: Medium (M=256, K=1024, N=2048)"
	./$(TARGET) --m=256 --k=1024 --n=2048
	@echo ""
	@echo "Test 3: Large (M=512, K=4096, N=14336)"
	./$(TARGET) --m=512 --k=4096 --n=14336
	@echo ""
	@echo "Test 4: GEMV (M=1, K=4096, N=14336)"
	./$(TARGET) --gemv --k=4096 --n=14336

# ============================================================================
# Clean
# ============================================================================

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o $(TARGET)

# Clean everything including HLS outputs
cleanall:
	@echo "Cleaning all outputs..."
	rm -rf work.out
	rm -f *.o $(TARGET) $(TARGET).xo
	rm -rf _x .Xil
	rm -f *.log *.jou

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Systolic Array Accelerator Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build the executable (default)"
	@echo "  make swsim        - Run software simulation (CSIM) with default params"
	@echo "  make swsim_gemv   - Run GEMV mode (M=1)"
	@echo "  make test_small   - Run with smaller dimensions for quick test"
	@echo "  make hls          - Run HLS synthesis to generate .xo"
	@echo "  make hwemu        - Run hardware emulation"
	@echo "  make perf         - Run performance tests with various sizes"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make cleanall     - Remove all outputs including HLS"
	@echo ""
	@echo "Options:"
	@echo "  --m=<val>         - Set M dimension (default: 512)"
	@echo "  --k=<val>         - Set K dimension (default: 4096)"
	@echo "  --n=<val>         - Set N dimension (default: 14336)"
	@echo "  --gemv            - Run in GEMV mode (M=1)"
	@echo "  --bitstream=<xo>  - Specify bitstream file for HW/HW-emu"
	@echo ""
	@echo "Examples:"
	@echo "  make swsim"
	@echo "  make test_small"
	@echo "  ./sa_test --m=128 --k=512 --n=1024"
	@echo "  ./sa_test --gemv --k=4096 --n=14336"

.PHONY: swsim swsim_gemv test_small hls hwemu perf clean cleanall help
