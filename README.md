````markdown
# FPGA-Project

This repository contains a Systolic Array implementation in C++ with HLS support. The project includes source code, a Makefile, and instructions for compilation and simulation.

## Directory Structure

```text
.
├── Makefile
└── src/
    ├── sa.h
    ├── sa.cpp
    └── main.cpp
````

* `src/sa.h` - Header file containing the Systolic Array definitions.
* `src/sa.cpp` - Implementation of the Systolic Array functions.
* `src/main.cpp` - Main program to test the Systolic Array.
* `Makefile` - Build configuration for compilation and simulation.

## Commands

Before building the project, make sure to set up the environment:

```bash
export PATH="$PATH:/home/coder/.rapidstream-tapa/usr/bin"
source /tools/Xilinx/Vitis/2023.2/settings64.sh
```

### Build and Simulation

```bash
# Clean previous build files
make clean

# Compile the project
make

# Run software simulation
make swsim

# Run HLS synthesis
make hls
```

## Configuration

You can change the dimensions `M`, `K`, and `N` of the Systolic Array **only** within the `src/sa.h` file:

```cpp
// Example (inside sa.h)
#define M 16
#define K 16
#define N 16
```

Make sure to rebuild the project after modifying these values:

```bash
make clean
make
```
