# RVV Kernel Tools

A comprehensive toolkit for converting RISC-V Vector (RVV) kernels to scalar C implementations for debugging and precision analysis.

## Overview

This project provides automated tools to:
- Convert RVV (RISC-V Vector Extension) kernel implementations to scalar C code
- Validate functional correctness by comparing RVV and scalar outputs
- Analyze precision errors and ULP (Units in the Last Place) differences
- Locate behavioral errors and precision issues in vectorized code

## Features

- **Automated Conversion**: Python-based converter that transforms RVV intrinsics to scalar operations
- **Comprehensive Validation**: Built-in test framework for comparing RVV vs scalar implementations
- **Precision Analysis**: ULP error tracking and visualization tools
- **Multiple Kernel Support**: Currently supports mathematical functions like `sin`, `exp`, and more
- **Configurable Testing**: JSON-based configuration for test parameters

## Prerequisites

- CMake 3.10 or higher
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- Python 3.8+
- RISC-V toolchain (for RVV compilation)
- matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rvv_kernel_tools.git
cd rvv_kernel_tools
```

2. Install Python dependencies:
```bash
pip install matplotlib numpy
```

3. Build the project:
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### 1. Convert RVV Kernel to Scalar

Run the conversion script and select your input file:

```bash
python convert_rvv_to_scalar.py
```

This will:
- Parse RVV intrinsics from your source file
- Generate equivalent scalar C code
- Output a header file with scalar implementations

### 2. Validate and Compare

Build and run the validation tool:

```bash
# Build the validator
cd build
make validate_scalar

# Run validation
./validate_scalar
```

The validator will:
- Generate test cases with various input values
- Execute both RVV and scalar implementations
- Compare results for correctness
- Report precision differences (ULP errors)

### 3. Analyze Results

Visualize ULP errors and precision analysis:

```bash
python plot_ulp_errors.py
```

## Project Structure

```
rvv_kernel_tools/
├── CMakeLists.txt                 # Build configuration
├── convert_rvv_to_scalar.py       # Main conversion script
├── validate_scalar.cpp            # Validation framework
├── rvv_conversion_config.json     # Conversion rules and mappings
├── plot_ulp_errors.py             # Visualization tools
├── *_scalar_functions.h           # Generated scalar implementations
├── *_test_config.json             # Test configurations
└── *_macro.txt                    # Macro definitions
```

## Configuration

### Conversion Configuration (`rvv_conversion_config.json`)

Defines mappings between RVV intrinsics and scalar operations:

```json
{
  "intrinsic_mappings": {
    "__riscv_vfadd": "scalar_add",
    "__riscv_vfmul": "scalar_mul",
    ...
  }
}
```

### Test Configuration (`*_test_config.json`)

Specifies test parameters for each kernel:

```json
{
  "test_range": [-10.0, 10.0],
  "test_points": 1000,
  "ulp_threshold": 4
}
```

## Supported Kernels

Currently supported mathematical functions:
- `sin` - Sine function
- `exp` - Exponential function
- More kernels can be added by extending the conversion rules

## How It Works

1. **Parsing Phase**: The converter parses RVV intrinsic calls and identifies vector operations
2. **Transformation**: Each RVV intrinsic is replaced with equivalent scalar loops
3. **Code Generation**: Scalar C code is generated maintaining the same computational logic
4. **Validation**: Test harness runs both implementations and compares results
5. **Analysis**: ULP errors and precision differences are calculated and visualized

## Debugging Workflow

1. Identify a problematic RVV kernel showing incorrect behavior
2. Convert it to scalar using the tool
3. Run validation to pinpoint exact operations causing issues
4. Analyze ULP error patterns to understand precision loss
5. Fix the original RVV implementation based on findings

## Contributing

Contributions are welcome! To add support for new RVV intrinsics:

1. Add the intrinsic mapping to `rvv_conversion_config.json`
2. Implement the scalar equivalent in the converter
3. Add test cases in the validation framework
4. Submit a pull request with your changes

## Examples

### Converting an Exp Kernel

```bash
# Select exp kernel for conversion
python convert_rvv_to_scalar.py
> Select file: rvv_exp_kernel.c

# Output: exp_scalar_functions.h generated

# Validate
./build/validate_exp
```

### Custom Test Configuration

Create a custom test configuration:

```json
{
  "test_range": [-100.0, 100.0],
  "test_points": 10000,
  "ulp_threshold": 2,
  "special_values": [0.0, 1.0, -1.0, "inf", "-inf", "nan"]
}
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Ensure RISC-V toolchain is properly installed
2. **Precision Differences**: Adjust ULP threshold in test configuration
3. **Unsupported Intrinsics**: Add new mappings to configuration file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as part of the nncase neural network compiler project
- Supports Kendryte K210/K230 AI accelerator development
- Uses RISC-V Vector Extension (RVV) for optimized tensor operations

## Contact

For questions or support, please open an issue on GitHub.