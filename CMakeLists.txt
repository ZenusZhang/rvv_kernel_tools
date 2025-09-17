cmake_minimum_required(VERSION 3.20)
project(rvv_kernel_tools CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toolchain configuration
set(USE_RISCV_TOOLCHAIN OFF CACHE BOOL "Use RISC-V cross-compilation toolchain")

# RISC-V toolchain configuration
set(RISCV_TOOLCHAIN_PATH "/workspace/riscv64-unknown-linux_gnu_14.2.0/bin" CACHE STRING "Path to RISC-V toolchain")
set(RISCV_PREFIX "riscv64-unknown-linux-gnu" CACHE STRING "RISC-V toolchain prefix")

# x86 toolchain configuration (default)
set(X86_TOOLCHAIN_PATH "/usr/bin" CACHE STRING "Path to x86 toolchain")
set(X86_PREFIX "gcc" CACHE STRING "x86 toolchain prefix")

# Set toolchain based on configuration
if(USE_RISCV_TOOLCHAIN)
    # Force RISC-V cross-compilation
    set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_PATH}/${RISCV_PREFIX}-gcc")
    set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PATH}/${RISCV_PREFIX}-g++")
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
else()
    # Use x86 native compilation (default)
    set(CMAKE_C_COMPILER "${X86_TOOLCHAIN_PATH}/gcc")
    set(CMAKE_CXX_COMPILER "${X86_TOOLCHAIN_PATH}/g++")
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Python script for generating header files
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Function to generate scalar header from macro input
function(generate_scalar_header INPUT_FILE KERNEL_TYPE)
    get_filename_component(INPUT_NAME ${INPUT_FILE} NAME_WE)
    set(OUTPUT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_TYPE}_scalar_functions.h")

    add_custom_command(
        OUTPUT ${OUTPUT_HEADER}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/convert_rvv_to_scalar.py
                -i ${INPUT_FILE}
                -o ${OUTPUT_HEADER}
                --config ${CMAKE_CURRENT_SOURCE_DIR}/rvv_conversion_config.json
                --kernel ${KERNEL_TYPE}
        DEPENDS ${INPUT_FILE}
                ${CMAKE_CURRENT_SOURCE_DIR}/convert_rvv_to_scalar.py
                ${CMAKE_CURRENT_SOURCE_DIR}/rvv_conversion_config.json
        COMMENT "Generating ${KERNEL_TYPE} scalar functions header from ${INPUT_FILE}"
        VERBATIM
    )

    # Create a target for the generated header
    add_custom_target(generate_${KERNEL_TYPE}_header DEPENDS ${OUTPUT_HEADER})
endfunction()

# Generate headers for available kernels
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sin_macro.txt")
    generate_scalar_header("${CMAKE_CURRENT_SOURCE_DIR}/sin_macro.txt" "sin")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/exp_macro.txt")
    generate_scalar_header("${CMAKE_CURRENT_SOURCE_DIR}/exp_macro.txt" "exp")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cos_macro.txt")
    generate_scalar_header("${CMAKE_CURRENT_SOURCE_DIR}/cos_macro.txt" "cos")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Function to create validator executable for a specific kernel
function(create_kernel_validator KERNEL_TYPE)
    string(TOUPPER ${KERNEL_TYPE} KERNEL_UPPER)
    set(TARGET_NAME "validate_${KERNEL_TYPE}")
    set(HEADER_FILE "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_TYPE}_scalar_functions.h")
    set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_TYPE}_test_config.json")

    # Only create target if header generation target exists
    if(TARGET generate_${KERNEL_TYPE}_header)
        add_executable(${TARGET_NAME} validate_scalar.cpp)

        # Add dependency on generated header
        add_dependencies(${TARGET_NAME} generate_${KERNEL_TYPE}_header)

        # Define kernel type
        target_compile_definitions(${TARGET_NAME} PRIVATE -DKERNEL_${KERNEL_UPPER})

        # Link math library
        target_link_libraries(${TARGET_NAME} m)

        # Set architecture-specific flags
        if(USE_RISCV_TOOLCHAIN)
            # Set RISC-V specific flags for cross-compiling
            target_compile_options(${TARGET_NAME} PRIVATE -march=rv64gcv -mabi=lp64d)
            target_link_options(${TARGET_NAME} PRIVATE -march=rv64gcv -mabi=lp64d)
        else()
            # Set x86 specific flags for native compilation
            target_compile_options(${TARGET_NAME} PRIVATE -march=native)
        endif()

        # Create test configuration if it doesn't exist
        if(NOT EXISTS ${CONFIG_FILE})
            if(${KERNEL_TYPE} STREQUAL "sin")
                file(WRITE ${CONFIG_FILE}
"{\n    \"golden\": \"sinf\",\n    \"threshold_ulp\": 2.0,\n    \"range_start\": -6.28,\n    \"range_end\": 6.28,\n    \"range_step\": 0.1,\n    \"include_specials\": true,\n    \"include_large_multiples\": false\n}")
            elseif(${KERNEL_TYPE} STREQUAL "exp")
                file(WRITE ${CONFIG_FILE}
"{\n    \"golden\": \"expf\",\n    \"threshold_ulp\": 2.0,\n    \"range_start\": -10.0,\n    \"range_end\": 10.0,\n    \"range_step\": 0.1,\n    \"include_specials\": false,\n    \"include_large_multiples\": false\n}")
            endif()
        endif()

        message(STATUS "Created validator target: ${TARGET_NAME}")

        # Create short alias for convenience
        add_custom_target(${KERNEL_TYPE})
        add_dependencies(${KERNEL_TYPE} ${TARGET_NAME})
    endif()
endfunction()

# Create validator executables for available kernels
create_kernel_validator("sin")
create_kernel_validator("exp")
create_kernel_validator("cos")

# Custom targets for convenience
add_custom_target(all_kernels)
foreach(kernel sin exp cos)
    if(TARGET validate_${kernel})
        add_dependencies(all_kernels validate_${kernel})
    endif()
endforeach()

# Help target
add_custom_target(usage
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "RVV Kernel Tools Build Targets:"
    COMMAND ${CMAKE_COMMAND} -E echo "  sin       - Build sin kernel validator"
    COMMAND ${CMAKE_COMMAND} -E echo "  exp       - Build exp kernel validator"
    COMMAND ${CMAKE_COMMAND} -E echo "  cos       - Build cos kernel validator"
    COMMAND ${CMAKE_COMMAND} -E echo "  all_kernels - Build all available validators"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
    COMMAND ${CMAKE_COMMAND} -E echo "  mkdir build && cd build"
    COMMAND ${CMAKE_COMMAND} -E echo "  # For x86 native compilation (default):"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    COMMAND ${CMAKE_COMMAND} -E echo "  # For RISC-V cross-compilation:"
    COMMAND ${CMAKE_COMMAND} -E echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_RISCV_TOOLCHAIN=ON"
    COMMAND ${CMAKE_COMMAND} -E echo "  make sin      # or exp, cos, all_kernels"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  ./validate_sin --config ../sin_test_config.json"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)

# Print configuration summary
message(STATUS "RVV Kernel Tools Configuration:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")

if(USE_RISCV_TOOLCHAIN)
    message(STATUS "  RISC-V Cross-compilation: ENABLED")
    message(STATUS "  RISC-V Toolchain: ${RISCV_TOOLCHAIN_PATH}")
else()
    message(STATUS "  x86 Native compilation: ENABLED")
    message(STATUS "  x86 Toolchain: ${X86_TOOLCHAIN_PATH}")
endif()