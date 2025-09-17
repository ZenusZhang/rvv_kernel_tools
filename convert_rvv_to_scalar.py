#!/usr/bin/env python3
"""
RISC-V Vector to Scalar Conversion Tool with SSA Transform

Converts RVV intrinsics in macro bodies to scalar C++ with SSA-based debug instrumentation.
Supports multiple kernels (sin, exp, etc.) with automatic kernel detection and variable discovery.
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ParsedOperation:
    intrinsic: str
    result: str
    operands: List[str]
    line_number: int
    original_line: str


@dataclass
class SSAOperation:
    result_ssa: str
    result_base: str
    rhs_expression: str
    original_line: str
    result_type: str


class RVVConverter:
    def __init__(self, config_file: str = "rvv_conversion_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.operations: List[ParsedOperation] = []
        self.variables: Dict[str, str] = {}
        self.ssa_variables: Dict[str, str] = {}
        self.ssa_versions: Dict[str, int] = defaultdict(int)
        self.kernel_type: str = "unknown"
        self.constants: Dict[str, str] = {}  # Store parsed #define constants

    def detect_kernel_from_content(self, content: str) -> str:
        """Detect kernel type from macro content"""
        if 'exp' in content or 'EXP' in content:
            return 'exp'
        elif 'sin' in content or 'SIN' in content:
            return 'sin'
        elif 'cos' in content or 'COS' in content:
            return 'cos'
        return 'unknown'

    def parse_constants_from_content(self, content: str) -> Dict[str, str]:
        """Parse #define constants from input content"""
        constants = {}

        # Match #define statements with constant values (including hex values)
        define_pattern = r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+((?:0x[0-9a-fA-F]+\.[0-9a-fA-F]+p[+-][0-9]+f?)|(?:[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?[fF]?))'

        for match in re.finditer(define_pattern, content):
            const_name = match.group(1)
            const_value = match.group(2)
            constants[const_name] = const_value

        return constants

    def infer_type_from_intrinsic(self, intrinsic: str, operands: List[str]) -> str:
        """Infer variable type from intrinsic pattern"""
        # Float operations
        if any(op in intrinsic for op in ['vfmadd', 'vfmul', 'vfadd', 'vfsub', 'vfmin', 'vfmax', 'vfabs', 'vfrsub']):
            return 'float'
        # Integer operations  
        if any(op in intrinsic for op in ['vfcvt_x_f', 'vadd_vv_i', 'vsll', 'vand', 'vxor']):
            return 'int32_t'
        # Mask operations
        if 'vmfgt' in intrinsic or 'vmflt' in intrinsic:
            return 'bool'
        # Bit reinterpret operations
        if 'vreinterpret' in intrinsic:
            if 'f32m' in intrinsic and 'i32m' in intrinsic:
                return 'uint32_t'  # for _bits variables
        # Float move operations
        if 'vfmv_v_f' in intrinsic:
            return 'float'
        return 'float'  # default
    
    def is_scalar_constant(self, operand: str) -> bool:
        stripped = operand.strip()

        # Check config-defined patterns
        for pattern in self.config["operand_detection"]["scalar_constant_patterns"]:
            if re.match(pattern, stripped):
                return True

        # Check parsed constants from #define
        if stripped in self.constants:
            return True

        # Also check for known constants
        return stripped.startswith('c_') or stripped in ['M', 'vl']
        
    def discover_variables_from_operations(self, operations: List[ParsedOperation]) -> Dict[str, str]:
        """Extract all variables used in operations and infer their types"""
        variables = {}
        
        for op in operations:
            # Infer type for result variable
            if op.result and op.result != "temp_var":
                var_type = self.infer_type_from_intrinsic(op.intrinsic, op.operands)
                variables[op.result] = var_type
            
            # Also track operand variables (non-constants)
            for operand in op.operands:
                if (not self.is_scalar_constant(operand) and 
                    re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand.strip()) and
                    operand.strip() not in variables):
                    # Try to infer type from usage context
                    variables[operand.strip()] = 'float'  # default assumption
        
        return variables

    def parse_intrinsic_call(self, line: str) -> Optional[ParsedOperation]:
        clean_line = re.sub(r"/\*.*?\*/", "", line).strip()
        clean_line = clean_line.rstrip("\\").strip()
        if not clean_line or clean_line.startswith("//"):
            return None

        if any(skip in clean_line for skip in ["#define", "inline", "const", "size_t"]):
            return None

        def extract_paren_content(s: str, open_idx: int) -> Optional[str]:
            # s[open_idx] should be '('
            if open_idx < 0 or open_idx >= len(s) or s[open_idx] != '(':
                return None
            depth = 0
            for i in range(open_idx, len(s)):
                ch = s[i]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        return s[open_idx + 1:i]
            return None

        # First try to match an assignment form and capture up to the opening '('
        m_assign = re.search(r"(?:auto\s+|vfloat32m\w+_t\s+)?(\w+)\s*=\s*(__riscv_[\w#]+)\s*\(", clean_line)
        if m_assign:
            result = m_assign.group(1).strip()
            intrinsic = m_assign.group(2).strip()
            open_idx = m_assign.end() - 1  # points to '('
            extracted = extract_paren_content(clean_line, open_idx)
            if extracted is None:
                return None
            operands_str = extracted.strip()
        else:
            # Match a direct intrinsic call without assignment
            m_direct = re.search(r"(__riscv_[\w#]+)\s*\(", clean_line)
            if not m_direct:
                return None
            result = "temp_var"
            intrinsic = m_direct.group(1)
            open_idx = m_direct.end() - 1
            extracted = extract_paren_content(clean_line, open_idx)
            if extracted is None:
                return None
            operands_str = extracted.strip()

        # Remove RVV macro concatenations like '##lmul' or '##lmul##'
        intrinsic = re.sub(r"##\w+(?:##)?", "", intrinsic)
        operands = self.parse_operands(operands_str)
        return ParsedOperation(intrinsic=intrinsic, result=result, operands=operands, line_number=0, original_line=line)

    def parse_operands(self, operands_str: str) -> List[str]:
        ops: List[str] = []
        depth = 0
        cur = ""
        for ch in operands_str:
            if ch == "," and depth == 0:
                ops.append(cur.strip())
                cur = ""
            else:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                cur += ch
        if cur.strip():
            ops.append(cur.strip())
        return [op for op in ops if op.strip() != "vl"]

    def match_intrinsic_pattern(self, intrinsic: str, pattern: str) -> bool:
        regex_pattern = pattern.replace("*", r"\w*")
        return bool(re.match(f"^{regex_pattern}$", intrinsic))

    def get_intrinsic_mapping(self, intrinsic: str) -> Optional[Dict]:
        clean = re.sub(r"##\w+(?:##)?", "", intrinsic)
        if clean in self.config["intrinsic_mappings"]:
            return self.config["intrinsic_mappings"][clean]
        for pat, mapping in self.config["intrinsic_mappings"].items():
            if self.match_intrinsic_pattern(clean, pat):
                return mapping
        for pat, mapping in self.config["bit_operations"].items():
            if self.match_intrinsic_pattern(clean, pat):
                return mapping
        return None

    def convert_constant_to_double(self, constant: str) -> str:
        c = constant.strip()
        if c in self.config["constants"]["double_version"]:
            return self.config["constants"]["double_version"][c]
        if c.endswith("f") and not c.startswith("0x"):
            return c[:-1]
        return c

    def apply_ssa_transform(self, operations: List[ParsedOperation]) -> List[SSAOperation]:
        """Convert operations to SSA form with version tracking"""
        ssa_ops = []
        input_variables_used = set()  # Track which input variables have been used

        for op in operations:
            if op.result == "temp_var" or not op.result:
                continue

            # Process operands FIRST (using current SSA versions)
            rhs_operands = []
            for operand in op.operands:
                if (not self.is_scalar_constant(operand) and
                    re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand.strip())):
                    operand_base = operand.strip()

                    # Special handling for first use of input variable (typically 'x' for exp, 'v' for sin)
                    if (operand_base not in input_variables_used and operand_base in ['x', 'v']):
                        # Use input parameter 'v' for first operation
                        rhs_operands.append('v')
                        input_variables_used.add(operand_base)
                        # Mark as having version 0 since we've used the input
                        if operand_base not in self.ssa_versions:
                            self.ssa_versions[operand_base] = 0
                    elif operand_base in self.ssa_versions and self.ssa_versions[operand_base] > 0:
                        # Use latest SSA version
                        rhs_operands.append(f"{operand_base}_f{self.ssa_versions[operand_base]}")
                    else:
                        # First use of non-input variable or version 0
                        if operand_base not in self.ssa_versions:
                            self.ssa_versions[operand_base] = 0
                        rhs_operands.append(f"{operand_base}_f0")
                else:
                    rhs_operands.append(operand)

            # NOW create SSA version for result (after processing operands)
            base_name = op.result
            self.ssa_versions[base_name] += 1
            result_ssa = f"{base_name}_f{self.ssa_versions[base_name]}"
            
            # Build RHS expression using scalar mapping
            mapping = self.get_intrinsic_mapping(op.intrinsic)
            setup_code = ""
            if mapping:
                subs = {"result": result_ssa}
                for i, operand in enumerate(rhs_operands):
                    subs[f"operand{i+1}"] = operand
                try:
                    rhs_expr = mapping["pattern"].format(**subs)
                    # Handle setup code for reinterpret operations
                    if "setup" in mapping:
                        setup_code = mapping["setup"].format(**subs)
                    # Remove result assignment since we handle it in SSA form
                    if " = " in rhs_expr:
                        rhs_expr = rhs_expr.split(" = ", 1)[1]
                except KeyError:
                    rhs_expr = f"/* mapping failed for {op.intrinsic} */"
            else:
                rhs_expr = f"/* no mapping for {op.intrinsic} */"
            
            # Determine result type
            result_type = self.infer_type_from_intrinsic(op.intrinsic, op.operands)
            
            # Store SSA variable info
            self.ssa_variables[result_ssa] = result_type
            
            ssa_ops.append(SSAOperation(
                result_ssa=result_ssa,
                result_base=base_name,
                rhs_expression=rhs_expr,
                original_line=op.original_line,
                result_type=result_type
            ))

            # Store setup code if needed
            if setup_code:
                setattr(ssa_ops[-1], 'setup_code', setup_code)
        
        return ssa_ops

    def parse_macro_definition(self, content: str) -> List[ParsedOperation]:
        operations: List[ParsedOperation] = []
        content = content.replace("\\\n", " ")
        m = re.search(r"\{(.*?)\}", content, re.DOTALL)
        if not m:
            return operations
        body = m.group(1)
        statements: List[str] = []
        depth = 0
        cur = ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth == 0:
                if cur.strip():
                    statements.append(cur.strip())
                cur = ""
                continue
            cur += ch
        if cur.strip():
            statements.append(cur.strip())
        for i, stmt in enumerate(statements):
            op = self.parse_intrinsic_call(stmt)
            if op:
                op.line_number = i
                operations.append(op)
        return operations

    def generate_kernel_debug_struct(self) -> str:
        """Generate kernel-specific debug struct with SSA variables"""
        struct_name = f"{self.kernel_type.title()}DebugValues"

        code = f"struct {struct_name} {{\n"

        # Standard fields for all kernels
        code += "    double input_v; // Original input value\n"
        code += "    double final_result; // Final computed result\n"

        # Add kernel-specific return fields
        if self.kernel_type == 'sin':
            code += "    double final_result_before_sign; // Result before sign application\n"

        # Add all discovered SSA variables
        ssa_vars = sorted(self.ssa_variables.items())
        for var_name, var_type in ssa_vars:
            # Convert type for debug struct storage (all use double for consistency)
            if var_type in ["float", "double"]:
                debug_type = "double"
            elif var_type == "uint32_t":
                debug_type = "uint64_t"  # Use uint64_t to handle both f32 and f64 cases
            elif var_type == "int32_t":
                debug_type = "int64_t"   # Use int64_t to handle both f32 and f64 cases
            else:
                debug_type = var_type
            code += f"    {debug_type} {var_name}; // SSA variable {var_name}\n"

        code += "};\n\n"
        return code

    def generate_constant_definitions(self) -> str:
        """Generate constant definitions from parsed #define statements"""
        code = "// Parsed constants from #define statements\n"

        # Add essential constants first
        if self.kernel_type == 'exp':
            code += "const int32_t MANTISSA_BITS = 23; // Mantissa bits for float32\n"
            code += "const int64_t MANTISSA_BITS_D = 52; // Mantissa bits for float64\n"

        if not self.constants:
            return code + "\n"

        for const_name, const_value in sorted(self.constants.items()):
            # Convert to appropriate precision
            if const_value.endswith('f'):
                # Float constant
                code += f"const float {const_name} = {const_value};\n"
                # Also provide double version
                double_value = const_value[:-1]  # Remove 'f' suffix
                code += f"const double {const_name}_d = {double_value};\n"
            else:
                # Double constant
                code += f"const double {const_name} = {const_value};\n"
                # Also provide float version
                code += f"const float {const_name}_f = {const_value}f;\n"

        code += "\n"
        return code

    def generate_variable_declarations(self, precision: str) -> str:
        """Generate all variable declarations needed for the function"""
        declarations = []
        
        # Collect all base variables (without SSA suffixes)
        base_vars = set()
        for ssa_var, var_type in self.ssa_variables.items():
            if '_f' in ssa_var:
                base_var = ssa_var.split('_f')[0]
                base_vars.add((base_var, var_type))
        
        # Generate declarations
        type_map = {"float": precision, "double": precision}
        for base_var, var_type in sorted(base_vars):
            actual_type = type_map.get(var_type, var_type)
            declarations.append(f"    {actual_type} {base_var};")
        
        return "\n".join(declarations)
    
    def substitute_constant_for_precision(self, operand: str, precision: str) -> str:
        """Replace constant names with precision-appropriate versions"""
        stripped = operand.strip()

        if stripped in self.constants:
            # Use precision-specific version of constants
            if precision == "double":
                # For double precision, use _d suffix if available, otherwise base name
                return f"{stripped}_d" if self.constants[stripped].endswith('f') else stripped
            else:
                # For float precision, use _f suffix if available, otherwise base name
                return f"{stripped}_f" if not self.constants[stripped].endswith('f') else stripped

        return operand

    def generate_ssa_operation_code(self, ssa_op: SSAOperation, precision: str) -> str:
        """Generate code for one SSA operation with debug instrumentation"""
        code = ""

        # Handle setup code for reinterpret operations
        if hasattr(ssa_op, 'setup_code') and ssa_op.setup_code:
            setup = ssa_op.setup_code
            # Replace precision-specific types in setup
            if precision == "double":
                setup = setup.replace('uint32_t', 'uint64_t')
            code += f"    {setup}\n"

        # Handle type conversion for precision
        if precision == "double" and ssa_op.result_type == "float":
            actual_type = "double"
        elif precision == "double" and ssa_op.result_type == "uint32_t":
            actual_type = "uint64_t"
        elif precision == "double" and ssa_op.result_type == "int32_t":
            actual_type = "int64_t"
        else:
            actual_type = ssa_op.result_type

        # Replace constants in RHS expression with precision-appropriate versions
        rhs_expr = ssa_op.rhs_expression
        for const_name in self.constants.keys():
            if const_name in rhs_expr:
                precision_const = self.substitute_constant_for_precision(const_name, precision)
                rhs_expr = rhs_expr.replace(const_name, precision_const)

        # Replace M constant with precision-appropriate version
        if 'M' in rhs_expr:
            if precision == "double":
                rhs_expr = rhs_expr.replace('M', 'MANTISSA_BITS_D')
            else:
                rhs_expr = rhs_expr.replace('M', 'MANTISSA_BITS')

        # Handle bit_cast operations for different precisions
        if 'std::bit_cast<uint32_t>' in rhs_expr and precision == "double":
            rhs_expr = rhs_expr.replace('std::bit_cast<uint32_t>', 'std::bit_cast<uint64_t>')

        # Generate the assignment and debug store
        code += f"    {actual_type} {ssa_op.result_ssa} = {rhs_expr};"
        code += f" // {ssa_op.original_line.strip()}\n"
        code += f"    dbg.{ssa_op.result_ssa} = {ssa_op.result_ssa};\n"
        return code
        
    def generate_f32_function(self, ssa_ops: List[SSAOperation]) -> str:
        struct_name = f"{self.kernel_type.title()}DebugValues"
        func_name = f"{self.kernel_type}_scalar_f32_debug"
        
        code = f"{struct_name} {func_name}(float v) {{\n"
        code += f"    {struct_name} dbg = {{}};\n"
        code += "    dbg.input_v = v;\n\n"
        
        code += "    // Variable declarations\n"
        code += self.generate_variable_declarations("float") + "\n\n"
        
        code += "    // SSA operations with debug instrumentation\n"
        for ssa_op in ssa_ops:
            code += self.generate_ssa_operation_code(ssa_op, "float")
        
        # Handle return value
        code += self.generate_return_handling("float")
        
        code += "\n    return dbg;\n"
        code += "}\n\n"
        return code

    def generate_f64_function(self, ssa_ops: List[SSAOperation]) -> str:
        struct_name = f"{self.kernel_type.title()}DebugValues"
        func_name = f"{self.kernel_type}_scalar_f64_debug"
        
        code = f"{struct_name} {func_name}(double v) {{\n"
        code += f"    {struct_name} dbg = {{}};\n"
        code += "    dbg.input_v = v;\n\n"
        
        code += "    // Variable declarations\n"
        code += self.generate_variable_declarations("double") + "\n\n"
        
        code += "    // SSA operations with debug instrumentation\n"
        for ssa_op in ssa_ops:
            code += self.generate_ssa_operation_code(ssa_op, "double")
        
        # Handle return value
        code += self.generate_return_handling("double")
        
        code += "\n    return dbg;\n"
        code += "}\n\n"
        return code
        
    def generate_return_handling(self, precision: str) -> str:
        """Generate kernel-specific return value handling"""
        code = "\n    // Handle final result\n"

        if self.kernel_type == 'exp':
            # exp returns: reinterpret int->float
            if 'ret' in [op.split('_f')[0] for op in self.ssa_variables.keys()]:
                # Find the latest ret version
                ret_version = max([int(var.split('_f')[1]) for var in self.ssa_variables.keys()
                                 if var.startswith('ret_f')])
                ret_var = f"ret_f{ret_version}"

                if precision == "float":
                    code += f"    float final_float;\n"
                    code += f"    memcpy(&final_float, &{ret_var}, sizeof(final_float));\n"
                    code += f"    dbg.final_result = final_float;\n"
                else:  # double
                    code += f"    double final_double;\n"
                    code += f"    memcpy(&final_double, &{ret_var}, sizeof(final_double));\n"
                    code += f"    dbg.final_result = final_double;\n"
            else:
                code += "    // No ret variable found for exp kernel\n"

        elif self.kernel_type == 'sin':
            # sin returns: reinterpret int->float with sign application
            if 'tmp' in [op.split('_f')[0] for op in self.ssa_variables.keys()]:
                # Find the latest tmp version (final reinterpreted result)
                tmp_versions = [int(var.split('_f')[1]) for var in self.ssa_variables.keys()
                               if var.startswith('tmp_f')]
                if tmp_versions:
                    tmp_version = max(tmp_versions)
                    tmp_var = f"tmp_f{tmp_version}"

                    if precision == "float":
                        code += f"    float final_float;\n"
                        code += f"    memcpy(&final_float, &{tmp_var}, sizeof(final_float));\n"
                        code += f"    dbg.final_result = final_float;\n"
                    else:  # double
                        code += f"    double final_double;\n"
                        code += f"    memcpy(&final_double, &{tmp_var}, sizeof(final_double));\n"
                        code += f"    dbg.final_result = final_double;\n"
                else:
                    code += "    // No tmp variable found for sin kernel\n"
            else:
                code += "    // No tmp variable found for sin kernel\n"
        else:
            code += f"    // Generic return handling for {self.kernel_type}\n"
            code += "    // dbg.final_result = /* determined from last operation */;\n"

        return code

    def generate_debug_print_function(self) -> str:
        """Generate a helper function to print all debug values"""
        struct_name = f"{self.kernel_type.title()}DebugValues"
        func_name = f"print_all_{self.kernel_type}_debug_values"

        code = f"""
// Comprehensive debug printing function for all {self.kernel_type} SSA variables
void {func_name}(const {struct_name}& f32_dbg, const {struct_name}& f64_dbg, const char* test_name = "test") {{
    auto print_row = [](const char *name, double f32v, double f64v) {{
        float a = static_cast<float>(f32v);
        float b = static_cast<float>(f64v);
        // Simple ULP difference calculation
        uint32_t ua, ub;
        std::memcpy(&ua, &a, sizeof(ua));
        std::memcpy(&ub, &b, sizeof(ub));
        uint32_t ulp = (ua > ub) ? (ua - ub) : (ub - ua);
        if ((ua ^ ub) & 0x80000000u) ulp = std::numeric_limits<uint32_t>::max(); // Different signs

        std::cout << std::left << std::setw(26) << name << "| "
                  << std::setw(20) << std::scientific << std::setprecision(8) << a << "| "
                  << std::setw(20) << b << "| ULP " << ulp << "\\n";
    }};

    std::cout << "\\n=== " << test_name << " - ALL {self.kernel_type.upper()} DEBUG VALUES ===\\n";
    std::cout << std::left << std::setw(26) << "Variable Name" << "| "
              << std::setw(20) << "f32 Value" << "| "
              << std::setw(20) << "f64 Value" << "| ULP Diff\\n";
    std::cout << std::string(80, '-') << "\\n";

    // Standard fields
    print_row("input_v", f32_dbg.input_v, f64_dbg.input_v);
    print_row("final_result", f32_dbg.final_result, f64_dbg.final_result);
"""

        # Add kernel-specific fields
        if self.kernel_type == 'sin':
            code += "    print_row(\"final_result_before_sign\", f32_dbg.final_result_before_sign, f64_dbg.final_result_before_sign);\n"

        # Add all SSA variables
        ssa_vars = sorted(self.ssa_variables.items())
        for var_name, var_type in ssa_vars:
            if var_type == "bool":
                code += f"    print_row(\"{var_name}\", f32_dbg.{var_name} ? 1.0 : 0.0, f64_dbg.{var_name} ? 1.0 : 0.0);\n"
            else:
                code += f"    print_row(\"{var_name}\", f32_dbg.{var_name}, f64_dbg.{var_name});\n"

        code += "    std::cout << std::string(80, '=') << \"\\n\\n\";\n"
        code += "}\n\n"
        return code

    def generate_header(self, ssa_ops: List[SSAOperation]) -> str:
        header = f"""#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <bit>

// Generated by convert_rvv_to_scalar.py
// Scalar versions of RISC-V vector {self.kernel_type} implementation for precision analysis

"""
        header += self.generate_constant_definitions()
        header += self.generate_kernel_debug_struct()
        header += self.generate_f32_function(ssa_ops)
        header += self.generate_f64_function(ssa_ops)
        header += self.generate_debug_print_function()
        return header

    def convert_file(self, input_file: str, output_file: str):
        try:
            with open(input_file, "r") as f:
                content = f.read()

            # Parse constants from #define statements
            self.constants = self.parse_constants_from_content(content)
            print(f"Parsed {len(self.constants)} constants: {list(self.constants.keys())}")

            # Detect kernel type
            self.kernel_type = self.detect_kernel_from_content(content)
            print(f"Detected kernel type: {self.kernel_type}")

            # Parse macro definition
            ops = self.parse_macro_definition(content)
            print(f"Parsed {len(ops)} operations from macro")

            # Discover variables and apply SSA transform
            self.variables = self.discover_variables_from_operations(ops)
            print(f"Discovered {len(self.variables)} base variables: {list(self.variables.keys())}")

            ssa_ops = self.apply_ssa_transform(ops)
            print(f"Generated {len(ssa_ops)} SSA operations")
            print(f"SSA variables: {list(self.ssa_variables.keys())}")

            # Generate header
            header = self.generate_header(ssa_ops)
            with open(output_file, "w") as f:
                f.write(header)

            print(f"Generated {self.kernel_type} scalar functions in {output_file}")
            print(f"Struct: {self.kernel_type.title()}DebugValues with {len(self.ssa_variables)} SSA fields")

        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    import os
    parser = argparse.ArgumentParser(description="Convert RISC-V vector intrinsics to scalar C++ functions with SSA debug")
    parser.add_argument("-i", "--input", required=True, help="Input file containing RISC-V macro definition")
    parser.add_argument("-o", "--output", required=False, help="Output C++ header file (optional)")
    parser.add_argument("--config", default="rvv_conversion_config.json", help="Configuration file")
    parser.add_argument("--kernel", help="Override kernel type detection")
    args = parser.parse_args()

    # Derive a sensible default output if not provided
    out = args.output
    if not out:
        base = os.path.basename(args.input).lower()
        # Heuristic: name by known kernel type
        if "exp" in base:
            out_name = "exp_scalar_functions.h"
        elif "sin" in base:
            out_name = "sin_scalar_functions.h"
        elif "cos" in base:
            out_name = "cos_scalar_functions.h"
        else:
            # Default to sin when ambiguous
            out_name = "sin_scalar_functions.h"
        out = os.path.join(os.path.dirname(args.input) or ".", out_name)

    converter = RVVConverter(args.config)
    if args.kernel:
        converter.kernel_type = args.kernel
    converter.convert_file(args.input, out)


if __name__ == "__main__":
    main()