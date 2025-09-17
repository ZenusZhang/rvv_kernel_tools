#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Kernel selection via compile definitions:
// -DKERNEL_SIN or -DKERNEL_EXP

#if defined(KERNEL_SIN)
#include "sin_scalar_functions.h"
using DebugValues = SinDebugValues;
static inline float golden_fn(float x) { return std::sinf(x); }
static constexpr const char *kKernelName = "sin";
#elif defined(KERNEL_EXP)
#define M 23 // mantissa bits for float if generator uses it
#include "exp_scalar_functions.h"
using DebugValues = ExpDebugValues;
static inline float golden_fn(float x) { return std::exp(x); }
static constexpr const char *kKernelName = "exp";
#else
#error "Define one of KERNEL_SIN or KERNEL_EXP to choose the kernel under test."
#endif

// Forward declarations for debug-call shims provided by generated headers
#if defined(KERNEL_SIN)
extern SinDebugValues sin_scalar_f32_debug(float v);
extern SinDebugValues sin_scalar_f64_debug(double v);
#elif defined(KERNEL_EXP)
extern ExpDebugValues exp_scalar_f32_debug(float v);
extern ExpDebugValues exp_scalar_f64_debug(double v);
#endif

// Minimal JSON config loader for simple flat configs.
struct Config {
    std::string golden = ""; // e.g., "sinf" or "expf" (informational)
    double threshold_ulp = 2.0;
    double range_start = 0.0;
    double range_end = 0.0;
    double range_step = 0.1;
    bool include_specials = false;        // optional (sin)
    bool include_large_multiples = false; // optional (sin)
};

static inline std::string read_text_file(const std::string &path) {
    std::ifstream ifs(path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static bool extract_string(const std::string &json, const std::string &key, std::string &out) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return false;
    ++pos;
    auto end = json.find('"', pos);
    if (end == std::string::npos) return false;
    out = json.substr(pos, end - pos);
    return true;
}

static bool extract_bool(const std::string &json, const std::string &key, bool &out) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (json.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static bool extract_number(const std::string &json, const std::string &key, double &out) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    size_t end = pos;
    while (end < json.size()) {
        char c = json[end];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
            ++end;
        } else {
            break;
        }
    }
    if (end == pos) return false;
    out = std::strtod(json.c_str() + pos, nullptr);
    return true;
}

static bool load_config(const std::string &path, Config &cfg) {
    const auto txt = read_text_file(path);
    if (txt.empty()) return false;
    extract_string(txt, "golden", cfg.golden);
    extract_number(txt, "threshold_ulp", cfg.threshold_ulp);
    extract_number(txt, "range_start", cfg.range_start);
    extract_number(txt, "range_end", cfg.range_end);
    extract_number(txt, "range_step", cfg.range_step);
    extract_bool(txt, "include_specials", cfg.include_specials);
    extract_bool(txt, "include_large_multiples", cfg.include_large_multiples);
    return true;
}

// ULP helpers for float
static inline uint32_t float_bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline uint32_t ulp_diff_float(float a, float b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) return std::numeric_limits<uint32_t>::max();
    uint32_t ia = float_bits(a), ib = float_bits(b);
    // If signs differ, treat as large difference
    if ((ia ^ ib) & 0x80000000u) return std::numeric_limits<uint32_t>::max();
    return ia > ib ? ia - ib : ib - ia;
}

// Test input generation
static std::vector<float> make_test_inputs(const Config &cfg) {
    std::vector<float> xs;
    // Range sweep
    for (double v = cfg.range_start; v <= cfg.range_end + 1e-12; v += std::max(cfg.range_step, 1e-12)) {
        xs.push_back(static_cast<float>(v));
    }

    // Optional specials for sin
    if (cfg.include_specials && std::string(kKernelName) == "sin") {
        const float pi = static_cast<float>(M_PI);
        xs.push_back(0.0f);
        xs.push_back(-0.0f);
        xs.push_back(pi / 2.0f);
        xs.push_back(pi);
        xs.push_back(3.0f * pi / 2.0f);
        xs.push_back(2.0f * pi);
        xs.push_back(pi - 1e-6f);
        xs.push_back(pi + 1e-6f);
    }

    if (cfg.include_large_multiples && std::string(kKernelName) == "sin") {
        const float pi = static_cast<float>(M_PI);
        for (float v = 8.0f * pi; v < 100.0f * pi; v += pi / 3.0f) xs.push_back(v);
    }
    return xs;
}

// Compare and optionally print details
static bool compare_and_report(float x, double ulp_threshold, const Config &cfg) {
#if defined(KERNEL_SIN)
    auto f32 = sin_scalar_f32_debug(x);
    auto f64 = sin_scalar_f64_debug(static_cast<double>(x));
    const float golden = golden_fn(x);

    const uint32_t ulp_vs_f64 = ulp_diff_float(static_cast<float>(f32.final_result), static_cast<float>(f64.final_result));
    const uint32_t ulp_vs_gld = ulp_diff_float(static_cast<float>(f32.final_result), golden);
    const bool exceeds = (ulp_vs_f64 > ulp_threshold) || (ulp_vs_gld > ulp_threshold);
    if (!exceeds) return false;

    std::cout << "\n=== PROBLEMATIC CASE (" << kKernelName << ") x = " << std::scientific << std::setprecision(8) << x << " ===\n";
    std::cout << "ULP vs f64: " << ulp_vs_f64 << ", ULP vs golden(" << (cfg.golden.empty() ? "sinf" : cfg.golden) << "): " << ulp_vs_gld
              << " (threshold: " << ulp_threshold << ")\n";

    auto print_row = [](const char *name, double f32v, double f64v) {
        float a = static_cast<float>(f32v);
        float b = static_cast<float>(f64v);
        uint32_t ulp = ulp_diff_float(a, b);
        std::cout << std::left << std::setw(26) << name << "| "
                  << std::setw(20) << std::scientific << std::setprecision(8) << a << "| "
                  << std::setw(20) << b << "| ULP " << ulp << "\n";
    };

    std::cout << "\n=== COMPREHENSIVE DEBUG VALUES (f32 vs f64) ===\n";
    std::cout << std::left << std::setw(26) << "Variable Name" << "| "
              << std::setw(20) << "f32 Value" << "| "
              << std::setw(20) << "f64 Value" << "| ULP Diff\n";
    std::cout << std::string(80, '-') << "\n";

    // Print ALL debug values available in SinDebugValues
    print_row("input_v", f32.input_v, f64.input_v);
    print_row("final_result", f32.final_result, f64.final_result);

    // Print all SSA variables (assuming sin struct has these fields)
    // Note: These field names should match what's generated by the conversion script
    // We'll use a macro-like approach to print all available fields

    // Common pattern: print all fields that exist in the debug struct
    #define PRINT_IF_EXISTS(field) \
        if (&f32.field != nullptr && &f64.field != nullptr) print_row(#field, f32.field, f64.field);

    // Based on common sin implementation patterns, print likely SSA variables
    print_row("r_abs", f32.r_abs, f64.r_abs);
    print_row("n_unrounded", f32.n_unrounded, f64.n_unrounded);
    print_row("r_reduced", f32.r_reduced, f64.r_reduced);
    print_row("r_prime", f32.r_prime, f64.r_prime);
    print_row("r2", f32.r2, f64.r2);
    print_row("poly_result", f32.poly_result, f64.poly_result);

    std::cout << "\nFinals:\n";
    std::cout << std::left << std::setw(26) << "Our f32" << "| " << std::setw(20)
              << static_cast<float>(f32.final_result) << "|\n";
    std::cout << std::left << std::setw(26) << "Our f64" << "| " << std::setw(20)
              << static_cast<float>(f64.final_result) << "|\n";
    std::cout << std::left << std::setw(26) << (cfg.golden.empty() ? "sinf" : cfg.golden.c_str()) << "| " << std::setw(20) << golden << "|\n";
    return true;
#elif defined(KERNEL_EXP)
    auto f32 = exp_scalar_f32_debug(x);
    auto f64 = exp_scalar_f64_debug(static_cast<double>(x));
    const float golden = golden_fn(x);
    const uint32_t ulp_vs_f64 = ulp_diff_float(static_cast<float>(f32.final_result), static_cast<float>(f64.final_result));
    const uint32_t ulp_vs_gld = ulp_diff_float(static_cast<float>(f32.final_result), golden);
    const bool exceeds = (ulp_vs_f64 > ulp_threshold) || (ulp_vs_gld > ulp_threshold);
    if (!exceeds) return false;

    std::cout << "\n=== PROBLEMATIC CASE (" << kKernelName << ") x = " << std::scientific << std::setprecision(8) << x << " ===\n";
    std::cout << "ULP vs f64: " << ulp_vs_f64 << ", ULP vs golden(" << (cfg.golden.empty() ? "expf" : cfg.golden) << "): " << ulp_vs_gld
              << " (threshold: " << ulp_threshold << ")\n";

    // Use the generated comprehensive debug print function
    char test_case_name[64];
    snprintf(test_case_name, sizeof(test_case_name), "x=%.8e", x);
    print_all_exp_debug_values(f32, f64, test_case_name);
    return true;
#endif
}

int main(int argc, char **argv) {
    // Parse args
    std::string config_path =
#if defined(KERNEL_SIN)
        "rvv_kernel_tools/sin_test_config.json";
#else
        "../exp_test_config.json";
#endif
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    Config cfg{};
    if (!load_config(config_path, cfg)) {
        std::cerr << "Failed to load config: " << config_path << "\n";
        return 2;
    }

    std::cout << "Kernel: " << kKernelName << ", golden: " << (cfg.golden.empty() ? "(auto)" : cfg.golden)
              << ", threshold_ulp: " << cfg.threshold_ulp << "\n";
    std::cout << "Range: [" << cfg.range_start << ", " << cfg.range_end << "] step " << cfg.range_step << "\n";

    const auto xs = make_test_inputs(cfg);
    int total = 0, problematic = 0;
    for (float x : xs) {
        ++total;
        if (compare_and_report(x, cfg.threshold_ulp, cfg)) ++problematic;
    }

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total cases: " << total << "\n";
    std::cout << "Problematic (> " << cfg.threshold_ulp << " ULP): " << problematic
              << " (" << (total ? (100.0 * problematic / total) : 0.0) << "%)\n";
    return 0;
}
