#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <gflags/gflags.h>

#include "sa.h"

using std::cout;
using std::endl;
using std::vector;

DEFINE_string(bitstream, "", "path to bitstream");

const int M = M_DIM;
const int K = K_DIM;
const int N = N_DIM;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

// Quantize weights to MXINT4 with group-wise scaling
void quantize_mxint4(
    const vector<float>& weights_fp32,
    aligned_vector<uint8_t>& weights_packed,
    aligned_vector<uint8_t>& scales,
    int K, int N
) {
    int total_weights = K * N;
    weights_packed.resize(total_weights / 2);
    scales.resize(total_weights / GROUP_SIZE);
    
    // Process each group
    for (int grp = 0; grp < total_weights / GROUP_SIZE; grp++) {
        int base_idx = grp * GROUP_SIZE;
        
        // Find max absolute value in group
        float max_abs = 0.0f;
        for (int i = 0; i < GROUP_SIZE; i++) {
            max_abs = std::max(max_abs, std::fabs(weights_fp32[base_idx + i]));
        }
        
        // Compute scale (shift amount for 4-bit range)
        int shift = 0;
        if (max_abs > 0.0f) {
            shift = (int)std::floor(std::log2(max_abs)) - 3;
            shift = std::max(0, std::min(3, shift));  // Limit to [0, 3] -> shift by 0,2,4,6
        }
        
        // Store scale (only Sw[1:0] used for now)
        scales[grp] = (uint8_t)shift;
        
        // Quantize weights in group
        float scale_val = std::pow(2.0f, shift * 2);  // shift * 2 because Sw[1:0]*2
        
        for (int i = 0; i < GROUP_SIZE; i += 2) {
            int idx0 = base_idx + i;
            int idx1 = base_idx + i + 1;
            
            // Quantize to 4-bit
            int8_t w0 = (int8_t)std::round(weights_fp32[idx0] / scale_val);
            int8_t w1 = (int8_t)std::round(weights_fp32[idx1] / scale_val);
            
            // Clamp to 4-bit signed range [-8, 7]
            w0 = std::max((int8_t)-8, std::min((int8_t)7, w0));
            w1 = std::max((int8_t)-8, std::min((int8_t)7, w1));
            
            // Pack: lower 4 bits = w0, upper 4 bits = w1
            uint8_t packed = (w0 & 0x0F) | ((w1 & 0x0F) << 4);
            weights_packed[idx0 / 2] = packed;
        }
    }
}

// CPU reference with MXINT4 dequantization
void cpu_reference(
    const aligned_vector<int8_t>& act,
    const aligned_vector<uint8_t>& wgt_packed,
    const aligned_vector<uint8_t>& scales,
    aligned_vector<int32_t>& out,
    int M, int K, int N
) {
    out.resize(M * N, 0);
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            
            for (int k = 0; k < K; k++) {
                int8_t a = act[m * K + k];
                
                // Get weight
                int w_idx = k * N + n;
                int packed_idx = w_idx / 2;
                uint8_t packed = wgt_packed[packed_idx];
                
                // Get scale
                int scale_idx = w_idx / GROUP_SIZE;
                uint8_t scale_factor = scales[scale_idx];
                
                // Dequantize
                bool is_upper = (w_idx % 2) == 1;
                int8_t w = unpack_dequantize_weight(packed, is_upper, scale_factor);
                
                sum += (int32_t)a * (int32_t)w;
            }
            
            out[m * N + n] = sum;
        }
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    cout << "16x16 Systolic Array with MXINT4" << endl;
    cout << "M=" << M << ", K=" << K << ", N=" << N << endl;
    cout << "GFLOPs: " << (2.0 * M * K * N / 1e9) << endl;
    
    // Generate test data
    vector<float> act_fp32(M * K);
    vector<float> wgt_fp32(K * N);
    
    for (int i = 0; i < M * K; i++) {
        act_fp32[i] = ((i % 17) - 8) / 8.0f;  // Range ~[-1, 1]
    }
    for (int i = 0; i < K * N; i++) {
        wgt_fp32[i] = ((i % 19) - 9) / 9.0f;  // Range ~[-1, 1]
    }
    
    // Quantize activations to INT8
    aligned_vector<int8_t> act_int8(M * K);
    for (int i = 0; i < M * K; i++) {
        float val = act_fp32[i] * 127.0f;
        act_int8[i] = (int8_t)std::max(-127.0f, std::min(127.0f, val));
    }
    
    // Quantize weights to MXINT4
    aligned_vector<uint8_t> wgt_packed;
    aligned_vector<uint8_t> scales;
    quantize_mxint4(wgt_fp32, wgt_packed, scales, K, N);
    
    cout << "Quantized data:" << endl;
    cout << "  Activations: " << act_int8.size() << " INT8" << endl;
    cout << "  Weights: " << wgt_packed.size() << " bytes (MXINT4 packed)" << endl;
    cout << "  Scales: " << scales.size() << " factors" << endl;
    
    // Allocate output
    aligned_vector<int32_t> out_hw(M * N);
    aligned_vector<int32_t> out_cpu;
    
    // CPU reference
    cout << "\nRunning CPU reference..." << endl;
    cpu_reference(act_int8, wgt_packed, scales, out_cpu, M, K, N);
    
    // Run accelerator
    cout << "Running accelerator..." << endl;
    tapa::invoke(
        SystolicArrayKernel,
        FLAGS_bitstream,
        tapa::read_only_mmap<int8_t>(act_int8),
        tapa::read_only_mmap<uint8_t>(wgt_packed),
        tapa::read_only_mmap<uint8_t>(scales),
        tapa::write_only_mmap<int32_t>(out_hw),
        M, K, N
    );
    
    // Verify
    cout << "\nFirst 10 results:" << endl;
    cout << "Index\tHW\tCPU\tDiff" << endl;
    int errors = 0;
    for (int i = 0; i < std::min(10, M * N); i++) {
        int diff = out_hw[i] - out_cpu[i];
        cout << i << "\t" << out_hw[i] << "\t" << out_cpu[i] << "\t" << diff << endl;
        if (diff != 0) errors++;
    }
    
    // Check all
    for (int i = 0; i < M * N; i++) {
        if (out_hw[i] != out_cpu[i]) errors++;
    }
    
    cout << "\nErrors: " << errors << " / " << (M * N) << endl;
    
    if (errors == 0) {
        cout << "PASS!" << endl;
        return 0;
    } else {
        cout << "FAIL!" << endl;
        return 1;
    }
}
