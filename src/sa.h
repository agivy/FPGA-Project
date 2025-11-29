#ifndef SA_H_
#define SA_H_

#include <tapa.h>
#include <cstdint>

const int M_DIM = 128;
const int K_DIM = 4096;
const int N_DIM = 512;

const int PE_ROWS = 16;
const int PE_COLS = 16;
const int GROUP_SIZE = 16;

// 4MB cache split intelligently
// Activation cache: 1MB → 16 M-tiles (128×512 = 128KB, can fit all!)
// Weight cache: 3MB → 24 N-tiles (24×16×512 = 192KB)
const int ACT_CACHE_SIZE = 8;   // Cache ALL 8 M-tiles (only 64KB!)
const int WGT_CACHE_SIZE = 24;  // Cache 24 out of 32 N-tiles

inline int8_t unpack_dequantize_weight(
    uint8_t packed_byte,
    bool is_upper,
    uint8_t scale_factor
) {
    #pragma HLS INLINE
    int8_t w_4bit = is_upper ? ((packed_byte >> 4) & 0x0F) : (packed_byte & 0x0F);
    if (w_4bit & 0x08) w_4bit |= 0xF0;
    uint8_t shift_amount = (scale_factor & 0x3) * 2;
    return w_4bit << shift_amount;
}

void SystolicArrayKernel(
    tapa::mmap<int8_t> activations,
    tapa::mmap<uint8_t> weights_packed,
    tapa::mmap<uint8_t> scales,
    tapa::mmap<int32_t> result,
    int M, int K, int N
);

#endif
