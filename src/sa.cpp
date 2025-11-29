#include "sa.h"

// ============================================================================
// SMART CACHE: 4MB with CONSTANT loop bounds
// ============================================================================

static int8_t A_cache[8][PE_ROWS][K_DIM];
static int8_t W_cache[32][PE_COLS][K_DIM];  // *** CHANGED: Cache ALL 32 N-tiles ***

static int8_t A_work[PE_ROWS][K_DIM];
static int8_t W_work[PE_COLS][K_DIM];
static int32_t C_work[PE_ROWS][PE_COLS];

void SystolicArrayKernel(
    tapa::mmap<int8_t> activations,
    tapa::mmap<uint8_t> weights_packed,
    tapa::mmap<uint8_t> scales,
    tapa::mmap<int32_t> result,
    int M, int K, int N
) {
    #pragma HLS INLINE off
    
    // ---- Array Partitioning ----
    #pragma HLS ARRAY_PARTITION variable=A_cache complete dim=2
    #pragma HLS ARRAY_PARTITION variable=W_cache complete dim=2
    #pragma HLS ARRAY_PARTITION variable=A_work complete dim=1
    #pragma HLS ARRAY_PARTITION variable=W_work complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_work complete dim=0
    
    #pragma HLS BIND_STORAGE variable=A_cache type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=W_cache type=RAM_2P impl=URAM
    #pragma HLS BIND_STORAGE variable=A_work type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=W_work type=RAM_2P impl=BRAM
    
    const int NUM_M_TILES = M / PE_ROWS;  // 8
    const int NUM_N_TILES = N / PE_COLS;  // 32
    
    // ============================================================================
    // PHASE 1: LOAD ALL ACTIVATIONS (64KB)
    // ============================================================================
    load_all_act: for (int m_tile = 0; m_tile < NUM_M_TILES; ++m_tile) {
        // #pragma HLS loop_tripcount min=8 max=8 avg=8
        #pragma HLS loop_tripcount min=M_DIM/PE_ROWS max=M_DIM/PE_ROWS avg=M_DIM/PE_ROWS
        
        load_act_tile: for (int idx = 0; idx < PE_ROWS * K; ++idx) {
            #pragma HLS PIPELINE II=1
            // #pragma HLS loop_tripcount min=8192 max=8192 avg=8192
            #pragma HLS loop_tripcount min=PE_ROWS*K_DIM max=PE_ROWS*K_DIM avg=PE_ROWS*K_DIM
            
            int i = idx / K;
            int k = idx % K;
            int m_idx = m_tile * PE_ROWS + i;
            A_cache[m_tile][i][k] = activations[m_idx * K + k];
        }
    }
    
    // ============================================================================
    // PHASE 2: LOAD ALL WEIGHTS (256KB with dequantization)
    // *** LOAD ONCE, REUSE FOREVER ***
    // ============================================================================
    load_all_wgt: for (int n_tile = 0; n_tile < NUM_N_TILES; ++n_tile) {
        // #pragma HLS loop_tripcount min=32 max=32 avg=32
        #pragma HLS loop_tripcount min=N_DIM/PE_COLS max=N_DIM/PE_COLS avg=N_DIM/PE_COLS
        
        load_wgt_tile: for (int idx = 0; idx < PE_COLS * K; ++idx) {
            #pragma HLS PIPELINE II=1
            // #pragma HLS loop_tripcount min=8192 max=8192 avg=8192
            #pragma HLS loop_tripcount min=PE_COLS*K_DIM max=PE_COLS*K_DIM avg=PE_COLS*K_DIM
            
            int j = idx / K;
            int k = idx % K;
            int n_idx = n_tile * PE_COLS + j;
            int w_linear = k * N + n_idx;
            
            uint8_t packed_byte = weights_packed[w_linear / 2];
            uint8_t scale_factor = scales[w_linear / GROUP_SIZE];
            bool is_upper = (w_linear & 1);
            
            W_cache[n_tile][j][k] = unpack_dequantize_weight(
                packed_byte, is_upper, scale_factor
            );
        }
    }
    
    // ============================================================================
    // PHASE 3: COMPUTE ALL TILES (everything on-chip now!)
    // ============================================================================
    m_loop: for (int m_tile = 0; m_tile < NUM_M_TILES; ++m_tile) {
        // #pragma HLS loop_tripcount min=8 max=8 avg=8
        #pragma HLS loop_tripcount min=M_DIM/PE_ROWS max=M_DIM/PE_ROWS avg=M_DIM/PE_ROWS
        
        n_loop: for (int n_tile = 0; n_tile < NUM_N_TILES; ++n_tile) {
            // #pragma HLS loop_tripcount min=32 max=32 avg=32
            #pragma HLS loop_tripcount min=N_DIM/PE_COLS max=N_DIM/PE_COLS avg=N_DIM/PE_COLS
            
            // ---- COPY TO WORKING BUFFERS ----
            copy_to_work: for (int k = 0; k < K; ++k) {
                #pragma HLS PIPELINE II=1
                // #pragma HLS loop_tripcount min=512 max=512 avg=512
                #pragma HLS loop_tripcount min=K_DIM max=K_DIM avg=K_DIM
                
                for (int i = 0; i < PE_ROWS; ++i) {
                    #pragma HLS UNROLL
                    A_work[i][k] = A_cache[m_tile][i][k];
                }
                
                for (int j = 0; j < PE_COLS; ++j) {
                    #pragma HLS UNROLL
                    W_work[j][k] = W_cache[n_tile][j][k];
                }
            }
            
            // ---- INITIALIZE OUTPUT ----
            for (int i = 0; i < PE_ROWS; ++i) {
                #pragma HLS UNROLL
                for (int j = 0; j < PE_COLS; ++j) {
                    #pragma HLS UNROLL
                    C_work[i][j] = 0;
                }
            }
            
            // ---- COMPUTE: 16×16 SYSTOLIC ARRAY ----
            compute: for (int k = 0; k < K; ++k) {
                #pragma HLS PIPELINE II=1
                // #pragma HLS loop_tripcount min=512 max=512 avg=512
                #pragma HLS loop_tripcount min=K_DIM max=K_DIM avg=K_DIM
                #pragma HLS DEPENDENCE variable=C_work inter false
                
                for (int i = 0; i < PE_ROWS; ++i) {
                    #pragma HLS UNROLL
                    for (int j = 0; j < PE_COLS; ++j) {
                        #pragma HLS UNROLL
                        int8_t a = A_work[i][k];
                        int8_t w = W_work[j][k];
                        C_work[i][j] += (int32_t)a * (int32_t)w;
                    }
                }
            }
            
            // ---- WRITE OUTPUT ----
            write_output: for (int idx = 0; idx < PE_ROWS * PE_COLS; ++idx) {
                #pragma HLS PIPELINE II=1
                // #pragma HLS loop_tripcount min=256 max=256 avg=256
                #pragma HLS loop_tripcount min=PE_ROWS*PE_COLS max=PE_ROWS*PE_COLS avg=PE_ROWS*PE_COLS
                
                int i = idx / PE_COLS;
                int j = idx % PE_COLS;
                
                int m_idx = m_tile * PE_ROWS + i;
                int n_idx = n_tile * PE_COLS + j;
                result[m_idx * N + n_idx] = C_work[i][j];
            }
        }
    }
}
