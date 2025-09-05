#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @init_barrier_for_tlx() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32

    // Create barriers with different arrival counts  
    %cst1 = arith.constant dense<0> : tensor<1xi64, #blocked0>
    %alloc1 = ttg.local_alloc %cst1 : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc1, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>

    %cst2 = arith.constant dense<0> : tensor<1xi64, #blocked0>
    %alloc2 = ttg.local_alloc %cst2 : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>  
    ttng.init_barrier %alloc2, 128 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>

    // GPU barrier for synchronization
    gpu.barrier

    tt.return
  }
}
