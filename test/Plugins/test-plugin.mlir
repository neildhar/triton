// RUN: TRITON_PASS_PLUGIN_PATH=/Users/neildhar/build_dbg/test/lib/Plugins/libTritonPluginsTestLib.dylib triton-opt -tritongpu-plugin %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

  // CHECK: func @foo()
  tt.func @bar() {
    tt.return
  }
}  // module
