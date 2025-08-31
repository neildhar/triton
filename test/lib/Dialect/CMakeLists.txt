add_triton_library(TritonTestDialect
  TestLoopPeeling.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonTransforms
)
