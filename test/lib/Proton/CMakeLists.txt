add_triton_library(TritonTestProton
  TestScopeIdAllocation.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
