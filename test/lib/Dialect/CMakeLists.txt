add_library(TritonTestDialect STATIC
  TestLoopPeeling.cpp
)

target_compile_options(TritonTestDialect PRIVATE -fno-rtti)
target_link_libraries(TritonTestDialect PUBLIC triton)
