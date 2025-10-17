add_library(TritonTestAnalysis STATIC
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestMembar.cpp
)

target_compile_options(TritonTestAnalysis PRIVATE -fno-rtti)
target_link_libraries(TritonTestAnalysis PUBLIC triton)
