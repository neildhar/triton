add_library(TritonTestProton STATIC
  TestScopeIdAllocation.cpp
)

target_compile_options(TritonTestProton PRIVATE -fno-rtti)
target_link_libraries(TritonTestProton PUBLIC triton)
