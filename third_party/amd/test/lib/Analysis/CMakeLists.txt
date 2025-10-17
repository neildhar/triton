add_library(TritonAMDGPUTestAnalysis STATIC
  TestAMDRangeAnalysis.cpp
  TestAMDGPUMembar.cpp
  TestAxisInfo.cpp
)

add_dependencies(TritonAMDGPUTestAnalysis
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)

target_compile_options(TritonAMDGPUTestAnalysis PRIVATE -fno-rtti)
target_link_libraries(TritonAMDGPUTestAnalysis PUBLIC triton)
