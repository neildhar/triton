add_triton_library(GluonTransforms
  Canonicalize.cpp
  Inline.cpp
  ResolveAutoEncodings.cpp
  SimplifyControlFlow.cpp

  DEPENDS
  GluonTransformsIncGen
)
