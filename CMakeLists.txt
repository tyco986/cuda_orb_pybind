cmake_minimum_required(VERSION 3.18)

project(CudaOrb LANGUAGES CXX CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(pybind11 CONFIG QUIET)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT;-Xcompiler=-fPIC") 
  endif()
endif()

set(cuda_sources
  orbd.cu
  orbd.h
  orb.cpp
  orb.h
  orb_structures.h
  cuda_utils.h
  warmup.cu
  warmup.h
)  

set(sources
  main.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cudaorb ${cuda_sources} ${sources} OPTIONS -arch=sm_75)

set_target_properties(cudaorb PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cudaorb ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Determinism test executable
set(test_sources determinism_test/test_determinism.cpp)
cuda_add_executable(test_determinism ${cuda_sources} ${test_sources} OPTIONS -arch=sm_75)
set_target_properties(test_determinism PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
target_link_libraries(test_determinism ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Homography test: CUDA ORB vs OpenCV ORB + findHomography
set(homography_sources homography_test/homography_test.cpp)
cuda_add_executable(homography_test ${cuda_sources} ${homography_sources} OPTIONS -arch=sm_75)
set_target_properties(homography_test PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
target_link_libraries(homography_test ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Python module (pybind11 + scikit-build)
if(pybind11_FOUND AND CUDA_FOUND)
  cuda_add_library(orb_core STATIC orbd.cu orb.cpp warmup.cu OPTIONS -arch=sm_75)
  set_target_properties(orb_core PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
  target_include_directories(orb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  pybind11_add_module(_cuda_orb src/orb_bindings.cpp)
  target_link_libraries(_cuda_orb PRIVATE orb_core)
  target_include_directories(_cuda_orb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS _cuda_orb LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_orb/__init__.py DESTINATION .)
endif()