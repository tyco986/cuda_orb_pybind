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

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda_orb_cpp)
set(cuda_sources
  ${SRC_DIR}/orbd.cu
  ${SRC_DIR}/orbd.h
  ${SRC_DIR}/orb.cpp
  ${SRC_DIR}/orb.h
  ${SRC_DIR}/orb_structures.h
  ${SRC_DIR}/cuda_utils.h
)

set(sources
  ${CPP_DIR}/orb_aligner.cpp
)

include_directories(${SRC_DIR} ${CPP_DIR})

# NVML for GPU peak monitoring
set(NVML_LIBRARY "")
if(EXISTS "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
  set(NVML_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
endif()
if(NOT NVML_LIBRARY)
  find_library(NVML_LIBRARY nvidia-ml
    PATHS /usr/local/cuda/targets/x86_64-linux/lib/stubs
          /usr/local/cuda-12/targets/x86_64-linux/lib/stubs)
endif()
find_path(NVML_INCLUDE_DIR nvml.h
  PATHS /usr/local/cuda/targets/x86_64-linux/include
        /usr/local/cuda/include
        /usr/include)
if(NVML_LIBRARY AND NVML_INCLUDE_DIR)
  message(STATUS "Found NVML: ${NVML_LIBRARY}")
  set(HAVE_NVML 1)
else()
  message(STATUS "NVML not found, GPU peak monitoring disabled")
  set(HAVE_NVML 0)
endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuda_orb ${cuda_sources} ${sources}
  ${CPP_DIR}/test_orb_aligner.cpp
  OPTIONS -arch=sm_75)

set_target_properties(cuda_orb PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS} -DHAVE_NVML=${HAVE_NVML}"
)

target_link_libraries(cuda_orb ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})
if(NVML_LIBRARY)
  target_link_libraries(cuda_orb ${NVML_LIBRARY})
  target_include_directories(cuda_orb PRIVATE ${NVML_INCLUDE_DIR})
endif()

# Python module (pybind11 + scikit-build)
if(pybind11_FOUND AND CUDA_FOUND)
  cuda_add_library(orb_core STATIC ${SRC_DIR}/orbd.cu ${SRC_DIR}/orb.cpp OPTIONS -arch=sm_75)
  set_target_properties(orb_core PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
  target_include_directories(orb_core PUBLIC ${SRC_DIR})

  pybind11_add_module(_cuda_orb ${SRC_DIR}/orb_bindings.cpp)
  target_link_libraries(_cuda_orb PRIVATE orb_core)
  target_include_directories(_cuda_orb PRIVATE ${SRC_DIR})
  install(TARGETS _cuda_orb LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_orb/__init__.py DESTINATION .)
endif()