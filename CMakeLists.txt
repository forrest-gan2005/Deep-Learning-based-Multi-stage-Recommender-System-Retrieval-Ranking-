cmake_minimum_required(VERSION 3.10)
project(RecEngine_DeepFM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/pytorch") 
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

add_executable(deepfm_inference src/cpp/main.cpp)

target_link_libraries(deepfm_inference ${TORCH_LIBRARIES})

if(APPLE)
    target_compile_definitions(deepfm_inference PRIVATE USE_MPS)
endif()