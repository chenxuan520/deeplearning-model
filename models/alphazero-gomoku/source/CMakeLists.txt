cmake_minimum_required(VERSION 3.10)
project(alphazero_gomoku CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_compile_options(-O2 -Wall)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- foundation library (copied from deeplearning repo WIP components) ----
file(GLOB LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/cnn/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/loss/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/optimizer/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/util/*.cpp)
add_library(azcore STATIC ${LIB_SOURCES})

# ---- game / mcts / train library ----
file(GLOB GAME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp)
add_library(azgame STATIC ${GAME_SOURCES})
target_link_libraries(azgame azcore pthread)

# ---- main CLI ----
add_executable(alphazero ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(alphazero azgame azcore pthread)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# ---- tests ----
add_executable(test_az ${CMAKE_CURRENT_SOURCE_DIR}/test/test_main.cpp)
target_link_libraries(test_az azgame azcore pthread)

# ---- browser-port parity probe (not used by training) ----
add_executable(az_model_probe ${CMAKE_CURRENT_SOURCE_DIR}/tools/model_probe.cpp)
target_link_libraries(az_model_probe azgame azcore pthread)
