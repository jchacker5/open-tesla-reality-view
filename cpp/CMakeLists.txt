cmake_minimum_required(VERSION 3.20)
project(otrv_core VERSION 0.1.0 LANGUAGES CXX)

# Open Tesla Reality View: C++ core.
#   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
# The Python extension (otrv_native) is built by scikit-build-core: `uv sync --extra native`.

option(OTRV_BUILD_TESTS "Build the C++ unit tests" ${PROJECT_IS_TOP_LEVEL})
option(OTRV_BUILD_TOOLS "Build the command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(OTRV_BUILD_PYTHON "Build the nanobind Python extension" OFF)
option(OTRV_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(OTRV_WERROR "Treat compiler warnings as errors" OFF)

if(DEFINED SKBUILD)
  set(OTRV_BUILD_PYTHON ON)
  set(OTRV_BUILD_TESTS OFF)
  set(OTRV_BUILD_TOOLS OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(otrv_warnings INTERFACE)
if(MSVC)
  target_compile_options(otrv_warnings INTERFACE /W4 $<$<BOOL:${OTRV_WERROR}>:/WX>)
else()
  target_compile_options(otrv_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                                 -Wno-sign-conversion $<$<BOOL:${OTRV_WERROR}>:-Werror>)
endif()
if(OTRV_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

add_library(otrv_core src/sei.cpp src/geo.cpp)
add_library(otrv::core ALIAS otrv_core)
target_include_directories(otrv_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(otrv_core PRIVATE otrv_warnings PUBLIC Threads::Threads)

if(OTRV_BUILD_TOOLS)
  add_executable(otrv-sei tools/otrv_sei.cpp)
  target_link_libraries(otrv-sei PRIVATE otrv::core otrv_warnings)
endif()

if(OTRV_BUILD_TESTS)
  enable_testing()
  include(cmake/doctest.cmake)
  add_executable(otrv_tests tests/main.cpp tests/test_sei.cpp tests/test_geo.cpp)
  target_link_libraries(otrv_tests PRIVATE otrv::core otrv_warnings doctest)
  add_test(NAME otrv_tests COMMAND otrv_tests)
endif()

if(OTRV_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC python/bindings.cpp)
  target_link_libraries(_core PRIVATE otrv::core)
  install(TARGETS _core LIBRARY DESTINATION otrv_native)
endif()
