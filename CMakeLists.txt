cmake_minimum_required(VERSION 3.18)

include(CMakePrintHelpers)

#################################################
## COMPILER OPTIONS

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g -Wall -fanalyzer -pedantic -Werror -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")                                          

set(CMAKE_CXX_EXTRA_FLAGS "-fmax-errors=10 -Wno-write-strings -Wno-sign-compare")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
endif()

if(CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_EXTRA_FLAGS}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

#################################################
## DEFINE PYTHON PROJECT

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "MyModule")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

#################################################
## IMPORT PYTHON

find_package(Python3 3.7
    COMPONENTS Interpreter Development.Module
    NumPy
    REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

cmake_print_variables(PYTHON_INCLUDE_DIRS)

#################################################
## IMPORT YAML-CPP

set(YAML_CPP_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu")
set(YAML_CPP_INCLUDE_DIR "/usr/include")

find_library(YAML_CPP_LIBRARY NAMES yaml-cpp HINTS ${YAML_CPP_LIBRARY_PATH})

if(NOT YAML_CPP_LIBRARY)
    message(FATAL_ERROR "yaml-cpp library not found. Please check if yaml-cpp is installed in the specified path.")
endif()

#################################################
## PYTHON MODULE

add_library(${SKBUILD_PROJECT_NAME} SHARED "")
set_target_properties(${SKBUILD_PROJECT_NAME}
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${SKBUILD_PROJECT_NAME}"
        LINKER_LANGUAGE CXX
        )

target_include_directories(${SKBUILD_PROJECT_NAME} PUBLIC inumpy ${YAML_CPP_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} Python3::NumPy)
target_link_libraries(${SKBUILD_PROJECT_NAME} PUBLIC inumpy ${YAML_CPP_LIBRARY} ${PYTHON_LIBRARIES} Python3::NumPy)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")

#################################################

add_subdirectory(src)