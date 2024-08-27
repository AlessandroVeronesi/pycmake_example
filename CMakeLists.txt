cmake_minimum_required(VERSION 3.18)

include(CMakePrintHelpers)

## Project Name
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

## Compiler Settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

## Python 
find_package(Python3 3.7
    COMPONENTS Interpreter Development.Module
    NumPy
    REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

cmake_print_variables(PYTHON_INCLUDE_DIRS)

if(NOT PYTHON_INCLUDE_DIRS)
    message(FATAL_ERROR "Python library not found")
endif()

## YAML-CPP
set(YAML_CPP_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu")
set(YAML_CPP_INCLUDE_DIR "/usr/include")

find_library(YAML_CPP_LIBRARY NAMES yaml-cpp HINTS ${YAML_CPP_LIBRARY_PATH})

if(NOT YAML_CPP_LIBRARY)
    message(FATAL_ERROR "yaml-cpp library not found. Please check if yaml-cpp is installed in the specified path.")
endif()


#################################
## Python Module Specifications

add_library(${SKBUILD_PROJECT_NAME} SHARED "")
set_target_properties(${SKBUILD_PROJECT_NAME}
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${SKBUILD_PROJECT_NAME}"
        LINKER_LANGUAGE CXX
        )

target_include_directories(${SKBUILD_PROJECT_NAME} PUBLIC ${YAML_CPP_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} Python3::NumPy)
target_link_libraries(${SKBUILD_PROJECT_NAME} PUBLIC ${YAML_CPP_LIBRARY} ${PYTHON_LIBRARIES} Python3::NumPy)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)

add_subdirectory(src)