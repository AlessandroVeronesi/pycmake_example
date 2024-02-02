cmake_minimum_required(VERSION 3.26)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_EXTENSIONS OFF)

project(MyModule
	VERSION 1.0
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python 3.7 REQUIRED Interpreter Development NumPy)
find_package(PythonLibs REQUIRED)

add_library(MyModule SHARED "")
set_target_properties(MyModule
	PROPERTIES
		PREFIX ""
		OUTPUT_NAME "MyModule"
		LINKER_LANGUAGE CXX
		)

target_sources(MyModule
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/src/python_wrapper.cpp
	)

target_include_directories(MyModule
	PUBLIC
		${CMAKE_CURRENT_LIST_DIR}/src/common
		${PYTHON_INCLUDE_DIRS}
	)

target_link_libraries(MyModule
	PUBLIC
		${PYTHON_LIBRARIES}
	)

install(TARGETS MyModule
	DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

