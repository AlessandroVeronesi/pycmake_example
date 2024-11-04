
project(inumpy
    VERSION 0.1
    LANGUAGES CXX)

add_library(inumpy INTERFACE)

target_include_directories(inumpy
    INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
        )
