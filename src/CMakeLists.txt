

target_sources(${SKBUILD_PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/wrapper.cpp
)

target_include_directories(${SKBUILD_PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)

add_subdirectory(INumPy)