cmake_minimum_required(VERSION 3.19)
project(decision_tree)

set(CMAKE_CXX_STANDARD 20)

add_executable(decision_tree
        main.cpp
        src/Tree.cpp
        src/Tree.h)