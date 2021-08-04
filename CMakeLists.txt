cmake_minimum_required(VERSION 3.19)
project(decision_tree)

find_package(PythonLibs 3)
find_package(Boost COMPONENTS python REQUIRED)

include_directories(/usr/include/python3.9)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module)

PYTHON_ADD_MODULE(decision_tree
        src/decision_tree_classifier.h
        src/decision_tree_classifier.cpp
        src/random_forest_classifier.h
        src/random_forest_classifier.cpp
        src/py_module.cpp)
target_link_libraries(decision_tree Boost::python)

#add_executable(decision_tree
#        main.cpp
#        src/decision_tree_classifier.h
#        src/decision_tree_classifier.cpp
#        src/random_forest_classifier.h
#        src/random_forest_classifier.cpp
#        )