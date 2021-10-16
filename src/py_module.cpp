#include "random_forest_classifier.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;



using array = py::list;
template<typename T>
using iterator = py::stl_input_iterator<T>;


/// std::vector to python list converter
template<typename T>
struct vector_to_list {
    static PyObject* convert(const std::vector<T>& vec) {
        auto l = boost::python::list();
        for (auto& i : vec) {
            l.append(i);
        }
        return l.ptr();
    }
};


template<typename T>
inline
std::vector<T> to_std_vector(const array& iterable) {
    return std::vector<T>(iterator<T>(iterable),
                          iterator<T>());

}

template<typename T>
inline
std::vector<std::vector<T>> to_2d_vector(const array& iterable) {
    std::vector<std::vector<T>> ans;
    for (auto iter = iterator<array>(iterable);
         iter != iterator<array>(); ++iter) {
        ans.push_back(to_std_vector<T>(*iter));
    }
    return ans;
}


/// class-helper to get container arguments
class clf_wrapper : public decision_tree_classifier {
public:
    explicit clf_wrapper(int num, double precision = 0.1) : decision_tree_classifier(num, precision) {}

    void fit(array& X, array& y) {
        decision_tree_classifier::fit(to_2d_vector<double>(X), to_std_vector<int>(y));
    }

    std::vector<int> predict(array& X) {
        return decision_tree_classifier::predict(to_2d_vector<double>(X));
    }

};

class forest_clf_wrapper : public random_forest_classifier{
public:
    forest_clf_wrapper(int num_of_trees, int num_of_classes) : random_forest_classifier(num_of_trees, num_of_classes){}

    void fit(array& X, array& y) {
        random_forest_classifier::fit(to_2d_vector<double>(X), to_std_vector<int>(y));
    }
    std::vector<int> predict(array& X) {
        return random_forest_classifier::predict(to_2d_vector<double>(X));
    }

};
BOOST_PYTHON_MODULE (decision_tree) {
    py::to_python_converter<std::vector<int>, vector_to_list<int>>();

    py::class_<clf_wrapper>("decision_tree_classifier", py::init<int, double>())
            .def(py::init<int>())
            .def("fit", &clf_wrapper::fit)
            .def("predict", &clf_wrapper::predict);

    py::class_<forest_clf_wrapper>("random_forest_classifier", py::init<int, int>())
            .def("fit", &forest_clf_wrapper::fit)
            .def("predict", &forest_clf_wrapper::predict);

}
