#include "Tree.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace py = boost::python;
namespace np = boost::python::numpy;


/// std::vector to python list converter
template<typename T>
struct vector_to_list {
    static PyObject* convert(const std::vector<T>& vec) {
        auto* l = new boost::python::list();
        for (auto& i : vec) {
            l->append(i);
        }
        return l->ptr();
    }
};

template<typename T>
inline
std::vector<T> to_std_vector(const py::object& iterable) {
    return std::vector<T>(py::stl_input_iterator<T>(iterable),
                          py::stl_input_iterator<T>());
}

template<typename T>
inline
std::vector<std::vector<T>> to_2d_vector(const py::object& iterable) {
    std::vector<std::vector<T>> ans;
    for (auto iter = py::stl_input_iterator<py::list>(iterable); iter != py::stl_input_iterator<py::list>(); ++iter) {
        ans.push_back(to_std_vector<T>(*iter));
    }
    return ans;
}


/// class-helper to get container arguments
class clf_wrapper : public decision_tree_classifier {
public:
    explicit clf_wrapper(int num) : decision_tree_classifier(num) {}

    void fit(py::list& X, py::list& y) {
        decision_tree_classifier::fit(to_2d_vector<double>(X), to_std_vector<int>(y));
    }

    std::vector<int> predict(py::list& X) {
        return decision_tree_classifier::predict(to_2d_vector<double>(X));
    }

    std::vector<int> predict(np::ndarray& X) {
        return decision_tree_classifier::predict(to_2d_vector<double>(X));
    }
};


BOOST_PYTHON_MODULE (decision_tree) {
    py::to_python_converter<std::vector<int>, vector_to_list<int>>();

    py::class_<clf_wrapper>("decision_tree_classifier", py::init<int>())
            .def("fit", &clf_wrapper::fit)
            .def("predict", &clf_wrapper::predict);
}
