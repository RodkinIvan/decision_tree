#include "Tree.h"

decision_tree_classifier::decision_tree_classifier(size_t num_of_classes,
                                                   std::vector<std::vector<double>>& X,
                                                   std::vector<int>& y) {
    assert(X.size() == y.size());

}


void decision_tree_classifier::generate_children(std::shared_ptr<decision_tree_classifier>& node,
                                                 std::vector<std::vector<double>>& X,
                                                 std::vector<int>& y) {
    assert(X.size() == y.size());
}


std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
decision_tree_classifier::best_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    assert(X.size() == y.size());

    /// copy of layer of x with necessary criterion for sorting

}

double decision_tree_classifier::quality(const std::vector<std::vector<double>>& X,
                                         const std::vector<int>& y,
                                         size_t class_num,
                                         double sep) {
    assert(X.size() == y.size());

}

double decision_tree_classifier::gini(const std::vector<int>& y) const {
    double sum = 0;

    /// the frequencies of class_num in y
    std::vector<double> ps(classes_num);

    for (auto class_num : y) {
        ps[class_num] += 1. / static_cast<double>(y.size());
    }

    for (auto p : ps) {
        sum += p * (1 - p);
    }
    return sum;

}


