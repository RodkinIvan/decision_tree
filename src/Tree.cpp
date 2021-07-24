#include "Tree.h"

decision_tree_classifier::decision_tree_classifier(size_t num_of_classes,
                                                   std::vector<std::vector<double>>& X,
                                                   std::vector<int>& y) {
    assert(X.size() == y.size());
    generate_children(X, y);
}


void decision_tree_classifier::generate_children(std::vector<std::vector<double>>& X,
                                                 std::vector<int>& y) {
    assert(X.size() == y.size());
    auto[left_X, left_y, right_X, right_y] = best_split(X, y);
    left = std::make_shared<decision_tree_classifier>(classes_num, left_X, left_y);
    right = std::make_shared<decision_tree_classifier>(classes_num, right_X, right_y);
    /// condition should be a field of the class
}


std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
decision_tree_classifier::best_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    assert(X.size() == y.size());

    /// copy of layer of features of x for sorting candidates
    size_t iter = 0;
    std::pair<size_t, double> the_best_separation;
    double best_quality = 0;
    for (size_t feature = 0; feature < X[0].size(); ++feature) {
        for (auto& x : X) {
            double q = quality(X, y, feature, x[feature]);
            if (q > best_quality) {
                the_best_separation = std::make_pair(feature, x[feature]);
                best_quality = q;
            }
        }
    }
    return split(X, y, the_best_separation.first, the_best_separation.second);
}

double decision_tree_classifier::quality(const std::vector<std::vector<double>>& X,
                                         const std::vector<int>& y,
                                         size_t feature,
                                         double sep) {
    assert(X.size() == y.size());

    auto[left_X, left_y, right_X, right_y] = split(X, y, feature, sep);

    return gini(y) - static_cast<double>(left_y.size()) / static_cast<double>(y.size()) * gini(left_y)
           - static_cast<double>(right_y.size()) / static_cast<double>(y.size()) * gini(right_y);
}


std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
decision_tree_classifier::split(const std::vector<std::vector<double>>& X, const std::vector<int>& y, size_t feature,
                                double sep) {
    assert(X.size() == y.size());

    std::vector<std::vector<double>> left_X;
    std::vector<int> left_y;
    std::vector<std::vector<double>> right_X;
    std::vector<int> right_y;

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature] < sep) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            left_y.push_back(y[i]);
        }
    }
    return {left_X, left_y, right_X, right_y};
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




