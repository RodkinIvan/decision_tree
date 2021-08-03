#include "decision_tree_classifier.h"

decision_tree_classifier::decision_tree_classifier(size_t num_of_classes,
                                                   const std::vector<std::vector<double>>& X,
                                                   const std::vector<int>& y,
                                                   double precision) : classes_num(num_of_classes),
                                                                       precision(precision) {
    assert(X.size() == y.size());
    generate_children(X, y);
}


void decision_tree_classifier::generate_children(const std::vector<std::vector<double>>& X,
                                                 const std::vector<int>& y, bool enable_random_subspace) {
    assert(X.size() == y.size());

    double inf = gini(y);
    if (inf == 0 || inf < precision) {
        left = nullptr;
        right = nullptr;
        decision = the_most_popular_class(y);
        return;
    }
    auto[left_X, left_y, right_X, right_y] = best_split(X, y, enable_random_subspace);
    left = std::make_shared<decision_tree_classifier>(classes_num, left_X, left_y);
    right = std::make_shared<decision_tree_classifier>(classes_num, right_X, right_y);
}


void decision_tree_classifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                                   bool enable_random_subspace) {
    assert(X.size() == y.size());
    generate_children(X, y, enable_random_subspace);
}


std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
decision_tree_classifier::best_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                                     bool enable_random_subspace) {
    assert(X.size() == y.size());
    int m = int(std::sqrt(classes_num));
    std::vector<int> features(X[0].size());
    std::vector<int> range(X[0].size());
    std::iota(range.begin(), range.end(), 0);
    if (enable_random_subspace) {
        /// making sample of features
        features.resize(m);
        std::sample(range.begin(), range.end(), features.begin(), m, std::mt19937(std::random_device{}()));
    } else {
        features = range;
    }

    std::pair<size_t, double> the_best_separation;
    double best_quality = 0;
    for (auto feature : features) {
        for (auto& x : X) {
            double q = quality(X, y, feature, x[feature]);
            if (q >= best_quality) {
                the_best_separation = std::make_pair(feature, x[feature]);
                best_quality = q;
            }
        }
    }
    condition = the_best_separation;
    return split(X, y, the_best_separation.first, the_best_separation.second);
}

double decision_tree_classifier::quality(const std::vector<std::vector<double>>& X,
                                         const std::vector<int>& y,
                                         size_t feature,
                                         double sep) {
    assert(X.size() == y.size());

    auto[left_X, left_y, right_X, right_y] = split(X, y, feature, sep);
    double left_inf = gini(left_y);
    double right_inf = gini(right_y);
    return gini(y) - static_cast<double>(left_y.size()) / static_cast<double>(y.size()) * left_inf
           - static_cast<double>(right_y.size()) / static_cast<double>(y.size()) * right_inf;
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
            right_y.push_back(y[i]);
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

int decision_tree_classifier::the_most_popular_class(const std::vector<int>& y) const {
    std::vector<int> num_of_each(classes_num);
    int the_most_class = 0;
    size_t the_biggest_num = 0;
    for (auto c : y) {
        ++num_of_each[c];
        if (num_of_each[c] > the_biggest_num) {
            the_biggest_num = num_of_each[c];
            the_most_class = c;
        }
    }
    return the_most_class;
}


int decision_tree_classifier::predict(const std::vector<double>& x) const {
    auto node = this;
    const decision_tree_classifier* parent = nullptr;
    while (node != nullptr) {
        parent = node;
        node = x[node->condition.first] < node->condition.second ? node->left.get() : node->right.get();
    }
    return parent->decision;
}


std::vector<int> decision_tree_classifier::predict(const std::vector<std::vector<double>>& X) const {

    std::vector<int> ans(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        ans[i] = predict(X[i]);
    }
    return ans;
}






