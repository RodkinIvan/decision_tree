#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

struct decision_tree_classifier : std::enable_shared_from_this<decision_tree_classifier> {

    ///constructs from num_of_classes and training dataset
    decision_tree_classifier(size_t num_of_classes,
                             const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y,
                             double precision = 0.1);

    int predict(const std::vector<double>& x) const;

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
private:

    /// the pair with the feature num and separator which responds to condition x[feature] >= separator
    std::pair<size_t, double> condition;

    /// if condition is false
    std::shared_ptr<decision_tree_classifier> left;
    /// if condition is true
    std::shared_ptr<decision_tree_classifier> right;

    /// number of classification classes
    size_t classes_num;

    /// the biggest gini informativty in leaf of the tree
    double precision;


    /// the leaf variable, which points to a class of an object
    int decision = 0;

    /// splits the data with the best condition and generates left and right children
    void generate_children(const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y);


    /// returns the best best_split of the data
    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
    best_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y);


    /// evaluates the quality of best_split
    double quality(const std::vector<std::vector<double>>& X,
                   const std::vector<int>& y,
                   size_t feature,
                   double sep);

    static std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
    split(const std::vector<std::vector<double>>& X, const std::vector<int>& y, size_t feature, double sep);

    /// calculates the gini's informativity
    double gini(const std::vector<int>& y) const;

    /// trivial act
    int the_most_popular_class(const std::vector<int>& y) const;


};