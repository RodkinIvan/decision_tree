#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <random>

using input_data = std::vector<std::vector<double>>;
using output_data = std::vector<int>;

struct decision_tree_classifier {


    /// default constructor
    explicit decision_tree_classifier(size_t num_of_classes, double precision = 0.1) : classes_num(num_of_classes),
                                                                                       precision(precision) {}

    ///constructs from num_of_classes and training dataset (default constructor + fit)
    decision_tree_classifier(size_t num_of_classes,
                             const input_data& X,
                             const output_data& y,
                             double precision = 0.1);

    void fit(const input_data& X, const output_data& y, bool enable_random_subspace = false);

    int predict(const std::vector<double>& x) const;

    std::vector<int> predict(const input_data& X) const;


protected:
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
    void generate_children(const input_data& X,
                           const output_data& y, bool enable_random_subspace = false);


    /// returns the best best_split of the data
    std::tuple<input_data, output_data, input_data, output_data>
    best_split(const input_data& X, const output_data& y, bool enable_random_subspace = false);


    /// evaluates the quality of best_split
    double quality(const input_data& X,
                   const output_data& y,
                   size_t feature,
                   double sep);

    static std::tuple<input_data, output_data, input_data, output_data>
    split(const input_data& X, const output_data& y, size_t feature, double sep);

    /// calculates the gini's informativity
    double gini(const output_data& y) const;

    /// trivial act
    int the_most_popular_class(const output_data& y) const;


};