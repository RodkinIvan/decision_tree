#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>

struct decision_tree_classifier : std::enable_shared_from_this<decision_tree_classifier> {

    ///constructs from num_of_classes and training dataset
    decision_tree_classifier(size_t num_of_classes,
                             std::vector<std::vector<double>>& X,
                             std::vector<int>& y);


private:

    /// if condition is false
    std::shared_ptr<decision_tree_classifier> left;
    /// if condition is true
    std::shared_ptr<decision_tree_classifier> right;

    ///
    size_t classes_num;


    /// splits the data with the best condition and generates left and right children
    void generate_children(std::vector<std::vector<double>>& X,
                           std::vector<int>& y);


    /// returns the best best_split of the data
    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
    best_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y);


    /// evaluates the quality of best_split
    double quality(const std::vector<std::vector<double>>& X,
                   const std::vector<int>& y,
                   size_t feature,
                   double sep);

    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
    split(const std::vector<std::vector<double>>& X, const std::vector<int>& y, size_t feature, double sep);

    /// calculates the gini's informativity
    double gini(const std::vector<int>& y) const;

};