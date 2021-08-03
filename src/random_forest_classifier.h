#include "decision_tree_classifier.h"
#include <cmath>


class random_forest_classifier{
public:
    random_forest_classifier(int num_of_trees, int num_of_classes);

    /// fits the training input with output
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

private:

    /// trees of an ensemble
    std::vector<decision_tree_classifier> trees;

    /// number of features, which are randomly selected in every node of tree to learn
    int m;

};

std::tuple<input_data, output_data> make_bagging(const input_data& X, const output_data& y, unsigned long seed=42);