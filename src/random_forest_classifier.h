#include "decision_tree_classifier.h"
#include <cmath>


class random_forest_classifier{
public:
    random_forest_classifier(int num_of_trees, int num_of_classes);

    /// fits the training input with output
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    output_data predict(const input_data& X) const;
private:
    /// number of classes
    int classes_num = 0;

    /// trees of an ensemble
    std::vector<decision_tree_classifier> trees;
};

std::tuple<input_data, output_data> make_bagging(const input_data& X, const output_data& y, unsigned long seed=42);