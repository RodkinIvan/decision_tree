#include "random_forest_classifier.h"


std::tuple<input_data, output_data> make_bagging(const input_data& X, const output_data& y, unsigned long seed){
    assert(X.size() == y.size());
    std::vector<int> prob(X.size(), 1);
    std::discrete_distribution distribution(prob.begin(), prob.end());
    std::default_random_engine generator(seed);
    input_data res_X(X.size());
    output_data res_y(y.size());
    for(int i = 0; i < X.size(); ++i){
        res_X[i] = X[distribution(generator)];
        res_y[i] = y[distribution(generator)];
    }
    return {res_X, res_y};
}

random_forest_classifier::random_forest_classifier(int num_of_trees, int num_of_classes) : trees(num_of_trees,
                                                                           decision_tree_classifier(num_of_classes)),
                                                                     m(int(std::sqrt(num_of_classes))) {

}

void random_forest_classifier::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    int num_of_trees = trees.size();

    for(int i = 0; i < num_of_trees; ++i){
        auto [X_beg, y_beg] = make_bagging(X, y, i*1000003);


    }
}

