#include <iostream>
#include <tuple>
#include "src/Tree.h"
#include <random>


void adecvatnost_test() {
    std::vector<std::vector<double>> X = {{0,   0},
                                          {0,   1},
                                          {0.5, 0.5},
                                          {1,   0},
                                          {1,   1}};
    std::vector<int> y = {0, 1, 1, 1, 0};
    decision_tree_classifier clf(2);
    clf.fit(X, y);
    std::vector<int> pred = clf.predict(X);
    assert(pred == y);
}

bool in_circle(double x, double y) {
    return x * x + y * y <= 1;
}

std::tuple<std::vector<std::vector<double>>, std::vector<int>>
make_binary_points_classification(size_t num, bool (* condition)(double x, double y), int x_min = -2, int x_max = 2,
                                  int y_min = -2,
                                  int y_max = 2) {
    std::vector<std::vector<double>> X(num);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> xs(x_min, x_max);
    std::uniform_real_distribution<> ys(y_min, y_max);

    for (size_t i = 0; i < num; ++i) {
        X[i] = {xs(gen), ys(gen)};
    }
    std::vector<int> y(num);
    for (size_t i = 0; i < num; ++i) {
        y[i] = condition(X[i][0], X[i][1]);
    }
    return {X, y};
}

void accuracy_on_training_set_test() {
    size_t num_of_points = 100;
    auto[X, y] = make_binary_points_classification(num_of_points, in_circle);
    decision_tree_classifier clf(2, X, y);
    auto pred = clf.predict(X);

    size_t correct = 0;
    for (size_t i = 0; i < num_of_points; ++i) {
        correct += (pred[i] == y[i]);
    }
    double acc = static_cast<double>(correct) / static_cast<double>(num_of_points);
    std::cout << "accuracy on training set = " << acc << '\n';
    assert(acc >= 0.9);
}

void accuracy_on_real_test() {
    size_t num_of_points = 100;
    auto[X, y] = make_binary_points_classification(num_of_points, in_circle);
    decision_tree_classifier clf(2);
    clf.fit(X, y);

    auto[test_X, test_y] = make_binary_points_classification(num_of_points, in_circle);
    auto pred = clf.predict(test_X);

    size_t correct = 0;
    for (size_t i = 0; i < num_of_points; ++i) {
        correct += (pred[i] == test_y[i]);
    }
    double acc = static_cast<double>(correct) / static_cast<double>(num_of_points);
    std::cout << "accuracy on real test = " << acc << '\n';
    assert(acc >= 0.8);
}

int main() {
    adecvatnost_test();
    accuracy_on_training_set_test();
    accuracy_on_real_test();

}
