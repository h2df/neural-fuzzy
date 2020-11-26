
#include <iostream>

#include "algorithm.h"

int main(int argc, char *argv[]) {
    auto test_params = NNParams{
        0.01, 0.01, 0.01,
        0,
        25};

    auto data = initialize_data("../train.dat", true, true);

    NeuralNetwork nn = initialize_network(data, test_params);

    double THRESHOLD = 0.01;
    for (auto sample : data.training_data) {
        std::vector<double> inputs;
        inputs.reserve(2);
        inputs.emplace_back(std::get<0>(sample));
        inputs.emplace_back(std::get<1>(sample));
        double label = std::get<2>(sample);
        nn.TrainOneIterate(inputs, label);
        double error = nn.CalcError(inputs, label);
        std::cout << error << std::endl;
        if (error < THRESHOLD) {
            break;
        }
    }

    for (Rule& rule : nn.GetRules()) {
        std::cout << "Rule Weight: " << rule.GetWeight() << std::endl;
        for (MemberFunc& func : rule.GetMemberFuncs()) {
            std::cout << "Center: " << func.GetCenter() << ", Width: " << func.GetWidth() << std::endl;
        }
    }
    return 0;
}
