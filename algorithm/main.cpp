
#include <iostream>

#include "algorithm.h"

int main(int argc, char *argv[]) {
    auto test_params = NNParams{
        0.01, 0.01, 0.01,
        0,
        25};

    auto data = initialize_data("../train.dat", true, true);

    NNTrainer trainer = NNTrainer(test_params);
    trainer.InitializeNNFromData(data);
    double THRESHOLD = 0.00001;
    for (auto sample : data.training_data) {
        std::vector<double> inputs;
        inputs.reserve(2);
        inputs.emplace_back(std::get<0>(sample));
        inputs.emplace_back(std::get<1>(sample));
        double label = std::get<2>(sample);
        trainer.TrainOneIterate(inputs, label);
        double error = trainer.CalcError(inputs, label);
        std::cout << error << std::endl;
        if (error < THRESHOLD) {
            std::cout << "Trained successfully." << std::endl;
            break;
        }
    }

    auto trained_rules = trainer.GetNN().GetRules();
    for (auto rule : trained_rules) {
        std::cout << "Rule Weight: " << rule.GetWeight() << std::endl;
        auto trained_funcs = rule.GetMemberFuncs();
        for (auto func : trained_funcs) {
            std::cout << "Center: " << func.GetCenter() << ", Width: " << func.GetWidth() << std::endl;
        }
    }
    return 0;
}
