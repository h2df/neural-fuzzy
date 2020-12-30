
#include <iostream>

#include "algorithm.h"

int main(int argc, char *argv[]) {
    auto test_params = NNParams{
        0.01, 0.01, 0.01,
        0,
        25};

    auto data = initialize_data("../train.dat", false, true);

    NNTrainer trainer = NNTrainer(test_params);
    trainer.InitializeNNFromData(data);
    const double THRESHOLD = 0.0001;

    double error = 1;
    do
    {
        trainer.TrainOneEpoch(data);
        error = trainer.CalcAvgError(data.training_data);
        std::cout << error << std::endl;
    } while (error > THRESHOLD);
    
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
