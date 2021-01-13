
#include <iostream>

#include "algorithm.h"

int main(int argc, char *argv[]) {
    std::cout << "Start..." << std::endl;
 
    auto test_params = NNParams{
        0.01, 0.01, 0.01,
        10,
        0,
        25};

    auto data = initialize_data("../poly.dat", true, true);

    NNTrainer trainer = NNTrainer(test_params);
    trainer.InitializeNNFromData(data);
    const double THRESHOLD = 0.000001;

    double error = 0;//1;
    double prevError = 0;

    int epoch=1;

    std::cout << "============================================" << std::endl;
    std::cout << "Training begins... " <<  std::endl; 
    do
    {
        
        trainer.TrainOneEpoch(data);
        prevError = error;
        error = trainer.CalcAvgError(data.training_data);
        std::cout.precision (6); 
        std::cout << std::fixed << "\n\t" << error << std::endl;
        std::cout << "---<< Epoch # " << epoch << " >>---" << ", Error difference = " << fabs(error - prevError) << std::endl;
        
        
        //let user adjust learning rate manually to be consistent with other hyper parameters 
        // if( fabs(prevError - error) < 0.000001){
        //     std::cout << "Error difference = " << fabs(error - prevError) << " \t\t\t\t--- adjusting learning rates... " <<  std::endl;
        //     trainer.AdjustLearningRates();
        // }
        
        epoch++;

    } while (error > THRESHOLD);

    std::cout << "============================================" << std::endl;
    std::cout << "---<< Trained rules >>---" <<  std::endl;
    
    auto trained_rules = trainer.GetNN().GetRules();
    for (unsigned i = 1; i <= trained_rules.size(); ++i) {
        auto rule = trained_rules[i];
        std::cout << "Rule " << i << ")" << std::endl; 
        
        auto trained_funcs = rule.GetMemberFuncs();
        for (auto func : trained_funcs) {
            std::cout << "\tCenter: " << func.GetCenter() << ", Width: " << func.GetWidth() << std::endl;
        }
        std::cout << "\tConsequent: " << rule.GetWeight() << std::endl;
        std::cout << "\tActual Consequent: " << rule.GetWeight() * (data.max_label - (data.min_label)) + (data.min_label) << std::endl;
    }
    std::cout << "============================================" << std::endl;
    return 0;
}