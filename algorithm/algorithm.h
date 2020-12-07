// #include <cmath>
#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

struct TrainingData {
    double max_pos, min_pos;
    double max_angle, min_angle;
    double max_label, min_label;
    std::vector<std::tuple<double, double, double>> training_data;
};

struct NNParams {
    double weight_learning_rate, func_center_learning_rate, func_width_learning_rate;
    double initial_rule_weight;
    unsigned rule_num;
};

class MemberFunc {
   private:
    double center, width;
    double last_output;

   public:
    MemberFunc(double center, double width) : center(center), width(width){};
    MemberFunc(){};
    double GetCenter() const { return center; }
    void SetCenter(double center_val);
    double GetWidth() const { return width; }
    void SetWidth(double width_val);
    double GetLastOutput() const { return last_output; }

    double CalcOutput(double input);
};

class Rule {
   private:
    std::vector<MemberFunc> member_funcs;
    double weight;
    double last_output;

   public:
    Rule(std::vector<MemberFunc>& member_funcs) : member_funcs(member_funcs){};
    Rule() = default;
    double GetWeight() const { return weight; }
    void SetWeight(double weight_val);
    std::vector<MemberFunc>& GetMemberFuncs() { return member_funcs; }
    double GetLastOutput() { return last_output; }
    double CalcOutput(const std::vector<double>& inputs);
};

class NeuralNetwork {
   private:
    std::vector<Rule> rules;
    double normalizer;

   public:
    NeuralNetwork() = default;
    NeuralNetwork(std::vector<Rule> rules);
    double GetNormalizer(){return normalizer;}
    std::vector<Rule>& GetRules() { return rules; }
    double CalcOutput(const std::vector<double>& inputs);
};

class NNTrainer {
   private:
    NeuralNetwork nn;
    NNParams params;

   public:
    NNTrainer() = default;
    NNTrainer(const NNParams& params);
    void InitializeNNFromData(const TrainingData& training_data);
    void TrainOneIterate(const std::vector<double>& inputs, double label);
    double CalcError(const std::vector<double>& inputs, double label);
    NeuralNetwork GetNN() {return nn;}
};

TrainingData initialize_data(const std::string training_data_f, bool shuffle, bool normalize);
std::vector<double> linspace(double start, double end, unsigned total_num);