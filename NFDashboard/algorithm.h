#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

struct TrainingData {
    bool valid = false;
    std::vector<std::tuple<double, double, double>> training_data;
};

struct NFTrainParams {
    double weight_learning_rate, func_center_learning_rate, func_width_learning_rate;
    unsigned center_move_iterate;
    double initial_rule_weight;
    unsigned rule_num;
    double error_threshold;
    unsigned max_epoch;
    double validation_factor;
};

struct TrainingDataParams {
    std::string training_data_path;
    bool shuffle;
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

class NFSystem {
   private:
    std::vector<Rule> rules;
    double normalizer;
    double min_pos, max_pos, min_angle, max_angle, min_label, max_label;

   public:
    NFSystem() = default;
    NFSystem(unsigned rule_num, double initial_rule_weight, double min_pox, double max_pos, double min_angle, double max_angle, double min_label, double max_label);
    double GetNormalizer(){return normalizer;}
    std::vector<Rule>& GetRules() { return rules; }
    double CalcOutput(const std::vector<double>& inputs);
    std::string GetRulesReport();
};

class NFTrainer {
   private:
    NFSystem nn;
    NFTrainParams params;
    TrainingData training_data;
    std::vector<std::tuple<double, double, double>> validation_data;
    void TrainOneIterate(const std::vector<double>& inputs, double label, unsigned& iterate_count);
    double CalcError(const std::vector<double>& inputs, double label);

   public:
    unsigned epoch_count;

    NFTrainer() = default;
    NFTrainer(const NFTrainParams& params);
    void Initialize(const TrainingDataParams& data_params);
    void TrainOneEpoch();
    double CalcTrainingError();
    double CalcValidationError();
    bool HasTrainingDataReady();
    bool ForceStopTraining();
    NFSystem GetNN() {return nn;}
};

double sgn (double val);

TrainingData initialize_data(const TrainingDataParams& data_params);
std::vector<double> linspace(double start, double end, unsigned total_num);

#endif
