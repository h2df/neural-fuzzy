#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

struct NFTrainParams {
    double weight_learning_rate, func_center_learning_rate, func_width_learning_rate;
    unsigned center_move_iterate;
    double error_threshold;
    unsigned max_epoch;
    double validation_factor;
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

class PendulumDataNormalizer {
private:
    double min_pos, max_pos, min_angle, max_angle, min_output, max_output;

public:
    void Initialize(std::vector<std::tuple<double, double, double>> data);
    std::vector<std::tuple<double, double, double>> Normalize(std::vector<std::tuple<double, double, double>> data) const;
    std::vector<std::tuple<double, double, double>> Denormalize(std::vector<std::tuple<double, double, double>> data) const;
};

class NFSystem {
   private:
    std::vector<Rule> rules;
    double normalizer;

   public:
    NFSystem(unsigned rule_num, double initial_rule_weight);
    double GetNormalizer(){return normalizer;}
    std::vector<Rule>& GetRules() { return rules; }
    double CalcOutput(const std::vector<double>& inputs);
    std::string GetRulesReport();
};

class NFTrainer {
   private:
    NFSystem *nf;
    NFTrainParams params;
    std::vector<std::tuple<double, double, double>>  training_data;
    std::vector<std::tuple<double, double, double>> validation_data;
    void TrainOneIterate(const std::vector<double>& inputs, double label, unsigned& iterate_count);

public:
    unsigned epoch_count;

    NFTrainer(NFSystem* nf, const NFTrainParams& params);
    bool Initialize(const std::string training_data_path);
    void TrainOneEpoch();
    bool ForceStopTraining();
    double GetErrorThreshold() {return params.error_threshold;}
    std::vector<std::tuple<double, double, double>> GetTrainingData(){return training_data; }
    std::vector<std::tuple<double, double, double>> GetValidationData(){return validation_data; }
    void NormalizeData(const PendulumDataNormalizer *normalizer);
};

class NFTester {
    NFSystem *nf;
public:
    NFTester(NFSystem *nf);
    double CalcAvgError(std::vector<std::tuple<double, double, double>> data);
private:
    double CalcError(std::vector<double> inputs, double label);
};

double sgn (double val);

std::vector<double> linspace(double start, double end, unsigned total_num);

#endif
