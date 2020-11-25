// #include <cmath>
#include <vector>
#include <limits>
#include <math.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <algorithm>

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

class MemberFunc
{
private:
    double center, width;
    double last_output;
public:
    MemberFunc(double center, double width) : center(center), width(width){};
    MemberFunc(){};
    double GetCenter() const {return center;}
    double GetWidth() const {return width;}
    double GetLastOutput() const {return last_output;}

    double CalcOutput(double input);
};

class Rule
{
private:
    std::vector<MemberFunc> member_funcs;
    double weight; 
    double last_output;
public:
    Rule(std::vector<MemberFunc>& member_funcs): member_funcs(member_funcs){};
    Rule(){};
    double GetWeight() const {return weight;}
    void SetWeight(double weight);
    std::vector<MemberFunc>& GetMemberFuncs() {return member_funcs;}
    double GetLastOutput() {return last_output;}

    double CalcOutput(const std::vector<double>& inputs);
};

class NeuralNetwork
{
private:
    std::vector<Rule> rules;
    double normalizer;
    NNParams params;

public:
    NeuralNetwork(std::vector<Rule>& rules, const NNParams& params);
    
    double CalcOutput(const std::vector<double>& inputs);
    void TrainOneIterate(std::vector<double>& inputs, double label);
    double CalcError(const std::vector<double>& inputs, double label);
};

TrainingData initialize_data(const std::string training_data_f, bool shuffle, bool normalize);
NeuralNetwork initialize_network(const TrainingData& training_data, const NNParams& params); 
std::vector<double> linspace(double start, double end, unsigned total_num);