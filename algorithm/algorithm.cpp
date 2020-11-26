#include "algorithm.h"

void MemberFunc::SetCenter(double center_val) {
    center = center_val;
}

void MemberFunc::SetWidth(double width_val) {
    width = width_val;
}

double MemberFunc::CalcOutput(double input) {
    double output = 0;
    if ((center - width / 2) < input && input < (center + width / 2)) {
        output = 1 - 2 * abs(input - center) / width;
    }
    last_output = output;
    return output;
}

void Rule::SetWeight(double weight_val) {
    if (weight_val > 1 || weight_val < 0) {
        throw "Invalid Weight";
    } else {
        weight = weight_val;
    }
}

double Rule::CalcOutput(const std::vector<double>& inputs) {
    double output = 1.0;
    for (unsigned i = 0; i < inputs.size(); i++) {
        double input = inputs[i];
        MemberFunc& func = member_funcs[i];
        output *= func.CalcOutput(input);
    }
    last_output = output;
    return output;
}

double NeuralNetwork::CalcOutput(const std::vector<double>& inputs) {
    double weighted_sum = 0;
    normalizer = 0;
    for (Rule& rule : rules) {
        double rule_output = rule.CalcOutput(inputs);
        weighted_sum += rule.GetWeight() * rule_output;
        normalizer += rule_output;
    }
    return weighted_sum / normalizer;
};

NeuralNetwork::NeuralNetwork(std::vector<Rule>& rules, const NNParams& params) : rules(rules), params(params) {
    for (Rule& rule : rules) {
        rule.SetWeight(params.initial_rule_weight);
    }
};

void NeuralNetwork::TrainOneIterate(std::vector<double>& inputs, double label) {
    double output = CalcOutput(inputs);  //step 3 in paper
    for (Rule& rule : rules) {
        double new_weight = rule.GetWeight() - params.weight_learning_rate * (rule.GetLastOutput() / normalizer) * (output - label);
        rule.SetWeight(new_weight);  //step 4 in paper
    }
    output = CalcOutput(inputs);  //step 5 in paper
    for (Rule& rule : rules) {
        if (rule.GetLastOutput() == 0) {
            continue;  //inactive rules should be skipped in this epoch's backpropagation
        }
        for (MemberFunc& func : rule.GetMemberFuncs()) {
            double new_center = func.GetCenter() - params.func_center_learning_rate * (rule.GetLastOutput() / normalizer) * (output - label) * (rule.GetWeight() - output) * (2 * sin(inputs[0] - func.GetCenter())) / (func.GetLastOutput() * func.GetWidth());
            func.SetCenter(new_center);
            double new_width = func.GetWidth() - params.func_center_learning_rate * (rule.GetLastOutput()/ normalizer) * (output - label) * (rule.GetWeight()  - output) * (1 - func.GetLastOutput())/func.GetLastOutput() * (1/func.GetWidth());
            func.SetWidth(new_width);
        }
    }
};

double NeuralNetwork::CalcError(const std::vector<double>& inputs, double label) {
    double output = CalcOutput(inputs);
    return 0.5 * pow(output - label, 2);
}

TrainingData initialize_data(const std::string training_data_f, bool shuffle, bool normalize) {
    std::ifstream training_f(training_data_f);
    if (!training_f.is_open()) {
        throw "Cannot open training data file.";
    }

    std::vector<std::tuple<double, double, double>> training_data;
    double max_pos = std::numeric_limits<double>::min();
    double min_pos = std::numeric_limits<double>::max();
    double max_angle = std::numeric_limits<double>::min();
    double min_angle = std::numeric_limits<double>::max();
    double max_label = std::numeric_limits<double>::min();
    double min_label = std::numeric_limits<double>::max();

    double pos_input, angle_input, label;
    while (training_f >> pos_input >> angle_input >> label) {
        if (pos_input > max_pos) {
            max_pos = pos_input;
        }
        if (pos_input < min_pos) {
            min_pos = pos_input;
        }
        if (angle_input > max_angle) {
            max_angle = angle_input;
        }
        if (angle_input < min_angle) {
            min_angle = angle_input;
        }
        if (label > max_label) {
            max_label = label;
        }
        if (label < min_label) {
            min_label = label;
        }
        training_data.push_back(std::tuple<double, double, double>(pos_input, angle_input, label));
    }

    if (shuffle) {
        std::random_shuffle(training_data.begin(), training_data.end());
    }

    if (normalize) {
        for (unsigned i = 0; i < training_data.size(); i++) {
            double pos = std::get<0>(training_data[i]);
            double new_pos = (pos - min_pos) / (max_pos - min_pos);

            double angle = std::get<1>(training_data[i]);
            double new_angle = (angle - min_angle) / (max_angle - min_angle);

            double label = std::get<2>(training_data[i]);
            double new_label = (label - min_label) / (max_label - min_label);

            training_data[i] = std::tuple<double, double, double>(new_pos, new_angle, new_label);
        }

        min_pos = min_angle = min_label = 0;
        max_pos = max_angle = max_label = 1;
    }

    return TrainingData{
        max_pos,
        min_pos,
        max_angle,
        min_angle,
        max_label,
        min_label,
        training_data};
}

NeuralNetwork initialize_network(const TrainingData& training_data, const NNParams& params) {
    unsigned function_num = (unsigned)sqrt(params.rule_num);

    double pos_width = (training_data.max_pos - training_data.min_pos) / (function_num / 2.0);
    std::vector<double> pos_centers = linspace(training_data.min_pos, training_data.max_pos, function_num);
    double angle_width = (training_data.max_angle - training_data.min_angle) / (function_num / 2.0);
    std::vector<double> angle_centers = linspace(training_data.min_angle, training_data.max_angle, function_num);

    std::vector<Rule> rules(params.rule_num);

    for (unsigned i = 0; i < params.rule_num; i++) {
        unsigned j = i / function_num;
        unsigned k = i % function_num;
        MemberFunc pos_func = MemberFunc(pos_centers[j], pos_width);
        MemberFunc angle_func = MemberFunc(pos_centers[k], angle_width);
        std::vector<MemberFunc> funcs;
        funcs.reserve(2);
        funcs.emplace_back(pos_func);
        funcs.emplace_back(angle_func);
        Rule r = Rule(funcs);
        rules[i] = r;
    }

    return NeuralNetwork(rules, params);
}

std::vector<double> linspace(double start, double end, unsigned total_num) {
    double step = (end - start) / (total_num - 1);
    std::vector<double> result(total_num);
    for (unsigned i = 0; i < (total_num - 1); i++) {
        result[i] = start + i * step;
    }
    result[total_num - 1] = end;  //to guarantee that the start and end is the same as given, required as float calculation is unprecise
    return result;
}
