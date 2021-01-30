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
    weight = weight_val;
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

NFSystem::NFSystem(unsigned rule_num, double initial_rule_weight, double min_pos, double max_pos, double min_angle, double max_angle, double min_label, double max_label) :
    min_pos(min_pos), max_pos(max_pos),
    min_angle(min_angle), max_angle(max_angle),
    min_label(min_label), max_label(max_label){
    unsigned function_num = (unsigned)sqrt(rule_num);

    double pos_width = 1 / (function_num / 2.0);
    std::vector<double> pos_centers = linspace(0, 1, function_num);
    double angle_width = 1 / (function_num / 2.0);
    std::vector<double> angle_centers = linspace(0, 1, function_num);

    rules = std::vector<Rule>(rule_num);

    for (unsigned i = 0; i < rule_num; i++) {
        unsigned j = i / function_num;
        unsigned k = i % function_num;
        MemberFunc pos_func = MemberFunc(pos_centers[j], pos_width);
        MemberFunc angle_func = MemberFunc(angle_centers[k], angle_width);
        std::vector<MemberFunc> funcs;
        funcs.reserve(2);
        funcs.emplace_back(pos_func);
        funcs.emplace_back(angle_func);
        Rule r = Rule(funcs);
        r.SetWeight(initial_rule_weight);
        rules[i] = r;
    }
}

double NFSystem::CalcOutput(const std::vector<double>& inputs) {
    double weighted_sum = 0;
    normalizer = 0;
    for (Rule& rule : rules) {
        double rule_output = rule.CalcOutput(inputs);
        weighted_sum += rule.GetWeight() * rule_output;
        normalizer += rule_output;
    }
    return weighted_sum / normalizer;
};

NFTrainer::NFTrainer(const NFTrainParams& params) : params(params) {}

void NFTrainer::Initialize(const TrainingDataParams &data_params) {
    std::ifstream training_f(data_params.training_data_path);
    if (!training_f.is_open()) {
        return;
    }

    double max_pos = std::numeric_limits<double>::min();
    double min_pos = std::numeric_limits<double>::max();
    double max_angle = std::numeric_limits<double>::min();
    double min_angle = std::numeric_limits<double>::max();
    double max_label = std::numeric_limits<double>::min();
    double min_label = std::numeric_limits<double>::max();

    double pos_input, angle_input, label;
    unsigned count = 0;
    while (training_f >> pos_input >> angle_input >> label) {
        if (params.use_validation) {
            //for the ease of implementation, 20% evenly sampled data will be used for validation
            if ((count++ % 4) == 0) {
                validation_data.push_back(std::tuple<double, double, double>(pos_input, angle_input, label));
                continue;
            }
        } else {
            validation_data.push_back(std::tuple<double, double, double>(pos_input, angle_input, label));
        }

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
        training_data.training_data.push_back(std::tuple<double, double, double>(pos_input, angle_input, label));
    }

    nn = NFSystem(params.rule_num, params.initial_rule_weight, min_pos, max_pos, min_angle, max_angle, min_label, max_label);

    if (data_params.shuffle) {
        std::random_shuffle(training_data.training_data.begin(), training_data.training_data.end());
    }

    //normalize training and testing data
    for (unsigned i = 0; i < training_data.training_data.size(); i++) {
        double pos = std::get<0>(training_data.training_data[i]);
        double new_pos = (pos - min_pos) / (max_pos - min_pos);

        double angle = std::get<1>(training_data.training_data[i]);
        double new_angle = (angle - min_angle) / (max_angle - min_angle);

        double label = std::get<2>(training_data.training_data[i]);
        double new_label = (label - min_label) / (max_label - min_label);

        training_data.training_data[i] = std::tuple<double, double, double>(new_pos, new_angle, new_label);
    }

    for (unsigned i = 0; i < validation_data.size(); i++) {
        double pos = std::get<0>(validation_data[i]);
        double new_pos = (pos - min_pos) / (max_pos - min_pos);

        double angle = std::get<1>(validation_data[i]);
        double new_angle = (angle - min_angle) / (max_angle - min_angle);

        double label = std::get<2>(validation_data[i]);
        double new_label = (label - min_label) / (max_label - min_label);

        validation_data[i] = std::tuple<double, double, double>(new_pos, new_angle, new_label);
    }

    training_data.valid = true;
}


void NFTrainer::TrainOneIterate(const std::vector<double>& inputs, double label, unsigned& iterate_count) {
    double output = nn.CalcOutput(inputs);  //step 3 in paper
    for (Rule& rule : nn.GetRules()) {
        double new_weight = rule.GetWeight() - params.weight_learning_rate * (rule.GetLastOutput() / nn.GetNormalizer()) * (output - label);
        rule.SetWeight(new_weight);  //step 4 in paper
    }
    output = nn.CalcOutput(inputs);  //step 5 in paper
    for (Rule& rule : nn.GetRules()) {
        if (rule.GetLastOutput() == 0) {
            continue;  //inactive rules should be skipped in this epoch's backpropagation
        }
        for (MemberFunc& func : rule.GetMemberFuncs()) {
            if((iterate_count % params.center_move_iterate)== 0){
                //std::cout << "\n\t\tAdjusting the centre parameter..." << std::endl;
                double new_center = func.GetCenter() - params.func_center_learning_rate * (rule.GetLastOutput() / nn.GetNormalizer()) * (output - label) * (rule.GetWeight() - output) * (2 * sgn(inputs[0] - func.GetCenter())) / (func.GetLastOutput() * func.GetWidth());
                func.SetCenter(new_center);
            }

            double new_width = func.GetWidth() - params.func_center_learning_rate * (rule.GetLastOutput() / nn.GetNormalizer()) * (output - label) * (rule.GetWeight() - output) * (1 - func.GetLastOutput()) / func.GetLastOutput() * (1 / func.GetWidth());
            func.SetWidth(new_width);
        }
    }
    ++iterate_count;
};

void NFTrainer::TrainOneEpoch() {
    unsigned iterate_count = 0;
    for (auto sample : training_data.training_data) {
        std::vector<double> inputs;
        inputs.reserve(2);
        inputs.emplace_back(std::get<0>(sample));
        inputs.emplace_back(std::get<1>(sample));
        double label = std::get<2>(sample);
        TrainOneIterate(inputs, label, iterate_count);
    }
    ++epoch_count;
}

// void NNTrainer::AdjustLearningRates(){
//     params.weight_learning_rate = params.weight_learning_rate / 10.0;
//     params.func_center_learning_rate = params.func_center_learning_rate / 10.0;
//     params.func_width_learning_rate = params.func_width_learning_rate / 10.0;
// }

double NFTrainer::CalcError(const std::vector<double>& inputs, double label) {
    double output = nn.CalcOutput(inputs);
    return pow(output - label, 2);
}

double NFTrainer::CalcValidationError() {
    double total_error = 0;
    for (auto sample: validation_data) {
        std::vector<double> inputs;
        inputs.reserve(2);
        inputs.emplace_back(std::get<0>(sample));
        inputs.emplace_back(std::get<1>(sample));
        double label = std::get<2>(sample);
        double error = CalcError(inputs, label);
        total_error += error;
    }
    return total_error/validation_data.size();
}

bool NFTrainer::HasTrainingDataReady() {
    return training_data.valid;
}

bool NFTrainer::ForceStopTraining() {
    return params.max_epoch > 0 && (epoch_count > params.max_epoch);
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

double sgn(double val) {
    auto sign =  (0 < val) - (val < 0);
    return (double)sign;
}
