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

double Rule::CalcOutput(const NFDataInput input) {
    MemberFunc& pos_Func = member_funcs[0];
    double pos_output = pos_Func.CalcOutput(input.pos);
    MemberFunc& angle_func = member_funcs[1];
    double angle_output = angle_func.CalcOutput(input.angle);

    double output = pos_output * angle_output;

    last_output = output;
    return output;
}

NFSystem::NFSystem(unsigned rule_num, double initial_rule_weight) {
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

double NFSystem::CalcOutput(const NFDataInput input) {
    double weighted_sum = 0;
    normalizer = 0;
    for (Rule& rule : rules) {
        double rule_output = rule.CalcOutput(input);
        weighted_sum += rule.GetWeight() * rule_output;
        normalizer += rule_output;
    }
    return weighted_sum / normalizer;
}

std::string NFSystem::GetRulesReport()
{
    std::string report = "Rules:\n";
    for (unsigned i = 0; i < rules.size(); i++) {
        report += "Rule " + std::to_string(i + 1) + ":\n";
        Rule r = rules[i];
        report += "Weight: " + std::to_string(r.GetWeight()) + "\n";
        for (unsigned j = 0; j < r.GetMemberFuncs().size(); j++) {
            std::string mem_name = j == 0 ? "Position: " : "Angle: ";
            auto m = r.GetMemberFuncs()[j];
            report += mem_name + "Center " + std::to_string(m.GetCenter()) + ", Width " + std::to_string(m.GetWidth()) + "\n";
        }
        report += "\n";
    }
    report += "\n";
    return report;
};

NFTrainer::NFTrainer(NFSystem* nf, const NFTrainParams& params) : nf(nf), params(params), epoch_count(0) {}

void NFTrainer::NormalizeData(const PendulumDataNormalizer* normalizer) {
    training_data = normalizer->Normalize(training_data);
    validation_data = normalizer->Normalize(validation_data);
}

bool NFTrainer::Initialize(const std::string training_data_path) {

    std::ifstream training_f(training_data_path);
    if (!training_f.is_open()) {
        return false;
    }

    double pos_input, angle_input, label;
    unsigned count = 0;
    while (training_f >> pos_input >> angle_input >> label) {
        count = (count + 1) % 10;
        if (count > (1 - params.validation_factor) * 10) {//e.g. if user set validation_factor to be 0.2, every 2 out of 10 samples will be used for validation
            validation_data.push_back(NFDataSample{{pos_input, angle_input}, label});
        } else {
            training_data.push_back(NFDataSample{{pos_input, angle_input}, label});
        }
    }
    training_f.close();

    if (params.shuffle) {
        std::random_shuffle(training_data.begin(), training_data.end());
        std::random_shuffle(validation_data.begin(), validation_data.end());
    }

    return true;
}


void NFTrainer::TrainOneIterate(const NFDataInput input, double label, unsigned& iterate_count) {
    double output = nf->CalcOutput(input);  //step 3 in paper
    for (Rule& rule : nf->GetRules()) {
        double new_weight = rule.GetWeight() - params.weight_learning_rate * (rule.GetLastOutput() / nf->GetNormalizer()) * (output - label);
        rule.SetWeight(new_weight);  //step 4 in paper
    }
    output = nf->CalcOutput(input);  //step 5 in paper
    for (Rule& rule : nf->GetRules()) {
        if (rule.GetLastOutput() == 0) {
            continue;  //inactive rules should be skipped in this epoch's backpropagation
        }
        for (unsigned j = 0; j < rule.GetMemberFuncs().size(); j++) {
            MemberFunc& func = rule.GetMemberFuncs()[j];
            if((iterate_count % params.center_move_iterate)== 0){
                //std::cout << "\n\t\tAdjusting the centre parameter..." << std::endl;
                double new_center = func.GetCenter() - params.func_center_learning_rate * (rule.GetLastOutput() / nf->GetNormalizer()) * (output - label) * (rule.GetWeight() - output) * (2 * sgn(j == 0? input.pos : input.angle - func.GetCenter())) / (func.GetLastOutput() * func.GetWidth());
                func.SetCenter(new_center);
            }

            double new_width = func.GetWidth() - params.func_center_learning_rate * (rule.GetLastOutput() / nf->GetNormalizer()) * (output - label) * (rule.GetWeight() - output) * (1 - func.GetLastOutput()) / func.GetLastOutput() * (1 / func.GetWidth());
            func.SetWidth(new_width);
        }
    }
    ++iterate_count;
};

void NFTrainer::TrainOneEpoch() {
    unsigned iterate_count = 0;
    for (auto sample : training_data) {
        TrainOneIterate(sample.input, sample.output, iterate_count);
    }
    ++epoch_count;
}

bool NFTrainer::ForceStopTraining() {
    return params.max_epoch > 0 && (epoch_count > params.max_epoch);
}

NFTester::NFTester(NFSystem *nf):nf(nf){}

double NFTester::CalcAvgError(std::vector<NFDataSample> data)
{
    double total_error = 0;
    for (auto sample: data) {
        double error = CalcError(sample.input, sample.output);

        //this will be changed later
        if (error != error) {
            continue;
        }
        //
        total_error += error;
    }
    return total_error/data.size();
}

double NFTester::CalcError(NFDataInput input, double label){
    double output = nf->CalcOutput(input);
    return pow(output - label, 2);
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


void PendulumDataNormalizer::Initialize(std::vector<NFDataSample> data)
{

    max_pos = std::numeric_limits<double>::min();
    min_pos = std::numeric_limits<double>::max();
    max_angle = std::numeric_limits<double>::min();
    min_angle = std::numeric_limits<double>::max();
    max_output = std::numeric_limits<double>::min();
    min_output = std::numeric_limits<double>::max();

    for (auto sample : data) {
        double pos_input = sample.input.pos;
        double angle_input = sample.input.angle;
        double output = sample.output;

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
        if (output > max_output) {
            max_output = output;
        }
        if (output < min_output) {
            min_output = output;
        }
    }
}

std::vector<NFDataSample> PendulumDataNormalizer::Normalize(const std::vector<NFDataSample> data) const
{
    auto result = std::vector<NFDataSample>(data.size());
    for (auto sample : data) {
        double pos = (sample.input.pos - min_pos) / (max_pos - min_pos);
        double angle = (sample.input.angle - min_angle) / (max_angle - min_angle);
        double output = (sample.output - min_output) / (max_output - min_output);
        result.emplace_back(NFDataSample{{pos, angle}, output});
    }
    return result;
}

std::vector<NFDataSample> PendulumDataNormalizer::Denormalize(const std::vector<NFDataSample> data) const
{
    auto result = std::vector<NFDataSample>(data.size());
    for (auto sample : data) {
        double pos = min_pos + sample.input.pos * (max_pos - min_pos);
        double angle = min_angle + sample.input.angle * (max_angle - min_angle);
        double output = min_output + sample.output * (max_output - min_output);
        result.emplace_back(NFDataSample{{pos, angle}, output});
    }
    return result;
}


