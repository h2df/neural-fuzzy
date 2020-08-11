#include "training-data-reader.h"
#include <iostream>
TrainingDataReader::TrainingDataReader() : min_angle_input(std::numeric_limits<double>::max()),
                                           max_angle_input(std::numeric_limits<double>::min()),
                                           min_pos_input(std::numeric_limits<double>::max()),
                                           max_pos_input(std::numeric_limits<double>::min()),
                                           min_output(std::numeric_limits<double>::max()),
                                           max_output(std::numeric_limits<double>::min()),
                                           training_data(new std::vector<PendulumData>()) {}

bool TrainingDataReader::LoadTrainingData(std::string training_file) {
    std::ifstream file(training_file);
    if (!file.is_open()) {
        return false;
    }

    double angle_input, pos_input, output;
    while (file >> angle_input >> pos_input >> output) {
        training_data->push_back(PendulumData{angle_input, pos_input, output});
        if (angle_input > max_angle_input) {
            max_angle_input = angle_input;
        }
        if (angle_input < min_angle_input) {
            min_angle_input = angle_input;
        }
        if (pos_input > max_pos_input) {
            max_pos_input = pos_input;
        }
        if (pos_input < min_pos_input) {
            min_pos_input = pos_input;
        }
        if (output > max_output) {
            max_output = output;
        }
        if (output < min_output) {
            min_output = output;
        }
    }
    file.close();
    return true;
};

std::vector<PendulumData>* TrainingDataReader::GetNormalizedTrainingData(){
    std::vector<PendulumData>* normalized = new std::vector<PendulumData>();
    normalized->reserve(training_data->size());
    for(unsigned i = 0; i < training_data->size(); i++) {
        normalized->push_back(PendulumData {
            training_data->at(i).angle_input,
            training_data->at(i).pos_input,
            -1 + (training_data->at(i).output - min_output)/(max_output - min_output) * 2
        });
    }
    return normalized;
};