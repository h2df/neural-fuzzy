#include <fstream>
#include <limits>
#include <vector>
#include "pendulum-data.h"


class TrainingDataReader {
    double min_angle_input, max_angle_input;
    double min_pos_input, max_pos_input;
    double min_output, max_output;
    std::vector<PendulumData>* training_data;

   public:
    TrainingDataReader();
    bool LoadTrainingData(std::string training_file);
    std::vector<PendulumData>* GetRawTrainingData() { return training_data; };
    std::vector<PendulumData>* GetNormalizedTrainingData();
    double GetMinAngleInput() { return min_angle_input; };
    double GetMaxAngleInput() { return max_angle_input; };
    double GetMinPosInput() { return min_pos_input; };
    double GetMaxPosInput() { return max_pos_input; };
};