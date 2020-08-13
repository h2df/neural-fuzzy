#include "nf-system.h"
#include "training-data-reader.h"

const std::string DATA_FILE = "train.dat";

const std::string debug_memberships[5] = {"NM", "NS", "ZR", "PS", "PM"};

void train_nf(std::vector<PendulumData>& training_data, MembershipFunc* membership_funcs, double* output_weight_factors, unsigned epoch = 0) {
    //declaration by layer
    Node input_layer[INPUT_NODES_NUM];

    std::vector<InputLink> input_links;
    const unsigned INPUT_LINKS_NUM = pow(MEMBER_FUNC_NUM, INPUT_NODES_NUM);
    input_links.reserve(INPUT_LINKS_NUM);

    Node hidden_layer[HIDDEN_NODES_NUM];

    std::vector<OutputLink> output_links;
    output_links.reserve(HIDDEN_NODES_NUM);

    Node output_node;

    //step 3: initialize links from input layer to hidden layer
    for (unsigned i = 0; i < MEMBER_FUNC_NUM; i++) {
        for (unsigned j = 0; j < MEMBER_FUNC_NUM; j++) {
            InputLink angle_link = InputLink(membership_funcs[i], input_layer[0], hidden_layer[i * MEMBER_FUNC_NUM + j]);
            input_links.push_back(angle_link);
            InputLink pos_link = InputLink(membership_funcs[MEMBER_FUNC_NUM + j], input_layer[1], hidden_layer[i * MEMBER_FUNC_NUM + j]);
            input_links.push_back(pos_link);
        }
    }

    //step 4: initialize links from hidden layer to output layer
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
        output_links.push_back(OutputLink(output_weight_factors[i], hidden_layer[i], output_node));
    }

    const unsigned epoch_number = epoch > 0 ? epoch : training_data.size();
    for (unsigned i = 0; i < epoch_number; i++) {
        PendulumData pattern = training_data[i];

        input_layer[0].SetValue(pattern.angle_input);
        input_layer[1].SetValue(pattern.pos_input);
        std::cout << "Angle input: " << input_layer[0].GetValue() << "; Pos input: " << input_layer[1].GetValue() << std::endl;

        //step 5: forward propagation
        std::cout << "Calculating rules:" << std::endl;
        for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
            double angle_member_strength = input_links[i * INPUT_NODES_NUM].GetOutput();
            double pos_member_strength = input_links[i * INPUT_NODES_NUM + 1].GetOutput();
            if (std::isnan(input_links[i * INPUT_NODES_NUM].GetMembershipFunctionWidth()) || std::isinf(input_links[i * INPUT_NODES_NUM].GetMembershipFunctionWidth())) {
                getchar();
            }
            std::cout << "Rule " << i << ":" << std::endl;
            std::cout << "If angle_" << debug_memberships[i / MEMBER_FUNC_NUM];
            std::cout << " and pos_" << debug_memberships[i % MEMBER_FUNC_NUM];

            hidden_layer[i].SetValue(angle_member_strength * pos_member_strength);

            std::cout << ", output force should be " << output_links[i].GetWeightFactor() << std::endl;
            std::cout << "In this epoch, the firing strength is " << angle_member_strength << ", " << pos_member_strength << " => " << output_links[i].GetOutput() << std::endl
                      << std::endl;
        }

        // step 3: calculate output force:
        double unnormalized_output = 0.0, sum_of_hidden = 0.0;
        for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
            unnormalized_output += output_links[i].GetOutput();
            sum_of_hidden += output_links[i].GetInputNode().GetValue();
        }
        output_node.SetValue(unnormalized_output / sum_of_hidden);

        std::cout << "Unormalized output:" << unnormalized_output << ". Sum of hidden layer value: " << sum_of_hidden << ". Final output: " << output_node.GetValue() << std::endl;
        std::cout << "Desired output: " << pattern.output << ". Now doing back propagation." << std::endl;

        // step 4: back propagation
        double diff = output_node.GetValue() - pattern.output;

        double output_link_diff[HIDDEN_NODES_NUM];
        for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {  //this needs to be done before updating output links
            output_link_diff[i] = output_links[i].GetWeightFactor() - output_node.GetValue();
        }

        for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
            output_links[i].UpdateLink(diff, sum_of_hidden);
        }
        std::cout << "OutputLinks updated: " << std::endl;
        for (auto link : output_links) {
            std::cout << "Current weight: " << link.GetWeightFactor() << std::endl;
        }

        for (unsigned i = 0; i < INPUT_LINKS_NUM; i++) {
            input_links[i].UpdateLink(diff, output_link_diff[i], sum_of_hidden);
        }
        std::cout << "IntputLinks updated: " << std::endl;
        for (auto link : input_links) {
            std::cout << "Current center: " << link.GetMembershipFunctionCenter() << ", current width: " << link.GetMembershipFunctionWidth() << std::endl;
        }
    }
}
int main() {
    //Step 1: load in training data and initialize membership functions and rule output force (weight factor of output link)
    TrainingDataReader reader;
    if (!reader.LoadTrainingData(DATA_FILE)) {
        std::cout << "Cannot read in training data." << std::endl;
        exit(1);
    }
    std::cout << "Angle data:\n\t  Min Angle Input: " << reader.GetMinAngleInput() << ", Max Angle Input: " << reader.GetMaxAngleInput() <<std::endl;
    std::cout << "Pos data:\n\t  Min Pos Input: " << reader.GetMinPosInput() << ", Max Pos Input: " << reader.GetMaxPosInput() <<std::endl;

    std::vector<PendulumData>* training_data = reader.GetNormalizedTrainingData();
    MembershipFunc membership_funcs[INPUT_NODES_NUM * MEMBER_FUNC_NUM];

    double angle_initial_width = (reader.GetMaxAngleInput() - reader.GetMinAngleInput()) / (MEMBER_FUNC_NUM / 2 + 1) + 0.1;
    for (unsigned i = 0; i < MEMBER_FUNC_NUM; i++) {
        membership_funcs[i] =
            {reader.GetMinAngleInput() + angle_initial_width / 2 + i * angle_initial_width,
             angle_initial_width};
    }

    double pos_initial_width = (reader.GetMaxPosInput() - reader.GetMinPosInput()) / (MEMBER_FUNC_NUM / 2 + 1) + 0.1;
    for (unsigned i = 0; i < MEMBER_FUNC_NUM; i++) {
        membership_funcs[MEMBER_FUNC_NUM + i] =
            {reader.GetMinPosInput() + pos_initial_width / 2 + i * pos_initial_width,
             pos_initial_width};
    }

    double output_weight_factors[HIDDEN_NODES_NUM];
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
        output_weight_factors[i] = 1;
    }

    for (unsigned i = 0; i < 10; i++) {
        std::cout<<membership_funcs[i].center << ", " << membership_funcs[i].width<<std::endl;
    }

    train_nf(*training_data, membership_funcs, output_weight_factors);
}