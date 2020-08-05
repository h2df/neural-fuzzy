#include "demo.h"
#include <iostream>

const PendulumData TRAINING_DATA{//1 training pattern
                                 angle_input : -1.69755,
                                 pos_input : 0.925215,
                                 force : 60
};

Link::Link(Node& input, Node& output) : input_node(input), output_node(output){}

InputLink::InputLink(MembershipFunc& member, Node& input, Node& output): Link(input, output), membership(member){ }

double InputLink::GetOutput() const
{
    //input is smaller than leftmost center or bigger than rightmost center
    if ((membership.type == MembershipFunc::Left && GetInputNode().GetValue() < membership.center) || (membership.type == MembershipFunc::Right && GetInputNode().GetValue() > membership.center))
    {
        return 1;
    }

    //normal memberhsip (formula is from the paper)
    if ((GetInputNode().GetValue() <= membership.center - membership.width / 2) || (GetInputNode().GetValue() >= membership.center + membership.width / 2))
    {
        return 0;
    }
    else
    {
        return 1 - 2 * std::abs(GetInputNode().GetValue() - membership.center) / membership.width;
    }
}

void InputLink::UpdateLink(double diff, double output_link_diff, double sum_of_mu) {
    double mu = GetOutputNode().GetValue();
    double sgn_xj_minus_centerj = GetInputNode().GetValue() - membership.center > 0? 1 : -1;
    membership.center = membership.center - LEARNING_RATE * (mu/sum_of_mu)*diff*(output_link_diff)*(2 * sgn_xj_minus_centerj/(GetOutput() * membership.width)) ;
    membership.width = membership.width - LEARNING_RATE * (mu/sum_of_mu)*diff*(output_link_diff)*((1-GetOutput())/GetOutput()) * (1/membership.width);
}

OutputLink::OutputLink(double factor, Node& input, Node& output): Link(input, output), weight_factor(factor){}

double OutputLink::GetOutput() const
{
    return weight_factor * GetInputNode().GetValue();
}

void OutputLink::UpdateLink(double diff, double sum_of_mu) {
    double mu = GetInputNode().GetValue();
    weight_factor = weight_factor - LEARNING_RATE * (mu/sum_of_mu) * diff;
}

int main()
{

    //Step 1: arbitrarily initialize membership functions and rule output force (weight factor of output link)
    MembershipFunc initial_membership_funcs[INPUT_NODES_NUM][MEMBER_FUNC_NUM] = {
        {{-1, 1, MembershipFunc::Left},     //angle nm
         {-0.5, 1, MembershipFunc::Normal}, //angle ns
         {0, 1, MembershipFunc::Normal},    //angle zr
         {0.5, 1, MembershipFunc::Normal},  //angle ps
         {1, 1, MembershipFunc::Right}},    //angle pm

        {{-1, 1, MembershipFunc::Left},     //pos nm
         {-0.5, 1, MembershipFunc::Normal}, //pos ns
         {0, 1, MembershipFunc::Normal},    //pos zr
         {0.5, 1, MembershipFunc::Normal},  //pos ps
         {1, 1, MembershipFunc::Right}}     //pos pm
    };

    double initial_output_weight_factors[HIDDEN_NODES_NUM];
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++)
    {
        initial_output_weight_factors[i] = 1;
    }

    //step 2: initalize nodes
    //input nodes:
    Node input_layer[INPUT_NODES_NUM];
    input_layer[0].SetValue(TRAINING_DATA.angle_input);
    input_layer[1].SetValue(TRAINING_DATA.pos_input);

    Node hidden_layer[HIDDEN_NODES_NUM];
    Node output_node;

    //step 3: initialize links from input layer to hidden layer
    std::vector<InputLink> input_links;
    input_links.reserve(HIDDEN_NODES_NUM * INPUT_NODES_NUM);
    for (unsigned i = 0; i < MEMBER_FUNC_NUM; i++)
    {
        for (unsigned j = 0; j < MEMBER_FUNC_NUM; j++)
        {
            InputLink angle_link = InputLink(initial_membership_funcs[0][i],input_layer[0],hidden_layer[i*MEMBER_FUNC_NUM+j]);
            input_links.push_back(angle_link);
            InputLink pos_link = InputLink(initial_membership_funcs[1][j],input_layer[1],hidden_layer[i*MEMBER_FUNC_NUM+j]);
            input_links.push_back(pos_link);
        }
    }

    //step 4: initialize links from hidden layer to output layer
    std::vector<OutputLink> output_links;
    output_links.reserve(HIDDEN_NODES_NUM);
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
        output_links.push_back(OutputLink(initial_output_weight_factors[i], hidden_layer[i], output_node));
    }

    //step 5: forward propagation
    std::cout << "Calculating rules:" << std::endl;
    std::string debug_memberships[5] = { "NM", "NS", "ZR", "PS", "PM"};
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++)
    {
        double angle_member_strength = input_links[i * 2].GetOutput();
        double pos_member_strength = input_links[i * 2 + 1].GetOutput();
        std::cout << "Rule " << i <<":" << std::endl;
        std::cout << "If angle_" << debug_memberships[i/MEMBER_FUNC_NUM];
        std::cout<< " and pos_" << debug_memberships[i%MEMBER_FUNC_NUM];

        hidden_layer[i].SetValue(angle_member_strength * pos_member_strength);

        std::cout<< ", output force should be " << output_links[i].GetWeightFactor() << std::endl;
        std::cout << "In this epoch, the firing strength is " << angle_member_strength << ", " << pos_member_strength <<" => " << output_links[i].GetOutput() << std::endl << std::endl; 
    }
    
    // step 3: calculate output force:
    double unnormalized_output = 0.0, sum_of_hidden = 0.0;
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++)
    {
        unnormalized_output += output_links[i].GetOutput();
        sum_of_hidden += output_links[i].GetInputNode().GetValue();
    }
    output_node.SetValue(unnormalized_output/sum_of_hidden);

    std::cout << "Unormalized output:" << unnormalized_output << ". Sum of hidden layer value: " << sum_of_hidden << ". Final output: " << output_node.GetValue() << std::endl;
    std::cout << "Desired output: " << TRAINING_DATA.force << ". Now doing back propagation." << std::endl;

    // step 4: back propagation
    double diff = output_node.GetValue() - TRAINING_DATA.force;

    double output_link_diff[HIDDEN_NODES_NUM];
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) { //this needs to be done before updating output links
        output_link_diff[i] = output_links[i].GetWeightFactor() - output_node.GetValue();
    }
    
    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++)
    {
        output_links[i].UpdateLink(diff, sum_of_hidden);
    }
    std::cout << "OutputLinks updated: " << std::endl;
    for (auto link: output_links) {
        std:: cout << "Current weight: " << link.GetWeightFactor() << std::endl;
    }

    for (unsigned i = 0; i < HIDDEN_NODES_NUM; i++) {
        input_links[i].UpdateLink(diff, output_link_diff[i], sum_of_hidden);
    }
    std::cout << "IntputLinks updated: " << std::endl;
    for (auto link : input_links) {
        std::cout << "Current center: " << link.GetMembershipFunctionCenter() << ", current width: " << link.GetMembershipFunctionWidth() << std::endl;
    }


}
