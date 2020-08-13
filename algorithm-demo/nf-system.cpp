#include "nf-system.h"


Link::Link(Node& input, Node& output) : input_node(input), output_node(output) {}

InputLink::InputLink(MembershipFunc& member, Node& input, Node& output) : Link(input, output), membership(member) {}

double InputLink::GetOutput() const {
    if ((GetInputNode().GetValue() <= membership.center - membership.width / 2) || (GetInputNode().GetValue() >= membership.center + membership.width / 2)) {
        return 0;
    } else {
        return 1 - 2 * std::abs(GetInputNode().GetValue() - membership.center) / membership.width;
    }
}

void InputLink::UpdateLink(double diff, double output_link_diff, double sum_of_mu) {
    double center = membership.center;
    double width = membership.width;
    double output = GetOutput();
    double sgn_xj_minus_centerj = GetInputNode().GetValue() - membership.center > 0 ? 1 : -1;

    if (sum_of_mu == 0.0) {
        std::cout << "\t\t(InputLink) Warning: sum of mu is ZERO." << std::endl;
        return;
    }

    double mu = GetOutputNode().GetValue();
    if (mu == 0.0) {
        std::cout << "\t\t(InputLink) Warning: mu is ZERO." << std::endl;
        return;
    }

    membership.center = center - LEARNING_RATE * (mu / sum_of_mu) * diff * output_link_diff * (2 * sgn_xj_minus_centerj / (output * width));
    membership.width = width - LEARNING_RATE * (mu / sum_of_mu) * diff * output_link_diff * ((1 - output) / output) * (1 / width);
}

OutputLink::OutputLink(double factor, Node& input, Node& output) : Link(input, output), weight_factor(factor) {}

double OutputLink::GetOutput() const {
    return weight_factor * GetInputNode().GetValue();
}

void OutputLink::UpdateLink(double diff, double sum_of_mu) {
    double mu = GetInputNode().GetValue();
    if (sum_of_mu == 0.0) {
        std::cout << "\t\t(OutputLink) Warning: sum of mu is zero." << std::endl;
        return;
    }
    weight_factor = weight_factor - LEARNING_RATE * (mu / sum_of_mu) * diff;
}


