#include <vector>
#include <cmath>
#include <iostream>

#define INPUT_NODES_NUM 2
#define MEMBER_FUNC_NUM 5
#define LEARNING_RATE 0.01
#define HIDDEN_NODES_NUM 25



class Node {
    double value;
    public:
    double GetValue(){return value;}
    void SetValue(double val) { value = val;}
};

class Link
{
    protected:
        Node& input_node;
        Node& output_node;
    public:
        Link(Node& input, Node& output);
        Node& GetInputNode() const {return input_node;};
        Node& GetOutputNode() const {return output_node;};
        virtual double GetOutput() const = 0;
};

struct MembershipFunc
{
    double center;
    double width;
};

class InputLink : public Link
{
    MembershipFunc &membership;
public:
    InputLink(MembershipFunc &member, Node& input, Node& output);
    double GetOutput() const;
    double GetMembershipFunctionCenter(){return membership.center;};
    double GetMembershipFunctionWidth(){return membership.width;};
    void UpdateLink(double diff, double output_link_diff, double sum_of_mu);
};

class OutputLink : public Link
{
    double weight_factor;

public:
    OutputLink(double factor, Node& input, Node& output);
    double GetWeightFactor() {return weight_factor;}
    double GetOutput() const;
    void UpdateLink(double diff, double sum_of_mu);
};