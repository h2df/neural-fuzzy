"""
This script is a prototype of implementation of the Nomura et al's neural-fuzzy system.
It reads from the train data file, trains the neural network with the descend method introduced in the paper, and show the parameters.
"""
from functools import reduce
import pandas as pd
import numpy as np

class MemberFunc():
    def __init__(self, center, width):
        self.center = center
        self.width = width

    def calc_value(self, input):
        if (self.center - self.width/2) < input and input < (self.center + self.width/2):
            self._value =  1 - 2 * np.abs(input - self.center)/self.width
        else:
            self._value =  0
    
    def __str__(self):
        return f"Center: {self.center:.3}, Width: {self.width:.3}"
    
    @property
    def value(self):
        return self._value

class Rule():
    def __init__(self, membership_funcs, weight):
        self.membership_funcs = membership_funcs
        self.weight = weight

    def calc_output(self, inputs):
        self._output = 1
        for input, func in zip(inputs, self.membership_funcs):
            func.calc_value(input)
            self._output *= func.value
    
    @property
    def output(self):
        return self._output
        

class NetWork():
    def __init__(self, rules):
        self._rules = rules

    def calc_output(self, inputs):
        weighted_sum = 0
        self._normalizer = 0
        for rule in self._rules:
            rule.calc_output(inputs)
            weighted_sum += rule.weight * rule.output
            self._normalizer += rule.output
        self._output = weighted_sum/self._normalizer

    def train_one_iterate(self, inputs, label):
        self.calc_output(inputs) #step 3 in paper
        for rule in self._rules:
            rule.weight = rule.weight - LEARNING_RATE * (rule.output/self._normalizer) * (self.output - label) # step 4 in paper
        self.calc_output(inputs) #step 5 in paper
        for rule in self._rules:
            if rule.output == 0:
                continue #inactive rules should be skipped in this epoch
            for func in rule.membership_funcs: #step 6 in paper
                func.center = func.center - LEARNING_RATE * (rule.output/self._normalizer) * (self.output - label) * (rule.weight - self.output) * (2 * np.sin(inputs[0] - func.center)/(func.value * func.width))
                func.width = func.width - LEARNING_RATE * (rule.output/self._normalizer) * (self.output - label) * (rule.weight - self.output) * ((1 - func.value)/func.value) * (1/func.width)
    
    def get_error(self, inputs, label):
        self.calc_output(inputs)
        return 0.5 * (self.output - label)**2

    @property
    def output(self):
        return self._output

if __name__ == "__main__":
    LEARNING_RATE = 0.01
    RULES = 25
    INITIAL_WEIGHT = 0.0
    NORMALIZE = False
    SHUFFLE = True
    THRESHOLD = 0.01

    estimated_function_num = int(np.sqrt(RULES))

    data = pd.read_csv('train.dat', header=None, sep=" ",index_col=False)
    # max_epoch = data.shape[0]

    if SHUFFLE:
        data = data.sample(frac=1.0)
    
    if NORMALIZE:
        data = (data - data.min()) / (data.max() - data.min())

    pos_min, pos_max = data[0].min(), data[0].max()
    pos_width = (pos_max - pos_min) / (estimated_function_num /2) # divide by 2 to ensure enough coverage
    pos_centers = np.repeat(np.linspace(pos_min, pos_max, num=estimated_function_num), estimated_function_num)
    pos_funcs = [MemberFunc(c, pos_width) for c in pos_centers]


    angle_min, angle_max = data[1].min(), data[1].max()
    angle_width = (angle_max - angle_min) / (estimated_function_num / 2)
    angle_centers = np.tile(np.linspace(angle_min, angle_max, num=estimated_function_num), estimated_function_num)
    angle_funcs = [MemberFunc(c, angle_width) for c in angle_centers]



    rules = [Rule((pos_func, angle_func), INITIAL_WEIGHT) for pos_func, angle_func in zip(pos_funcs, angle_funcs)]
    nn = NetWork(rules)
    for row in data.iterrows():
        nn.train_one_iterate(row[1][:2], row[1][2])
        error=nn.get_error(row[1][:2], row[1][2])
        print(error)
        if error < THRESHOLD:
            print("Trained Successfully")
            break

    for rule in nn._rules:
        print(f"Weight- {rule.weight:.3}", "Membership Functions- Pos: " , rule.membership_funcs[0], " Angle: ", rule.membership_funcs[1])
            




