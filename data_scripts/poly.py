import pandas as pd
import math

NUM = 500

def poly(x1, x2):
    return 2 * (x1 ** 2) + 4 * (x2 ** 3)

data = pd.DataFrame([(x1, x2, poly(x1, x2)) for x1 in range(NUM) for x2 in range(NUM)], columns=["x1", "x2", "y"]).sample(frac=1)
split = math.floor(NUM * NUM * 0.8)
training = data.iloc[: split, :]
test = data.iloc[split:, :]
training.to_csv("poly_training.dat", sep=" ", index=False, header=False)
training.to_csv("poly_test.dat", sep=" ", index=False, header=False)
