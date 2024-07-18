import numpy as np
def caculate_moment(samples, order):
    return (sum(samples ** order) / len(samples))   

def calculate_joint_moment(samples_X, samples_Y):
    return np.mean(samples_X * samples_Y)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

print(calculate_joint_moment(x, y))

parameter = 3
def distance(x, y):
    return np.sqrt(np.power(x, parameter) + np.power(y, parameter))

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

print(distance(x, y))

