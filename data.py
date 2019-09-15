import numpy as np


def triple_xor(sample_size):
    '''
    Generating a random dataset of 3 binary values as input and their xor value as output
    :param sample_size: size of the dataset
    :return: input values and expected output values
    '''
    X = np.random.rand(sample_size, 3)
    X = (X < 0.5).astype(np.int)

    Y = (np.sum(X, axis=1) == 1).astype(np.int)

    return X, Y
