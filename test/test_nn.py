# TODO: import dependencies and write unit tests below
import numpy as np

# required functions from nn module
from nn.io import read_text_file, read_fasta_file
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs,  sample_seqs


def test_single_forward():
    pass

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    """
    Ensure that the mean_squared_error function correctly calculates the mean squared error
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    mse = mean_squared_error(y_true, y_pred)

    assert np.isclose(mse, 0.035), "Mean squared error is incorrect"

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    """
    Ensure that the one_hot_encode_seqs function correctly encodes sequences
    """
    encoded_seqs = one_hot_encode_seqs('AGA') 


    assert np.allequal(encoded_seqs, [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), "One-hot encoding is incorrect"