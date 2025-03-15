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
    """
    Check that a single backpropagation of the neural network is correct.
    """
    pass

def test_predict():
    """
    Check the that the calculated prediction of the neural network is correct.
    """
    pass

def test_binary_cross_entropy():

    nn_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                                                          {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                            lr = 0.5, 
                            seed = 42, 
                            batch_size = 5, 
                            epochs = 1, 
                            loss_function='bce'
                                                    )
    
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    bce = nn_eg_model._binary_cross_entropy(y_true, y_pred)

    assert np.isclose(bce, 0.164252033486018), "Binary cross entropy loss calculation was incorrect"
    

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    """
    Ensure that the mean_squared_error function correctly calculates the mean squared error
    """
   # initialize the neural network model so that we can use the mean_squared_error function
    nn_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                                                          {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                                                        lr = 0.5, 
                                                        seed = 42, 
                                                        batch_size = 5, 
                                                        epochs = 1, 
                                                        loss_function='mse'
                                                        )
   # make an sample example of true and predicted y values to test the mean_squared_error function
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # calculate the mean squared error, compare to the value calculated by hand
    mse = nn_eg_model._mean_squared_error(y_true, y_pred)

    assert np.isclose(mse, 0.035), "Mean squared error loss calculation was incorrect"

def test_mean_squared_error_backprop():
    """
    Ensure that the mean_squared_error_backprop function correctly calculates the mean squared error backpropagation
    """
    pass


def test_sample_seqs():
    """
    Ensure that the sample_seqs function correctly samples sequences
    """

    seqs = ['AGA', 'TGC', 'CTA', 'GAT']
    labels = [True, True, True, False]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    assert len(sampled_seqs) == 6, "Sampled sequences are incorrect"


def test_one_hot_encode_seqs():
    """
    Ensure that the one_hot_encode_seqs function correctly one-hot-encodes sequences. 
    Checks that one-hot-encoded sequence is correct length, and correct encoding.
    """
    encoded_seqs = one_hot_encode_seqs(['AGA']) 
   
    assert len(encoded_seqs[0]) == 12, "One-hot encoding is incorrect"

    assert np.array_equal(encoded_seqs, [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), "One-hot encoding is incorrect"