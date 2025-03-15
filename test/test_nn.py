# TODO: import dependencies and write unit tests below
import numpy as np

# required functions from nn module
from nn.io import read_text_file, read_fasta_file
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs,  sample_seqs

# Use scikit-learn to check the correctness of our loss functions (bce, mse)
from sklearn.metrics import log_loss # bce
from sklearn.metrics import mean_squared_error # mse

def test_single_forward():
    """
    Check that a single forward pass of the neural network is correct.
    """

    pass

def test_forward():
    """
    Check that a forward pass of the neural network is correct.
    """

    pass

def test_single_backprop():
    """
    Check that a single backpropagation of the neural network is correct.
    """

    pass

def test_predict():
    """
    Ensure that the calculated prediction of the neural network is correct.
    """
    pass

def test_binary_cross_entropy():
    """
    Ensure that the binary_cross_entropy function correctly calculates the binary cross entropy loss
    by comparing to the value calculated by sklearn's log_loss function.

    The example tested:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    binary cross entropy = 0.164252

    """
    # initialize the neural network model so that we can use the binary_cross_entropy function
    nn_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                                                          {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                                                        lr = 0.5, 
                                                       seed = 42, 
                                                       batch_size = 5, 
                                                       epochs = 1, 
                                                      loss_function='bce'
                                                      )
    
    # Make an sample example of true and predicted y values to test the binary_cross_entropy function
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # Calculate loss using scikit learn's log_loss function
    sklearn_loss = log_loss(y_true, y_pred)

    # Calculate the binary cross entropy using nn module _binary_cross_entropy function
    nn_loss = nn_eg_model._binary_cross_entropy(y_true, y_pred)

    # Compare binary cross entropy loss to sklean's calculation
    assert np.isclose(nn_loss, sklearn_loss), "Binary cross entropy loss calculation was incorrect, does not match sklearn's log_loss function"
    

def test_binary_cross_entropy_backprop():
    """
    Ensure that the binary_cross_entropy_backprop function correctly calculates the binary cross entropy backpropagation. 
    """
    pass

def test_mean_squared_error():
    """
    Ensure that the mean_squared_error function correctly calculates the mean squared error loss 
    by comparing to the value calculated by sklearn's mean_squared_error function.

    The example tested:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    mean squared error = 0.025

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
   # Make an sample example of true and predicted y values to test the mean_squared_error function
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

   # Calculate loss using scikit learn's log_loss function
    sklearn_loss = mean_squared_error(y_true, y_pred)

    # Calculate the mean squared error using nn module _mean_squared_error function
    nn_loss = nn_eg_model._mean_squared_error(y_true, y_pred)
    
    # Compare mean squared error loss to sklean's calculation
    assert np.isclose(nn_loss, sklearn_loss), "Mean squared error loss calculation was incorrect"

def test_mean_squared_error_backprop():
    """
    Ensure that the mean_squared_error_backprop function correctly calculates the mean squared error backpropagation
    """

   # initialize the neural network model 
    nn_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
                                                                          {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}],
                                                        lr = 0.5, 
                                                        seed = 42, 
                                                        batch_size = 5, 
                                                        epochs = 1, 
                                                        loss_function='mse'
                                                        )
   # Make an sample example of true and predicted y values to test the mean_squared_error backpropagation
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # Calculate the mean squared error backpropagation
    mse_backprop = nn_eg_model._mean_squared_error_backprop(y_true, y_pred)

    # Check that the mean squared error backpropagation is correct
    assert np.allclose(mse_backprop, np.array([-0.1, 0.1, -0.2, 0.2])), "Mean squared error backpropagation was incorrect"



def test_sample_seqs():
    """
    Ensure that the sample_seqs function correctly samples sequences, and labels to account for class imbalance.
    Checks that the sampled sequences and labels are the expected length, and that the sampling balances the classes.

    """
    # Create a list of sequences and labels to test the sample_seqs function
    seqs = ['AGA', 'TGC', 'CTA', 'GAT']
    labels = [True, True, True, False]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # Check that the sampled sequences and labels are the expected format / contain the expeted values
    assert all(isinstance(x, bool) for x in labels), "Sampling of sequences is incorrect, sampledlabels should contain only True and False values"
    #assert set(sample_seqs).issubset(set(seqs)), "Sampling of sequences is incorrect, sampled seqs should only contain sequences in subset original list of sequences"

   # Check the length of the sampled sequences and labels are correct, and that the sampling balances the classes
    assert len(sampled_seqs) == 6, "Sampling of sequences is incorrect, resampling by sample_seqs produced the wrong number of sequences"
    assert sum(sampled_labels) == 3, "Sampling of labels is incorrect, resampling by sample_seqs produced the wrong number of positive labels"
    assert sampled_labels.count(True) == sampled_labels.count(False), "Sampling of labels is incorrect, resampling by sample_seqs produced an imbalanced number of positive and negative labels"


def test_one_hot_encode_seqs():
    """
    Ensure that the one_hot_encode_seqs function correctly one-hot-encodes sequences. 
    Checks that one-hot-encoded sequence is correct length, and encoding matches expected output.

    The example tested:
    AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encoded_seqs = one_hot_encode_seqs(['AGA']) 
   
    assert len(encoded_seqs[0]) == 12, "One-hot encoding is incorrect"

    print(encoded_seqs)

    assert np.array_equal(encoded_seqs[0], [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), "One-hot encoding is incorrect"