# TODO: import dependencies and write unit tests below
import numpy as np

# required functions from nn module
from nn.io import read_text_file, read_fasta_file
from nn.nn import NeuralNetwork
from nn.preprocess import one_hot_encode_seqs,  sample_seqs

# Use scikit-learn to check the correctness of our loss functions (bce, mse)
from sklearn.metrics import log_loss # bce
from sklearn.metrics import mean_squared_error # mse

# Initialize a neural network model to use across tests (with loss function set to binary cross entropy)
nn_bce_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'},
                                                                          {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'}],
                                                    lr = 0.5, 
                                                    seed = 42, 
                                                    batch_size = 5, 
                                                    epochs = 1, 
                                                    loss_function='bce'
                                                    )

# Fake data to train model on
#X_train = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]])
#y_train = np.array([1, 0, 1, 0, 1])

#X_val = np.array([[2, 3, 4, 5], [3, 4, 5, 6]])
#y_val = np.array([0, 1])

#nn_bce_eg_model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

# set own weights and biases for the neural network for testing
nn_bce_eg_model._param_dict = {"W1": np.array([[0.5, 1, 5, 1],
                                                                               [1, 0.5, 1, 1]]),
                                                      "b1": np.array([[1], [1]]),
                                                      "W2": np.array([[2, 1]]),
                                                      "b2": np.array([[1]])
                                                     }

# Initialize a neural network model to use across tests (with loss function set to mean squared error entropy)
nn_mse_eg_model  = NeuralNetwork(nn_arch = [{'input_dim': 4, 'output_dim': 2, 'activation': 'relu'},
                                                                          {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'}],
                                                        lr = 0.5, 
                                                        seed = 42, 
                                                        batch_size = 5, 
                                                        epochs = 5, 
                                                        loss_function='mse'
                                                           )




def test_single_forward():
    """
    Check that a single forward pass of the neural network is correct:
    - check fails correctly when invalid activation function is used.
    - compare the calculated activation values and Z values to the expected values, when using the relu activation function
    - compare the calculated activation values and Z values to the expected values, when using the sigmoid activation function

    """

    # Create example to test the single forward pass of the neural network
    W_curr = np.array([[2, 1, 0.5, 1], [1,0.5, 0.5, 1]]) #(m,n) where m is the number of neurons in the current layer and n is the number of neurons in the prior layer
    A_prev = np.array([[1, 3, 1, 0], [1,0, 1, 1], [1,2, 1,0]]).T # size (n,p) where n is the number of features (neurons in prior layer) and p is the number of examples
    b_curr = np.array([[-4],[-1]]) # size (m,1) where m is the number of neurons in the current layer

    # Expected output
    true_Z_curr = np.array([[ 1.5, -0.5, 0.5], [ 2, 1.5, 1.5]])
    true_relu_A_curr = np.array([[ 1.5, 0, 0.5], [ 2, 1.5, 1.5]])
    true_sigmoid_A_curr = np.array([[ 0.81757448,  0.37754067, 0.62245933], [ 0.88079708, 0.81757448, 0.81757448]])

    # Check that the single forward pass fails correctly when an invalid activation function is used
    try:
        A_curr, Z_curr = nn_bce_eg_model._single_forward(W_curr, b_curr, A_prev, activation='invalid')
        assert False, "Single forward pass did not fail correctly, an invalid activation function was used"
    except ValueError:
        pass

    # Check that the single forward pass is correct when using the relu activation function
    A_relu_curr, Z_curr = nn_bce_eg_model._single_forward(W_curr, b_curr, A_prev, activation='relu')


    assert np.allclose(Z_curr, true_Z_curr, rtol = 1e-4), "Single forward pass was incorrect, the calculated linear transformed activation values do not match the expected values"
    assert np.allclose(A_relu_curr, true_relu_A_curr, rtol = 1e-4), "Single forward pass was incorrect, the calculated Z values with relu activation do not match the expected values"

   # Check that the single forward pass is correct when using the sigmoid activation function
    A_sigmoid_curr, Z_curr = nn_bce_eg_model._single_forward(W_curr, b_curr, A_prev, activation='sigmoid')
    assert np.allclose(A_sigmoid_curr, true_sigmoid_A_curr, rtol = 1e-4), "Single forward pass was incorrect, the calculated Z values with sigmoid activation do not match the expected values"


def test_forward():
    """
    Check that a forward pass of the neural network is correct.
    """
    # Calculate forward
    output, cache = nn_bce_eg_model.forward(np.array([2, 1, 1, 2]))


    print(cache)

    print(output)

    # Check that the forward pass output is the correct length
    #assert len(output) == 1, "Forward pass was incorrect, output dimension should have been a single value"

    # Check that the forward pass output is the correct value
    #assert np.allclose(output, 0.9990889488055994, rtol = 1e-4), "Forward pass was incorrect, the output value does not match the expected value"

    # Check that the forward pass output matches the cache
    #assert np.allclose(output, cache['A_curr'], rtol = 1e-4), "Forward pass was incorrect, the output value does not match the cache value"





    pass




def test_single_backprop():
    """
    Check that a single backpropagation of the neural network is correct: 
    - check fails correctly when invalid activation function is used.
    - compare the calculated dA_prev, dW_curr, and db_curr values to the expected values, when using the relu activation function
    """

    # Create example to test the single backpropagation of the neural network
    W_curr = np.array([[2, 1, 0.5, 1], [1,0.5, 0.5, 1]]) #(m,n) where m is the number of neurons in the current layer and n is the number of neurons in the prior layer
    A_prev = np.array([[1, 3, 1, 0], [1,0, 1, 1], [1,2, 1,0]]).T # size (n,p) where n is the number of features (neurons in prior layer) and p is the number of examples
    b_curr = np.array([[-4],[-1]]) # size (m,1) where m is the number of neurons in the current layer
    dA_curr = np.array([[1, 2, 1], [1, 1, 1]]) # size (m,p) where m is the number of neurons in the current layer and p is the number of examples
    Z_curr = np.array([[ 1.5, -0.5, 0.5], [ 2, 1.5, 1.5]]) # size (m,p) where m is the number of neurons in the current layer and p is the number of examples
    A_curr = np.array([[ 1.5, 0, 0.5], [ 2, 1.5, 1.5]]) # size (m,p) where m is the number of neurons in the current layer and p is the number of examples

    try:
        dA_prev, dW_curr, db_curr=nn_bce_eg_model._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'invalid')
        assert False, "Single backpropagation did not fail correctly, an invalid activation function was used"
    except ValueError:
        pass
   
    # Now calculate and check the single backpropagation with the relu activation function
    dA_prev, dW_curr, db_curr = nn_bce_eg_model._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'relu')

    # Check that the single backpropagation output is the correct length
    assert len(dA_prev) == len(A_prev), "Single backpropagation was incorrect, the length of the dA_prev should be the same as the length of the A_prev"
    assert len(dW_curr) == len(W_curr), "Single backpropagation was incorrect, the length of the dW_curr should be the same as the length of the W_curr"
    assert len(db_curr) == len(b_curr), "Single backpropagation was incorrect, the length of the db_curr should be the same as the length of the b_curr"

    # set expected output - calculated by hand
    # to check the correctness of the backpropagation 
    true_dA_prev = np.array([[
        0.5, 0.5, 0.5, 0.5],
        [ 1.5, 1.5, 1.5, 1.5],
        [ 0.5, 0.5, 0.5, 0.5]
    ]).T

    true_dW_curr = np.array([
        [ 1.5, 1.5, 1.5, 1.5],
        [ 1.5, 1.5, 1.5, 1.5]
    ])

    true_db_curr = np.array([
        [ 3],  
        [ 3]
    ])


    # Check that the single backpropagation output matches the expected values
    print(dA_prev)
    print(dW_curr)
    print(db_curr)

    assert np.allclose(dA_prev, true_dA_prev, rtol = 1e-4), "Single backpropagation was incorrect, the calculated dA_prev values do not match the expected values"
    assert np.allclose(dW_curr, true_dW_curr, rtol = 1e-4), "Single backpropagation was incorrect, the calculated dW_curr values do not match the expected values"  
    assert np.allclose(db_curr, true_db_curr, rtol = 1e-4), "Single backpropagation was incorrect, the calculated db_curr values do not match the expected values"





def test_predict():
    """
    Ensure that the calculated prediction of the neural network is correct: 
    - (?TO DO): check won't predict if the model has not been trained
    - check that the prediction is the correct length
    - check that the prediction is the correct value
    - check that the prediction matches the output of the forward pass
    """
    
    pred = nn_bce_eg_model.predict(np.array([2, 1, 1, 2]))

    assert len(pred) == 1, "Prediction was incorrect, output dimension should have been a single value"

    print(pred)

    assert np.allclose(pred, 0.9990889488055994, rtol = 1e-4), "Prediction was incorrect, the predicted value does not match the expected value"

    assert np.allclose(pred, nn_bce_eg_model.forward(np.array([2, 1, 1, 2]))[0], rtol = 1e-4), "Prediction was incorrect, the predicted value does not match the forward pass output"





def test_binary_cross_entropy():
    """
    Ensure that the binary_cross_entropy function correctly calculates the binary cross entropy loss
    by comparing to the value calculated by sklearn's log_loss function.

    The example tested:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    binary cross entropy = 0.164252

    """

    # Make an sample example of true and predicted y values to test the binary_cross_entropy function
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

    # Calculate loss using scikit learn's log_loss function
    sklearn_loss = log_loss(y_true, y_pred)

    # Calculate the binary cross entropy using nn module _binary_cross_entropy function
    nn_loss = nn_bce_eg_model._binary_cross_entropy(y_true, y_pred)

    # Compare binary cross entropy loss to sklean's calculation
    assert np.isclose(nn_loss, sklearn_loss), "Binary cross entropy loss calculation was incorrect, does not match sklearn's log_loss function"
    



def test_binary_cross_entropy_backprop():
    """
    Ensure that the binary_cross_entropy_backprop function correctly calculates the binary cross entropy backpropagation. 

   The example tested:
    y = np.array([1, 0, 1, 0])
    y_hat = np.array([0.9, 0.1, 0.8, 0.2])

    true_bce_backprop = np.array([-0.2778,-0.2778, -0.3125, -0.3125])

    """

   # Make an example of true and predicted y values to test the binary cross entropy backpropagation
    y = np.array([1, 0, 1, 0])
    y_hat = np.array([0.9, 0.1, 0.8, 0.2])
   
   # Calculate the binary cross entropy backpropagation
    bce_backprop = nn_bce_eg_model._binary_cross_entropy_backprop(y = y, y_hat = y_hat)

    # True binary cross entropy values, calculated by hand
    true_bce_backprop = np.array([-0.2778, -0.2778, -0.3125, -0.3125])

    print(bce_backprop)
    # Check that binary cross entropy backpropagation output is the correct length
    assert len(bce_backprop) == len(y), "Mean squared error backpropagation was incorrectly performed, the length of the backpropagation should be the same as the length of the true values"

    # Check that the binary cross entropy backpropagation match the expected values calculated by hand
    assert np.allclose(bce_backprop, true_bce_backprop, rtol = 1e-4), "Mean squared error backpropagation was incorrect, the backpropagation values do not match expected values"



def test_mean_squared_error():
    """
    Ensure that the mean_squared_error function correctly calculates the mean squared error loss 
    by comparing to the value calculated by sklearn's mean_squared_error function.

    The example tested:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    mean squared error = 0.025

    """

   # Make an sample example of true and predicted y values to test the mean_squared_error function
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])

   # Calculate loss using scikit learn's log_loss function
    sklearn_loss = mean_squared_error(y_true, y_pred)

    # Calculate the mean squared error using nn module _mean_squared_error function
    nn_loss = nn_mse_eg_model._mean_squared_error(y_true, y_pred)
    
    # Compare mean squared error loss to sklean's calculation
    assert np.isclose(nn_loss, sklearn_loss), "Mean squared error loss calculation was incorrect, does not match sklearn's mean_squared_error function"




def test_mean_squared_error_backprop():
    """
    Ensure that the mean_squared_error_backprop function correctly calculates the mean squared error backpropagation 
    by comparing its output to the value calculated by hand.

    The example tested:
    y = np.array([1, 0, 1, 0])
    y_hat = np.array([0.9, 0.1, 0.8, 0.2])

    true_mse_backprop = np.array([-0.05,  0.05, -0.10,  0.10])

    """
   # Make an example of true and predicted y values to test the mean_squared_error backpropagation
    y = np.array([1, 0, 1, 0])
    y_hat = np.array([0.9, 0.1, 0.8, 0.2])

    # Calculate the mean squared error backpropagation
    mse_backprop = nn_mse_eg_model._mean_squared_error_backprop(y = y, y_hat = y_hat)

    # True mean squared error backpropagation values, calculated by hand
    true_mse_backprop = np.array([-0.05,  0.05, -0.10,  0.10])
   
    # Check that the mean squared error backpropagation output is the correct length
    assert len(mse_backprop) == len(y), "Mean squared error backpropagation was incorrectly performed, the length of the backpropagation should be the same as the length of the true values"
   
    # Check that the mean squared error backpropagation match the expected values calculated by hand
    assert np.allclose(mse_backprop, true_mse_backprop, rtol = 1e-4), "Mean squared error backpropagation was incorrect, the backpropagation values do not match expected values"




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
    # Test the one_hot_encode_seqs function with an example sequence
    encoded_seqs = one_hot_encode_seqs(['AGA']) 
   
    # Check that the one-hot-encoded sequence is the correct length (4 * sequence length)
    assert len(encoded_seqs[0]) == 12, "One-hot encoding is incorrect"
   
    # Check that the one-hot-encoded sequence matches the expected output
    assert np.array_equal(encoded_seqs[0], [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]), "One-hot encoding is incorrect"