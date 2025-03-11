# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Linear transformed matrix = weight times input + bias
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        
        # then apply transformation with activation function: 
        if activation.lower() == "sigmoid":
           A_curr = self._sigmoid(Z_curr)
           
        elif activation.lower() == "relu":
           A_curr = self._relu(Z_curr)
           
        else:
          raise ValueError("Activation function should be one of: sigmoid, relu")
        
        return (A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialise variables
        cache = {}
        
        # Add first layer to cache (this is just the input layer)
        cache['A0'] = X
        
        # First A_prev is the input
        A_prev = X
        
        # Traverse each layer .. 
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            
            # Get parameters required for _single_forward
            activation = layer['activation']
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            
            A_curr, Z_curr = self._single_forward(
                                                  W_curr,
                                                  b_curr,
                                                  A_prev,
                                                  activation
                                                  ) 
           
            # Update cache
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr
            
            # Update A_prev for next layer
            A_prev = A_curr
        
        output = A_prev
        
        return (output, cache)

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        
        # Derivative of activation function
        if activation_curr.lower() == "sigmoid":
           dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() == "relu":
           dZ = self._relu_backprop(dA_curr, Z_curr)
        else:
          raise ValueError("Activation function should be one of: sigmoid, relu")
        
        # Then, partial derivative of loss function, with respect to:
        # previous layer activation matrix
        dA_prev = np.dot(dZ, W_curr)
        # current layer weight matrix: from fomula slide 29/43 in neural networks lecture
        dW_curr = np.dot(dZ, A_prev)
        # current layer bias matrix: from formula slide 29/43 in neural networks lecture
        db_curr = np.sum(dZ)
        
        return (dA_prev, dW_curr, db_curr)
        
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]) -> Dict[str, ArrayLike]:
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Initialize variables
        grad_dict = {}
        
        # Calculate dA_curr - as required argument for _single_backprop 
        if self._loss_func.lower() == "mse":
           dA_curr = self.mean_squared_error_backprop(y, y_hat)
        elif self._loss_func.lower() == "bce":
           dA_curr = self.binary_cross_entropy_backprop(y, y_hat)
        else:
           raise ValueError("Loss function should be one of: mse, bce")
        
        # Traverse backwards across every layer ... 
        for idx in range(len(self.arch) - 1, -1, -1):
            
            layer = self.arch[idx]
            layer_idx = idx + 1 

            # Get additional required parameter values for _single_backprop step
            activation_curr = layer['activation']
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx - 1)]
            
            # Perform single backprop step            
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr,
                                                              b_curr,
                                                              Z_curr,
                                                              A_prev,
                                                              dA_curr,
                                                              activation_curr)
                                                              
            
            # Update grad_dict                                                   
            grad_dict['W' + str(layer_idx)] =  dW_curr
            grad_dict['b' + str(layer_idx)] =  db_curr
            
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        
        # for each node parameters across every layer ... 
        for idx, layer in enumerate(self.arch):
          
            layer_idx = idx + 1
            
            # update using gradient descent, i.e.
            # using formula on slide 12/43 neural network lecture
            # update by subtracting learning rate * gradient
            self.param_dict['W' + str(layer_idx)] -= self.lr * grad_dict['W' + str(layer_idx)]
            self.param_dict['b' + str(layer_idx)] -= self.lr * grad_dict['b' + str(layer_idx)]
        

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # initialise variables
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        
        # define error function
        if self._loss_func.lower() == "mse":
           error_fn = self.mean_squared_error()
        elif self._loss_func.lower() == "bce":
           error_fn = self.binary_cross_entropy()
        else:
           raise ValueError("Loss function should be one of: mse, bce")
        
        for epoch in range(self._epochs):
          
            # Shuffling the training data for each epoch of training
            # Shuffling code taken from HW7-regression/regression/logreg.py
            shuffle_arr = np.concatenate([X_train, np.expand_dims(y_train, 1)], axis=1)
            np.random.shuffle(shuffle_arr)
            X_train = shuffle_arr[:, :-1]
            y_train = shuffle_arr[:, -1].flatten()
                  
            # Create batches (also taken from HW7-regression/regression/logreg.py)
            num_batches = int(X_train.shape[0] / self._batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)
            
            per_batch_loss = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X_train, y_train in zip(X_batch, y_batch):            
            
                   # steps taken from slide 27/43 of neural networks lecture
                   
                   # step 1: forward pass
                   y_pred, cache = self.forward(X_train)
                   
                   # step 2. measure error
                   error = error_fn(y_train, y_pred)
                   per_batch_loss.append(error)
                   
                   # step 3. backward pass
                   grad_dict = self.backprop(y_train, y_pred, cache)
                  
                   # step 4. do standard gradient descent 
                   self._update_params(grad_dict)
            
            
            # Calculate per epoch loss on training set  
            per_epoch_loss_train.append(np.mean(per_batch_loss))
            
            # Calculate per epoch loss on validation set 
            y_pred_val = self.predict(X_val)
            per_epoch_loss_val.append(self.error_fn(y_val, y_pred_val))
            
        
        return (per_epoch_loss_train, per_epoch_loss_val)
      
        
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # prediction is same as one forward pass
        y_hat, _ = self.forward(X)
        
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        
        nl_transform = 1 / (1 + np.exp(-Z))
        
        return nl_transform
      
    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # dervivative of sigmoid with respect to Z
        # d/dZ( 1 / (1 + np.exp(-Z)) ) = exp (-Z) / (1 + exp(-Z)) ^ 2
        # equiv to: sigmoid(Z) * (1 - sigmoid(Z))
        # https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60 
        sig_Z = self._sigmoid(Z)
        
        dZ = dA * sig_Z * (1 - sig_Z)
        
        return dZ
        

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        
        nl_transform = np.maximum(0, Z)
        
        return nl_transform


    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # reLu gradient is either 0, or 1
        # if Z>0:
        #    # reLu gradient is 1, so dZ = 1 * dA
        #    dZ = dA
        # else: 
        #   # reLu gradient is 0, so dZ = 9 * dA
        #    dZ = 0
        #    
        dZ = np.where(Z > 0, dA, 0)
        
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Step 1. Prepare inputs
        # Handle potential numerical stability issues
        # By adding small epsilon to prevent log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        # Step 1. Calculate loss
        # calculate binary cross-entropy loss = - (y * log(p) + (1 - y) * log(1 - p))
        neg_losses = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        loss = -1 * np.mean(neg_losses)

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        
        # Step 1. Prepare inputs
        # Handle potential numerical stability issues
        # By adding small epsilon to prevent log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Take derivative of losses: - y * (np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        dA_unscaled = - (y *  1 /(y_hat) + (1- y) * 1 / (1 - y_hat))
        # scale dervivative by batch size
        dA = dA_unscaled / len(y)
        
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        
        mse = np.mean((y - y_hat) ** 2)
        
        return mse
      

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # calculate mse dervivative, scaled by batch size: (y - y_hat) ** 2 / n
        dA = - 2 * (y - y_hat) / len(y)
        
        return dA
