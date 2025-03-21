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

        # Check input matrix dimensions
        # W_curr shape = (m,n) where m is the number of neurons in the current layer and n is the number of neurons in the prior layer
        # A_prev shape = (n, batch_size) where n is the number of neurons in the prior layer and batch_size is the number of samples
        # b_curr shape = (m, 1) where m is the number of neurons in the current layer

        if W_curr.shape[1] != A_prev.shape[1]: 
            raise ValueError(f"Matrix dimensions do not match: W_curr.shape={W_curr.shape}, A_prev.shape={A_prev.shape}")
    
        if W_curr.shape[0] != b_curr.shape[0]:
            raise ValueError(f"Matrix dimensions do not match: W_curr.shape={W_curr.shape}, b_curr.shape={b_curr.shape}")

        # Linear transformed matrix = weight times input + bias
        #Z_curr = np.dot(W_curr, A_prev) + b_curr
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T
        
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
        dW_curr = np.dot(dZ.T, A_prev) / dZ.shape[0]  # ? Normalize by batch size
        # current layer bias matrix: from formula slide 29/43 in neural networks lecture
        db_curr= np.sum(dZ, axis=0, keepdims=True).T  #np.sum(dZ, axis=0).reshape(b_curr.shape)
        
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
           dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func.lower() == "bce":
           dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
           raise ValueError("Loss function should be one of: mse, bce")
        
        # Traverse backwards across every layer ... 
        for idx in range(len(self.arch) - 1, -1, -1):
            
            layer = self.arch[idx]
            layer_idx = idx + 1 

            activation_curr = layer['activation']
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx -1)]
            
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

            # Update dA_curr for next layer
            dA_curr = dA_prev
            
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
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['W' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['b' + str(layer_idx)]
        

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
        # Validate shapes - X should be (samples, features), y should be (samples,)
        if len(X_train.shape) != 2:
           raise ValueError(f"X_train should be 2D array, got shape {X_train.shape}")
        
        if len(X_val.shape) != 2:
           raise ValueError(f"X_val should be 2D array, got shape {X_val.shape}")
        
        # Ensure y has correct shape (samples, output_dim)
        if y_train.shape[0] != X_train.shape[0]:
           raise ValueError(f"X_train and y_train should have the same number of samples, got shapes {X_train.shape} and {y_train.shape}")
        
        try:
            if y_train.shape[1] != self.arch[-1]['output_dim']:
                raise ValueError(f"Output dimension of y_train should match output dimension of last layer in neural network architecture, got {y_train.shape[1]} and {self.arch[-1]['output_dim']}")
        except:
            if self.arch[-1]['output_dim'] == 1 and len(y_train.shape) != 1:
                raise ValueError(f"Output dimension of y_train should match output dimension of last layer in neural network architecture, got {len(y_train.shape)} and {self.arch[-1]['output_dim']}")
        
        # check requested error function is valid
        if self._loss_func.lower() != "mse" and  self._loss_func.lower() != "bce":
           raise ValueError("Loss function should be one of: mse, bce")
        
        # Deal with boolean y values
        y_train = np.asarray(y_train, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        # initialise variables
        per_epoch_loss_train = []
        per_epoch_loss_val = []


        for epoch in range(self._epochs):
          
            # Shuffling the training data for each epoch of training
            shuffled_idx = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[shuffled_idx]
            y_train_shuffled = y_train[shuffled_idx]
                  
            # Create batches (also taken from HW7-regression/regression/logreg.py)
            num_batches =  int(np.ceil(X_train_shuffled.shape[0] / self._batch_size))
            X_batch = np.array_split(X_train_shuffled, num_batches)
            y_batch = np.array_split(y_train_shuffled, num_batches)
            
            per_batch_loss = []

            # Iterate through batches (one of these loops is one epoch of training)
            for X, y in zip(X_batch, y_batch): 

                   # steps taken from slide 27/43 of neural networks lecture
                   # Ensure y is properly shaped for the network
                   if len(y.shape) == 1:
                        y = y.reshape(-1, 1)
                   
                   # step 1: forward pass
                   y_pred, cache = self.forward(X)
                   
                   # step 2. measure error
                   if self._loss_func.lower() == "mse":
                      error = self._mean_squared_error(y, y_pred)
                   elif self._loss_func.lower() == "bce":
                       error = self._binary_cross_entropy(y, y_pred)
                   
                   per_batch_loss.append(error)
                   
                   # step 3. backward pass
                   grad_dict = self.backprop(y, y_pred, cache)
                  
                   # step 4. do standard gradient descent 
                   self._update_params(grad_dict)
            
            
            # Calculate per epoch loss on training set  
            per_epoch_loss_train.append(np.mean(per_batch_loss))
            
            # Calculate per epoch loss on validation set 
            y_pred_val = self.predict(X_val)

            if self._loss_func.lower() == "mse":
                      val_error = self._mean_squared_error(y_val, y_pred_val)
            elif self._loss_func.lower() == "bce":
                       val_error = self._binary_cross_entropy(y_val, y_pred_val)

            per_epoch_loss_val.append(val_error)
            
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
        # Make sure dA and Z have the same shape
        if dA.shape != Z.shape:
            raise ValueError(f"Shape mismatch in _relu_backprop: dA shape {dA.shape}, Z shape {Z.shape}")
    
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
        #y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        # Take derivative of losses: - y * (np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        dA_unscaled =  (-y) /(y_hat + epsilon) + (1- y)  / (1 - y_hat + epsilon)

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
