import pickle

import numpy as np


class NeuralNetwork:
    """
    3-layer neural network for AI paddle control in Pong.

    Implements a neural network with two hidden layers using ReLU activation
    for hidden layers and sigmoid for output. Includes optimizations:
    momentum, L2 regularization, gradient clipping, dropout, and batch normalization.

    Attributes:
        input_size (int): Input layer size
        hidden_size (int): Hidden layer size
        output_size (int): Output layer size
        learning_rate (float): Learning rate
        momentum (float): Momentum coefficient for optimization
        l2_lambda (float): L2 regularization coefficient
        dropout_rate (float): Neuron dropout probability
    """

    def __init__(self, input_size, hidden_size=32, output_size=1, learning_rate=0.001):
        """
        Initialize neural network.

        Args:
            input_size (int): Input layer size
            hidden_size (int): Hidden layer size (default 32)
            output_size (int): Output layer size
            learning_rate (float): Learning rate (default 0.001)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size) * np.sqrt(
            2.0 / input_size
        )
        self.weights_hidden1_hidden2 = np.random.randn(
            hidden_size, hidden_size
        ) * np.sqrt(2.0 / hidden_size)
        self.weights_hidden2_output = np.random.randn(
            hidden_size, output_size
        ) * np.sqrt(2.0 / hidden_size)

        self.bias_hidden1 = np.zeros((1, hidden_size))
        self.bias_hidden2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        self.bn_gamma1 = np.ones((1, hidden_size))
        self.bn_beta1 = np.zeros((1, hidden_size))
        self.bn_gamma2 = np.ones((1, hidden_size))
        self.bn_beta2 = np.zeros((1, hidden_size))
        self.bn_gamma3 = np.ones((1, output_size))
        self.bn_beta3 = np.zeros((1, output_size))

        self.bn_mean1 = np.zeros((1, hidden_size))
        self.bn_var1 = np.ones((1, hidden_size))
        self.bn_mean2 = np.zeros((1, hidden_size))
        self.bn_var2 = np.ones((1, hidden_size))
        self.bn_mean3 = np.zeros((1, output_size))
        self.bn_var3 = np.ones((1, output_size))

        self.momentum = 0.9
        self.velocity_w_ih1 = np.zeros_like(self.weights_input_hidden1)
        self.velocity_w_h1h2 = np.zeros_like(self.weights_hidden1_hidden2)
        self.velocity_w_h2o = np.zeros_like(self.weights_hidden2_output)
        self.velocity_b_h1 = np.zeros_like(self.bias_hidden1)
        self.velocity_b_h2 = np.zeros_like(self.bias_hidden2)
        self.velocity_b_o = np.zeros_like(self.bias_output)

        self.l2_lambda = 0.0001

        self.dropout_rate = 0.2
        self.training = True

    def relu(self, x):
        """ReLU activation function.
        Args:
            x: Input data
        Returns:
            ReLU activation function
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU derivative.
        Args:
            x: Input data
        Returns:
            ReLU derivative
        """
        return (x > 0).astype(float)

    def sigmoid(self, x):
        """Sigmoid activation function with overflow protection.
        Args:
            x: Input data
        Returns:
            Sigmoid activation function
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid derivative.
        Args:
            x: Input data
        Returns:
            Sigmoid derivative
        """
        return x * (1 - x)

    def normalize_inputs(self, inputs):
        """Normalize input data using z-score.
        Args:
            inputs: Input data
        Returns:
            Normalized input data
        """
        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        return (inputs - mean) / (std + 1e-8)

    def batch_norm(self, x, gamma, beta, mean, var, training=True):
        """
        Batch normalization with proper inference mode handling.

        Args:
            x: Input data
            gamma: Scaling parameter
            beta: Shift parameter
            mean: Mean value
            var: Variance
            training: Training/inference mode
        Returns:
            Normalized data
        """
        if training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)

            momentum = 0.9
            self.bn_mean1 = momentum * self.bn_mean1 + (1 - momentum) * batch_mean
            self.bn_var1 = momentum * self.bn_var1 + (1 - momentum) * batch_var

            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
        else:
            x_norm = (x - mean) / np.sqrt(var + 1e-8)

        return gamma * x_norm + beta

    def dropout(self, x, rate):
        """Apply dropout to input data.
        Args:
            x: Input data
            rate: Dropout rate
        Returns:
            Dropout applied data
        """
        if not self.training:
            return x

        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: Input data

        Returns:
            Network output (float): The output of the network
        """
        self.inputs = inputs

        self.hidden1_input = (
            np.dot(inputs, self.weights_input_hidden1) + self.bias_hidden1
        )
        self.hidden1_norm = self.batch_norm(
            self.hidden1_input,
            self.bn_gamma1,
            self.bn_beta1,
            self.bn_mean1,
            self.bn_var1,
            self.training,
        )
        self.hidden1_layer = self.relu(self.hidden1_norm)
        self.hidden1_dropout = self.dropout(self.hidden1_layer, self.dropout_rate)

        self.hidden2_input = (
            np.dot(self.hidden1_dropout, self.weights_hidden1_hidden2)
            + self.bias_hidden2
        )
        self.hidden2_norm = self.batch_norm(
            self.hidden2_input,
            self.bn_gamma2,
            self.bn_beta2,
            self.bn_mean2,
            self.bn_var2,
            self.training,
        )
        self.hidden2_layer = self.relu(self.hidden2_norm)
        self.hidden2_dropout = self.dropout(self.hidden2_layer, self.dropout_rate)

        self.output_input = (
            np.dot(self.hidden2_dropout, self.weights_hidden2_output) + self.bias_output
        )
        self.output_norm = self.batch_norm(
            self.output_input,
            self.bn_gamma3,
            self.bn_beta3,
            self.bn_mean3,
            self.bn_var3,
            self.training,
        )
        self.output_layer = self.sigmoid(self.output_norm)

        return self.output_layer

    def backward(self, targets, learning_rate=None):
        """
        Backward pass with batch normalization.
        Args:
            targets: Target values
            learning_rate: Learning rate (uses default if None)
        Returns:
            None
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        batch_size = targets.shape[0] if len(targets.shape) > 1 else 1

        output_error = targets - self.output_layer
        output_delta = output_error * self.sigmoid_derivative(self.output_layer)

        dout_norm = output_delta
        dout_input = dout_norm * self.bn_gamma3 / np.sqrt(self.bn_var3 + 1e-8)

        hidden2_error = np.dot(dout_input, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * self.relu_derivative(self.hidden2_norm)

        dhidden2_norm = hidden2_delta
        dhidden2_input = dhidden2_norm * self.bn_gamma2 / np.sqrt(self.bn_var2 + 1e-8)

        hidden1_error = np.dot(dhidden2_input, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * self.relu_derivative(self.hidden1_norm)

        dhidden1_norm = hidden1_delta
        dhidden1_input = dhidden1_norm * self.bn_gamma1 / np.sqrt(self.bn_var1 + 1e-8)

        grad_w_h2o = (
            np.dot(self.hidden2_dropout.T, dout_input) / batch_size
            + self.l2_lambda * self.weights_hidden2_output
        )
        grad_b_o = np.sum(dout_input, axis=0, keepdims=True) / batch_size

        grad_w_h1h2 = (
            np.dot(self.hidden1_dropout.T, dhidden2_input) / batch_size
            + self.l2_lambda * self.weights_hidden1_hidden2
        )
        grad_b_h2 = np.sum(dhidden2_input, axis=0, keepdims=True) / batch_size

        grad_w_ih1 = (
            np.dot(self.inputs.T, dhidden1_input) / batch_size
            + self.l2_lambda * self.weights_input_hidden1
        )
        grad_b_h1 = np.sum(dhidden1_input, axis=0, keepdims=True) / batch_size

        grad_w_h2o = np.clip(grad_w_h2o, -1, 1)
        grad_w_h1h2 = np.clip(grad_w_h1h2, -1, 1)
        grad_w_ih1 = np.clip(grad_w_ih1, -1, 1)

        self.velocity_w_h2o = (
            self.momentum * self.velocity_w_h2o + learning_rate * grad_w_h2o
        )
        self.velocity_w_h1h2 = (
            self.momentum * self.velocity_w_h1h2 + learning_rate * grad_w_h1h2
        )
        self.velocity_w_ih1 = (
            self.momentum * self.velocity_w_ih1 + learning_rate * grad_w_ih1
        )
        self.velocity_b_o = self.momentum * self.velocity_b_o + learning_rate * grad_b_o
        self.velocity_b_h2 = (
            self.momentum * self.velocity_b_h2 + learning_rate * grad_b_h2
        )
        self.velocity_b_h1 = (
            self.momentum * self.velocity_b_h1 + learning_rate * grad_b_h1
        )

        self.weights_hidden2_output += self.velocity_w_h2o
        self.weights_hidden1_hidden2 += self.velocity_w_h1h2
        self.weights_input_hidden1 += self.velocity_w_ih1
        self.bias_output += self.velocity_b_o
        self.bias_hidden2 += self.velocity_b_h2
        self.bias_hidden1 += self.velocity_b_h1

        self.bn_gamma3 += (
            learning_rate
            * np.sum(output_delta * self.output_norm, axis=0, keepdims=True)
            / batch_size
        )
        self.bn_beta3 += (
            learning_rate * np.sum(output_delta, axis=0, keepdims=True) / batch_size
        )
        self.bn_gamma2 += (
            learning_rate
            * np.sum(hidden2_delta * self.hidden2_norm, axis=0, keepdims=True)
            / batch_size
        )
        self.bn_beta2 += (
            learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True) / batch_size
        )
        self.bn_gamma1 += (
            learning_rate
            * np.sum(hidden1_delta * self.hidden1_norm, axis=0, keepdims=True)
            / batch_size
        )
        self.bn_beta1 += (
            learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True) / batch_size
        )

    def train_batch(self, inputs, targets, learning_rate=None):
        """
        Train on a batch of data.

        Args:
            inputs: Input data
            targets: Target values
            learning_rate: Learning rate (uses default if None)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        if len(targets.shape) == 1:
            targets = targets.reshape(1, -1)

        self.forward(inputs)
        self.backward(targets, learning_rate)

    def predict(self, inputs):
        """
        Predict output for given inputs.
        Args:
            inputs: Input data
        Returns:
            Network output (float): The output of the network
        """
        return self.forward(inputs)

    def mse(self, predictions, targets):
        """
        Calculate mean squared error.
        Args:
            predictions: Predicted values
            targets: Target values
        Returns:
            Mean squared error (float)
        """
        return np.mean((predictions - targets) ** 2)

    def accuracy(self, predictions, targets, threshold=0.5):
        """
        Calculate binary classification accuracy.

        Args:
            predictions: Predicted values
            targets: Target values
            threshold: Threshold for binarization (default 0.5)

        Returns:
            Classification accuracy
        """
        pred_binary = (predictions > threshold).astype(int)
        targ_binary = (targets > threshold).astype(int)
        return np.mean(pred_binary == targ_binary)

    def save_model(self, filepath):
        """
        Save model to file.
        Args:
            filepath: Path to save the model
        Returns:
            None
        """
        model_data = {
            "weights_input_hidden1": self.weights_input_hidden1,
            "weights_hidden1_hidden2": self.weights_hidden1_hidden2,
            "weights_hidden2_output": self.weights_hidden2_output,
            "bias_hidden1": self.bias_hidden1,
            "bias_hidden2": self.bias_hidden2,
            "bias_output": self.bias_output,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """
        Load model from file.
        Args:
            filepath: Path to load the model
        Returns:
            None
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.weights_input_hidden1 = model_data["weights_input_hidden1"]
        self.weights_hidden1_hidden2 = model_data["weights_hidden1_hidden2"]
        self.weights_hidden2_output = model_data["weights_hidden2_output"]
        self.bias_hidden1 = model_data["bias_hidden1"]
        self.bias_hidden2 = model_data["bias_hidden2"]
        self.bias_output = model_data["bias_output"]
        self.input_size = model_data["input_size"]
        self.hidden_size = model_data["hidden_size"]
        self.output_size = model_data["output_size"]
        self.learning_rate = model_data["learning_rate"]

    def get_loss_with_regularization(self, predictions, targets):
        """
        Calculate loss with L2 regularization.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Loss value with regularization (float)
        """
        mse_loss = self.mse(predictions, targets)
        l2_loss = self.l2_lambda * (
            np.sum(self.weights_input_hidden1**2)
            + np.sum(self.weights_hidden1_hidden2**2)
            + np.sum(self.weights_hidden2_output**2)
        )
        return mse_loss + l2_loss
