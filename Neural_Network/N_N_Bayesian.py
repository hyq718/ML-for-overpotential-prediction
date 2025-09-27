import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class BayesianNeuralNetwork:
    def __init__(self, input_size, hidden_layers=[64], activation='relu',
                 learning_rate=0.01, epochs=2000, l2_lambda=0.001, kl_weight=1e-4):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.kl_weight = kl_weight

        self._initialize_bayesian_parameters()
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

    def _initialize_bayesian_parameters(self):
        np.random.seed(42)
        layer_sizes = [self.input_size] + self.hidden_layers + [1]
        self.mu_weights, self.log_var_weights = [], []
        self.mu_biases, self.log_var_biases = [], []

        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            # Xavier/Glorot initialization for mu weights
            std = np.sqrt(2.0 / (in_dim + out_dim))
            self.mu_weights.append(np.random.randn(in_dim, out_dim) * std)
            self.log_var_weights.append(np.full((in_dim, out_dim), -10.0))
            self.mu_biases.append(np.zeros(out_dim))
            self.log_var_biases.append(np.full(out_dim, -10.0))

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _forward(self, x, weights, biases):
        self.activations = [x]
        self.z_values = []
        
        for i in range(len(weights) - 1):
            z = np.dot(self.activations[-1], weights[i]) + biases[i]
            self.z_values.append(z)
            self.activations.append(self._activation(z))
        
        # Linear activation for output layer
        z = np.dot(self.activations[-1], weights[-1]) + biases[-1]
        self.z_values.append(z)
        self.activations.append(z)
        
        return self.activations[-1]

    def _sample_weights(self):
        weights = []
        for mu, log_var in zip(self.mu_weights, self.log_var_weights):
            epsilon = np.random.randn(*mu.shape)
            weights.append(mu + np.exp(0.5 * log_var) * epsilon)
            
        biases = []
        for mu, log_var in zip(self.mu_biases, self.log_var_biases):
            epsilon = np.random.randn(*mu.shape)
            biases.append(mu + np.exp(0.5 * log_var) * epsilon)
            
        return weights, biases

    def _compute_kl_divergence(self):
        kl = 0
        for mu_w, log_var_w, mu_b, log_var_b in zip(self.mu_weights, self.log_var_weights, 
                                                   self.mu_biases, self.log_var_biases):
            # KL divergence for weights
            kl += 0.5 * np.sum(np.exp(log_var_w) + mu_w**2 - 1 - log_var_w)
            # KL divergence for biases
            kl += 0.5 * np.sum(np.exp(log_var_b) + mu_b**2 - 1 - log_var_b)
        return kl

    # 修改后的N_N_Bayesian.py中的_backward方法
    def _backward(self, x, y, weights, biases):
        # Number of layers
        n_layers = len(weights)
        
        # Initialize gradients
        grad_mu_weights = [np.zeros_like(mu) for mu in self.mu_weights]
        grad_log_var_weights = [np.zeros_like(log_var) for log_var in self.log_var_weights]
        grad_mu_biases = [np.zeros_like(mu) for mu in self.mu_biases]
        grad_log_var_biases = [np.zeros_like(log_var) for log_var in self.log_var_biases]
        
        # Output error
        error = self.activations[-1] - y
        delta = error
        
        # Backpropagate through layers
        for l in range(n_layers - 1, -1, -1):
            # Gradient for weights
            a_prev = self.activations[l]
            
            # Gradient for mu weights
            grad_mu_weights[l] = np.dot(a_prev.T, delta)
            
            # Gradient for log_var weights
            epsilon_w = (weights[l] - self.mu_weights[l]) / (np.exp(0.5 * self.log_var_weights[l]) + 1e-8)
            # 修改这里：确保delta和epsilon_w形状匹配
            if delta.shape[1] != epsilon_w.shape[1]:
                delta = delta.reshape(-1, epsilon_w.shape[1])
            grad_log_var_weights[l] = 0.5 * np.dot(a_prev.T, delta * epsilon_w)
            
            # Gradient for biases
            grad_mu_biases[l] = np.sum(delta, axis=0)
            
            # Gradient for log_var biases
            epsilon_b = (biases[l] - self.mu_biases[l]) / (np.exp(0.5 * self.log_var_biases[l]) + 1e-8)
            grad_log_var_biases[l] = 0.5 * np.sum(delta * epsilon_b, axis=0)
            
            # Propagate error backward if not the first layer
            if l > 0:
                delta = np.dot(delta, weights[l].T) * self._activation_derivative(self.z_values[l-1])
        
        return grad_mu_weights, grad_log_var_weights, grad_mu_biases, grad_log_var_biases
    def train(self, X, y, verbose=True):
        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y.reshape(-1, 1))
        
        for epoch in range(self.epochs):
            # Sample weights and biases
            weights, biases = self._sample_weights()
            
            # Forward pass
            preds = self._forward(X, weights, biases)
            
            # Compute losses
            mse_loss = np.mean((preds - y)**2)
            kl_loss = self._compute_kl_divergence()
            total_loss = mse_loss + self.kl_weight * kl_loss
            
            # Backward pass
            grad_mu_w, grad_log_var_w, grad_mu_b, grad_log_var_b = self._backward(X, y, weights, biases)
            
            # Update parameters with gradients
            for i in range(len(self.mu_weights)):
                # Add L2 regularization
                grad_mu_w[i] += self.l2_lambda * self.mu_weights[i]
                
                # Update parameters
                self.mu_weights[i] -= self.learning_rate * grad_mu_w[i]
                self.log_var_weights[i] -= self.learning_rate * grad_log_var_w[i]
                self.mu_biases[i] -= self.learning_rate * grad_mu_b[i]
                self.log_var_biases[i] -= self.learning_rate * grad_log_var_b[i]
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, MSE: {mse_loss:.4f}, KL: {kl_loss:.4f}, Total Loss: {total_loss:.4f}")

    def predict(self, X, n_samples=50):
        X = self.x_scaler.transform(X)
        predictions = []
        for _ in range(n_samples):
            weights, biases = self._sample_weights()
            preds = self._forward(X, weights, biases)
            predictions.append(preds)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return (self.y_scaler.inverse_transform(mean_pred), 
                self.y_scaler.scale_[0] * std_pred)

# ========= 评估与工具函数 =========

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)
    y_train_pred, _ = model.predict(X_train)
    y_test_pred, y_test_std = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    plot_predicted_vs_true_scatter(y_train, y_train_pred, y_test, y_test_pred)
    plot_prediction_with_confidence_interval(model, X_test, y_test)

def cross_validate(model_class, X, y, k=5, **model_kwargs):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []

    for train_idx, test_idx in kf.split(X):
        model = model_class(input_size=X.shape[1], **model_kwargs)
        model.train(X[train_idx], y[train_idx], verbose=False)
        y_pred, _ = model.predict(X[test_idx])
        mse = mean_squared_error(y[test_idx], y_pred)
        mse_scores.append(mse)

    print(f"{k}-Fold Cross-Validation MSEs: {mse_scores}")
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    return np.mean(mse_scores)

def grid_search_cv(X, y, param_grid, k=5):
    best_mse = float("inf")
    best_params = None
    results = []

    for hidden in param_grid['hidden_layers']:
        for act in param_grid['activation']:
            print(f"\nTrying: hidden={hidden}, activation={act}")
            mse = cross_validate(BayesianNeuralNetwork, X, y, k=k,
                                hidden_layers=hidden, activation=act,
                                learning_rate=0.01, epochs=1000)
            results.append({'hidden': hidden, 'activation': act, 'mse': mse})
            if mse < best_mse:
                best_mse = mse
                best_params = {'hidden_layers': hidden, 'activation': act}

    print(f"\nBest Params: {best_params}, MSE: {best_mse:.4f}")
    return best_params, best_mse

# ========= 可视化函数 =========

def plot_prediction_with_confidence_interval(model, X, y_true, n_samples=50):
    y_pred, y_std = model.predict(X, n_samples=n_samples)
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True", color='blue')
    plt.plot(y_pred, label="Predicted", color='orange')
    plt.fill_between(range(len(y_pred)),
                     y_pred - 1.96 * y_std,
                     y_pred + 1.96 * y_std,
                     alpha=0.3, label="95% Confidence", color='orange')
    plt.xlabel("Sample")
    plt.ylabel("Target Value")
    plt.title("Prediction with Confidence Interval")
    plt.legend()
    plt.grid()
    plt.show()

def plot_predicted_vs_true_scatter(y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_pred, color='blue', label='Train', marker='+')
    plt.scatter(y_test, y_test_pred, color='red', label='Test', marker='x')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k-', lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True")
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparison(baseline_score, best_score, baseline_params, best_params):
    labels = ['Baseline', 'Best Grid Search']
    scores = [baseline_score, best_score]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, scores, color=['gray', 'green'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}", ha='center', va='bottom')
    plt.ylabel("Cross-Validation MSE")
    plt.title("Model Comparison")
    plt.grid(axis='y')
    plt.show()

def plot_k_comparison(X, y, best_params, k_values=[2, 5, 10, 15]):
    avg_mses = []
    for k in k_values:
        print(f"\nEvaluating with K={k}")
        mse = cross_validate(BayesianNeuralNetwork, X, y, k=k, **best_params)
        avg_mses.append(mse)

    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(k) for k in k_values], avg_mses, color='skyblue')
    for bar, mse in zip(bars, avg_mses):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{mse:.2f}", ha='center', va='bottom')
    plt.xlabel("K-Fold")
    plt.ylabel("Average MSE")
    plt.title("Cross-Validation MSE vs K")
    plt.grid(axis='y')
    plt.show()