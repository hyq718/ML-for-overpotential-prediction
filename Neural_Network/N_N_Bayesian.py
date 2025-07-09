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
            self.mu_weights.append(np.random.randn(in_dim, out_dim) * 0.1)
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

    def _forward(self, x, weights, biases):
        for i in range(len(weights) - 1):
            x = self._activation(np.dot(x, weights[i]) + biases[i])
        return np.dot(x, weights[-1]) + biases[-1]

    def _sample_weights(self):
        weights = [mu + np.exp(0.5 * log_var) * np.random.randn(*mu.shape)
                   for mu, log_var in zip(self.mu_weights, self.log_var_weights)]
        biases = [mu + np.exp(0.5 * log_var) * np.random.randn(*mu.shape)
                  for mu, log_var in zip(self.mu_biases, self.log_var_biases)]
        return weights, biases

    def train(self, X, y):
        X = self.x_scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(y.reshape(-1, 1))

        for epoch in range(self.epochs):
            weights, biases = self._sample_weights()
            preds = self._forward(X, weights, biases)
            error = preds - y
            loss = np.mean(error**2)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X, n_samples=50):
        X = self.x_scaler.transform(X)
        predictions = []
        for _ in range(n_samples):
            weights, biases = self._sample_weights()
            preds = self._forward(X, weights, biases)
            predictions.append(preds)
        mean_pred = np.mean(predictions, axis=0)
        return self.y_scaler.inverse_transform(mean_pred)

# ========= 评估与工具函数 =========

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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
        model.train(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        mse = mean_squared_error(y[test_idx], y_pred)
        mse_scores.append(mse)

    print(f"{k}-Fold Cross-Validation MSEs: {mse_scores}")
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    return np.mean(mse_scores)

def grid_search_cv(X, y, param_grid):
    best_mse = float("inf")
    best_params = None

    for hidden in param_grid['hidden_layers']:
        for act in param_grid['activation']:
            print(f"\nTrying: hidden={hidden}, activation={act}")
            model = BayesianNeuralNetwork(input_size=X.shape[1], hidden_layers=hidden,
                                          activation=act, learning_rate=0.01, epochs=1000)
            model.train(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            print(f"MSE: {mse:.4f}")
            if mse < best_mse:
                best_mse = mse
                best_params = {'hidden_layers': hidden, 'activation': act}

    print(f"\nBest Params: {best_params}, MSE: {best_mse:.4f}")
    return best_params, best_mse

# ========= 可视化函数 =========

def plot_prediction_with_confidence_interval(model, X, y_true, n_samples=50):
    X_scaled = model.x_scaler.transform(X)
    predictions = []
    for _ in range(n_samples):
        weights, biases = model._sample_weights()
        pred_scaled = model._forward(X_scaled, weights, biases)
        predictions.append(pred_scaled)

    predictions = np.array(predictions).squeeze(axis=-1)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    mean_pred = model.y_scaler.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
    std_pred = model.y_scaler.scale_[0] * std_pred

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True", color='blue')
    plt.plot(mean_pred, label="Predicted", color='orange')
    plt.fill_between(range(len(mean_pred)),
                     mean_pred - 1.96 * std_pred,
                     mean_pred + 1.96 * std_pred,
                     alpha=0.3, label="95% Confidence", color='orange')
    plt.xlabel("Sample")
    plt.ylabel("Overpotential (mV)")
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
