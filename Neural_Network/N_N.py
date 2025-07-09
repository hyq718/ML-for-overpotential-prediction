import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers=[64], activation='relu', learning_rate=0.01, epochs=2000, l2_lambda=0.001):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda

        self._initialize_weights()
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

    def _initialize_weights(self):
        np.random.seed(42)
        layer_sizes = [self.input_size] + self.hidden_layers + [1]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def _activate_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def fit(self, X_train, y_train):
        X_scaled = self.x_scaler.fit_transform(X_train)
        y_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))

        for epoch in range(self.epochs):
            # Forward
            A = X_scaled
            Zs, As = [], [A]
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                Z = np.dot(A, W) + b
                A = self._activate(Z)
                Zs.append(Z)
                As.append(A)

            Z_final = np.dot(A, self.weights[-1]) + self.biases[-1]
            A_final = Z_final

            mse_loss = np.mean((y_scaled - A_final) ** 2)
            l2_loss = self.l2_lambda * sum(np.sum(W**2) for W in self.weights)
            total_loss = mse_loss + l2_loss

            # Backward
            m = X_scaled.shape[0]
            dA = 2 * (A_final - y_scaled) / m
            dWs = []
            dbs = []

            for i in reversed(range(len(self.weights))):
                dW = np.dot(As[i].T, dA) + 2 * self.l2_lambda * self.weights[i]
                db = np.sum(dA, axis=0, keepdims=True)
                dWs.insert(0, dW)
                dbs.insert(0, db)
                if i > 0:
                    dA = np.dot(dA, self.weights[i].T) * self._activate_derivative(Zs[i-1])

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dWs[i]
                self.biases[i] -= self.learning_rate * dbs[i]

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, MSE Loss: {mse_loss:.6f}, Total Loss: {total_loss:.6f}")

    def predict(self, X_test):
        X_scaled = self.x_scaler.transform(X_test)
        A = X_scaled
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = np.dot(A, W) + b
            A = self._activate(Z)
        Z_final = np.dot(A, self.weights[-1]) + self.biases[-1]
        return self.y_scaler.inverse_transform(Z_final)
    
    #  绘制预测值与真实值折线图，并带有真实值±10%的置信区间阴影
    def plot_prediction_with_confidence_interval(self, y_true, y_pred):
        y_true = pd.Series(y_true.flatten())
        y_pred = pd.Series(y_pred.flatten())

        lower_bound = y_true * 0.9
        upper_bound = y_true * 1.1

        plt.figure(figsize=(6, 6))
        plt.plot(y_true.values, color='red', label='Actual Value')
        plt.plot(y_pred.values, color='blue', label='Predicted Value')
        plt.fill_between(range(len(y_true)), lower_bound, upper_bound,
                        color='red', alpha=0.3, label='Confidence Interval (±10%)')

        x_ticks = np.arange(len(y_true))
        x_tick_labels = np.arange(1, len(y_true) + 1)
        plt.xticks(x_ticks, x_tick_labels)

        plt.title('Actual VS Predicted Value')
        plt.xlabel('No.')
        plt.ylabel('Overpotential')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 训练集与测试集散点图
    def plot_predicted_vs_true_scatter(self, y_train_true, y_train_pred, y_test_true, y_test_pred):
        y_train_true = pd.Series(y_train_true.flatten())
        y_train_pred = pd.Series(y_train_pred.flatten())
        y_test_true = pd.Series(y_test_true.flatten())
        y_test_pred = pd.Series(y_test_pred.flatten())

        plt.figure(figsize=(6, 6))
        plt.scatter(y_train_pred, y_train_true, color='blue', label='Train', marker='+')
        plt.scatter(y_test_pred, y_test_true, color='red', label='Test', marker='o')
        min_val = min(y_train_true.min(), y_test_true.min(), y_train_pred.min(), y_test_pred.min())
        max_val = max(y_train_true.max(), y_test_true.max(), y_train_pred.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=1, label='y=x')

        plt.xlabel("Predicted Overpotential (mV)")
        plt.ylabel("True Overpotential (mV)")
        plt.legend()
        plt.title("Predicted vs. True Overpotential")
        plt.grid(True)
        plt.show()





def cross_validate(model_class, X, y, n_splits=5,**kwargs):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model = model_class(input_size=X.shape[1], **kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = mean_squared_error(y_val, preds)
        scores.append(score)
    return np.mean(scores)

def grid_search_cv(X, y, param_grid):
    best_score = float('inf')
    best_params = None
    for hidden_layers in param_grid['hidden_layers']:
        for activation in param_grid['activation']:
            print(f"Evaluating: hidden_layers={hidden_layers}, activation={activation}")
            score = cross_validate(NeuralNetwork, X, y,
                                   hidden_layers=hidden_layers,
                                   activation=activation,
                                   learning_rate=0.01,
                                   epochs=1000,
                                   l2_lambda=0.001)
            print(f"Score: {score:.6f}\n")
            if score < best_score:
                best_score = score
                best_params = {'hidden_layers': hidden_layers, 'activation': activation}
    return best_params, best_score

def plot_comparison(baseline_score, best_score, baseline_params, best_params):
    best_hidden_layers = best_params['hidden_layers']
    best_activation = best_params['activation']
    best_num_layers = len(best_hidden_layers)
    baseline_num_layers = len(baseline_params['hidden_layers'])

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.bar(['Baseline: ' + baseline_params['activation'], f'Best: {best_activation}'],
            [baseline_score, best_score], color=['gray', 'green'])
    plt.title('Activation Function Comparison')
    plt.ylabel('CV MSE')

    plt.subplot(1, 3, 2)
    plt.bar([str(baseline_params['hidden_layers']), str(best_hidden_layers)],
            [baseline_score, best_score], color=['gray', 'blue'])
    plt.title('Hidden Layers Comparison')
    plt.ylabel('CV MSE')

    plt.subplot(1, 3, 3)
    plt.bar([f'{baseline_num_layers} layer(s)\n(Baseline)', f'{best_num_layers} layer(s)\n(Best)'],
            [baseline_score, best_score], color=['gray', 'orange'])
    plt.title('Network Depth Comparison')
    plt.ylabel('CV MSE')

    plt.tight_layout()
    plt.show()

def plot_k_comparison(X, y, best_params, k_values):
    scores = []
    for k in k_values:
        score = cross_validate(NeuralNetwork, X, y, n_splits=k,
                                hidden_layers=best_params['hidden_layers'],
                                activation=best_params['activation'],
                                learning_rate=0.01,
                                epochs=1000,
                                l2_lambda=0.001)
        scores.append(score)

    # 可视化
    plt.figure(figsize=(6, 4))
    plt.bar([f'K={k}' for k in k_values], scores, color='skyblue')
    plt.title('K-Fold Cross-Validation Comparison')
    plt.ylabel('CV MSE')
    plt.tight_layout()
    plt.show()



