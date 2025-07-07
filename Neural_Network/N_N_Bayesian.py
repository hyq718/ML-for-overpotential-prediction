import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
            in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
            self.mu_weights.append(np.random.randn(in_dim, out_dim) * np.sqrt(1. / in_dim))
            self.log_var_weights.append(np.full((in_dim, out_dim), -10.0))
            self.mu_biases.append(np.zeros((1, out_dim)))
            self.log_var_biases.append(np.full((1, out_dim), -10.0))

    def _sample_weights(self):
        weights, biases = [], []
        for mu_w, log_var_w, mu_b, log_var_b in zip(
            self.mu_weights, self.log_var_weights, self.mu_biases, self.log_var_biases):
            eps_w = np.random.randn(*mu_w.shape)
            eps_b = np.random.randn(*mu_b.shape)
            w = mu_w + np.exp(0.5 * log_var_w) * eps_w
            b = mu_b + np.exp(0.5 * log_var_b) * eps_b
            weights.append(w)
            biases.append(b)
        return weights, biases

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation")

    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def _kl_divergence(self):
        kl = 0
        for mu, log_var in zip(self.mu_weights + self.mu_biases, self.log_var_weights + self.log_var_biases):
            kl += 0.5 * np.sum(np.exp(log_var) + mu**2 - 1 - log_var)
        return kl

    def forward(self, X, weights, biases):
        self.zs, self.activations = [], [X]
        a = X
        for i in range(len(weights) - 1):
            z = a.dot(weights[i]) + biases[i]
            a = self._activation(z)
            self.zs.append(z)
            self.activations.append(a)
        z = a.dot(weights[-1]) + biases[-1]
        self.zs.append(z)
        self.activations.append(z)
        return z

    def backward(self, y_pred, y_true, weights, biases):
        grads_mu_w, grads_logvar_w = [], []
        grads_mu_b, grads_logvar_b = [], []

        m = y_true.shape[0]
        delta = (y_pred - y_true) / m

        for i in reversed(range(len(weights))):
            a_prev = self.activations[i]
            dW = a_prev.T.dot(delta)
            dB = np.sum(delta, axis=0, keepdims=True)

            # 采样的 epsilon
            eps_w = (weights[i] - self.mu_weights[i]) / np.exp(0.5 * self.log_var_weights[i])
            eps_b = (biases[i] - self.mu_biases[i]) / np.exp(0.5 * self.log_var_biases[i])

            # 梯度
            d_mu_w = dW + self.kl_weight * self.mu_weights[i]
            d_logvar_w = 0.5 * (np.exp(self.log_var_weights[i]) - 1) + \
                        self.kl_weight * np.exp(0.5 * self.log_var_weights[i]) * eps_w * dW

            d_mu_b = dB + self.kl_weight * self.mu_biases[i]
            d_logvar_b = 0.5 * (np.exp(self.log_var_biases[i]) - 1) + \
                        self.kl_weight * np.exp(0.5 * self.log_var_biases[i]) * eps_b * dB

            grads_mu_w.insert(0, d_mu_w)
            grads_logvar_w.insert(0, d_logvar_w)
            grads_mu_b.insert(0, d_mu_b)
            grads_logvar_b.insert(0, d_logvar_b)

            # 更新 delta 供下一轮使用
            if i > 0:
                d_act = self._activation_derivative(self.zs[i - 1])
                delta = delta.dot(weights[i].T) * d_act

        return grads_mu_w, grads_logvar_w, grads_mu_b, grads_logvar_b

    def train(self, X, y):
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

        for epoch in range(self.epochs):
            weights, biases = self._sample_weights()
            y_pred = self.forward(X_scaled, weights, biases)

            grads_mu_w, grads_logvar_w, grads_mu_b, grads_logvar_b = self.backward(
                y_pred, y_scaled, weights, biases)

            for i in range(len(self.mu_weights)):
                self.mu_weights[i] -= self.learning_rate * grads_mu_w[i]
                self.log_var_weights[i] -= self.learning_rate * grads_logvar_w[i]
                self.mu_biases[i] -= self.learning_rate * grads_mu_b[i]
                self.log_var_biases[i] -= self.learning_rate * grads_logvar_b[i]

            if (epoch + 1) % 500 == 0:
                mse = np.mean((y_pred - y_scaled) ** 2)
                kl = self._kl_divergence()
                total_loss = mse + self.kl_weight * kl
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.6f}, MSE: {mse:.6f}, KL: {kl:.2f}")

    def predict(self, X, n_samples=10):
        X_scaled = self.x_scaler.transform(X)
        preds = []
        for _ in range(n_samples):
            weights, biases = self._sample_weights()
            y_pred = self.forward(X_scaled, weights, biases)
            preds.append(y_pred)
        preds = np.mean(preds, axis=0)
        return self.y_scaler.inverse_transform(preds)
