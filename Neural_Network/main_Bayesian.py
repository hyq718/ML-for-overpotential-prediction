import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from N_N_Bayesian import *

# 数据加载
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

X_train = train_data.drop(['Overpotential(mV)'], axis=1).values
y_train = train_data['Overpotential(mV)'].values
X_test = test_data.drop(['Overpotential(mV)'], axis=1).values
y_test = test_data['Overpotential(mV)'].values

# Baseline 模型
baseline_params = {'hidden_layers': [64, 32], 'activation': 'tanh'}
baseline_score = cross_validate(BayesianNeuralNetwork, X_train, y_train,
                                 hidden_layers=baseline_params['hidden_layers'],
                                 activation=baseline_params['activation'],
                                 learning_rate=0.01,
                                 epochs=1000,
                                 l2_lambda=0.001)

# 网格搜索
param_grid = {
    'hidden_layers': [[64], [128], [64, 32], [128, 64]],
    'activation': ['relu', 'tanh']
}
best_params, best_score = grid_search_cv(X_train, y_train, param_grid)
print("\n最优参数:", best_params)

# 最终模型
final_model = BayesianNeuralNetwork(input_size=X_train.shape[1], epochs=5000, **best_params)
final_model.train(X_train, y_train)

# 预测与评估
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)

plot_prediction_with_confidence_interval(final_model, X_test, y_test)
plot_predicted_vs_true_scatter(y_train, y_train_pred, y_test, y_test_pred)

print("\n预测值 : 实际值 对比如下：")
for pred, actual in zip(y_test_pred.flatten(), y_test.flatten()):
    print(f"{pred:.3f} : {actual:.3f}")
print(f"\n测试集 MSE: {mse:.6f}")

# 结果对比图
plot_comparison(baseline_score, best_score, baseline_params, best_params)
plot_k_comparison(X_train, y_train, best_params, k_values=[2, 5, 10, 15])

# 随机样本预测
random_dataset = pd.read_csv('random_features.csv')
y_random_pred = final_model.predict(random_dataset)
y_random_pred_df = pd.DataFrame(y_random_pred, columns=['Overpotential(mV)'])
random_dataset_with_predictions = pd.concat([random_dataset, y_random_pred_df], axis=1)
random_dataset_with_predictions.to_csv('random_dataset_with_predictions.csv', index=False)

top_10 = y_random_pred_df.nsmallest(10, 'Overpotential(mV)')
top_10_indices = top_10.index
top_10_full_data = pd.concat([random_dataset.iloc[top_10_indices].reset_index(drop=True), top_10.reset_index(drop=True)], axis=1)
top_10_full_data.to_csv('top_10_lowest_overpotential.csv', index=False)

print("\n结果已保存完毕。")
