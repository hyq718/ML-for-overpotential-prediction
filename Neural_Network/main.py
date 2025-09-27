import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from N_N import *

# 数据导入
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

X_train = train_data.drop(['Overpotential(mV)'], axis=1).values
y_train = train_data['Overpotential(mV)'].values

X_test = test_data.drop(['Overpotential(mV)'], axis=1).values
y_test = test_data['Overpotential(mV)'].values

# baseline 参数
baseline_params = {'hidden_layers': [64, 32], 'activation': 'tanh'}
baseline_score = cross_validate(NeuralNetwork, X_train, y_train,
                                 hidden_layers=baseline_params['hidden_layers'],
                                 activation=baseline_params['activation'],
                                 learning_rate=0.01,
                                 epochs=1000,
                                 l2_lambda=0.001)

# 网格搜索优化参数
param_grid = {
    'hidden_layers': [[64], [128], [64, 32], [128, 64]],
    'activation': ['relu', 'tanh', 'sigmoid']
}

best_params, best_score = grid_search_cv(X_train, y_train, param_grid)
print("\n最优参数:", best_params)

# 使用最优参数训练模型
final_model = NeuralNetwork(input_size=X_train.shape[1], epochs=5000, **best_params)
final_model.fit(X_train, y_train)

# 在测试集上进行预测
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)

# 绘图调用
final_model.plot_prediction_with_confidence_interval(y_test, y_test_pred)
final_model.plot_predicted_vs_true_scatter(y_train, y_train_pred, y_test, y_test_pred)

# 输出预测与实际值
print("\n所有预测值 : 实际值 对比如下：")
for pred, actual in zip(y_test_pred.flatten(), y_test.flatten()):
    print(f"{pred:.3f} : {actual:.3f}")

# 输出 MSE
print(f"\n测试集上的 MSE 值: {mse:.6f}")

# 可视化参数优化结果
plot_comparison(baseline_score, best_score, baseline_params, best_params)
# 绘制不同 K 值的柱状图比较
plot_k_comparison(X_train, y_train, best_params, k_values=[2, 5, 10,15])


#随机预测
#调用随机生成的数据进行预测
random_dataset = pd.read_csv('random_features.csv')
y_random_pred = final_model.predict(random_dataset)

# 将随机生成的数据与预测结果合并
y_random_pred_df = pd.DataFrame(y_random_pred, columns=['Overpotential(mV)'])
random_dataset_with_predictions = pd.concat([random_dataset, y_random_pred_df], axis=1)

# 将随机生成的数据及其预测结果保存到CSV文件
random_dataset_with_predictions.to_csv('random_dataset_with_predictions.csv', index=False)
print("随机数据和预测结果已保存到 random_dataset_with_predictions.csv 文件中。")

# 寻找最低的10个过电位及其对应的x值
top_10 = y_random_pred_df.nsmallest(10, 'Overpotential(mV)')

# 合并最低过电位数据与对应的输入特征
top_10_indices = top_10.index
top_10_full_data = pd.concat([random_dataset.iloc[top_10_indices].reset_index(drop=True), top_10.reset_index(drop=True)], axis=1)

# 将最低10个过电位的完整数据保存到CSV文件
top_10_full_data.to_csv('top_10_lowest_overpotential.csv', index=False)
print("最低10个过电位及其对应数据已保存到 top_10_lowest_overpotential.csv 文件中。")