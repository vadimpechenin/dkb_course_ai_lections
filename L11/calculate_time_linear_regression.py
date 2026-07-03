import numpy as np
import matplotlib.pyplot as plt
from linear_regression_algorithm import linear_regression

def predict(x, w, b):
    return w * x + b

# Исходные данные
features = np.array([2, 3, 4, 6, 7, 8])
features_for_predict = np.array([5,])
labels = np.array([14, 16, 22, 28, 30, 37])

# Подбор параметров модели
time_per_weight, base_time = linear_regression(features, labels, learning_rate = 0.01, epochs = 10000)

# Построение графика
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# scatter
sc = ax.scatter(features, labels, label="Данные")

# линия
line, = ax.plot(features, predict(features, time_per_weight, base_time), 'r-', label="Модель")
sc_predict = ax.scatter(features_for_predict, predict(features_for_predict, time_per_weight, base_time), label="Прогноз")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
plt.show()