import numpy as np
import matplotlib.pyplot as plt
from L11.logistic_regression_algorithm import logistic_regression_algorithm


features = np.array([[2,4],[3,4],[4,3],[5,3],[6,1],[7,2],[8,1]])
labels = np.array([0, 0, 0, 0, 1, 1, 1])

weights, bias, errors = logistic_regression_algorithm(features, labels)


print("Массив весов: " + str(weights))
print("Свободный член: " + str(bias))
print("Ошибка: " + str(errors[-1]))


details = ['Д1', 'Д2', 'Д3', 'Д4', 'Д5', 'Д6', 'Д7']
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(6, 4))

# 4. Построение точечного графика (scatter plot)
# c=time и cmap='viridis' задают плавное изменение цвета точек от времени
scatter = ax.scatter(features[:,0], features[:,1], c=labels.tolist(), cmap='viridis',
                     alpha=0.75, edgecolors='black', linewidths=1.5)

# 5. Добавление подписей к каждой точке (Д1, Д2 и т.д.)
for i, txt in enumerate(details):
    ax.annotate(txt, (features[i,0], features[i,1]),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=10, weight='bold')

# Уравнение прямой: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2
x_line = np.array([min(features[:,0]) - 1, max(features[:,0]) + 1])
y_line = -(weights[0] * x_line + bias) / weights[1]

ax.plot(x_line, y_line, color='red', linestyle='-', linewidth=2.5,
        label='Разделяющая линия (Логистическая регрессия)', zorder=2)

# 6. Оформление осей и заголовков
#ax.set_title('Зависимость времени обработки от массы детали', fontsize=14, weight='bold', pad=15)
ax.set_xlabel('Масса детали, кг', fontsize=12, labelpad=10)
ax.set_ylabel('Количество отверстий', fontsize=12, labelpad=10)

# Настройка отступов на осях, чтобы крайние точки не прижимались к рамке
ax.set_xlim(min(features[:,0]) - 1, max(features[:,0]) + 1)
ax.set_ylim(min(features[:,1]) - 4, max(features[:,1]) + 4)

# Включение красивой сетки
ax.grid(True, linestyle='--', alpha=0.5)

# Автоматическое выравнивание полей
plt.tight_layout()
plt.show()