import matplotlib.pyplot as plt
import numpy as np
from L9.k_means import k_means

# Температура и вибрация станков
X = np.array([
    [42, 1.2], [45, 1.3], [41, 1.1],
    [78, 3.8], [80, 4.0], [82, 3.9],
    [60, 2.2], [63, 2.5], [61, 2.3]
])

# Запуск вашей кастомной функции кластеризации
cluster_centers, labels = k_means(X, k=3)

# Словарь со смысловыми названиями для каждого кластера
cluster_names = {
    0: "Нормальный режим",
    1: "Критическое состояние",
    2: "Повышенная нагрузка"
}

# Построение графика с размерами 5x5 дюймов
plt.figure(figsize=(5, 4), dpi=100)

# Цветовая палитра для трех технологических состояний
colors = {0: '#3498db', 1: '#e74c3c', 2: '#f1c40f'}

# Отрисовка точек каждого кластера отдельно
for cluster_idx in range(3):
    cluster_data = X[labels == cluster_idx]
    if len(cluster_data) > 0:
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    color=colors[cluster_idx],
                    s=80,
                    edgecolor='black',
                    label=cluster_names[cluster_idx],
                    zorder=5)

# Отображение вычисленных центроидов
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            color='black',
            marker='X',
            s=180,
            edgecolor='white',
            label='Центроиды',
            zorder=6)

# Настройка подписей, сетки и компактной легенды
plt.xlabel("Температура, °C", fontsize=10)
plt.ylabel("Вибрация, мм/с", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Легенда уменьшена (prop) для корректного отображения в маленьком окне 5x5
plt.legend(loc='upper left', fontsize=8, prop={'size': 7.5})

plt.tight_layout()
#plt.show()
output_filename = 'cluster_image.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')