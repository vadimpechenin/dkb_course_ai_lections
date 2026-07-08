import matplotlib.pyplot as plt
import numpy as np

from L9.k_means import k_means

# Температура и вибрация станков
X = np.array([
    [42, 1.2], [45, 1.3], [41, 1.1],
    [78, 3.8], [80, 4.0], [82, 3.9],
    [60, 2.2], [63, 2.5], [61, 2.3]
])

cluster_centers, labels = k_means(X,k=3)

plt.scatter(X[:,0], X[:,1], c=labels, s=80)
plt.scatter(cluster_centers[:,0],
            cluster_centers[:,1],
            marker='X',
            s=200)

plt.xlabel("Температура, °C")
plt.ylabel("Вибрация, мм/с")
plt.show()