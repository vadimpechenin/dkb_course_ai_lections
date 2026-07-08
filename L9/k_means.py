import numpy as np


def k_means(data, k, max_iters=100, tol=1e-4):
    """Реализация алгоритма кластеризации K-Means.

    Параметры:
    ----------
    data : np.ndarray
        Массив входных данных формы (n_samples, n_features).
    k : int
        Целевое число кластеров.
    max_iters : int
        Максимальное количество итераций цикла.
    tol : float
        Порог сходимости (если центры сдвинулись меньше чем на tol,
        алгоритм останавливается).

    Возвращает:
    -----------
    centroids : np.ndarray
        Финальные координаты центров кластеров.
    labels : np.ndarray
        Индексы кластеров для каждого объекта выборки.
    """
    # Преобразуем входные данные в массив NumPy, если они переданы списком
    data = np.array(data)
    n_samples, n_features = data.shape

    # --- 1. Инициализация центров ---
    # Выбираем k случайных уникальных точек из исходного набора данных
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[random_indices]

    labels = np.zeros(n_samples, dtype=int)

    # --- 2. Основной цикл (Процедура) ---
    for iteration in range(max_iters):
        old_centroids = centroids.copy()

        # ШАГ РАСПРЕДЕЛЕНИЯ: Для каждого объекта находим ближайший центр
        for i, point in enumerate(data):
            # Вычисляем Евклидово расстояние от точки до каждого из K центроидов
            distances = np.linalg.norm(point - centroids, axis=1)
            # Относим объект к кластеру с минимальным расстоянием
            labels[i] = np.argmin(distances)

        # ШАГ ОБНОВЛЕНИЯ: Пересчитываем положение центра для каждого кластера
        new_centroids = np.zeros((k, n_features))
        for cluster_idx in range(k):
            # Отбираем все точки, попавшие в текущий кластер
            cluster_points = data[labels == cluster_idx]

            # Если кластер не пустой, считаем среднее арифметическое его точек
            if len(cluster_points) > 0:
                new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                # Если кластер пустой, оставляем центр прежним
                new_centroids[cluster_idx] = centroids[cluster_idx]

        centroids = new_centroids

        # ПРОВЕРКА КРИТЕРИЯ ОСТАНОВКИ: Изменяются ли центры?
        # Если суммарный сдвиг всех центроидов меньше порога tol — завершаем цикл
        center_shift = np.sum(np.linalg.norm(centroids - old_centroids, axis=1))
        if center_shift < tol:
            break

    return centroids, labels