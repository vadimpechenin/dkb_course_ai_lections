import random
from square_trick import square_trick


def linear_regression(features, labels, learning_rate=0.01, epochs = 1000):
    time_per_weight = random.random()
    base_time = random.random()
    for epoch in range(epochs):
        i = random.randint(0, len(features) - 1)
        weight = features[i]
        time = labels[i]
        time_per_weight, base_time = square_trick(base_time,time_per_weight,weight, time,
                    learning_rate=learning_rate)
    return time_per_weight, base_time
