import random

from L11.logisitc_trick import total_log_loss, logistic_trick


def logistic_regression_algorithm(features, labels, learning_rate = 0.01, epochs = 200):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features) - 1)
        weights, bias = logistic_trick(weights, bias, features[j], labels[j])

    return weights, bias, errors
