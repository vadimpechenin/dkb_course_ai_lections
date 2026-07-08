def square_trick(base_time, time_per_weight, weight, time, learning_rate):
    predicted_time = base_time + time_per_weight * weight
    base_time += learning_rate * (time - predicted_time)
    time_per_weight += learning_rate * weight * (time - predicted_time)
    return time_per_weight, base_time