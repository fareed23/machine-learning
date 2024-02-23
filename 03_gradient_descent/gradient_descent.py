import numpy as np

def gradient_descent(x, y):
    m = b = 0
    iterations = 1000
    learning_rate = 0.0001
    n = len(x)
    for i in range(iterations):
        y_predicted = m * x + b
        # derivatives of m and b
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        # update m and b --> (m = m - learning_rate * md) & (b = b - learning_rate * bd)
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print(f"m {m}, b {b}, iteration {i}")


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)