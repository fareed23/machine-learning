import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def using_sklearn():
    df = pd.read_csv('03_gradient_descent/test_scores.csv')
    model = LinearRegression()
    model.fit(df[['math']], df.cs)
    return model.coef_, model.intercept_

def gradient_descent(x, y):
    m = b = 0
    iterations = 1000000
    learning_rate = 0.00011
    n = len(x)

    cost_prev = 0
    for i in range(iterations):
        y_predicted = m * x + b
        # cost function
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        # derivatives of m and b
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m = m - learning_rate * md
        b = b - learning_rate * bd

        if math.isclose(cost, cost_prev, rel_tol=1e-20):
            break
        cost_prev = cost
        # print(f"m {m}, b {b}, cost {cost}, iteration {i}")

        return m, b

if __name__ == '__main__':    
    df = pd.read_csv('03_gradient_descent/test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)
    print(f"Using gradient descent function: Coef: {m}, Intercept: {b}")

    m_sklearn, b_sklearn = using_sklearn()
    print(f"Using sklearn: Coef: {m_sklearn}, Intercept: {b_sklearn}")