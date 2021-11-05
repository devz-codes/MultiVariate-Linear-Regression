import numpy as np
from numpy.linalg import inv

class Linear_Regression():
    def __init__(self,x_values,y_values):
        self.x_values = x_values
        self.y_values = y_values
    
    def coefficients(self):
        size = len(self.x_values)
        y_matrix = np.array([[sum(self.y_values)]])
        x_matrix = np.array([[len(self.y_values)]])

        for i in range(0, size):
            val = sum(list(map(lambda x, y: x*y, self.x_values[i], self.y_values)))
            y_matrix = np.vstack((y_matrix, np.array([[val]])))

        lst = [sum(self.x_values[j]) for j in range(0, size)]

        x_matrix = np.hstack((x_matrix, np.array([lst])))
        for i in range(0, size):
            lst = []
            lst.append(sum(self.x_values[i]))
            for j in range(0, size):
                lst.append(
                    sum(list(map(lambda x, y: x*y, self.x_values[i], self.x_values[j]))))
            x_matrix = np.vstack((x_matrix, np.array([lst])))

        inverse = inv(x_matrix)
        product = np.dot(inverse, y_matrix)

        return product

    def predict(self, x_val):
        sum = self.coefficients()[0][0]

        for i in range(1, len(x_val)+1):
            sum += self.coefficients()[i][0] * x_val[i-1]

        return sum
