import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import Kernel, LS_SVM, f



if __name__ == "__main__":
    #заполнение данных, k массивов по n элементов
    data_in = []
    with open("noise_data.txt") as f:
        for line in f:
            data_in.append([float(val) for val in line.split()])
    x_tr = data_in[0]
    y_tr = data_in[1]

    l = len(x_tr)

    x_test = []
    for i in range(l):                                      # генерация неповторяющихся вещественных чисел
        r = random.uniform(0, 2 * math.pi)
        if r not in x_test: 
            x_test.append(r)
    
    noise = np.random.normal(0,0.1, l)
    y =  list(map(math.sin, x_test))
    y_test = list(map(lambda a, b: a + b, noise, y))

    C = 120
    sigma = 1.5
    nv  = 10                              # оптимальное клоличество опорных векторов


    K = [[0] * l for i in range(l)]
    for i in range(l):
        for j in range(l):
            K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)

    [B, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv, False)

    y_res = []
    RMS = 0
    for i in range(l):
        y_res.append(f(x_test[i], x_tr, S, B, sigma))
        RMS += (y_res[i] - y_test[i]) ** 2
    RMS  = math.sqrt(RMS / l)
    print("RMS = ", RMS)

    plt.plot(x_test, y_res, '.')  
    plt.plot(x_test, y_test, '*')
    for i in S:
        plt.plot(x_tr[i], y_tr[i], 'ok')
    plt.show()


