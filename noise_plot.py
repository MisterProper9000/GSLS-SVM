import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import Kernel, LS_SVM, f



if __name__ == "__main__":
    #заполнение данных, k массивов по n элементов
    data_in = []
    with open("Data//noise_data.txt") as f_in:
        for line in f_in:
            data_in.append([float(val) for val in line.split()])
    x_tr = data_in[0]
    y_tr = data_in[1]

    l = len(x_tr)

    x_test = []
    for i in range(l // 10):                                      # генерация неповторяющихся вещественных чисел
        r = random.uniform(0, 2 * math.pi)
        if r not in x_test: 
            x_test.append(r)
    
    noise = np.random.normal(0,0.1, l // 10)
    y =  list(map(math.sin, x_test))
    y_test = list(map(lambda a, b: a + b, noise, y))

    C = 16384
    sigma = 3.5
    nv  = 4                              # оптимальное клоличество опорных векторов


    K = [[0] * l for i in range(l)]
    for i in range(l):
        for j in range(l):
            K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)

    [B, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv, True)

    y_res_3 = []
    y_res_4 = []
    RMS = 0
    for i in range(l // 10):
        y_res_3.append(f(x_test[i], x_tr, S, B[2], sigma))
        y_res_4.append(f(x_test[i], x_tr, S, B[3], sigma))
        # RMS += (y_res[i] - y_test[i]) ** 2
    # RMS  = math.sqrt(RMS / l)
    # print("RMS = ", RMS)

    x_ref = []
    y_ref = []
    ideal = zip(x_test, y_test)
    ideal_s = sorted(ideal, key=lambda tup: tup[0])
    x_test_s = [x[0] for x in ideal_s]
    y_test_s  = [x[1] for x in ideal_s]

    res_3= zip(x_test, y_res_3)
    res_3_s = sorted(res_3, key=lambda tup: tup[0])
    x_test_s = [x[0] for x in res_3_s ]
    y_res_3_s  = [x[1] for x in res_3_s ]

    res_4 = zip(x_test, y_res_4)
    res_4_s = sorted(res_4, key=lambda tup: tup[0])
    x_test_s = [x[0] for x in res_4_s ]
    y_res_4_s  = [x[1] for x in res_4_s ]


    plt.plot(x_test_s, y_test_s, '.', label = 'y = sin(x) + N(0, 0.1)')
    # plt.plot(x_test, y_res_3, '*', label = 'y = f(x), n_ref = 3')  
    # plt.plot(x_test, y_res_4, '*', label = 'y = f(x), n_ref = 4') 
    plt.plot(x_test_s, y_res_3_s, '*-', label = 'y = f(x), n_ref = 3')  
    plt.plot(x_test_s, y_res_4_s, '*-', label = 'y = f(x), n_ref = 4') 
    for i in S:
        x_ref.append(x_tr[i])
        y_ref.append(y_tr[i])
    plt.plot(x_ref, y_ref, 'ok', label = 'reference vectors')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

