import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import Kernel, LS_SVM, f



if __name__ == "__main__":
    #заполнение данных, k массивов по n элементов
    l = 300                                              # количество элементов в тестовой последовательности
    x_tr = []
    for i in range(l):                                      # генерация неповторяющихся вещественных чисел
        r = random.uniform(0, 2 * math.pi)
        if r not in x_tr: 
            x_tr.append(r)
    
    y_tr =  list(map(np.sinc, x_tr))


    x_test = []
    for i in range(l):                                      # генерация неповторяющихся вещественных чисел
        r = random.uniform(0, 2 * math.pi)
        if r not in x_test: 
            x_test.append(r)
    
    y_test =  list(map(np.sinc, x_test))
    

    C = 524288 # сюда запишем те два числа с первого запуска
    sigma = 0.7 
    nv  = 6                              # оптимальное клоличество опорных векторов (пали график №2)


    K = [[0] * l for i in range(l)]
    for i in range(l):
        for j in range(l):
            K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)

    [B, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv, True)

    y_res_3 = []
    y_res_4 = []
    RMS = 0
    for i in range(l):
        y_res_3.append(f(x_test[i], x_tr, S, B[nv - 2], sigma)) 
        y_res_4.append(f(x_test[i], x_tr, S, B[nv - 1], sigma))
        # RMS += (y_res[i] - y_test[i]) ** 2
    # RMS  = math.sqrt(RMS / l)
    # print("RMS = ", RMS)

    x_ref = []
    y_ref = []
    ideal = zip(x_test, y_test)
    ideal_s = sorted(ideal, key=lambda tup: tup[0])
    x_test_s = [x[0] for x in ideal_s]
    y_test_s  = [x[1] for x in ideal_s]

    plt.plot(x_test_s, y_test_s, '-', label = 'y = sinc(x)')
    plt.plot(x_test, y_res_3, '*', label = "y = f(x), n_ref = " + str(nv-1))  
    plt.plot(x_test, y_res_4, '*', label = "y = f(x), n_ref = " + str(nv)) 
    for i in S:
        x_ref.append(x_tr[i])
        y_ref.append(y_tr[i])
    plt.plot(x_ref, y_ref, 'ok', label = 'reference vectors')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


