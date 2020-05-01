import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import k_training

if __name__ == "__main__":

    #заполнение данных, k массивов по n элементов
    n = 50                                                  # количество элементов в каждой из частей выборки
    k = 3                                                   # кратность перекрестной проверки
    x=[]
    for i in range(k):                                      # генерация неповторяющихся вещественных чисел
        x.append([]) 
        j = 0
        while j < n:
            r = random.uniform(0, 2 * math.pi)
            if r not in x[i]: 
                x[i].append(r)
                j += 1
    
    y = []
    for i in range(k):
        y.append([]) 
        y[i] =  list(map(math.sin, x[i]))

    # настройка машины:
    # перебор C
    d = n * (k - 1)
    C_begin = 2 ** (1)
    C_end = (2 ** (16)) * d
    C = C_begin

    sigma_begin = 2
    sigma_end = 5
    sigma_step = 0.2
    s_range =np.arange(sigma_begin, sigma_end, sigma_step)

    for sigma in s_range:
        # перебор sigma
        C = C_begin
        print("sigma = " + str(sigma))
        while C <= C_end:
            print("C = " + str(C))
            inf_cur = k_training(k, n, x, y, C, sigma, d // 10, False)
            if (C == C_begin and sigma == sigma_begin) or (inf_cur < inf_min):
                inf_min = inf_cur
                C_min = C
                sigma_min = sigma
            C *= 2

    C = C_min
    # уточнение sigma
    sigma_begin = sigma_min - sigma_step
    sigma_end = sigma_min + sigma_step
    sigma_step = sigma_step / 10
    s_range =np.arange(sigma_begin, sigma_end, sigma_step)
    for sigma in s_range:
            print("sigma = " + str(sigma))
            inf_cur = k_training(k, n, x, y, C, sigma, d // 10, False)
            if (sigma == sigma_begin) or (inf_cur < inf_min):
                inf_min = inf_cur
                sigma_min = sigma

    sigma = sigma_min

    print("C_min = " + str(C_min))
    print("sigma_min = " + str(sigma_min))

    # вычисление среднеквадратичной ошибки для любого количества опорных векторов
    inf = k_training(k, n, x, y, C, sigma, n, True)

    plt.plot(range(n), inf, '.-')

    plt.title('Cross-validation error of GSLSSVM')
    plt.xlabel('number of vectors')
    plt.ylabel('RMS error')
    plt.show()



    # y_res = []
    # for i in range(n):
    #   y_res.append(f(x[2][i], x_tr, S, B, sigma))

    # plt.plot(x[2], y_res, '.')  


    # plt.plot(x[2], y[2], '*')
    # for i in S:
    #     plt.plot(x_tr[i], y_tr[i], 'ok')
    # plt.show()