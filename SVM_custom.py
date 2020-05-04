import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import k_training

if __name__ == "__main__":

    #заполнение данных, k массивов по n элементов
    #n = 50                                                  # количество элементов в каждой из частей выборки
    k = 3                                                   # кратность перекрестной проверки
    data_in = []
    x = []
    y = []

    file_name = "Data//noise_data.txt"
    #считывание данных
    data_in = []
    with open(file_name) as f:
        for line in f:
            data_in.append([float(val) for val in line.split()])

    # f_out_RMS = open( "RMS_model_data.csv", "w") 

    n = len(data_in[0]) // k

    #разбиение данных на k частей
    part = n
    for i in range(k):                                      
        x.append(data_in[0][part - n:part]) 
        y.append(data_in[1][part - n:part]) 
        part += n


    # настройка машины:
    # перебор C
    d = n * (k - 1)
    C_begin = 2 ** (1)
    C_end = (2 ** (13)) * d
    C = C_begin

    sigma_begin = 0.5
    sigma_end = 4
    sigma_step = 0.2
    s_range =np.arange(sigma_begin, sigma_end, sigma_step)

    # for sigma in s_range:
    #     # перебор sigma
    #     C = C_begin
    #     print("sigma = " + str(sigma))
    #     f_out_RMS.write(str(sigma) + "; ")
    #     while C <= C_end:
    #         print("C = " + str(C))
    #         inf_cur = k_training(k, n, x, y, C, sigma, d // 10, False)
    #         if (C == C_begin and sigma == sigma_begin) or (inf_cur < inf_min):
    #             inf_min = inf_cur
    #             C_min = C
    #             sigma_min = sigma
    #             print("sigma_min = ", sigma_min, "C_min = ", C, "inf_cur = ", inf_cur)
    #             f_out_RMS.write(str(inf_cur) + "; ")
    #         else:
    #             f_out_RMS.write("0; ")
    #         C *= 2
    #     f_out_RMS.write('\n')

    # f_out_RMS.close()
    # C = C_min

    C = 1048576
    sigma = 2.9

    # sigma = sigma_min

    # print("C_min = " + str(C_min))
    # print("sigma_min = " + str(sigma_min))

    # вычисление среднеквадратичной ошибки для любого количества опорных векторов
    inf = k_training(k, n, x, y, C, sigma, n, True)

    plt.plot(range(3,21,1), inf[3:21:1], '.-')

    plt.title('Cross-validation error of GSLSSVM')
    plt.xlabel('number of vectors')
    plt.ylabel('RMS error')
    plt.show()
