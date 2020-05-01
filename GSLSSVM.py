import math
import numpy as np
import time

# функция вычисления значения ядра для двух элементов
# x1, x2 - элементы
# s - sigma, параметр ядра
def Kernel(x1, x2, s):
    return math.exp(-(s**(-2)) * ((x1 - x2) ** 2))

# целевая функция задачи оптимизации
# B - вектор коэффициентов бэта и последний элемент это b
# S - вектор индексов опорных векторов
# K - матрица ядра l на l
# l - размер тренировочной последовательности
# y - вектор компонент y_i тренировочной последовательности
# gamma - параментр машины 
# Возвращаемое значение: значение целевой функции
def LS(B, S, K, l, y, gamma):
    
    # вычисление левой части выражения
    val_left = 0
    for i in range(len(S)):
        for j in range(len(S)):
            val_left += B[i] * B[j] * K[S[i]][S[j]]
    val_left *= 0.5

    # вычисление правой части выражения
    val_right = 0
    for i in range(l):
        val = y[i]
        for j in range(len(S)):
            val -= B[j] * K[i][S[j]]
        val -= B[len(B) - 1]
        val_right += val ** 2 

    val_right *= gamma / l
    return val_left + val_right

# функция регрессии для элемента x
# x - элемент
# x_ref - тренировочная последовательность
# S - индексы опорных векторов
# B - вектор коэффициентов бэта и последний элемент это b
# sigma - параметр машины
def f(x, x_ref, S, B, sigma):
    res = 0
    for i in range(len(B) - 1):
        res += B[i] * Kernel(x_ref[S[i]], x, sigma)
    res += B[len(B) - 1]
    return res


# функция поиска индексов опорных векторов
# C, sigma - парметры машины
# K - матрица ядра lxl
# x_tr - вектор компонент x_i тренировочной последовательности
# y_tr - вектор компонент y_i тренировочной последовательности
# nv_max - искомое количество опорных векторов
# return_array - [True, False] - параметр, определяющий возвращаются ли
#           коэффициенты бэта для каждого количества опорных векторов
#           или только для максимального
# Возвращаемое значение: список векторов с коэффициентами бэта для каждого вектора[B] и индексами опорных векторов [S]
def LS_SVM(C, sigma, K, x_tr, y_tr, nv_max, return_array):
    gamma = C
    y_sum = sum(y_tr)
    l = len(x_tr)
    
    B_arr = []
    S = []                                  # Вектор индексов опорных векторов
    v = [y_sum]                             # Вектор (правая часть системы) 
    F = []                                  # Вектор Ф
    Omega = np.zeros((1,1))                 # матрица Omega

    for nv_i in range(nv_max):
        Ind = list(set(range(l)) - set(S)) + [-1]  # список свободных индексов
        S.append(0)                                # добавление к ОВ нового индекса
        #millis_start = int(round(time.time() * 1000))

        #np_millis = 0
        for i in Ind:                              # пеербор всех свободных индексов, для каждого строится и решается СЛАУ
            S[nv_i] = i

            # построение v и F
            if len(F) == nv_i: 
                v.insert(nv_i, 0)
                F.append(0)
            else:
                v[nv_i] = 0
                F[nv_i] = 0

            for j in range(l):
                v[nv_i] += y_tr[j] * K[i][j] 
                F[nv_i] += K[i][j]


            
            # построение матрицы Omega
            if Omega.size == nv_i ** 2:
                Omega = np.vstack((Omega, np.zeros((nv_i), dtype=float)))    # добавление строки в матрицу 
                Omega = np.hstack((Omega, np.zeros((nv_i + 1, 1), dtype=float))) # добавление столбца в матрицу 
            #millis_start1 = int(round(time.time() * 1000))
            for j in range(nv_i + 1):
                Omega[nv_i][j] = l / (2.*gamma) * K[i][S[j]]
                for r in range(l):
                    Omega[nv_i][j]  += K[r][S[j]] * K[r][i]
                Omega[j][nv_i] = Omega[nv_i][j]
            #np_millis += int(round(time.time() * 1000)) - millis_start1
           
            
            # конструирование матрицы СЛАУ из Omega и F
            H = Omega
            F_v = np.array(F) 
            H = np.vstack((H, F_v))
            F_v = np.insert(F_v, nv_i + 1, l)
            F_v = F_v.reshape(nv_i + 2, 1)
            H = np.hstack((H, F_v))
            
            if np.linalg.det(H) == 0:                           # проверка определителя матрицы
                print("H", H)
                print("F_v", F_v)
                print('Error: nv = ' + str(nv_i) + ' from ' + str(nv_max))
            
            
            B = np.linalg.solve(H, np.array(v))                 # решение СЛАУ
            
            if Ind[len(Ind) - 1] == -1:                         # если не найден минимальный индекс

                LS_cur = LS(B, S, K, l, y_tr, gamma)            # вычисление целевой функции

                if i == Ind[0] or LS_cur < LS_min:              # обновление минимума
                    LS_min = LS_cur
                    Ind_min = i
            else:
                if return_array:
                    B_arr.append(B)
                break

            if i == Ind[len(Ind) - 2]:                          # если пройдены все свободные индексы добавляем в список найденный минимальный индекс и подготавливаем Omega, F, v
                Ind[len(Ind) - 1] = Ind_min

        #print("cur size:", nv_i, int(round(time.time() * 1000)) - millis_start)
        #print("cur size:", nv_i, np_millis)
        #9:230
    if return_array:
        return [B_arr, S] 
    else:
        return [B, S]

# функция перекресной проверки
# k - кратнорсть проверки
# n - размер одного подмножества элементов
# x - валидационное множество 
# C, sigma - параметры машины
# nv_max - количество опорных векторов
# return_array - [True, False] - параметр, определяющий возвращается ли
#           среднеквакдратическая ошибка для каждого количества опорных векторов
#           или только для максимального
# Возвращаемое значение: среднеквадратическая ошибка

K = []
cashed_sigma = 0

def k_training(k, n, x, y, C, sigma, nv_max, return_array):
    if return_array:
        inf_arr = []
        for i in range(nv_max):
            inf_arr.append(0)
    else:
        inf = 0
    

    for n_test in range(k):
        x_tr = []
        y_tr = []
        
        # подготовка тренировочной и тестовой последовательности
        for n_tr in range(k):
            if n_tr != n_test:
                x_tr += x[n_tr]
                y_tr += y[n_tr]

        l = len(x_tr)
        # заполнение матрицы ядра 
        if (sigma != cashed_sigma):
            K = [[0] * l for i in range(l)]
            for i in range(l):
                for j in range(l):
                    K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)
        
        if return_array:
            [B_arr, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv_max, True)  # построение машины
            # вычисление среднеквартичной ошибки для каждого количества опорных векторов
            for nv in range(len(S)):                                    
                y_res = []
                for i in range(n):
                    y_res = f(x[n_test][i], x_tr, S, B_arr[nv], sigma)
                    inf_arr[nv] += (y_res - y[n_test][i]) ** 2
        else:
            [B, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv_max, False)
            y_res = []
            for i in range(n):
                y_res = f(x[n_test][i], x_tr, S, B, sigma)
                inf += (y_res - y[n_test][i]) ** 2

    if return_array:
        for i in range(nv_max):
             inf_arr[i] = math.sqrt(inf_arr[i] / (n * k))
        return inf_arr
    else:
        inf /= n * k
        return math.sqrt(inf)