import random
import math
import numpy as np
import matplotlib.pyplot as plt


def Kernel(x1, x2, s):
    return math.exp(-(s**(-2)) * ((x1 - x2) ** 2))

def IndexMinLS(B, S, K, l, y, ind):
    minInd = -1
    for i in range(l):
        if i not in ind:
            curLS = LS(B, S, K, y[i], i)
            if minInd == -1 or abs(curLS) < abs(minLS):
                minLS = curLS
                minInd = i
    return minInd
                

def LS(B, S, K, yi, ind):
    for j in range(S):
        yi -= B[j]*K[ind][j]
    yi -= B[S]
    return yi

def f(x, x_ref, B, sigma):
    res = 0
    for i in range(len(x_ref)):
        res += B[i] * Kernel(x_ref[i], x, sigma)
    res += B[len(x_ref)]
    return res

n = 30                                                  # количество элементов в каждой из частей выборки
k = 3                                                   # кратность перекрестной проверки
x=[]
for i in range(k):                                      # генерация неповторяющихся вещественных чисел   
    x.append([]) 
    for j in range(n):
        r = random.uniform(0, 2 * math.pi)
        if r not in x[i]: x[i].append(r)

y = []
for i in range(k):
    y.append([]) 
    y[i] =  list(map(math.sin, x[i]))                            

# настройка машины:
# перебор C
C_begin = 0.5
C_end = 5.5
C_step = 0.5
for C in range(C_begin, C_end, C_step):
    # перебор sigma
    sigma_begin = 0.1
    sigma_end = 1.5
    sigma_step = 0.1
    for sigma in range(sigma_begin, sigma_end, sigma_step):
        gamma = C

        x_tr = []
        y_tr = []
        # перекресная проверка кратности k
        for n_test in range(k):
            for n_tr in range(k):
                if n_tr != n_test:
                    x_tr += x[n_tr]
                    y_tr += y[n_tr]
            y_sum = sum(y_tr)
            l = len(x_tr)

            # заполнение матрицы ядра 
            K = [[0] * l for i in range(l)]
            for i in range(l):
            for j in range(l):
                K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)  


            x_ref = []
            ind_ref = []
            y_res = []
            v = [y_sum]                                         # Вектор (правая часть системы) 
            F = []
            Omega = np.array([l / (2.*gamma) * K[0][0]])
            for r in range(l):
                Omega[0] +=  K[r][0]** 2


            nv = 15                                       # число опорных векторов
            for i in range(nv):
                c_i = 0
                F_i = 0
                for j in range(l):
                    c_i += y_tr[j] * K[i][j] 
                    F_i += K[i][j]

                v.insert(i, c_i)
                F.append(F_i)

                if i:
                    Omega_col = np.zeros((i), dtype=float)
                    for j in range(i):

                    Omega_col[j] += l / (2.*gamma) * K[i][j]
                    for r in range(l):
                            Omega_col[j] += K[r][j] * K[r][i]
                

                    Omega = np.vstack((Omega, Omega_col))    
                    Omega_col = np.insert(Omega_col, i, 0)
                    for r in range(l):
                        Omega_col[i] +=  K[r][i] ** 2  
                    Omega_col = Omega_col.reshape(i + 1, 1)
                    Omega = np.hstack((Omega, Omega_col)) 

                H = Omega
                F_v = np.array(F) 
                H = np.vstack((H, F_v))
                F_v = np.insert(F_v, i + 1, l)
                F_v = F_v.reshape(i + 2, 1)
                H = np.hstack((H, F_v))

                B = np.linalg.solve(H, np.array(v))
                IndMin = IndexMinLS(B, i, K, l, y_tr, ind_ref)
                ind_ref.append(IndMin)
                x_ref.append(x_tr[IndMin])

                #вычисление значения функции для тестового множества
                y_res = []
                measure = 0
                for i in range(n):
                    y_res.append(f(x[n_test][i], x_ref, B, sigma))
                    # вычисление ошибки
                    measure += abs(y_res[i] - y[n_test][i])
                measure /= n

                
                
            
                #plt.plot(x_tr, y_res, '.')






plt.plot(x_tr, y_tr, '*')
plt.show()