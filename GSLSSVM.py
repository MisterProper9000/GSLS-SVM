import math
import numpy as np


def Kernel(x1, x2, s):
    return math.exp(-(s**(-2)) * ((x1 - x2) ** 2))

   
def LS(B, S, K, l, y, gamma):
    val_left = 0

    for i in range(len(S)):
        for j in range(len(S)):
            val_left += B[i] * B[j] * K[S[i]][S[j]]
    val_left *= 0.5

    val_right = 0
    for i in range(l):
        val = y[i]
        for j in range(len(S)):
            val -= B[j] * K[i][S[j]]
        val -= B[len(B) - 1]
        val_right += val ** 2 

    val_right *= gamma / l
    return val_left + val_right

def f(x, x_ref, S, B, sigma):
    res = 0
    for i in range(len(B) - 1):
        res += B[i] * Kernel(x_ref[S[i]], x, sigma)
    res += B[len(B) - 1]
    return res

def LS_SVM(C, sigma, K, x_tr, y_tr, nv_max, return_array):
    gamma = C
    y_sum = sum(y_tr)
    l = len(x_tr)
    
    B_arr = []
    S = []
    v = [y_sum]                                         # Вектор (правая часть системы) 
    F = []
    Omega = np.zeros((1,1)) 

    for nv_i in range(nv_max):
        Ind = list(set(range(l)) - set(S)) + [-1]
        S.append(0)
        for i in Ind:
            S[nv_i] = i

            if len(F) == nv_i: 
                v.insert(nv_i, 0)
                F.append(0)
            else:
                v[nv_i] = 0
                F[nv_i] = 0

            for j in range(l):
                v[nv_i] += y_tr[j] * K[i][j] 
                F[nv_i] += K[i][j]


            if Omega.size == nv_i ** 2:
                Omega = np.vstack((Omega, np.zeros((nv_i), dtype=float)))    # добавление строки в матрицу 
                Omega = np.hstack((Omega, np.zeros((nv_i + 1, 1), dtype=float))) # добавление столбца в матрицу 

            for j in range(nv_i + 1):
                Omega[nv_i][j] = l / (2.*gamma) * K[i][S[j]]
                for r in range(l):
                    Omega[nv_i][j]  += K[r][S[j]] * K[r][i]
                Omega[j][nv_i] = Omega[nv_i][j]
            

            H = Omega
            F_v = np.array(F) 
            H = np.vstack((H, F_v))
            F_v = np.insert(F_v, nv_i + 1, l)
            F_v = F_v.reshape(nv_i + 2, 1)
            H = np.hstack((H, F_v))

            if np.linalg.det(H) == 0:
                print('here')
            B = np.linalg.solve(H, np.array(v))

            if Ind[len(Ind) - 1] == -1: 
                LS_cur = LS(B, S, K, l, y_tr, gamma)

                if i == Ind[0] or LS_cur < LS_min:
                    LS_min = LS_cur
                    Ind_min = i
            else:
                if return_array:
                    B_arr.append(B)
                break

            if i == Ind[len(Ind) - 2]:
                Ind[len(Ind) - 1] = Ind_min
    if return_array:
        return [B_arr, S] 
    else:
        return [B, S]

# перекресная проверка кратности k
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
        
        for n_tr in range(k):
            if n_tr != n_test:
                x_tr += x[n_tr]
                y_tr += y[n_tr]

        l = len(x_tr)
        # заполнение матрицы ядра 
        K = [[0] * l for i in range(l)]
        for i in range(l):
            for j in range(l):
                K[i][j] = Kernel(x_tr[i], x_tr[j], sigma)  

        
        if return_array:
            [B_arr, S] = LS_SVM(C, sigma, K, x_tr, y_tr, nv_max, True)
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