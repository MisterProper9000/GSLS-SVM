import random
import math
import numpy as np
import matplotlib.pyplot as plt
from GSLSSVM import k_training



#заполнение данных, k массивов по n элементов
n = 30                                                 # количество элементов в каждой из частей выборки
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

d = n * (k - 1)
C_begin = 2 ** (1)
C_end = (2 ** (16)) * d
C = C_begin

sigma_begin = 0.1
sigma_end = 2.3
sigma_step = 0.2
s_range =np.arange(sigma_begin, sigma_end, sigma_step)

while C <= C_end:
    # перебор sigma
    print("C = " + str(C))
    for sigma in s_range:
        print("sigma = " + str(sigma))
        inf_cur = k_training(k, n, x, y, C, sigma, d // 10, False)
        if (C == C_begin and sigma == sigma_begin) or (inf_cur < inf_min):
            inf_min = inf_cur
            C_min = C
            sigma_min = sigma
    C *= 2
        

print("C_min = " + str(C_min))
print("sigma_min = " + str(sigma_min))
C = C_min
sigma = sigma_min
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