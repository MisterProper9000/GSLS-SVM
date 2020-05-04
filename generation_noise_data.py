import random
import math
import numpy as np
from pathlib import Path
  
n = 50                                                  # количество элементов в каждой из частей выборки
k = 3                                                   # кратность перекрестной проверки
x = []
for i in range(k * n):                                  # генерация неповторяющихся вещественных чисел
    r = random.uniform(0, 2 * math.pi)
    if r not in x: 
        x.append(r) 

noise = np.random.normal(0,0.1,k * n)
y =  list(map(math.sin, x))
y_noise = list(map(lambda a, b: a + b, noise, y))

with open("noise_data.txt", "w") as file:
    print(*x, file=file)
    print(*y_noise, file=file)
