import random
import math
from pathlib import Path
  
n = 100                                                  # количество элементов в каждой из частей выборки
k = 3                                                   # кратность перекрестной проверки
x = []
for i in range(k * n):                                  # генерация неповторяющихся вещественных чисел
    r = random.uniform(0, 2 * math.pi)
    if r not in x: 
        x.append(r) 

y =  list(map(math.sin, x))


with open("Data//model_data.txt", "w") as file:
    print(*x, file=file)
    print(*y, file=file)
