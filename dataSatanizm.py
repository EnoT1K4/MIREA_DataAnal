from collections import Counter
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/dubrovskijvladislav/Downloads/МАД - Лист1-3.csv")
months = list(data["Month"].dropna().apply(lambda x : int(x)))
height = list(data["Height"].dropna().apply(lambda x : float(x.replace(',','.'))))


frequency_dict = Counter(months)
sorted_items = sorted(frequency_dict.items())
total_observations = len(months)
relative_frequencies = [(value / total_observations) for _, value in sorted_items]
print("вариационный ряд с абсолютными и относительными частотами:")
for key, value in sorted_items:
    print(f"{key}: абсолютная частота: {value}, относительная частота: {round(value / total_observations,2)}")
 
 
    
frequency = Counter(months)
total_values = len(months)
relative_frequency = [count / total_values for count in frequency.values()]
plt.plot(sorted(list(frequency.keys())), relative_frequency, marker='o')
plt.xlabel('Значения')
plt.ylabel('Относительные частоты')
plt.title('Полигон относительных частот')
plt.show()

def empirical_cdf(months):
    sorted_data = np.sort(months)
    n = len(sorted_data)
    y = np.arange(1, n+1) / n
    return sorted_data, y
#график


x, y = empirical_cdf(months)
plt.plot(x, y, drawstyle='steps-post')
plt.xlabel('Значения выборки')
plt.ylabel('Значения функции')
plt.title('Эмпирическая функция распределения')
plt.show()


def opisatelnaya_statka(data):
# Выборочное среднее (среднее значение выборки) можно рассчитать путем сложения всех значений выборки и деления на количество значений в выборке:
    mean = sum(data) / len(data)
    # Выборочная дисперсия можно рассчитать как среднее арифметическое квадратов отклонений каждого значения выборки от выборочного среднего:
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    # Выборочное стандартное отклонение можно рассчитать как квадратный корень выборочной дисперсии:
    std_deviation = variance ** 0.5
    # Выборочная медиана - это значение, которое находится в середине упорядоченной выборки. Для расчета выборочной медианы сначала нужно отсортировать выборку, а затем найти значение, которое находится посередине:
    sorted_data = sorted(data)
    median = sorted_data[len(sorted_data) // 2]
    # Коэффициент вариации - это отношение стандартного отклонения к среднему значению выборки. Он позволяет оценить относительную изменчивость выборки:
    coefficient_of_variation = (std_deviation / mean) * 100
    
    print('Выборочное среднее:' , mean , '\n', 
          'Выборочная дисперсия' , variance , '\n',
          'Выборочная медиана' , median , '\n',
          'Коэффициент вариации' , coefficient_of_variation , '\n', 
          'Выборочное стандартное отклонениe' , std_deviation)
opisatelnaya_statka(months)

#part 2

def calculate_groups(n):
    m = 1 + math.log2(n)
    return math.ceil(m)

# Example usage
data_points = len(height) #enter u dataset
groups = calculate_groups(data_points)
print(groups, 'groups in data')

min_value = min(height)  
max_value = max(height) 
m = len(height)/groups  
interval_width = (max_value - min_value) / m

def calculate_boundaries(sample, h):
    min_value = min(sample)
    boundaries = [min_value + (i * h) for i in range(len(sample) + 1)]
    return boundaries

boundaries = calculate_boundaries(height, m)
print(boundaries, 'boundaries in data')

# Создание списка интервальных данных
intervals = []
for i in range(len(height)-1):
    intervals.append((height[i], height[i+1]))
# Сортировка списка
intervals_1 = sorted(intervals)
intervals_1.sort()
midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals_1]
midpoints.sort()
print(midpoints, 'midpoints')

# список значений интервалов выборки и соответствующих относительных частот
intervals = [max(val[1],val[0]) - min(val[1],val[0]) for val in intervals_1]
frequency_dict = Counter(intervals)
abs_frec = [val for key,val in frequency_dict.items()]
sorted_items = sorted(frequency_dict.items())
total_observations = len(intervals)
relative_frequencies = [round((value / total_observations),5) for _, value in sorted_items]
print(intervals_1, 'Inter', '\n', abs_frec, 'abs', '\n', relative_frequencies, '\n', 'rel')
#plt.bar(intervals, frequencies)
#plt.xlabel('Интервалы выборки')
#plt.ylabel('Относительные частоты')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot (111)
ax.hist (height, edgecolor='black', weights=np.ones_like(height) / len (height)) 
plt.show()

x, y = empirical_cdf(height)
plt.plot(x, y, drawstyle='steps-post')
plt.xlabel('Значения выборки')
plt.ylabel('Значения функции')
plt.title('Эмпирическая функция распределения')
plt.show()

opisatelnaya_statka(height)



