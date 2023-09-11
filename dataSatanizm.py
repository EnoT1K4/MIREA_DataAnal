from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]  # Replace this with your own data

# Step 2: Count the frequency of each unique value
frequency_dict = Counter(data)

    # Step 3: Sort the dictionary items by keys
sorted_items = sorted(frequency_dict.items())

    # Step 4: Calculate relative frequencies
total_observations = len(data)
relative_frequencies = [(value / total_observations) for _, value in sorted_items]

    # Print the variation series with absolute and relative frequencies
print("вариационный ряд с абсолютными и относительными частотами:")
for key, value in sorted_items:
    print(f"{key}: абсолютная частота: {value}, относительная частота: {value / total_observations}")
        


frequency = Counter(data)
total_values = len(data)
relative_frequency = [count / total_values for count in frequency.values()]

plt.plot(list(frequency.keys()), relative_frequency, marker='o')
plt.xlabel('Значения')
plt.ylabel('Относительные частоты')
plt.title('Полигон относительных частот')
plt.show()




def empirical_cdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n+1) / n
    return sorted_data, y


#график
x, y = empirical_cdf(data)
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
    
    print('Выборочное среднее:' , mean , '\n' , 
          'Выборочная дисперсия' , variance , '\n' ,
          'Выборочная медиана' , median , '\n' ,
          'Коэффициент вариации' , coefficient_of_variation , '\n' , 
          'Выборочное стандартное отклонениe' , std_deviation)
opisatelnaya_statka(data)





#part 2 with  new data only
def calculate_groups(n):
    m = 1 + math.log2(n)
    return math.ceil(m)

# Example usage
data_points = len(data) #enter u dataset
groups = calculate_groups(data_points)
print(groups, 'group')


min_value = min(data)  
max_value = max(data) 
m = len(data)/groups  
interval_width = (max_value - min_value) / m

boundaries = []
for i in range(int(m)):
    boundary = min_value + i * interval_width
    boundaries.append(boundary)

print(boundaries, 'bound')

# Создание списка интервальных данных
intervals = []
for i in range(len(data)-1):
    intervals.append((data[i], data[i+1]))
# Сортировка списка
intervals_1 = sorted(intervals)
# Вывод отсортированного списка
    
    # Step 2: Sort intervals in ascending order
intervals_1.sort()
    
    # Step 3 and 4: Calculate midpoints and store in a new list
midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals_1]
    
    # Step 5: Sort midpoints in ascending order
midpoints.sort()
    
    # Step 6: Print the variational series
print(midpoints, 'midpoints')


# Здесь предполагается, что у вас есть список значений интервалов выборки и соответствующих относительных частот
intervals = [val[1]-val[0] for val in intervals_1]

frequency_dict = Counter(intervals)


    # Step 3: Sort the dictionary items by keys
sorted_items = sorted(frequency_dict.items())

    # Step 4: Calculate relative frequencies
total_observations = len(intervals)
relative_frequencies = [(value / total_observations) for _, value in sorted_items]
frequencies = relative_frequencies

print(len(intervals), len(frequencies))

# Построение гистограммы
plt.bar(intervals, frequencies)

# Добавление подписей осей
plt.xlabel('Интервалы выборки')
plt.ylabel('Относительные частоты')

# Отображение гистограммы
plt.show()

x, y = empirical_cdf(data)
plt.plot(x, y, drawstyle='steps-post')
plt.xlabel('Значения выборки')
plt.ylabel('Значения функции')
plt.title('Эмпирическая функция распределения')
plt.show()

opisatelnaya_statka(data)
