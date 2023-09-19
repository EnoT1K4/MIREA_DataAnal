import math
import numpy as np
import pandas as pd
from scipy.stats import norm, t, chi2

data1 = pd.read_csv('/Users/dubrovskijvladislav/Desktop/lists/1.txt', header = None)[0].values.tolist()
data2 = pd.read_csv('/Users/dubrovskijvladislav/Desktop/lists/2.txt', header = None)[0].values.tolist()
data3 = pd.read_csv('/Users/dubrovskijvladislav/Desktop/lists/3.txt', header = None)[0].values.tolist()
data4 = pd.read_csv('/Users/dubrovskijvladislav/Desktop/lists/4.txt', header = None)[0].values.tolist()

def calculate_point_estimates(data):
    # Calculate the point estimate of the mean (𝜇)
    mean = sum(data) / len(data)

    # Calculate the point estimate of the standard deviation (𝜎)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    std_dev = math.sqrt(squared_diff_sum / (len(data) - 1))

    return round(mean,3), round(std_dev,3)


def calculate_confidence_interval_normal(data, confidence_level=0.95):
    # Calculate the point estimate of the mean (𝜇)
    mean = sum(data) / len(data)

    # Calculate the point estimate of the standard deviation (𝜎)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    std_dev = math.sqrt(squared_diff_sum / (len(data) - 1))

    # Calculate the critical value (𝑧) from the standard normal distribution
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z * (std_dev / math.sqrt(len(data)))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return round(lower_bound,3), round(upper_bound,3)


def calculate_confidence_interval_t(data, confidence_level = 0.95):
    # Calculate the point estimate of the mean (𝜇)
    mean = sum(data) / len(data)

    # Calculate the point estimate of the standard deviation (𝜎)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    std_dev = math.sqrt(squared_diff_sum / (len(data) - 1))

    # Calculate the critical value (𝑡) from the t-distribution
    dof = len(data) - 1
    t_value = t.ppf(1 - (1 - confidence_level) / 2, dof)

    # Calculate the margin of error
    margin_of_error = t_value * (std_dev / math.sqrt(len(data)))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return round(lower_bound,3), round(upper_bound,3)


def confidence_interval_std_dev(data, confidence_level = 0.95 ):
    sample_size = len(data)
    sample_mean = sum(data) / sample_size
    sample_variance = sum((x - sample_mean) ** 2 for x in data) / (sample_size - 1)
    lower_bound = math.sqrt((sample_size - 1) * sample_variance / chi2.ppf((1 + confidence_level) / 2, sample_size - 1))
    upper_bound = math.sqrt((sample_size - 1) * sample_variance / chi2.ppf((1 - confidence_level) / 2, sample_size - 1))
    return (round(lower_bound,3), round(upper_bound,3))


print('точечные оценки математического ожидания и стандартного отклонения для списка 1:', calculate_point_estimates(data1), '\n')
print('границы доверительного интервала мат. ожидания по правилу нормального распределения для списка 1:', calculate_confidence_interval_normal(data1), '\n')
print('По правилу t-распределения Стьюдента, используя таблицу критических значений t-распределения при значении уверенности = 0.95 для списка 1:', calculate_confidence_interval_t(data1), '\n')
print('границы доверительного интервала для среднеквадратического отклонения по оценке 𝜒2-распределения при значении уверенности = 0.95 для списка 1:', confidence_interval_std_dev(data1))
print('-------------------------------------------------------------------------------------')
print('точечные оценки математического ожидания и стандартного отклонения для списка 2:', calculate_point_estimates(data2), '\n')
print('границы доверительного интервала мат. ожидания по правилу нормального распределения для списка 2:', calculate_confidence_interval_normal(data2), '\n')
print('По правилу t-распределения Стьюдента, используя таблицу критических значений t-распределения при значении уверенности = 0.95 для списка 2:', calculate_confidence_interval_t(data2), '\n')
print('границы доверительного интервала для среднеквадратического отклонения по оценке 𝜒2-распределения при значении уверенности = 0.95 для списка 2:', confidence_interval_std_dev(data2))
print('-------------------------------------------------------------------------------------')
print('точечные оценки математического ожидания и стандартного отклонения для списка 3:', calculate_point_estimates(data3), '\n')
print('границы доверительного интервала мат. ожидания по правилу нормального распределения для списка 3:', calculate_confidence_interval_normal(data3), '\n')
print('По правилу t-распределения Стьюдента, используя таблицу критических значений t-распределения при значении уверенности = 0.95 для списка 3:', calculate_confidence_interval_t(data3), '\n')
print('границы доверительного интервала для среднеквадратического отклонения по оценке 𝜒2-распределения при значении уверенности = 0.95 для списка 3:', confidence_interval_std_dev(data3))
print('-------------------------------------------------------------------------------------')
print('точечные оценки математического ожидания и стандартного отклонения для списка 4:', calculate_point_estimates(data4), '\n')
print('границы доверительного интервала мат. ожидания по правилу нормального распределения для списка 4:', calculate_confidence_interval_normal(data4), '\n')
print('По правилу t-распределения Стьюдента, используя таблицу критических значений t-распределения при значении уверенности = 0.95 для списка 4:', calculate_confidence_interval_t(data4), '\n')
print('границы доверительного интервала для среднеквадратического отклонения по оценке 𝜒2-распределения при значении уверенности = 0.95 для списка 4:', confidence_interval_std_dev(data4))
print('-------------------------------------------------------------------------------------')


