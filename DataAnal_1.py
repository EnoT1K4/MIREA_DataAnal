import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def sliding_window(x_array, w, step):
    num_rows = (len(x_array) - w) // step + 1
    A_matrix = np.zeros((num_rows, w))

    for i in range(num_rows):
        start = i * step
        end = start + w
        A_matrix[i] = x_array[start:end]

    return A_matrix

window = 3
step_s = 1
x1 = np.array([8, 1, 4, 5, -2, 5, 9, 0])
A1 = np.array([[8, 1, 4],
                    [1, 4, 5],
                    [4, 5, -2],
                    [5, -2, 5],
                    [-2, 5, 9],
                    [5, 9, 0]])

print(np.array_equal(sliding_window(x1, w=window, step=step_s),A1))

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep = ", ", header=None,
                   names = ["age", "workclass", "fnlwgt", "education",
                    "education-num", "marital-status", "occupation", "relationship",
                    "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"])

list_of_objects = data.dtypes.axes[0][data.dtypes == "object"]
data[list_of_objects] = data[list_of_objects].astype("string")

sex_counts = data['sex'].value_counts()
Fem_counter = sex_counts[1]
# Visualize 
sex_counts.plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Number of Males and Females')
plt.show()
# Mean
su = 0  
for id,val in enumerate(data['sex']):
    if val == 'Female':
        su += data['age'][id]
print('Средний возраст женщин = ' + str(su/Fem_counter))

#Germany
all = data['native-country'].value_counts().sum()
ger = 0
for elem in data['native-country']:
    if elem == 'Germany':
        ger += 1
print('Процент немцев = ' + str(ger/all) + '%')



# Выборка данных для тех, кто получает более 50K в год
high_income = data[data['salary'] == '>50K']


# Выборка данных для тех, кто получает менее 50K в год
low_income = data[data['salary'] == '<=50K']

# Расчет среднего значения и среднеквадратичного отклонения возраста для каждой группы
high_income_mean = high_income['age'].mean()
high_income_std = high_income['age'].std()

low_income_mean = low_income['age'].mean()
low_income_std = low_income['age'].std()

# Вывод результатов
print(f"Средний возраст : {high_income_mean:.2f}")
print(f"Среднеквадратичное отклонение возраста : {high_income_std:.2f}")

print(f"Средний возраст менее 50K в год: {low_income_mean:.2f}")
print(f"Среднеквадратичное отклонение возраста менее 50K в год: {low_income_std:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.bar(['>50K', '<=50K'], [high_income_mean, low_income_mean], yerr=[high_income_std, low_income_std])
plt.xlabel('Уровень дохода')
plt.ylabel('Средний возраст')
plt.title('Сравнение среднего возраста для разных')
plt.show()



education_counts = data[data['salary'] == '>50K']['education'].value_counts()

# Вывод результатов
print(education_counts)



stats = data.groupby(['race', 'sex'])['age'].describe()
print(stats)
max_age = stats.loc[('Amer-Indian-Eskimo', 'Male'), 'max']
print("Максимальный возраст:", max_age)




men_data = data[data['sex'] == 'Male']

married_men = men_data[men_data['marital-status'].str.startswith('Married')]
single_men = men_data[~men_data['marital-status'].str.startswith('Married')]
married_high_earners = married_men[married_men['salary'] == '>50K'].shape[0]
single_high_earners = single_men[single_men['salary'] == '>50K'].shape[0]
married_proportion = married_high_earners / married_men.shape[0]
single_proportion = single_high_earners / single_men.shape[0]
if married_proportion > single_proportion:
    print("Женатые")
else:
    print("Счастливые")
    

max_hours = data['hours-per-week'].max()

count_max_hours = data[data['hours-per-week'] == max_hours].shape[0]
percent_high_income = (data[(data['hours-per-week'] == max_hours) & (data['salary'] == '>50K')].shape[0] / count_max_hours) * 100
print(f"Максимальное количество часов : {max_hours}")
print(f"Количество людей: {count_max_hours}")
print(f"Процент людей: {percent_high_income}%")


average_hours = data.groupby(['native-country', 'salary'])['hours-per-week'].mean()
average_hours.unstack().plot(kind='bar')
plt.xlabel('Cтрана')
plt.ylabel('Часы/неделя')
plt.show()