'''
Получите данные по безработице в Москве:

https://video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv

Найдите, с какого года процент людей с ограниченными возможностями (UnemployedDisabled)

среди всех безработных (UnemployedTotal) стал меньше 2%.

Вопросы к этому заданию
С какого года безработных инвалидов меньше 2% в Москве?

---

Пример преподавателя

С какого года безработных инвалидов меньше 2% в Москве?

2018
'''
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pandas.read_csv('https://video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv', sep = ";")

def task1(data):
    percent = (data["UnemployedDisabled"] / data["UnemployedTotal"]) * 100
    for key, val in enumerate(percent):
        if val < 2:
            print(data["Year"][key])
            break
    
df_grouped = data.groupby('Year').filter(lambda x: len(x) >= 6)

data["UDP"] = 100*data[ "UnemployedDisabled"]/data[ "UnemployedTotal"]
data_group = data. groupby ("Year"). filter (lambda x: (x["UDP"].count()) > 5)
data_group = data_group. groupby ("Year"). mean()
x = np. array (data_group. index).reshape(len (data_group.index), 1)
y = np.array(data_group["UDP" ]).reshape(len(data_group.index), 1)
model = LinearRegression()
model. fit(x, y)
print (np.round(model.predict(np.array(2020).reshape(1,1)),2))

'''
Bозьмите данные по безработице в городе Москва:

video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv


Постройте модель линейной регрессии по годам среднего значения отношения UnemployedDisabled к UnemployedTotal (процента людей с ограниченными возможностями) за месяц и ответьте, какое ожидается значение в 2020 году при сохранении текущей политики города Москвы?

Ответ округлите до сотых. Например, 2,32

Вопросы к этому заданию
Какое ожидается значение в 2020 году при сохранении текущей политики города Москвы?

---

Пример преподавателя

Какое ожидается значение в 2020 году при сохранении текущей политики города Москвы?

1,52

'''

import numpy as np
import matplotlib.pyplot as plt
