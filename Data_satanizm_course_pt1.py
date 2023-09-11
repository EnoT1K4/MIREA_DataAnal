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


data = pandas.read_csv('https://video.ittensive.com/python-advanced/data-9753-2019-07-25.utf.csv', sep = ";")
percent = (data["UnemployedDisabled"] / data["UnemployedTotal"]) * 100
for key, val in enumerate(percent):
    if val < 2:
        print(data["Year"][key])
        break