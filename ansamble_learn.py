from matplotlib import pyplot as plt
from sklearn import tree
from xgboost import XGBClassifier
import catboost as cb
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import ensemble
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
#part1
data = pd.read_csv('customer_classification_data.csv', sep=',')
X = data.drop('Education', axis =1)
y = data['Education']
my_file = open("otvet.txt", "w")
param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [i for i in range(1,20)],
        'min_samples_split': [i for i in range(1,17)],
        'min_samples_leaf': [i for i in range(1,10)]
    }



tree = DecisionTreeClassifier()
grid_search = GridSearchCV(tree, param_grid, cv=5)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_
plt.figure(figsize=(10, 6))
plot_tree(grid_search.best_estimator_)
plt.show()
my_file.write(best_params, ' Best params', '\n',  best_score, ' Best score')
print(best_params, ' Best params')
print(best_score, ' Best score')

#part2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение моделей для базовых алгоритмов
estimators = [
    ('svm', SVC()),
    ('logistic', LogisticRegression())
]

# Определение модели стекинга
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Обучение модели стекинга
stacking_model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = stacking_model.predict(X_test)


# Оценка качества полученной модели
accuracy = accuracy_score(y_test, y_pred)
my_file.write('Accuracy:', accuracy)
print(f"Accuracy: {accuracy}")

#part 3
base_model = DecisionTreeClassifier()


bagging_model = BaggingClassifier(base_model, n_estimators=10)

bagging_model.fit(X_train, y_train)

y_pred = bagging_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
my_file.write('Accuracy:', accuracy)
print(f"Accuracy: {accuracy}")

#part 4
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
boosting_model = XGBClassifier()

boosting_model.fit(X_train, y_train)

y_pred = boosting_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
my_file.write('Accuracy:', accuracy)
print(f"Accuracy XGBC: {accuracy}")

boosting_model = CatBoostClassifier()
boosting_model.fit(X_train, y_train)

y_pred = boosting_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
my_file.write('Accuracy:', accuracy)
print(f"Accuracy Cat: {accuracy}")

boosting_model = LGBMClassifier()
boosting_model.fit(X_train, y_train)

y_pred = boosting_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
my_file.write('Accuracy:', accuracy)
print(f"Accuracy Light: {accuracy}")


my_file.close()
