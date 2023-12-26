import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap

df = pd.read_csv('./smoking_driking_dataset_Ver01.csv')
print(df.head())
sc=StandardScaler()


le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["DRK_YN"] = le.fit_transform(df["DRK_YN"])
X = df.drop("DRK_YN",axis=1)
y = df["DRK_YN"]

linear_model = LogisticRegression()
linear_model.fit(X, y)

p_values = linear_model.coef_
sorted_features = sorted(zip(X.columns, p_values), key=lambda x: x[1], reverse=True)

feature_weights = abs(linear_model.coef_)
sorted_features = sorted(zip(X.columns, feature_weights), key=lambda x: x[1], reverse=True)

# Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

lasso_selector = SelectFromModel(lasso_model, prefit=True)
selected_features = X.columns[lasso_selector.get_support()]

# Calculate and display p-values
X_with_constant = sm.add_constant(X[selected_features])
linear_model_with_constant = sm.OLS(y, X_with_constant).fit()
p_values = linear_model_with_constant.pvalues[1:]

print("Selected features:")
for feature, p_value in zip(selected_features, p_values):
    print(f"{feature}: {p_value}")
#part 2

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3,random_state=22)

# Создаем экземпляр класса SelectKBest и задаем метод оценки f_regression
selector = SelectKBest(score_func=f_regression) 
X_new = selector.fit_transform(X, y)

# Получаем индексы выбранных признаков
selected_feature_indices = selector.get_support(indices=True)

# Выводим имена выбранных признаков и их веса
selected_features = [X.columns[i] for i in selected_feature_indices]

# Обучаем модель LinearRegression на выбранных признаках
model = LinearRegression()
model.fit(X_new, y)
print('используя метод весов')
# Выводим имена выбранных признаков и их веса
for feature, weight in zip(selected_features, model.coef_):
    print(f"Признак: {feature}, Вес: {weight}")



#part2
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3,random_state=22)
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
feature_names = X.columns.values.tolist()

rfc = RandomForestClassifier(n_estimators=150,criterion= "gini",n_jobs=-1)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
rfc_imp = rfc.feature_importances_
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
acc_score_rfc = accuracy_score(y_test,y_pred_rfc)
print("confusion matrix :", cm_rfc)
print("accuracy score Random Forest :", acc_score_rfc)


std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
forest_importances = pd.Series(rfc_imp, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()






gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb_model = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_gb_model)
cm_gb_model = confusion_matrix(y_test, y_pred_gb_model)
print("Gradient Boost:", accuracy)
print("confusion matrix:", cm_gb_model)
importances = gb_model.feature_importances_
plt.figure(figsize=(15,3))
plt.bar(feature_names, importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()




explainer = shap.Explainer(rfc)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
print(shap_values, 'RandForCl')
explainer = shap.Explainer(gb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
print(shap_values,'GradBoost')



pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(pca.explained_variance_ratio_)



#tsne = TSNE(n_components=2, perplexity=40, random_state=123)
#tsne_features = tsne.fit_transform(df)

#plt.scatter(tsne_features[:, 0], tsne_features[:, 1])
#plt.show()