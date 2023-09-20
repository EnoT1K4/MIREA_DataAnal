import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
color_pal = sns.color_palette()
plt.style.use('seaborn-dark-palette')
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/3703888/6420753/Top_100_Youtube_Channels_2023.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230918%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230918T115838Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=342ad55df6a4a11d15510c2c3fb68565f2fc96be26371a4d75331d39690dd01a7d7edc271971e58d6611a5bec288cd4d10ba5b8c19431a447edb8b2b0dae729468474137fa5f26636f6106fcbeb84598d665a94454d45b9e375679114455796b6954ef1ddbb766a3d3322203da3a04e73de661d7b542cacea4f971443a54d7b6774588bd571cef27d30deffb7fecdd88bac8aa00b5e0f5ef3401dec4876213d23c4d1e8246ac732eca5f53e1589d74fbf530c7aa9c8cbc815d4be2d16ffe386cc09b18f0d4ac1787f39203a7157d81114787a66c41171fe4c9c05c5cb5d2c1aaa1f3b81355004725fa4c733c8fe511727157811ac9fcf6662918749e225992ea', sep = ',')
print(data.head())
data.drop('Unnamed: 0',axis=1,inplace=True)
data.fillna("No description for this channel",inplace=True)
def convert_subscribers(subscribers):
    if 'M' in subscribers:
        return float(subscribers.replace('M', ''))   # Convert to millions
    else:
        return float(subscribers)

# Apply the function to the 'Subscribers' column
data['Subscribers'] = data['Subscribers'].apply(convert_subscribers)

# Change the data type of 'Subscribers' to int or float
data['Subscribers'] = data['Subscribers'].astype(int)  # Change to int

def convert_views(views):
    if 'B' in views:
        return float(views.replace('B', '')) * 10**3 # Convert to millions
    else:
        return float(views)

# Apply the function to the 'Subscribers' column
data['Views'] = data['Views'].apply(convert_views)

# Change the data type of 'Subscribers' to int or float
data['Views'] = data['Views'].astype(int)  # Change to int
    
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Views', y='Subscribers', data=data,)
plt.title('Views vs. Subscribers')
plt.xlabel('Views')
plt.ylabel('Subscribers')
plt.show()

# Load the data
X = data[['Views', 'Videos']]
y = data['Subscribers']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LinearRegression()
model.fit(X_train_scaled, y_train) 
print('Коэффициенты:', model.coef_)
print('Свободный член:', model.intercept_)   
new_data = data[['Views', 'Videos']]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print('Предсказанное значение:', prediction)
# Train Lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)

# Train Ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train_scaled, y_train)

# Train another linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Make predictions using the trained models
lasso_predictions = lasso_model.predict(X_test_scaled)
ridge_predictions = ridge_model.predict(X_test_scaled)
linear_predictions = linear_model.predict(X_test_scaled)

# Calculate mean squared error for each model
lasso_mse = mean_squared_error(y_test, lasso_predictions)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)

# Calculate R-squared score for each model
linear_regression_r2 = r2_score(y_test, linear_predictions)
ridge_regression_r2 = r2_score(y_test, ridge_predictions)
lasso_regression_r2 = r2_score(y_test, lasso_predictions)

print("Метрики для линейной регрессии без регуляризации:")
print("Value:", round(linear_predictions.mean(),3), 'примерное количество подписчиков')
print("MSE:", linear_mse)
print("R2 score:", linear_regression_r2)

print("Метрики для регуляризации Тихонова:")
print("Value:", round(ridge_predictions.mean(),3), 'примерное количество подписчиков')
print("MSE:", ridge_mse)
print("R2 score:", ridge_regression_r2)

print("Метрики для лассо регуляризации:")
print("Value:", round(lasso_predictions.mean(),3), 'примерное количество подписчиков')
print("MSE:", lasso_mse)
print("R2 score:", lasso_regression_r2)