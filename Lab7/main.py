import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

# 1. Завантаження даних
df = pd.read_csv('merc.csv')

# 2. Попередня обробка
df = df.dropna()

# 3. Вибір ознак для регресії
X = df[['year', 'mileage', 'engineSize']]
y = df['price']

# 4. Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 5. Масштабування ознак для лінійної регресії
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Лінійна регресія
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 7. Випадковий ліс
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# 8. Оцінка моделей
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nОцінка моделі: {model_name}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")


evaluate_model(y_test, y_pred_lr, "Лінійна регресія")
evaluate_model(y_test, y_pred_rf, "Випадковий ліс")

# 9. Покращення моделі
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Оцінка покращеної моделі
evaluate_model(y_test, y_pred_best_rf, "Покращений випадковий ліс (GridSearchCV)")

# пайплайн: StandardScaler + LinearRegression
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

# Налаштування параметрів для підбору
param_grid_lr = {
    'lr__fit_intercept': [True, False],
    'lr__positive': [True, False]
}

grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='r2')
grid_search_lr.fit(X_train, y_train)

# Краща модель
best_lr_pipeline = grid_search_lr.best_estimator_
y_pred_best_lr = best_lr_pipeline.predict(X_test)

# Оцінка покращеної лінійної регресії
evaluate_model(y_test, y_pred_best_lr, "Покращена Лінійна регресія (GridSearchCV)")

# 10. Кластеризація
cluster_data = df[['mileage', 'engineSize', 'mpg']].dropna()

scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(cluster_data)

# Підбір оптимальної кількості кластерів
inertia_list = []
silhouette_list = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_scaled)
    inertia_list.append(kmeans.inertia_)
    silhouette_list.append(silhouette_score(cluster_scaled, kmeans.labels_))

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), inertia_list, marker='o')
plt.title('Метод ліктя для визначення кількості кластерів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Inertia')
plt.grid()
plt.show()

optimal_k = 3

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_final.fit_predict(cluster_scaled)

cluster_data['Cluster'] = clusters

print("\nОцінка кластеризації:")
print(f"Silhouette Score: {silhouette_score(cluster_scaled, clusters):.2f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(cluster_scaled, clusters):.2f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(cluster_scaled, clusters):.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=cluster_data['mileage'], y=cluster_data['mpg'], hue=cluster_data['Cluster'], palette='viridis')
plt.title('Кластери автомобілів')
plt.xlabel('Пробіг (mileage)')
plt.ylabel('Витрати палива (mpg)')
plt.legend()
plt.grid()
plt.show()
