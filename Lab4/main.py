import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'bike.csv'
data = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")

# 1a. Стовпчикова діаграма: кількість покупців різних професій
plt.figure(figsize=(12, 6))
sns.countplot(y='Occupation', data=data, order=data['Occupation'].value_counts().index)
plt.xticks(ticks=range(0, max(data['Occupation'].value_counts()) + 1, 10))  # Зменшений крок осі X
plt.title('Кількість покупців різних професій')
plt.xlabel('Кількість')
plt.ylabel('Професія')
plt.show()

print("Кількість покупців за професіями:")
print(data['Occupation'].value_counts())

# 1b. Медіанний дохід покупців різних професій (без довірчих інтервалів)
plt.figure(figsize=(12, 6))
income_median_order = data.groupby('Occupation')['Income'].median().sort_values(ascending=True).index
sns.barplot(y='Occupation', x='Income', data=data, estimator=np.median, order=income_median_order, errorbar=None)
plt.title('Медіанний дохід покупців різних професій')
plt.xlabel('Дохід')
plt.ylabel('Професія')
plt.show()

print("Медіанний дохід за професіями:")
print(data.groupby('Occupation')['Income'].median())

# 1c. Середній вік покупців різних професій з розподілом за покупкою велосипеда
plt.figure(figsize=(12, 6))
sns.barplot(y='Occupation', x='Age', hue='Purchased Bike', data=data, estimator=lambda x: x.mean(), errorbar=None)
plt.title('Середній вік покупців різних професій')
plt.xlabel('Вік')
plt.ylabel('Професія')
plt.legend(title='Купив велосипед')
plt.show()

print("Середній вік покупців за професіями:")
print(data.groupby(['Occupation', 'Purchased Bike'])['Age'].mean())

# 2. Гістограма кількості дітей загальна та залежно від покупки велосипеда
# Гістограма кількості дітей загальна
plt.figure(figsize=(10, 5))
sns.histplot(data['Children'], bins=range(0, data['Children'].max() + 2), kde=False)
plt.title('Гістограма кількості дітей (загальна)')
plt.xlabel('Кількість дітей')
plt.ylabel('Частота')
plt.show()

# Гістограма кількості дітей залежно від покупки велосипеда.
plt.figure(figsize=(10, 5))
sns.histplot(data, x='Children', hue='Purchased Bike', bins=range(0, data['Children'].max() + 2), kde=False)
plt.title('Гістограма кількості дітей в залежності від покупки велосипеда')
plt.xlabel('Кількість дітей')
plt.ylabel('Частота')
plt.show()

print("Розподіл кількості дітей:")
print(data['Children'].value_counts())
print("Розподіл кількості дітей в залежності від покупки велосипеда:")
print(data.groupby('Purchased Bike')['Children'].value_counts())

# 3.Побудувати діаграму розмаху доходу (загальну і залежно від рівня освіти), визначити чи присутні викиди.
# Діаграма розмаху доходу загальна
q1 = data['Income'].quantile(0.25)
q2 = data['Income'].median()
q3 = data['Income'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data['Income'] < lower_bound) | (data['Income'] > upper_bound)]

plt.figure(figsize=(10, 6))
sns.boxplot(y=data['Income'])
plt.title('Діаграма розмаху доходу (загальна)')
plt.ylabel('Дохід')
plt.show()

print(f"Q1 (25%): {q1:.2f}")
print(f"Q2 (Медіана): {q2:.2f}")
print(f"Q3 (75%): {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Нижня межа вуса: {lower_bound:.2f}")
print(f"Верхня межа вуса: {upper_bound:.2f}")
print(f"Кількість викидів: {len(outliers)}")
print("Викиди (доходи):\n", outliers['Income'].values)

# Діаграма розмаху доходу залежно від рівня освіти
plt.figure(figsize=(12, 6))
sns.boxplot(x='Education', y='Income', data=data)
plt.title('Діаграма розмаху доходу в залежності від освіти')
plt.xlabel('Освіта')
plt.ylabel('Дохід')
plt.show()

print("\nХарактеристики доходу за рівнями освіти:")
edu_groups = data.groupby('Education')
for edu, group in edu_groups:
    q1 = group['Income'].quantile(0.25)
    q2 = group['Income'].median()
    q3 = group['Income'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = group[(group['Income'] < lower) | (group['Income'] > upper)]

    print(f"\nОсвіта: {edu}")
    print(f"  Q1 (25%): {q1:.2f}")
    print(f"  Q2 (Медіана): {q2:.2f}")
    print(f"  Q3 (75%): {q3:.2f}")
    print(f"  IQR: {iqr:.2f}")
    print(f"  Нижня межа вуса: {lower:.2f}")
    print(f"  Верхня межа вуса: {upper:.2f}")
    print(f"  Кількість викидів: {len(outliers)}")
    print("  Викиди:", outliers['Income'].values)

# 4a. Діаграма розсіювання: доходи та вік
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Income', data=data)
plt.title('Залежність між доходами і віком')
plt.xlabel('Вік')
plt.ylabel('Дохід')
plt.show()

correlation_income_age = data[['Age', 'Income']].corr().iloc[0, 1]
print(f'Коефіцієнт Кореляції між віком та доходом: {correlation_income_age:.2f}')

# 4b. Діаграма розсіювання: кількість дітей і машин
plt.figure(figsize=(10, 6))
sns.stripplot(x='Children', y='Cars', data=data, jitter=True)
plt.title('Залежність між кількістю дітей і машин')
plt.xlabel('Кількість дітей')
plt.ylabel('Кількість машин')
plt.show()

correlation_children_cars = data[['Children', 'Cars']].corr().iloc[0, 1]
print(f'Коефіцієнт Кореляції між кількістю дітей та машин: {correlation_children_cars:.2f}')
