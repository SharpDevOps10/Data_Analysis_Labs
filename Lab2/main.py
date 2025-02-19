import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

data = pd.read_csv('Birthweight.csv')

# 1. Знайти середній зріст дітей і його медіану
mean_length = data['Length'].mean()
median_length = data['Length'].median()
print(f'Середній зріст дітей: {mean_length:.2f} см')
print(f'Медіана зросту дітей: {median_length:.2f} см')

# 2. Перевірити чи нормально розподілена кількість сигарет в день
shapiro_stat, shapiro_p_value = stats.shapiro(data['mnocig'])
print(f'Статистика тесту Шапіро-Уілка: {shapiro_stat:.4f}, p-значення: {shapiro_p_value:.4e}')
if shapiro_p_value < 0.05:
    print('Кількість сигарет в день не розподілена нормально.')
else:
    print('Кількість сигарет в день розподілена нормально.')

# 3. Перевірити чи у матерів, що старші 35, легші діти
birthweight_above_35 = data[data['mage35'] == 1]['Birthweight']
birthweight_below_35 = data[data['mage35'] == 0]['Birthweight']
ttest_stat, ttest_p_value = stats.ttest_ind(birthweight_above_35, birthweight_below_35, alternative='less')
print(f'Статистика t-тесту: {ttest_stat:.4f}, p-значення: {ttest_p_value:.4f}')
if ttest_p_value < 0.05:
    print('У матерів старших за 35 років діти легші.')
else:
    print('Немає статистично значущої різниці у вазі дітей матерів старших за 35 років.')

# 4. Чи є зв’язок між тривалістю вагітності та вагою дитини
gestation_stat, gestation_p = stats.shapiro(data['Gestation'])
birthweight_stat, birthweight_p = stats.shapiro(data['Birthweight'])

print(f'Перевірка нормальності тривалості вагітності: p-значення = {gestation_p:.4e}')
print(f'Перевірка нормальності ваги дитини: p-значення = {birthweight_p:.4e}')

if gestation_p >= 0.05 and birthweight_p >= 0.05:
    correlation, p_value = stats.pearsonr(data['Gestation'], data['Birthweight'])
    print(f'Коефіцієнт кореляції Пірсона: {correlation:.4f}, p-значення: {p_value:.4e}')
else:
    correlation, p_value = stats.spearmanr(data['Gestation'], data['Birthweight'])
    print(f'Коефіцієнт кореляції Спірмена: {correlation:.4f}, p-значення: {p_value:.4e}')

if p_value < 0.05:
    print('Є статистично значущий зв’язок між тривалістю вагітності та вагою дитини.')
else:
    print('Немає статистично значущого зв’язку між тривалістю вагітності та вагою дитини.')

plt.scatter(data['Gestation'], data['Birthweight'])
plt.xlabel('Тривалість вагітності (тижні)')
plt.ylabel('Вага дитини при народженні (кг)')
plt.title('Зв’язок між тривалістю вагітності та вагою дитини')
plt.grid(True)
plt.show()
