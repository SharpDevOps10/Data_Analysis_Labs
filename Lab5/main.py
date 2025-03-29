import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Завантаження даних
df = pd.read_csv("Delhi_Climate.csv")

# Перетворення дати у формат datetime і встановлення індексу
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Перетворення wind_speed у числовий формат
df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')


# --- ЗАВДАННЯ 1: Графіки вологості ---

def plot_humidity(data, title):
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data['humidity'])
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Вологість (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# а) Загальний графік вологості
plot_humidity(df, 'а) Загальна зміна вологості')

# б) За 2013 рік
plot_humidity(df.loc['2013'], 'б) Вологість за 2013 рік')

# в) За лютий 2016 року
plot_humidity(df.loc['2016-02'], 'в) Вологість за лютий 2016 року')

# г) Січень 2014 – Березень 2016
plot_humidity(df.loc['2014-01':'2016-03'], 'г) Вологість з січня 2014 до березня 2016')

# д) 2014 та 2015 на одному графіку
plt.figure(figsize=(12, 4))
plt.plot(df.loc['2014'].index, df.loc['2014']['humidity'], label='2014')
plt.plot(df.loc['2015'].index, df.loc['2015']['humidity'], label='2015')
plt.title('д) Вологість у 2014 та 2015 роках')
plt.xlabel('Дата')
plt.ylabel('Вологість (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- ЗАВДАННЯ 2: Аналіз швидкості вітру ---

# а) Середнє значення за 2015 рік
wind_2015_mean = df.loc['2015', 'wind_speed'].mean()
print(f"2а) Середня швидкість вітру за 2015 рік: {wind_2015_mean:.2f} м/с")

# б) Середнє значення по місяцях 2014 року
wind_2014_monthly = df.loc['2014'].resample('ME').mean()['wind_speed']
plt.figure(figsize=(10, 4))
sns.barplot(x=wind_2014_monthly.index.strftime('%b'), y=wind_2014_monthly.values)
plt.title('2б) Середня швидкість вітру за місяцями 2014 року')
plt.ylabel('Швидкість вітру')
plt.xlabel('Місяць')
plt.tight_layout()
plt.show()

print("\n2б) Середня швидкість вітру по місяцях 2014 року:")
for date, value in wind_2014_monthly.items():
    print(f"{date.strftime('%B')}: {value:.2f} м/с")

# в) Середня швидкість вітру по тижнях осені 2016 року
wind_autumn_2016 = df.loc['2016-09':'2016-11'].resample('W').mean()['wind_speed']
plt.figure(figsize=(10, 4))
plt.plot(wind_autumn_2016.index, wind_autumn_2016.values, marker='o')
plt.title('2в) Середня швидкість вітру за тижнями осені 2016 року')
plt.ylabel('Швидкість вітру')
plt.xlabel('Дата')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n2в) Середня швидкість вітру за тижнями осені 2016 року:")
for date, value in wind_autumn_2016.items():
    print(f"{date.strftime('%Y-%m-%d')}: {value:.2f} м/с")

# г) Відсоткові зміни за весну 2013 року
wind_spring_2013_pct = df.loc['2013-03':'2013-05']['wind_speed'].pct_change() * 100
plt.figure(figsize=(12, 4))
plt.plot(wind_spring_2013_pct.index, wind_spring_2013_pct.values)
plt.title('2г) Зміни швидкості вітру у % за весну 2013 року')
plt.ylabel('Зміна (%)')
plt.xlabel('Дата')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n2г) Відсоткові зміни швидкості вітру за весну 2013 року:")
for date, change in wind_spring_2013_pct.dropna().items():
    print(f"{date.date()}: {change:+.2f}%")

# д) Ковзне середнє (вікно 4 дні) за лютий 2015 року
wind_feb_2015 = df.loc['2015-02']['wind_speed']
wind_feb_2015_rolling = wind_feb_2015.rolling(window=4).mean()

plt.figure(figsize=(10, 4))
plt.plot(wind_feb_2015.index, wind_feb_2015, label='Швидкість вітру')
plt.plot(wind_feb_2015_rolling.index, wind_feb_2015_rolling, label='Ковзне середнє (4 дні)', linewidth=2)
plt.title('2д) Ковзне середнє швидкості вітру (лютий 2015)')
plt.xlabel('Дата')
plt.ylabel('Швидкість вітру')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n2д) Ковзне середнє швидкості вітру за лютий 2015 року:")
for date, avg in wind_feb_2015_rolling.dropna().items():
    print(f"{date.date()}: {avg:.2f} м/с")
