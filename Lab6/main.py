import pandas as pd

# 1. Читання HTML-файлу
file_path = "Version 6.html"
tables = pd.read_html(file_path)
df = tables[0]
df = df.iloc[:, 1:]

# 2. Зміна назв стовпців
df.columns = ['Дата', 'Середня температура (°C)', 'Вологість (%)', 'Швидкість вітру (км/год)', 'Атмосферний тиск (гПа)']

# 3. Попередня обробка даних
print("Інформація про дані до обробки:")
print(df.info())

# 3.1 Пошук відсутніх значень
print("\nКількість пропущених значень у кожному стовпці:")
print(df.isna().sum())

# 3.2 Видалення повністю порожніх рядків
df = df.dropna(how='all')

# 3.3 Заповнення пропущених значень середнім по стовпцю
df_filled = df.fillna(df.mean(numeric_only=True))

# 3.4 Перевірка дублікатів
duplicates = df_filled.duplicated().sum()
print(f"\nКількість дублікатів: {duplicates}")

# 3.5 Видалення дублікатів
df_cleaned = df_filled.drop_duplicates()

# 3.6 Перетворення типу "Дата" у формат datetime
df_cleaned.loc[:, 'Дата'] = pd.to_datetime(df_cleaned['Дата'], errors='coerce')

# Підсумкова інформація
print("\nІнформація про дані після обробки:")
print(df_cleaned.info())

# Збереження очищеного датасету
df_cleaned.to_csv("cleaned_weather_data.csv", index=False)
df_cleaned.to_html("cleaned_weather_data.html", index=False)

print("\nФайли 'cleaned_weather_data.csv' та 'cleaned_weather_data.html' збережено.")
