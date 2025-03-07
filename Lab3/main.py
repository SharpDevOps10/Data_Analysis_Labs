import pandas as pd

file_path = 'penguins.csv'
data = pd.read_csv(file_path)

# 1.1 Вивести інформацію про набір даних, типи ознак.
print('1.1. Інформація про набір даних:')
data.info()

# 1.2 Які ознаки є категоріальними, а які – кількісними?
categorical_features = data.select_dtypes(include='object').columns.tolist()
numerical_features = data.select_dtypes(include='number').columns.tolist()
print('1.2. Категоріальні ознаки: ', categorical_features)
print('1.2. Числові ознаки: ', numerical_features)

# 2. a) зберегти назви стовпців у окрему змінну і вивести її;
columns_names = data.columns.tolist()
print('2a. Назви стовпців: ', columns_names)

# 2. б) вивести кількість пінгвінів на кожному острові;
penguins_count_per_island = data['island'].value_counts()
print('2б. Кількість пінгвінів на кожному острові: ')
print(penguins_count_per_island)

# 2. в) вивести дані випадкового самця Аделі з масою понад 3 кг;
adelie_penguin_over_3kg = data[(data['species'] == 'Adelie') & (data['sex'] == 'MALE') & (data['body_mass_g'] > 3000)]
random_adelie_penguin = adelie_penguin_over_3kg.sample(n=1)
print("2в. Дані випадкового самця Аделі з масою понад 3 кг:")
if random_adelie_penguin is not None:
    print(random_adelie_penguin.to_string(index=False))
else:
    print("Немає відповідних записів.")

# 2. г) додати новий рядок до DataFrame з довільними даними;
new_data = pd.DataFrame([{'species': 'Adelie', 'island': 'Dream', 'culmen_length_mm': 40.0,
                          'culmen_depth_mm': 18.0, 'flipper_length_mm': 190,
                          'body_mass_g': 3500, 'sex': 'FEMALE'}])

data = pd.concat([data, new_data], ignore_index=True)

print("2г. Додано новий рядок:")
print(data.tail(1).to_string(index=False))

# 3. Робота із групованими даними.

# a) знайти максимальну довжину дзьобу для кожного виду;
max_culmen_length_per_species = data.groupby('species')['culmen_length_mm'].max()
print("3a. Максимальна довжина дзьобу для кожного виду:")
print(max_culmen_length_per_species)

# б) додати новий стовпець, який містить середню вагу пінгвінів даного виду;
data['mean_body_mass'] = data.groupby('species')['body_mass_g'].transform('mean')
print("3б. Додано новий стовпець із середньою вагою:")
print(data[['species', 'mean_body_mass']].drop_duplicates())

# в) вивести дані пінгвінів лише тих видів, для яких середня маса менша за 4000.
filtered_df = data[data['mean_body_mass'] < 4000]
print("3в. Пінгвіни видів, де середня маса менша за 4000:")
print(filtered_df)

# 4.1. За допомогою pivot_table створити нову таблицю, що буде містити середню довжину і глибину дзьобу пінгвінів різних видів та різних статей.
pivot_table = data.pivot_table(values=['culmen_length_mm', 'culmen_depth_mm'], index='species', columns='sex',
                               aggfunc='mean')
print("4.1. Pivot table:")
print(pivot_table)

# Збереження середньої довжини дзьобу самок Аделі
adelie_female_culmen_length = pivot_table.loc['Adelie', ('culmen_length_mm', 'FEMALE')]
print("4.2. Середня довжина дзьобу самок Аделі:", adelie_female_culmen_length)

pivot_table_reset = pivot_table.reset_index()
pivot_table_reset.columns = ['species'] + [f"{col[0]}_{col[1]}" for col in pivot_table_reset.columns[1:]]
pivot_table_reset.to_csv('pivot_table.csv', index=False, header=True)
print("Pivot table збережено у файл pivot_table.csv")
