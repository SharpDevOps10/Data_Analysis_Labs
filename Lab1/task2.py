import pandas as pd


def analyze_stars(input_file, output_file):
    df = pd.read_csv(input_file)

    quantitative_columns = ["Temperature", "L", "R", "A_M"]

    print("Статистичні характеристики:")
    print(df[quantitative_columns].describe())

    main_sequence_stars = df[(df["Type"] == 3) & (df["A_M"] < 0)]
    print("\nЗірки головної послідовності з від’ємною абсолютною величиною:")
    print(main_sequence_stars)

    df["Absolute_Radius_km"] = df["R"] * 696340
    print("\nОновлена таблиця з абсолютним радіусом:")
    print(df.head())

    df.to_csv(output_file, index=False)
    print(f"Нова таблиця збережена у файл: {output_file}")

    return df


file_path = "Stars.csv"
output_path = "Stars_with_Absolute_Radius.csv"
analyze_stars(file_path, output_path)
