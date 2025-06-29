import pandas as pd
import os

# Находим последний сохраненный файл
files = [f for f in os.listdir('output') if f.startswith('claude_output_part_') and f.endswith('.csv')]
if not files:
    print("Не найдены файлы с промежуточными результатами")
    exit(1)

# Сортируем по номеру в имени файла
latest_file = sorted(files, key=lambda x: int(x.split('_part_')[1].replace('.csv', '')))[-1]
print(f"Конвертируем файл: {latest_file}")

# Читаем CSV
df = pd.read_csv(f'output/{latest_file}')
print(f"Загружено записей: {len(df)}")

# Создаем имя для Excel файла
excel_file = f'output/{latest_file.replace(".csv", ".xlsx")}'

# Сохраняем в Excel
df.to_excel(excel_file, index=False)
print(f"Файл сохранен как: {excel_file}")

# Выводим список колонок для проверки
print("\nСписок колонок в файле:")
for col in df.columns:
    print(f"- {col}") 