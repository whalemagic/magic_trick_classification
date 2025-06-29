import pandas as pd
import sys

# Получаем путь к файлу из аргументов командной строки
file_path = sys.argv[1] if len(sys.argv) > 1 else 'output/claude_output.csv'

# Читаем CSV файл
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f'Количество записей в CSV: {len(df)}')
    print('\nПервые несколько строк:')
    print(df.head())
    print('\nСписок колонок:')
    print(df.columns.tolist())
except Exception as e:
    print(f'Ошибка при чтении файла {file_path}:')
    print(e) 