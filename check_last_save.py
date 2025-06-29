import pandas as pd

# Читаем последний сохраненный файл
df = pd.read_csv('output/claude_output_part_30.csv')

# Выводим информацию
print(f'Количество записей в последнем сохранении: {len(df)}')
print('\nПоследние 5 строк:')
print(df.tail()) 