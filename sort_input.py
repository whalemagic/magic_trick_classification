import pandas as pd
import re

# Читаем исходный файл
df = pd.read_csv('data/input_cleaned.csv')

# Функция для извлечения номера страницы из URL
def extract_page_number(url):
    match = re.search(r'/p/(\d+)', url)
    return int(match.group(1)) if match else 0

# Создаем колонку с номером страницы
df['page'] = df['url'].apply(extract_page_number)

# Сортируем по номеру страницы в порядке возрастания
df_sorted = df.sort_values('page')

# Сохраняем отсортированный файл
df_sorted.to_csv('data/input_cleaned_sorted.csv', index=False)

print(f'Файл отсортирован. Всего записей: {len(df_sorted)}')
print(f'Диапазон номеров страниц: от {df_sorted["page"].min()} до {df_sorted["page"].max()}') 