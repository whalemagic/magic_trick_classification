import pandas as pd
import tiktoken

def num_tokens_from_string(string: str) -> int:
    """Возвращает количество токенов в строке используя cl100k_base энкодер"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def calculate_cost(num_tokens: int, model: str) -> float:
    """Рассчитывает стоимость в USD на основе количества токенов и модели"""
    costs = {
        'claude-3-haiku-20240307': 0.25 / 1000000,  # $0.25/1M токенов
        'claude-3-sonnet-20240229': 3.0 / 1000000,  # $3.00/1M токенов
        'claude-3-opus-20240229': 15.0 / 1000000,   # $15.00/1M токенов
    }
    return num_tokens * costs[model]

def build_prompt(name: str, description: str) -> str:
    """Воссоздаем промпт из нашего основного скрипта"""
    return f"""You are a magic trick classifier. Analyze this magic product and provide a detailed classification.

Product Name: {name}
Description: {description}

Requirements:
1. Main magical effect MUST be chosen ONLY from this list: Production, Vanish, Transformation, Teleportation, Restoration, Multiplication, Penetration, Prediction, Mind Reading, Mentalism, Time Manipulation, Levitation, Animation, Anti-gravity, Impossible Object, Control / Forcing, Escape, Choice Revelation
2. Props should be primarily chosen from this list (but you can suggest others if they fit better): cards, coins, phone, paper, bills, magic wand, magic box, magic apparatus, gimmick, prop, device
3. Product type should be one of: physical product, digital download, video, book, magazine, kit

Return your analysis as a Python list containing:
1. effects: list of tuples (effect, probability, reason)
2. props: list of tuples (prop, probability, reason)
3. product_types: list of tuples (type, probability, reason)"""

# Загружаем данные
print("\n1. Подсчет исходного количества строк:")
df = pd.read_csv("data/input.csv")
print(f"Исходное количество строк: {len(df)}")

# Очищаем пустые описания
print("\n2. Очистка пустых описаний:")
df_cleaned = df[df["description"].notna() & (df["description"] != "")]
print(f"Количество строк после очистки пустых описаний: {len(df_cleaned)}")

# Проверяем дубликаты
print("\n3. Проверка дубликатов по URL:")
duplicates = df_cleaned[df_cleaned.duplicated(subset=['url'], keep=False)]
print(f"Найдено дубликатов: {len(duplicates)}")

# Оставляем только записи с самым длинным описанием для каждого URL
df_no_dupes = df_cleaned.sort_values('description').drop_duplicates(
    subset=['url'], 
    keep='last'
)
print(f"Количество строк после удаления дубликатов: {len(df_no_dupes)}")

# Подсчет токенов и стоимости
print("\n4. Оценка количества токенов и стоимости:")
total_tokens = 0
sample_size = min(100, len(df_no_dupes))  # Берем выборку для оценки
sample_df = df_no_dupes.sample(n=sample_size, random_state=42)

for _, row in sample_df.iterrows():
    prompt = build_prompt(row['name'], row['description'])
    tokens = num_tokens_from_string(prompt)
    total_tokens += tokens

# Экстраполируем на весь датасет
avg_tokens_per_row = total_tokens / sample_size
estimated_total_tokens = avg_tokens_per_row * len(df_no_dupes)

print(f"\nСредняя длина промпта в токенах: {avg_tokens_per_row:.1f}")
print(f"Оценочное общее количество токенов: {estimated_total_tokens:,.0f}")

# Расчет стоимости для разных моделей
models = ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229']
print("\nОценка стоимости классификации:")
for model in models:
    cost = calculate_cost(estimated_total_tokens, model)
    print(f"{model}: ${cost:,.2f}")

# Сохраняем очищенный датасет
df_no_dupes.to_csv("data/input_cleaned.csv", index=False)
print("\nОчищенный датасет сохранен в data/input_cleaned.csv") 