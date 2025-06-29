import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ZeroShotClassificationPipeline
from hashlib import md5
import pickle
import json
import argparse
import os
import logging
from typing import Dict, List, Optional
from collections import defaultdict
import logging.handlers
from magic_classifier import (
    tag_csv_with_progress,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_BATCH_SIZE,
    DEFAULT_RULES_PRIORITY
)
import time
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('classification.log', encoding='utf-8', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Добавляем обработчик для отладки
debug_handler = logging.FileHandler('debug.log', encoding='utf-8', mode='w')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

# Переводы меток
LABEL_TRANSLATIONS = {
    # Эффекты
    "mentalism": "ментализм",
    "prediction": "предсказание",
    "mind reading": "чтение мыслей",
    "mental revelation": "ментальное откровение",
    "vanish": "исчезновение",
    "appearance": "появление",
    "transformation": "трансформация",
    "restoration": "восстановление",
    "levitation": "левитация",
    "penetration": "проникновение",
    "transposition": "транспозиция",
    "teleportation": "телепортация",
    "card revelation": "откровение карты",
    "card control": "контроль карты",
    "card location": "локация карты",
    "coin magic": "фокусы с монетами",
    "coin vanish": "исчезновение монеты",
    "coin production": "появление монеты",
    
    # Реквизит
    "cards": "карты",
    "coins": "монеты",
    "phone": "телефон",
    "paper": "бумага",
    "bills": "купюры",
    "magic wand": "волшебная палочка",
    "magic box": "волшебная коробка",
    "magic apparatus": "магический аппарат",
    "gimmick": "гиммик",
    "prop": "реквизит",
    "device": "устройство",
    
    # Масштаб и стиль
    "close-up magic": "микромагия",
    "parlour magic": "салонная магия",
    "stage magic": "сценическая магия",
    "comedy magic": "комедийная магия",
    "serious magic": "серьёзная магия",
    "interactive magic": "интерактивная магия",
    
    # Уровень сложности
    "beginner": "начинающий",
    "intermediate": "средний",
    "advanced": "продвинутый",
    "professional": "профессиональный",
    "easy to learn": "легко учится",
    "requires practice": "требует практики",
    "requires skill": "требует навыка",
    
    # Тип товара
    "physical product": "физический товар",
    "digital download": "цифровая загрузка",
    "video": "видео",
    "book": "книга",
    "magazine": "журнал",
    "kit": "набор"
}

# 1. Основные эффекты
effects_eng = [
    "mentalism", "prediction", "mind reading", "mental revelation",
    "vanish", "appearance", "transformation", "restoration",
    "levitation", "penetration", "transposition", "teleportation",
    "card revelation", "card control", "card location",
    "coin magic", "coin vanish", "coin production"
]

# 2. Реквизит
props_eng = [
    "cards", "coins", "phone", "paper", "bills",
    "magic wand", "magic box", "magic apparatus",
    "gimmick", "prop", "device"
]

# 3. Масштаб и стиль
scale_eng = [
    "close-up magic", "parlour magic", "stage magic",
    "comedy magic", "serious magic", "interactive magic"
]

# 4. Уровень сложности
difficulty_eng = [
    "beginner", "intermediate", "advanced", "professional",
    "easy to learn", "requires practice", "requires skill"
]

# 5. Тип товара
product_type_eng = [
    "physical product", "digital download", "video",
    "book", "magazine", "kit"
]

def rule_based_tags(name: str, description: str) -> Dict[str, List[str]]:
    """Применяет правила для определения тегов на основе имени и описания."""
    logger.debug(f"Применение правил для: {name}")
    tags = defaultdict(list)
    
    # Правила для типа товара
    if any(x in name.lower() for x in ["instant download", "digital download", "download"]):
        tags["product_type"].extend(["digital download", "video"])
        logger.debug(f"Найден тип товара: digital download, video")
    if any(x in name.lower() for x in ["dvd", "video", "tutorial", "streaming"]):
        tags["product_type"].append("video")
    if any(x in name.lower() for x in ["book", "manual", "guide", "pdf", "ebook"]):
        tags["product_type"].append("book")
    if any(x in name.lower() for x in ["magazine", "journal", "periodical"]):
        tags["product_type"].append("magazine")
    if any(x in name.lower() for x in ["kit", "set", "bundle", "collection"]):
        tags["product_type"].append("kit")
    if not tags["product_type"]:
        tags["product_type"].append("physical product")
    
    # Правила для сложности
    if any(x in name.lower() + description.lower() for x in ["beginner", "easy", "basic", "starter", "novice"]):
        tags["difficulty"].extend(["beginner", "easy to learn"])
    if any(x in name.lower() + description.lower() for x in ["intermediate", "medium", "moderate"]):
        tags["difficulty"].append("intermediate")
    if any(x in name.lower() + description.lower() for x in ["advanced", "expert", "complex", "difficult"]):
        tags["difficulty"].append("advanced")
    if any(x in name.lower() + description.lower() for x in ["professional", "pro", "master"]):
        tags["difficulty"].append("professional")
    
    # Правила для реквизита
    if any(x in name.lower() + description.lower() for x in ["card", "deck", "playing cards", "bicycle"]):
        tags["props"].append("cards")
    if any(x in name.lower() + description.lower() for x in ["coin", "coins", "dollar", "quarter"]):
        tags["props"].append("coins")
    if any(x in name.lower() + description.lower() for x in ["phone", "smartphone", "mobile", "iphone", "android"]):
        tags["props"].append("phone")
    if any(x in name.lower() + description.lower() for x in ["paper", "bill", "note", "money"]):
        tags["props"].append("paper")
    if any(x in name.lower() + description.lower() for x in ["wand", "stick", "magic wand"]):
        tags["props"].append("magic wand")
    if any(x in name.lower() + description.lower() for x in ["box", "case", "container"]):
        tags["props"].append("magic box")
    if any(x in name.lower() + description.lower() for x in ["gimmick", "gimmicked", "special prop"]):
        tags["props"].append("gimmick")
    if any(x in name.lower() + description.lower() for x in ["apparatus", "device", "equipment", "machine"]):
        tags["props"].append("magic apparatus")
    
    # Правила для масштаба
    if any(x in name.lower() + description.lower() for x in ["close-up", "closeup", "table", "intimate"]):
        tags["scale"].append("close-up magic")
    if any(x in name.lower() + description.lower() for x in ["parlour", "parlor", "salon", "living room"]):
        tags["scale"].append("parlour magic")
    if any(x in name.lower() + description.lower() for x in ["stage", "show", "performance", "theater"]):
        tags["scale"].append("stage magic")
    if any(x in name.lower() + description.lower() for x in ["comedy", "funny", "humor", "entertaining"]):
        tags["scale"].append("comedy magic")
    if any(x in name.lower() + description.lower() for x in ["serious", "professional", "dramatic"]):
        tags["scale"].append("serious magic")
    if any(x in name.lower() + description.lower() for x in ["interactive", "audience", "participation", "spectator"]):
        tags["scale"].append("interactive magic")
    
    # Правила для эффектов
    if any(x in name.lower() + description.lower() for x in ["mental", "mind", "psychic", "psychological"]):
        tags["effect"].append("mentalism")
    if any(x in name.lower() + description.lower() for x in ["predict", "prediction", "forecast", "foretell"]):
        tags["effect"].append("prediction")
    if any(x in name.lower() + description.lower() for x in ["mind reading", "telepathy", "thought"]):
        tags["effect"].append("mind reading")
    if any(x in name.lower() + description.lower() for x in ["vanish", "disappear", "vanishing", "invisible"]):
        tags["effect"].append("vanish")
    if any(x in name.lower() + description.lower() for x in ["appear", "production", "produce", "materialize"]):
        tags["effect"].append("appearance")
    if any(x in name.lower() + description.lower() for x in ["transform", "change", "morph", "transmutation"]):
        tags["effect"].append("transformation")
    if any(x in name.lower() + description.lower() for x in ["restore", "restoration", "repair", "fix"]):
        tags["effect"].append("restoration")
    if any(x in name.lower() + description.lower() for x in ["levitate", "levitation", "float", "suspend"]):
        tags["effect"].append("levitation")
    if any(x in name.lower() + description.lower() for x in ["penetrate", "penetration", "through", "pass through"]):
        tags["effect"].append("penetration")
    if any(x in name.lower() + description.lower() for x in ["transpose", "transposition", "switch", "exchange"]):
        tags["effect"].append("transposition")
    
    return dict(tags)

def get_args():
    parser = argparse.ArgumentParser(description='Классификация CSV файла с тегами')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_PATH, help='Путь к входному CSV файлу')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH, help='Путь к выходному CSV файлу')
    parser.add_argument('--limit', type=int, default=30, help='Ограничение количества строк для обработки')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Порог вероятности для тегов')
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K, help='Количество тегов для каждой категории')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча для обработки')
    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE_PATH, help='Путь к файлу кэша')
    parser.add_argument('--force', action='store_true', help='Игнорировать существующий кэш')
    parser.add_argument('--rules_priority', type=str, default=DEFAULT_RULES_PRIORITY, 
                      choices=['append', 'prefer_rules', 'prefer_model'],
                      help='Как обрабатывать rule-based теги vs предсказания модели')
    return parser.parse_args()

def limit_tags_json(tag_scores: dict, max_tags: int = 10) -> dict:
    """Ограничивает количество тегов в tags_json."""
    all_scores = []
    for cat_scores in tag_scores.values():
        all_scores.extend(cat_scores.items())
    
    return dict(sorted(all_scores, key=lambda x: -x[1])[:max_tags])

def translate_tags(tags: List[str]) -> List[str]:
    """Переводит теги на русский язык."""
    return [LABEL_TRANSLATIONS.get(tag, tag) for tag in tags]

def save_partial_results(df: pd.DataFrame, output_path: str, row_count: int):
    """Сохраняет промежуточные результаты."""
    base, ext = os.path.splitext(output_path)
    partial_path = f"{base}.autosave_{row_count}{ext}"
    df.to_csv(partial_path, index=False)
    logging.info(f"Сохранены промежуточные результаты: {partial_path}")

def generate_report(stats: dict) -> str:
    """Генерирует отчет о классификации."""
    report = []
    report.append("=== Отчет о классификации ===\n")
    
    # Общая статистика
    report.append("Общая статистика:")
    report.append(f"- Всего обработано товаров: {stats.get('total_items', 0)}")
    report.append(f"- Успешно классифицировано: {stats.get('successful_items', 0)}")
    report.append(f"- Использовано кэширование: {stats.get('cached_items', 0)} раз")
    report.append(f"- Среднее время на товар: {stats.get('avg_time_per_item', 0):.2f} сек")
    
    # Статистика по категориям
    if 'category_stats' in stats:
        report.append("\nСтатистика по категориям:")
        for category, count in stats['category_stats'].items():
            report.append(f"- {LABEL_TRANSLATIONS.get(category, category)}: {count} тегов")
    
    # Топ тегов
    if 'top_tags' in stats:
        report.append("\nСамые частые теги:")
        for tag, count in sorted(stats['top_tags'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"- {LABEL_TRANSLATIONS.get(tag, tag)}: {count} раз")
    
    # Статистика по источникам тегов
    if 'tag_sources' in stats:
        report.append("\nИсточники тегов:")
        report.append(f"- Правила: {stats['tag_sources'].get('rules', 0)}")
        report.append(f"- Модель: {stats['tag_sources'].get('model', 0)}")
        report.append(f"- Комбинированные: {stats['tag_sources'].get('combined', 0)}")
    
    # Предупреждения и ошибки
    if stats.get('warnings', []):
        report.append("\nПредупреждения:")
        for warning in stats['warnings']:
            report.append(f"- {warning}")
    
    if stats.get('errors', []):
        report.append("\nОшибки:")
        for error in stats['errors']:
            report.append(f"- {error}")
    
    # Рекомендации
    report.append("\nРекомендации:")
    if stats.get('low_confidence_items', 0) > 0:
        report.append("- Обнаружены товары с низкой уверенностью классификации")
        report.append(f"  Количество: {stats['low_confidence_items']}")
        report.append("  Рекомендуется проверить эти товары вручную")
    
    if stats.get('missing_categories', []):
        report.append("- Следующие категории редко встречаются в результатах:")
        for category in stats['missing_categories']:
            report.append(f"  * {LABEL_TRANSLATIONS.get(category, category)}")
        report.append("  Возможно, стоит добавить больше правил для этих категорий")
    
    return "\n".join(report)

def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

def batch_infer(classifier, descs, labels):
    # descs: список описаний, labels: список меток
    results = classifier(descs, labels, multi_label=True)
    # results — список словарей (или один словарь, если 1 описание)
    if isinstance(results, dict):
        results = [results]
    return results

def collect_statistics(df: pd.DataFrame, cache: dict, start_time: float) -> dict:
    """Собирает статистику для отчета."""
    stats = {
        'total_items': len(df),
        'successful_items': 0,
        'cached_items': 0,
        'avg_time_per_item': 0,
        'category_stats': defaultdict(int),
        'top_tags': defaultdict(int),
        'tag_sources': {'rules': 0, 'model': 0, 'combined': 0},
        'warnings': [],
        'errors': [],
        'low_confidence_items': 0,
        'missing_categories': []
    }
    
    # Подсчет успешных классификаций и кэшированных элементов
    for _, row in df.iterrows():
        desc = str(row.get("description", "")).strip()
        if not desc or desc.lower() == 'nan':
            continue
        h = md5(desc.encode()).hexdigest()
        if h in cache:
            stats['cached_items'] += 1
            cache_data = cache[h]
            
            # Подсчет тегов по категориям
            for category, tags in cache_data.items():
                stats['category_stats'][category] += len(tags)
                for tag, confidence in tags.items():
                    stats['top_tags'][tag] += 1
                    if confidence < 0.3:  # Порог низкой уверенности
                        stats['low_confidence_items'] += 1
            
            # Определение источника тегов
            for category, tags in cache_data.items():
                if all(score == 1.0 for score in tags.values()):
                    stats['tag_sources']['rules'] += 1
                elif all(score < 1.0 for score in tags.values()):
                    stats['tag_sources']['model'] += 1
                else:
                    stats['tag_sources']['combined'] += 1
            
            stats['successful_items'] += 1
    
    # Проверка на отсутствующие категории
    all_categories = set(['effect', 'props', 'scale', 'difficulty', 'product_type'])
    found_categories = set(stats['category_stats'].keys())
    stats['missing_categories'] = list(all_categories - found_categories)
    
    # Среднее время на элемент
    if start_time:
        total_time = time.time() - start_time
        stats['avg_time_per_item'] = total_time / stats['total_items'] if stats['total_items'] > 0 else 0
    
    return stats

def validate_classification_results(df: pd.DataFrame, cache: dict, threshold: float = 0.3) -> List[str]:
    """Проверяет результаты классификации на наличие проблем."""
    warnings = []
    
    # Проверка пустых описаний
    empty_desc_count = df['description'].isna().sum() + (df['description'] == '').sum()
    if empty_desc_count > 0:
        warnings.append(f"Найдено {empty_desc_count} записей с пустыми описаниями")
    
    # Проверка дубликатов
    duplicate_desc = df[df['description'].duplicated()]['description'].count()
    if duplicate_desc > 0:
        warnings.append(f"Найдено {duplicate_desc} дубликатов описаний")
    
    # Проверка результатов классификации
    for _, row in df.iterrows():
        desc = str(row.get("description", "")).strip()
        if not desc or desc.lower() == 'nan':
            continue
            
        h = md5(desc.encode()).hexdigest()
        if h in cache:
            cache_data = cache[h]
            
            # Проверка на пустые категории
            empty_categories = [cat for cat, tags in cache_data.items() if not tags]
            if empty_categories:
                warnings.append(f"Запись '{row.get('name', 'Unknown')}' не имеет тегов в категориях: {', '.join(empty_categories)}")
            
            # Проверка на низкую уверенность
            low_confidence_tags = []
            for cat, tags in cache_data.items():
                low_conf = [tag for tag, conf in tags.items() if conf < threshold]
                if low_conf:
                    low_confidence_tags.extend(low_conf)
            
            if low_confidence_tags:
                warnings.append(f"Низкая уверенность ({threshold}) для тегов {', '.join(low_confidence_tags)} в записи '{row.get('name', 'Unknown')}'")
    
    return warnings

def export_results(df: pd.DataFrame, results: List[dict], all_json: List[str], base_output_path: str):
    """Экспортирует результаты в различных форматах."""
    # CSV с основными результатами
    result_df = pd.DataFrame(results)
    result_df['tags_json'] = all_json
    result_df.to_csv(base_output_path, index=False)
    logger.info(f"Основные результаты сохранены в {base_output_path}")
    
    # JSON с детальной информацией
    base, _ = os.path.splitext(base_output_path)
    json_path = f"{base}_detailed.json"
    detailed_results = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        result = {
            'name': row.get('name', ''),
            'description': row.get('description', ''),
            'tags': json.loads(all_json[i]) if i < len(all_json) else {},
            'metadata': {
                'source': row.get('source', ''),
                'url': row.get('url', ''),
                'date_added': row.get('date_added', '')
            }
        }
        detailed_results.append(result)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Детальные результаты сохранены в {json_path}")
    
    # Excel с вкладками для каждой категории
    excel_path = f"{base}_categories.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        # Основной лист
        result_df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Листы по категориям
        categories = ['effect', 'props', 'scale', 'difficulty', 'product_type']
        for cat in categories:
            cat_data = []
            for i, row in df.iterrows():
                if i >= len(all_json):
                    continue
                tags = json.loads(all_json[i])
                if cat in tags:
                    cat_tags = tags[cat]
                    if cat_tags:  # Если есть теги в этой категории
                        cat_data.append({
                            'name': row.get('name', ''),
                            'description': row.get('description', ''),
                            'tags': ', '.join([f"{tag} ({conf:.2f})" for tag, conf in cat_tags.items()]),
                            'tags_ru': ', '.join([f"{LABEL_TRANSLATIONS.get(tag, tag)} ({conf:.2f})" 
                                                for tag, conf in cat_tags.items()])
                        })
            
            if cat_data:  # Если есть данные для категории
                cat_df = pd.DataFrame(cat_data)
                cat_df.to_excel(writer, sheet_name=cat, index=False)
    
    logger.info(f"Результаты по категориям сохранены в {excel_path}")
    
    # Текстовый отчет с статистикой
    txt_path = f"{base}_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        # Общая статистика
        f.write("=== Статистика классификации ===\n\n")
        f.write(f"Всего обработано: {len(df)} записей\n")
        f.write(f"Уникальных описаний: {df['description'].nunique()}\n\n")
        
        # Статистика по категориям
        f.write("Распределение тегов по категориям:\n")
        for cat in categories:
            cat_count = sum(1 for j in all_json if cat in json.loads(j))
            f.write(f"{cat}: {cat_count} записей\n")
        
        # Топ тегов
        f.write("\nСамые частые теги:\n")
        tag_counts = defaultdict(int)
        for j in all_json:
            tags_dict = json.loads(j)
            for cat_tags in tags_dict.values():
                for tag in cat_tags:
                    tag_counts[tag] += 1
        
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{tag} ({LABEL_TRANSLATIONS.get(tag, tag)}): {count}\n")
    
    logger.info(f"Текстовый отчет сохранен в {txt_path}")

def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает и предобрабатывает данные перед классификацией."""
    logger.info("Начало очистки и предобработки данных...")
    
    # Создаем копию DataFrame
    df_clean = df.copy()
    
    # Очистка пустых значений
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['description'])
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.warning(f"Удалено {dropped_rows} строк с пустыми описаниями")
    
    # Очистка текстовых полей
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Приводим к строке
        text = str(text)
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        # Удаляем HTML-теги
        text = re.sub(r'<[^>]+>', '', text)
        # Удаляем специальные символы
        text = re.sub(r'[^\w\s\-.,!?]', ' ', text)
        # Удаляем повторяющиеся знаки препинания
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        # Удаляем лишние пробелы после очистки
        text = ' '.join(text.split())
        return text
    
    df_clean['description'] = df_clean['description'].apply(clean_text)
    df_clean['name'] = df_clean['name'].apply(clean_text)
    
    # Удаление дубликатов
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['description'])
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.warning(f"Удалено {dropped_rows} дубликатов описаний")
    
    # Фильтрация по длине описания
    min_desc_length = 10  # Минимальная длина описания в символах
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['description'].str.len() >= min_desc_length]
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.warning(f"Удалено {dropped_rows} строк с слишком короткими описаниями (< {min_desc_length} символов)")
    
    # Добавление дополнительных признаков
    df_clean['description_length'] = df_clean['description'].str.len()
    df_clean['name_length'] = df_clean['name'].str.len()
    df_clean['word_count'] = df_clean['description'].str.split().str.len()
    
    # Сортировка по длине описания (более длинные описания обычно содержат больше информации)
    df_clean = df_clean.sort_values('description_length', ascending=False)
    
    logger.info(f"Предобработка завершена. Итоговое количество записей: {len(df_clean)}")
    
    return df_clean

def tag_csv_with_progress(input_file, output_file, limit=None, threshold=0.3, top_k=3, batch_size=8, cache_file='cache.pkl', force=False, rules_priority='append'):
    try:
        start_time = time.time()
        logger.info(f"Параметр limit: {limit}")
        
        # Загружаем и предобрабатываем данные
        df = pd.read_csv(input_file)
        logger.info(f"Загружен файл {input_file}, исходных строк: {len(df)}")
        df = clean_and_preprocess_data(df)
        
        if limit:
            df = df.head(limit)
            logger.info(f"Применено ограничение: {limit} строк")
        
        # Загружаем кэш, если он существует и не используется флаг force
        cache = {}
        if os.path.exists(cache_file) and not force:
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logging.info(f"Загружен кеш из {cache_file}, записей: {len(cache)}")
            except Exception as e:
                logging.error(f"Ошибка при загрузке кеша: {str(e)}")
                cache = {}

        logger.info("Начинаем обработку файла...")
        logger.info("Загрузка модели и токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        classifier = ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, batch_size=batch_size)
        logger.info("Модель и токенизатор успешно загружены")

        categories = [
            ("effect", effects_eng),
            ("props", props_eng),
            ("scale", scale_eng),
            ("difficulty", difficulty_eng),
            ("product_type", product_type_eng)
        ]

        results = []
        all_json = []
        progress = tqdm(total=len(df), desc="Classifying", dynamic_ncols=True)

        # Список индексов, которые нужно обработать (нет в кеше)
        to_process = []
        for i, row in df.iterrows():
            if limit and i >= limit:
                break
            desc = str(row.get("description", "")).strip()
            name = str(row.get("name", "")).strip()
            if not desc or desc.lower() == 'nan':
                logger.debug(f"Пропущена строка {i}: пустое описание")
                continue
            h = md5(desc.encode()).hexdigest()
            if h not in cache:
                to_process.append((i, desc, name))
        
        logger.info(f"Найдено {len(to_process)} строк для обработки (не в кеше)")

        # Батчинг описаний
        for start in range(0, len(to_process), batch_size):
            batch = to_process[start:start+batch_size]
            logger.debug(f"Обработка батча {start//batch_size + 1}, размер: {len(batch)}")
            
            batch_indices = [i for i, _, _ in batch]
            batch_descs = [desc for _, desc, _ in batch]
            batch_names = [name for _, _, name in batch]
            batch_results = {}
            
            # Получаем rule-based теги
            rule_tags = [rule_based_tags(name, desc) for name, desc in zip(batch_names, batch_descs)]
            logger.debug(f"Получены rule-based теги для батча: {rule_tags}")
            
            for cat, labels in categories:
                cat_results = batch_infer(classifier, batch_descs, labels)
                for idx, res, rule_tag in zip(batch_indices, cat_results, rule_tags):
                    desc_val = str(df.loc[idx, "description"]).strip()
                    if not desc_val or desc_val.lower() == 'nan':
                        continue
                    h = md5(desc_val.encode()).hexdigest()
                    if h not in cache:
                        cache[h] = {}
                    
                    # Применяем правила в зависимости от приоритета
                    if rules_priority == 'prefer_rules' and rule_tag.get(cat):
                        cache[h][cat] = {tag: 1.0 for tag in rule_tag[cat]}
                    elif rules_priority == 'prefer_model':
                        cache[h][cat] = dict(zip(res["labels"], res["scores"]))
                    else:  # append
                        model_scores = dict(zip(res["labels"], res["scores"]))
                        rule_scores = {tag: 1.0 for tag in rule_tag.get(cat, [])}
                        # Объединяем теги, отдавая предпочтение правилам
                        combined = {**model_scores, **rule_scores}
                        cache[h][cat] = combined
                        
            progress.update(len(batch))

        # Формируем результаты для каждой строки
        tags_per_category = {cat: 0 for cat, _ in categories}
        cache_hits = 0
        for i, row in df.iterrows():
            desc = str(row.get("description", "")).strip()
            name = str(row.get("name", "")).strip()
            if not desc or desc.lower() == 'nan':
                tag_json = {}
                all_json.append(json.dumps(tag_json, ensure_ascii=False))
                tags = {cat: [] for cat, _ in categories}
                tags_ru = {cat: [] for cat, _ in categories}
                results.append({**tags, **{cat+"_ru": tags_ru[cat] for cat in tags_ru}})
                continue
            h = md5(desc.encode()).hexdigest()
            tag_json = cache.get(h, {})
            if h in cache:
                cache_hits += 1
            all_json.append(json.dumps(tag_json, ensure_ascii=False))
            tags = {cat: [] for cat, _ in categories}  # Инициализируем пустые списки для всех категорий
            tags_ru = {cat: [] for cat, _ in categories}  # Инициализируем пустые списки для всех категорий
            for cat, labels in categories:
                tag_scores = tag_json.get(cat, {})
                filtered = [label for label, score in sorted(tag_scores.items(), key=lambda x: -x[1]) if score >= threshold]
                if top_k:
                    filtered = filtered[:top_k]
                tags[cat] = filtered
                tags_ru[cat] = [LABEL_TRANSLATIONS.get(label, label) for label in filtered]
                tags_per_category[cat] += len(filtered)
            results.append({**tags, **{cat+"_ru": tags_ru[cat] for cat in tags_ru}})
        progress.close()
        # Добавляем колонки
        for cat, _ in categories:
            df[cat] = [r[cat] for r in results]
            df[cat+"_ru"] = [r[cat+"_ru"] for r in results]
        df["tags_json"] = all_json
        df.to_csv(output_file, index=False)
        print(f"✔ Done! Saved to: {output_file}")
        save_cache(cache, cache_file)
        print(f"Кеш сохранён в {cache_file}")
        # Финальный отчёт
        total_rows = len(df)
        print("\n📊 Отчёт:")
        print(f"✔ Всего обработано: {total_rows} строк")
        for cat in tags_per_category:
            avg = tags_per_category[cat] / total_rows if total_rows else 0
            print(f"   {cat:<12} — в среднем {avg:.1f} тега(ов)/строку")
        print(f"✔ Кеш-хитов: {cache_hits} из {total_rows}")

        # Валидация результатов
        warnings = validate_classification_results(df, cache, threshold)
        if warnings:
            logger.warning("Обнаружены проблемы при валидации:")
            for warning in warnings:
                logger.warning(f"- {warning}")
        
        # Собираем статистику
        stats = collect_statistics(df, cache, start_time)
        stats['warnings'].extend(warnings)  # Добавляем предупреждения в статистику
        report = generate_report(stats)
        logger.info("\n" + report)
        
        # Экспортируем результаты в разных форматах
        export_results(df, results, all_json, output_file)
        
        # Сохраняем кэш
        save_cache(cache, cache_file)
        logger.info(f"Кэш сохранен в {cache_file}")
        
        return True, "Success"
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    args = get_args()
    tag_csv_with_progress(args.input, args.output, args.limit, args.threshold, args.top_k, args.batch_size, args.cache, args.force, args.rules_priority) 