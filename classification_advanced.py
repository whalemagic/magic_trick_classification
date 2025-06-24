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
    if any(x in name.lower() for x in ["dvd", "video", "tutorial"]):
        tags["product_type"].append("video")
    if any(x in name.lower() for x in ["book", "manual", "guide"]):
        tags["product_type"].append("book")
    if "magazine" in name.lower():
        tags["product_type"].append("magazine")
    if "kit" in name.lower():
        tags["product_type"].append("kit")
    if not tags["product_type"]:
        tags["product_type"].append("physical product")
    
    # Правила для сложности
    if any(x in name.lower() for x in ["beginner", "easy", "basic"]):
        tags["difficulty"].extend(["beginner", "easy to learn"])
    if "intermediate" in name.lower():
        tags["difficulty"].append("intermediate")
    if any(x in name.lower() for x in ["advanced", "expert"]):
        tags["difficulty"].append("advanced")
    if any(x in name.lower() for x in ["professional", "pro"]):
        tags["difficulty"].append("professional")
    
    # Правила для реквизита
    if any(x in name.lower() for x in ["card", "deck"]):
        tags["props"].append("cards")
    if any(x in name.lower() for x in ["coin", "coins"]):
        tags["props"].append("coins")
    if any(x in name.lower() for x in ["phone", "smartphone"]):
        tags["props"].append("phone")
    if any(x in name.lower() for x in ["wand", "stick"]):
        tags["props"].append("magic wand")
    if any(x in name.lower() for x in ["box", "case"]):
        tags["props"].append("magic box")
    if any(x in name.lower() for x in ["gimmick", "gimmicked"]):
        tags["props"].append("gimmick")
    if any(x in name.lower() for x in ["apparatus", "device"]):
        tags["props"].append("magic apparatus")
    
    # Правила для масштаба
    if any(x in name.lower() for x in ["close-up", "closeup", "table"]):
        tags["scale"].append("close-up magic")
    if any(x in name.lower() for x in ["parlour", "parlor", "parlour"]):
        tags["scale"].append("parlour magic")
    if any(x in name.lower() for x in ["stage", "show"]):
        tags["scale"].append("stage magic")
    if any(x in name.lower() for x in ["comedy", "funny"]):
        tags["scale"].append("comedy magic")
    if any(x in name.lower() for x in ["serious", "professional"]):
        tags["scale"].append("serious magic")
    if any(x in name.lower() for x in ["interactive", "audience"]):
        tags["scale"].append("interactive magic")
    
    # Правила для эффектов
    if any(x in name.lower() for x in ["mental", "mind", "psychic"]):
        tags["effect"].append("mentalism")
    if any(x in name.lower() for x in ["predict", "prediction"]):
        tags["effect"].append("prediction")
    if any(x in name.lower() for x in ["vanish", "disappear"]):
        tags["effect"].append("vanish")
    if any(x in name.lower() for x in ["appear", "production"]):
        tags["effect"].append("appearance")
    if any(x in name.lower() for x in ["transform", "change"]):
        tags["effect"].append("transformation")
    if any(x in name.lower() for x in ["restore", "restoration"]):
        tags["effect"].append("restoration")
    if "levitation" in name.lower():
        tags["effect"].append("levitation")
    
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
    """Генерирует отчёт о результатах обработки."""
    report = ["📊 Отчёт:"]
    report.append(f"✔ Всего обработано: {stats['total_rows']} строк")
    
    for cat in ['effect', 'props', 'scale', 'difficulty', 'product_type']:
        avg_tags = stats['tags_per_category'][cat] / stats['total_rows']
        report.append(f"   {cat:<12} — в среднем {avg_tags:.1f} тега(ов)/строку")
    
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

def tag_csv_with_progress(input_file, output_file, limit=None, threshold=0.3, top_k=3, batch_size=8, cache_file='cache.pkl', force=False, rules_priority='append'):
    try:
        logger.info(f"Параметр limit: {limit}")
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
        df = pd.read_csv(input_file)
        logger.info(f"Загружен файл {input_file}, строк: {len(df)}")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Применено ограничение: {limit} строк")

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Используется устройство: {'CUDA' if device == 0 else 'CPU'}")
        
        logger.info("Загрузка модели и токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        classifier = ZeroShotClassificationPipeline(model=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
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
    except Exception as e:
        logging.error(f"Ошибка при обработке файла: {str(e)}")

if __name__ == "__main__":
    args = get_args()
    tag_csv_with_progress(args.input, args.output, args.limit, args.threshold, args.top_k, args.batch_size, args.cache, args.force, args.rules_priority) 