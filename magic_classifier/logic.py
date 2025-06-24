import pandas as pd
from tqdm import tqdm
import json
import logging
from hashlib import md5
from typing import Dict, List

from .categories import CATEGORIES, LABEL_TRANSLATIONS
from .classifier import load_model_and_tokenizer, batch_infer
from .rules import rule_based_tags
from .cache_utils import load_cache, save_cache
from .config import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_BATCH_SIZE,
    DEFAULT_RULES_PRIORITY
)

logger = logging.getLogger(__name__)

def translate_tags(tags: List[str]) -> List[str]:
    """Переводит теги на русский язык."""
    return [LABEL_TRANSLATIONS.get(tag, tag) for tag in tags]

def save_partial_results(df: pd.DataFrame, output_path: str, row_count: int):
    """Сохраняет промежуточные результаты."""
    base, ext = os.path.splitext(output_path)
    partial_path = f"{base}.autosave_{row_count}{ext}"
    df.to_csv(partial_path, index=False)
    logger.info(f"Сохранены промежуточные результаты: {partial_path}")

def generate_report(stats: dict) -> str:
    """Генерирует отчёт о результатах обработки."""
    report = ["📊 Отчёт:"]
    report.append(f"✔ Всего обработано: {stats['total_rows']} строк")
    
    for cat in ['effect', 'props', 'scale', 'difficulty', 'product_type']:
        avg_tags = stats['tags_per_category'][cat] / stats['total_rows']
        report.append(f"   {cat:<12} — в среднем {avg_tags:.1f} тега(ов)/строку")
    
    return "\n".join(report)

def tag_csv_with_progress(
    input_file=DEFAULT_INPUT_PATH,
    output_file=DEFAULT_OUTPUT_PATH,
    limit=None,
    threshold=DEFAULT_THRESHOLD,
    top_k=DEFAULT_TOP_K,
    batch_size=DEFAULT_BATCH_SIZE,
    cache_file=DEFAULT_CACHE_PATH,
    force=False,
    rules_priority=DEFAULT_RULES_PRIORITY
):
    """Основная функция для обработки CSV файла."""
    try:
        # Загружаем кэш
        cache = load_cache(cache_file) if not force else {}

        logger.info("Начинаем обработку файла...")
        df = pd.read_csv(input_file)
        logger.info(f"Загружен файл {input_file}, строк: {len(df)}")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Применено ограничение: {limit} строк")

        # Загружаем модель
        classifier = load_model_and_tokenizer(batch_size)

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
            
            # Получаем rule-based теги
            rule_tags = [rule_based_tags(name, desc) for name, desc in zip(batch_names, batch_descs)]
            logger.debug(f"Получены rule-based теги для батча: {rule_tags}")
            
            for cat, labels in CATEGORIES:
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
        tags_per_category = {cat: 0 for cat, _ in CATEGORIES}
        cache_hits = 0
        for i, row in df.iterrows():
            desc = str(row.get("description", "")).strip()
            name = str(row.get("name", "")).strip()
            if not desc or desc.lower() == 'nan':
                tag_json = {}
                all_json.append(json.dumps(tag_json, ensure_ascii=False))
                tags = {cat: [] for cat, _ in CATEGORIES}
                tags_ru = {cat: [] for cat, _ in CATEGORIES}
                results.append({**tags, **{cat+"_ru": tags_ru[cat] for cat in tags_ru}})
                continue
            h = md5(desc.encode()).hexdigest()
            tag_json = cache.get(h, {})
            if h in cache:
                cache_hits += 1
            all_json.append(json.dumps(tag_json, ensure_ascii=False))
            tags = {cat: [] for cat, _ in CATEGORIES}  # Инициализируем пустые списки для всех категорий
            tags_ru = {cat: [] for cat, _ in CATEGORIES}  # Инициализируем пустые списки для всех категорий
            for cat, labels in CATEGORIES:
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
        for cat, _ in CATEGORIES:
            df[cat] = [r[cat] for r in results]
            df[cat+"_ru"] = [r[cat+"_ru"] for r in results]
        df["tags_json"] = all_json
        df.to_csv(output_file, index=False)
        print(f"✔ Done! Saved to: {output_file}")
        
        # Сохраняем кеш
        save_cache(cache, cache_file)
        
        # Финальный отчёт
        total_rows = len(df)
        print("\n📊 Отчёт:")
        print(f"✔ Всего обработано: {total_rows} строк")
        for cat in tags_per_category:
            avg = tags_per_category[cat] / total_rows if total_rows else 0
            print(f"   {cat:<12} — в среднем {avg:.1f} тега(ов)/строку")
        print(f"✔ Кеш-хитов: {cache_hits} из {total_rows}")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {str(e)}")
        raise 