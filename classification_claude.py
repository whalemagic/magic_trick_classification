import pandas as pd
import os
import anthropic
from dotenv import load_dotenv
import time
import re
import ast
from typing import List, Tuple, Dict, Any
import logging
from tqdm import tqdm
import argparse
import sys

# Настройка логирования в файл
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Загрузка переменных окружения
load_dotenv()
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

if not CLAUDE_API_KEY:
    raise ValueError("CLAUDE_API_KEY не найден в файле .env")

# Инициализация клиента
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def clean_text_for_excel(text: str) -> str:
    """Очищает текст от недопустимых для Excel символов"""
    if not isinstance(text, str):
        return text
    # Заменяем недопустимые символы на пробел
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)

def parse_claude_response(content: str) -> Dict[str, Any]:
    """Парсит ответ от Claude API и группирует результаты по категориям."""
    try:
        # Очищаем текст от лишних пояснений
        content = content.strip()
        if content.startswith('```python'):
            content = content[8:]
        if content.endswith('```'):
            content = content[:-3]
            
        # Пытаемся найти список в тексте
        match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if not match:
            return None
            
        # Парсим список кортежей
        items = []
        for item_match in re.finditer(r'\(([^)]+)\)', match.group(1)):
            item_str = item_match.group(1)
            # Разбиваем на компоненты, учитывая возможные запятые внутри строк
            parts = []
            current_part = ""
            in_quotes = False
            for char in item_str:
                if char in ["'", '"']:
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            if current_part:
                parts.append(current_part.strip())
            
            # Очищаем части от кавычек и преобразуем числа
            cleaned_parts = []
            for part in parts:
                part = part.strip().strip("'").strip('"')
                try:
                    if '.' in part:
                        cleaned_parts.append(float(part))
                    else:
                        cleaned_parts.append(int(part))
                except ValueError:
                    cleaned_parts.append(part)
            
            if len(cleaned_parts) >= 3:
                items.append(tuple(cleaned_parts[:3]))  # берем только первые три элемента

        # Группируем по категориям
        effects = []
        props = []
        product_types = []
        
        effect_types = [
            "Production", "Vanish", "Transformation", "Teleportation", 
            "Restoration", "Multiplication", "Penetration", "Prediction",
            "Mind Reading", "Mentalism", "Time Manipulation", "Levitation",
            "Animation", "Anti-gravity", "Impossible Object", "Control",
            "Forcing", "Escape", "Choice Revelation"
        ]
        
        prop_types = [
            "cards", "coins", "phone", "paper", "bills", "magic wand",
            "magic box", "magic apparatus", "gimmick", "prop", "device",
            "clips"  # добавляем clips, так как он часто встречается
        ]
        
        product_type_list = [
            "physical product", "digital download", "video",
            "book", "magazine", "kit"
        ]
        
        for item in items:
            name, prob, reason = item
            name_lower = name.lower()
            
            # Определяем категорию
            if "Custom Effect:" in name or any(effect.lower() in name_lower for effect in effect_types):
                effects.append(item)
            elif any(prop.lower() in name_lower for prop in prop_types):
                props.append(item)
            elif any(ptype.lower() in name_lower for ptype in product_type_list):
                product_types.append(item)
        
        return {
            'effects': effects,
            'props': props,
            'product_types': product_types
        }
        
    except Exception as e:
        logging.error(f"Ошибка при парсинге ответа: {str(e)}")
        logging.error(f"Содержимое ответа: {content}")
        return None

def classify_product(name: str, description: str) -> Dict[str, Any]:
    """
    Классифицирует магический продукт используя Claude API.
    """
    prompt = f"""You are a magic trick classifier. Analyze this magic product and provide a detailed classification.

Product Name: {name}
Description: {description}

Requirements:
1. Main magical effect should be primarily chosen from this list:
   Production, Vanish, Transformation, Teleportation, Restoration, 
   Multiplication, Penetration, Prediction, Mind Reading, Mentalism, 
   Time Manipulation, Levitation, Animation, Anti-gravity, 
   Impossible Object, Control / Forcing, Escape, Choice Revelation

   If you are highly confident (probability >= 0.9) that the effect is something 
   different, you can suggest your own effect category. In this case, provide a 
   detailed explanation of why the standard categories don't fit and why your 
   suggestion is more appropriate.

2. Props should be primarily chosen from this list (but you can suggest others if they fit better): 
   cards, coins, phone, paper, bills, magic wand, magic box, magic apparatus, 
   gimmick, prop, device

3. Product type should be one of: physical product, digital download, video, 
   book, magazine, kit

Return your analysis as a list of tuples, where each tuple contains (name, probability, reason).
The probability should be a number between 0 and 1.
Include ALL relevant effects, props, and product types in the same list.

Example response format:
[
    ("Prediction", 0.9, "explicitly mentioned in title"),
    ("Custom Effect: Visual Morphing", 0.95, "The effect involves smooth visual transformation of objects that goes beyond standard transformation - objects actually morph and flow like liquid"),
    ("cards", 0.8, "main prop used in the effect"),
    ("digital download", 1.0, "product is instant download")
]"""

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Получаем текст ответа
        content = response.content[0].text if isinstance(response.content, list) else response.content
        
        # Парсим ответ
        return parse_claude_response(content)
            
    except Exception as e:
        logging.error(f"Ошибка при классификации продукта {name}: {str(e)}")
        logging.error(f"Полный ответ API: {response.content if 'response' in locals() else 'нет ответа'}")
        return None

def format_classification_with_probabilities(classification: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, str]:
    """Форматирует классификацию с вероятностями для каждой категории."""
    result = {}
    
    # Форматируем эффекты
    effects = classification.get('effects', [])
    effects_str = "; ".join([f"{effect[0]} ({effect[1]:.2f})" for effect in effects])
    result['effects'] = effects_str
    
    # Форматируем реквизит
    props = classification.get('props', [])
    props_str = "; ".join([f"{prop[0]} ({prop[1]:.2f})" for prop in props])
    result['props'] = props_str
    
    return result

def format_classification_with_comments(classification: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, str]:
    """Форматирует классификацию с комментариями для каждой категории."""
    result = {}
    
    # Форматируем эффекты с комментариями
    effects = classification.get('effects', [])
    effects_str = "; ".join([f"{effect[0]} ({effect[1]:.2f}) - {effect[2]}" for effect in effects])
    result['effects'] = effects_str
    
    # Форматируем реквизит с комментариями
    props = classification.get('props', [])
    props_str = "; ".join([f"{prop[0]} ({prop[1]:.2f}) - {prop[2]}" for prop in props])
    result['props'] = props_str
    
    return result

def get_high_probability_classification(classification: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, str]:
    """Получает только классификации с высокой вероятностью (>= 0.7)."""
    result = {}
    
    # Фильтруем эффекты
    effects = classification.get('effects', [])
    high_prob_effects = [effect for effect in effects if effect[1] >= 0.7]
    effects_str = "; ".join([f"{effect[0]} ({effect[1]:.2f})" for effect in high_prob_effects])
    result['effects'] = effects_str
    
    # Фильтруем реквизит
    props = classification.get('props', [])
    high_prob_props = [prop for prop in props if prop[1] >= 0.7]
    props_str = "; ".join([f"{prop[0]} ({prop[1]:.2f})" for prop in high_prob_props])
    result['props'] = props_str
    
    return result

def get_high_probability_classification_with_comments(classification: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, str]:
    """Получает только классификации с высокой вероятностью (>= 0.7) с комментариями."""
    result = {}
    
    # Фильтруем эффекты с комментариями
    effects = classification.get('effects', [])
    high_prob_effects = [effect for effect in effects if effect[1] >= 0.7]
    effects_str = "; ".join([f"{effect[0]} ({effect[1]:.2f}) - {effect[2]}" for effect in high_prob_effects])
    result['effects'] = effects_str
    
    # Фильтруем реквизит с комментариями
    props = classification.get('props', [])
    high_prob_props = [prop for prop in props if prop[1] >= 0.7]
    props_str = "; ".join([f"{prop[0]} ({prop[1]:.2f}) - {prop[2]}" for prop in high_prob_props])
    result['props'] = props_str
    
    return result

def save_to_excel(df: pd.DataFrame, output_file: str):
    """Безопасное сохранение в Excel с очисткой данных"""
    # Создаем копию датафрейма
    df_clean = df.copy()
    
    # Очищаем все строковые колонки
    for column in df_clean.select_dtypes(include=['object']).columns:
        df_clean[column] = df_clean[column].apply(clean_text_for_excel)
    
    try:
        df_clean.to_excel(output_file, index=False)
        logging.info(f"Результаты успешно сохранены в {output_file}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Excel: {str(e)}")
        # Пробуем сохранить в CSV как запасной вариант
        csv_file = output_file.replace('.xlsx', '.csv')
        df_clean.to_csv(csv_file, index=False)
        logging.info(f"Результаты сохранены в CSV: {csv_file}")

def save_intermediate_results(df: pd.DataFrame, filename: str):
    """Сохраняет промежуточные результаты только в CSV формат."""
    try:
        df.to_csv(f"{filename}.csv", index=False, encoding='utf-8-sig')
        logging.info(f"Промежуточные результаты сохранены в {filename}.csv")
    except Exception as e:
        logging.error(f"Ошибка при сохранении в CSV: {str(e)}")

def save_final_results(df: pd.DataFrame, base_filename: str):
    """Сохраняет финальные результаты в Excel и CSV форматы."""
    try:
        # Сохраняем в CSV
        df.to_csv(f"{base_filename}.csv", index=False, encoding='utf-8-sig')
        logging.info(f"Финальные результаты сохранены в {base_filename}.csv")
        
        # Сохраняем в Excel
        save_to_excel(df, f"{base_filename}.xlsx")
        logging.info(f"Финальные результаты сохранены в {base_filename}.xlsx")
    except Exception as e:
        logging.error(f"Ошибка при сохранении финальных результатов: {str(e)}")

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Ограничение количества записей для обработки')
    parser.add_argument('--start_from', type=int, help='Начать с указанной позиции (может быть отрицательным)')
    parser.add_argument('--input', type=str, default='data/input.csv', help='Путь к входному CSV файлу')
    args = parser.parse_args()

    try:
        # Читаем CSV файл
        df = pd.read_csv(args.input)
        total_rows = len(df)
        
        # Определяем диапазон записей для обработки
        start_idx = args.start_from if args.start_from is not None else 0
        if start_idx < 0:
            start_idx = total_rows + start_idx
        
        end_idx = start_idx + args.limit if args.limit is not None else total_rows
        end_idx = min(end_idx, total_rows)
        
        logging.info(f'Обрабатываем записи с {start_idx} по {end_idx}')
        
        # Получаем срез данных
        df_slice = df.iloc[start_idx:end_idx].copy()
        logging.info(f'Загружено {len(df_slice)} строк')

        # Сбрасываем индекс для корректной работы с DataFrame
        df_slice = df_slice.reset_index(drop=True)

        # Добавляем новые колонки для эффектов
        df_slice['effect_all_classification'] = None
        df_slice['effect_all_classification_with_comments'] = None
        df_slice['effect_res_classification'] = None
        df_slice['effect_res_classification_with_comments'] = None
        
        # Добавляем новые колонки для реквизита
        df_slice['props_all_classification'] = None
        df_slice['props_all_classification_with_comments'] = None
        df_slice['props_res_classification'] = None
        df_slice['props_res_classification_with_comments'] = None

        # Настройки сохранения
        save_interval = 10  # Сохраняем каждые 10 записей
        base_output_path = "output/claude_output"

        # Создаем директорию для выходных файлов, если её нет
        os.makedirs("output", exist_ok=True)

        # Создаем прогресс-бар с отключенным выводом в консоль
        pbar = tqdm(total=len(df_slice), desc="Классификация", unit="запись", file=sys.stdout, mininterval=1)

        try:
            # Обработка каждой строки
            for idx, row in df_slice.iterrows():
                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_description(f"Обработка {idx + 1}/{len(df_slice)}")
                
                # Логируем в файл
                logging.info(f"\nНазвание: {row['name']}")
                logging.info(f"Описание: {row['description'][:100]}...")

                classification = classify_product(row['name'], row['description'])
                
                if classification:
                    logging.info("Успешная классификация:")
                    logging.info(f"Effects: {classification['effects']}")
                    logging.info(f"Props: {classification['props']}")
                    logging.info(f"Product types: {classification['product_types']}")
                    
                    # Получаем форматированные результаты
                    all_class = format_classification_with_probabilities(classification)
                    all_class_comments = format_classification_with_comments(classification)
                    res_class = get_high_probability_classification(classification)
                    res_class_comments = get_high_probability_classification_with_comments(classification)
                    
                    # Сохраняем результаты эффектов
                    df_slice.at[idx, 'effect_all_classification'] = all_class['effects']
                    df_slice.at[idx, 'effect_all_classification_with_comments'] = all_class_comments['effects']
                    df_slice.at[idx, 'effect_res_classification'] = res_class['effects']
                    df_slice.at[idx, 'effect_res_classification_with_comments'] = res_class_comments['effects']
                    
                    # Сохраняем результаты реквизита
                    df_slice.at[idx, 'props_all_classification'] = all_class['props']
                    df_slice.at[idx, 'props_all_classification_with_comments'] = all_class_comments['props']
                    df_slice.at[idx, 'props_res_classification'] = res_class['props']
                    df_slice.at[idx, 'props_res_classification_with_comments'] = res_class_comments['props']
                    
                    logging.info("Форматированные результаты:")
                    logging.info(f"Все эффекты: {df_slice.at[idx, 'effect_all_classification']}")
                    logging.info(f"Все эффекты с комментариями: {df_slice.at[idx, 'effect_all_classification_with_comments']}")
                    logging.info(f"Высокая вероятность эффектов: {df_slice.at[idx, 'effect_res_classification']}")
                    logging.info(f"Высокая вероятность эффектов с комментариями: {df_slice.at[idx, 'effect_res_classification_with_comments']}")
                    logging.info(f"Весь реквизит: {df_slice.at[idx, 'props_all_classification']}")
                    logging.info(f"Весь реквизит с комментариями: {df_slice.at[idx, 'props_all_classification_with_comments']}")
                    logging.info(f"Высокая вероятность реквизита: {df_slice.at[idx, 'props_res_classification']}")
                    logging.info(f"Высокая вероятность реквизита с комментариями: {df_slice.at[idx, 'props_res_classification_with_comments']}")
                else:
                    logging.error(f"Не удалось получить классификацию для записи {idx + 1}")
                
                # Сохраняем промежуточные результаты только в CSV
                if (idx + 1) % save_interval == 0:
                    processed_count = idx + 1
                    # Обновляем описание прогресс-бара при сохранении
                    pbar.set_description(f"Сохранение результатов... ({processed_count}/{len(df_slice)})")
                    
                    # Сохраняем текущий прогресс
                    current_output_path = f"{base_output_path}_part_{processed_count}"
                    save_intermediate_results(df_slice[:processed_count], current_output_path)
                    
                    time.sleep(1)  # Небольшая пауза чтобы не перегружать API
                    
                    # Возвращаем описание прогресс-бара
                    pbar.set_description(f"Классификация")

        except Exception as e:
            logging.error(f"Произошла ошибка: {str(e)}")
            # Сохраняем результаты при ошибке
            error_output_path = f"{base_output_path}_error_{idx + 1}"
            save_intermediate_results(df_slice[:idx + 1], error_output_path)
            raise e
        finally:
            # Закрываем прогресс-бар
            pbar.close()

            # Сохраняем финальные результаты в оба формата
            logging.info("\nСохранение финальных результатов...")
            save_final_results(df_slice, base_output_path)
            logging.info("Классификация завершена")

    except Exception as e:
        logging.error(f"Произошла ошибка при чтении CSV файла: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 