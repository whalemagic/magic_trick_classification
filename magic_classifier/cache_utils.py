import os
import pickle
import logging

logger = logging.getLogger(__name__)

def load_cache(cache_path):
    """Загружает кеш из файла."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            logger.info(f"Загружен кеш из {cache_path}, записей: {len(cache)}")
            return cache
        except Exception as e:
            logger.error(f"Ошибка при загрузке кеша: {str(e)}")
    return {}

def save_cache(cache, cache_path):
    """Сохраняет кеш в файл."""
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f"Кеш сохранён в {cache_path}") 