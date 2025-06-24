import os

# Пути к файлам по умолчанию
DEFAULT_INPUT_PATH = os.path.join('data', 'input.csv')
DEFAULT_OUTPUT_PATH = os.path.join('data', 'output.csv')
DEFAULT_CACHE_PATH = 'cache.pkl'

# Параметры по умолчанию
DEFAULT_THRESHOLD = 0.3
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 8
DEFAULT_RULES_PRIORITY = 'append' 