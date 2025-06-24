# Переводы меток
LABEL_TRANSLATIONS = {
    # Эффекты
    "mentalism": "ментализм",
    "prediction": "предсказание",
    "mind reading": "чтение мыслей",
    "vanish": "исчезновение",
    "appearance": "появление",
    "transformation": "трансформация",
    "restoration": "восстановление",
    "levitation": "левитация",
    "penetration": "проникновение",
    "transposition": "транспозиция",
    "teleportation": "телепортация",
    "cards magic" : "карточная магия",
    "card control": "контроль карты",
    "card location": "локация карты",
    "coin magic": "фокусы с монетами",
    
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
    
    # Масштаб 
    "close-up magic": "микромагия",
    "parlour magic": "салонная магия",
    "stage magic": "сценическая магия",
    
    # Уровень сложности
    "beginner": "начинающий",
    "intermediate": "средний",
    "advanced": "продвинутый",
    "professional": "профессиональный",
    
    # Тип товара
    "physical product": "физический товар",
    "digital download": "цифровая загрузка",
    "video": "видео",
    "book": "книга",
    "magazine": "журнал",
}

# 1. Основные эффекты
effects_eng = [
    "mentalism", "prediction", "mind reading",
    "vanish", "appearance", "transformation", "restoration",
    "levitation", "penetration", "transposition", "teleportation",
    "cards magic", "card control", "card location",
    "coin magic"
]

# 2. Реквизит
props_eng = [
    "cards", "coins", "phone", "paper", "bills",
    "magic wand", "magic box", "magic apparatus",
    "gimmick", "die", "electronic device"
]

# 3. Масштаб и стиль
scale_eng = [
    "close-up magic", "parlour magic", "stage magic"
]

# 4. Уровень сложности
difficulty_eng = [
    "beginner", "intermediate", "advanced", "professional"
]

# 5. Тип товара
product_type_eng = [
    "physical product", "digital download", "video",
    "book", "magazine"
]

# Все категории
CATEGORIES = [
    ("effect", effects_eng),
    ("props", props_eng),
    ("scale", scale_eng),
    ("difficulty", difficulty_eng),
    ("product_type", product_type_eng)
] 