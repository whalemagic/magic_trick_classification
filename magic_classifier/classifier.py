import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ZeroShotClassificationPipeline
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Определяет доступное устройство для модели."""
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Используется устройство: {'CUDA' if device == 0 else 'CPU'}")
    return device

def load_model_and_tokenizer(batch_size=8):
    """Загружает модель и токенизатор."""
    logger.info("Загрузка модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    device = get_device()
    classifier = ZeroShotClassificationPipeline(
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        batch_size=batch_size
    )
    logger.info("Модель и токенизатор успешно загружены")
    return classifier

def batch_infer(classifier, descs, labels):
    """Выполняет пакетное предсказание для списка описаний."""
    # descs: список описаний, labels: список меток
    results = classifier(descs, labels, multi_label=True)
    # results — список словарей (или один словарь, если 1 описание)
    if isinstance(results, dict):
        results = [results]
    return results 