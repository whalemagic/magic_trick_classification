import logging
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)

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