"""
Конфігурація параметрів для Artificial Bee Colony (ABC)
"""
from dataclasses import dataclass


@dataclass
class ABCConfig:
    """
    Параметри алгоритму ABC
    
    Attributes:
        colony_size: Розмір колонії (кількість джерел їжі)
        limit: Ліміт спроб для scout bees (якщо джерело не покращується limit разів - замінити)
        max_cycles: Максимальна кількість циклів (ітерацій)
        seed: Seed для генератора випадкових чисел
    """
    colony_size: int = 30
    limit: int = 100
    max_cycles: int = 500
    seed: int = 42
