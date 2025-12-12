"""
Реалізація алгоритму Artificial Bee Colony (ABC)
"""
import numpy as np
from typing import Callable, Dict, List
from abc_config import ABCConfig


def calculate_fitness(f_value: float) -> float:
    """
    Обчислює fitness на основі значення цільової функції.
    
    fitness = 1/(1+f),  якщо f ≥ 0
    fitness = 1+|f|,    якщо f < 0
    
    Для задач мінімізації: більший fitness означає краще рішення
    """
    if f_value >= 0:
        return 1.0 / (1.0 + f_value)
    else:
        return 1.0 + abs(f_value)


def abc_classic(f: Callable[[np.ndarray], float], 
                bounds: np.ndarray,
                cfg: ABCConfig) -> Dict:
    """
    Класичний алгоритм Artificial Bee Colony (ABC)
    
    Параметри:
    - f: цільова функція для мінімізації
    - bounds: межі області пошуку, shape (d, 2)
    - cfg: конфігурація параметрів ABC
    
    Повертає:
    - dict з результатами:
        - best_solution: найкраще знайдене рішення
        - best_fitness: значення f(best_solution)
        - history: список best-so-far значень по циклах
        - evaluations: кількість обчислень цільової функції
    """
    np.random.seed(cfg.seed)
    d = len(bounds)  # розмірність
    
    # Ініціалізація джерел їжі (food sources)
    food_sources = np.random.uniform(
        low=bounds[:, 0],
        high=bounds[:, 1],
        size=(cfg.colony_size, d)
    )
    
    # Обчислення значень цільової функції та fitness для кожного джерела
    f_values = np.array([f(x) for x in food_sources])
    fitness = np.array([calculate_fitness(fv) for fv in f_values])
    
    # Лічильники спроб (trial counters) - скільки разів джерело не покращувалось
    trials = np.zeros(cfg.colony_size, dtype=int)
    
    # Найкраще рішення (мінімум цільової функції, максимум fitness)
    best_idx = np.argmin(f_values)
    best_solution = food_sources[best_idx].copy()
    best_f_value = f_values[best_idx]
    
    # Історія збіжності (зберігаємо значення цільової функції, не fitness)
    history = [best_f_value]
    evaluations = cfg.colony_size  # початкові обчислення
    
    # Основний цикл ABC
    for cycle in range(cfg.max_cycles):
        
        # ===== 1. EMPLOYED BEES PHASE =====
        # Кожна employed bee намагається покращити своє джерело
        for i in range(cfg.colony_size):
            # Вибір випадкової розмірності для модифікації
            j = np.random.randint(d)
            
            # Вибір випадкового сусіда k (k != i)
            k = i
            while k == i:
                k = np.random.randint(cfg.colony_size)
            
            # Генерація нового кандидата за формулою:
            # v_ij = x_ij + φ_ij * (x_ij - x_kj)
            # де φ_ij ∈ [-1, 1]
            phi = np.random.uniform(-1, 1)
            candidate = food_sources[i].copy()
            candidate[j] = food_sources[i, j] + phi * (food_sources[i, j] - food_sources[k, j])
            
            # Обмеження меж
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            
            # Обчислення значення функції та fitness кандидата
            candidate_f_value = f(candidate)
            candidate_fitness = calculate_fitness(candidate_f_value)
            evaluations += 1
            
            # Greedy selection - порівнюємо fitness (більший = кращий)
            if candidate_fitness > fitness[i]:
                food_sources[i] = candidate
                f_values[i] = candidate_f_value
                fitness[i] = candidate_fitness
                trials[i] = 0  # скидаємо лічильник
            else:
                trials[i] += 1  # збільшуємо лічильник невдач
        
        # ===== 2. ONLOOKER BEES PHASE =====
        # Обчислення ймовірностей вибору джерел на основі fitness
        # Більший fitness = краще рішення = більша ймовірність бути обраним
        probabilities = fitness / np.sum(fitness)
        
        # Onlooker bees вибирають джерела пропорційно ймовірностям
        onlooker_count = 0
        i = 0
        while onlooker_count < cfg.colony_size:
            # Рулетка для вибору джерела
            if np.random.random() < probabilities[i]:
                # Вибір випадкової розмірності
                j = np.random.randint(d)
                
                # Вибір випадкового сусіда
                k = i
                while k == i:
                    k = np.random.randint(cfg.colony_size)
                
                # Генерація кандидата
                phi = np.random.uniform(-1, 1)
                candidate = food_sources[i].copy()
                candidate[j] = food_sources[i, j] + phi * (food_sources[i, j] - food_sources[k, j])
                
                # Обмеження меж
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                
                # Обчислення значення функції та fitness
                candidate_f_value = f(candidate)
                candidate_fitness = calculate_fitness(candidate_f_value)
                evaluations += 1
                
                # Greedy selection - порівнюємо fitness
                if candidate_fitness > fitness[i]:
                    food_sources[i] = candidate
                    f_values[i] = candidate_f_value
                    fitness[i] = candidate_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
                
                onlooker_count += 1
            
            i = (i + 1) % cfg.colony_size
        
        # ===== 3. SCOUT BEES PHASE =====
        # Якщо джерело не покращувалось більше ніж limit разів - замінити на нове
        for i in range(cfg.colony_size):
            if trials[i] >= cfg.limit:
                # Scout bee знаходить нове випадкове джерело
                food_sources[i] = np.random.uniform(
                    low=bounds[:, 0],
                    high=bounds[:, 1],
                    size=d
                )
                f_values[i] = f(food_sources[i])
                fitness[i] = calculate_fitness(f_values[i])
                evaluations += 1
                trials[i] = 0
        
        # Оновлення найкращого рішення (мінімум цільової функції)
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_f_value:
            best_f_value = f_values[current_best_idx]
            best_solution = food_sources[current_best_idx].copy()
        
        history.append(best_f_value)
    
    return {
        'best_solution': best_solution,
        'best_fitness': best_f_value,  # повертаємо значення цільової функції
        'history': history,
        'evaluations': evaluations
    }


def abc_modified(f: Callable[[np.ndarray], float], 
                 bounds: np.ndarray,
                 cfg: ABCConfig,
                 c: float = 0.3) -> Dict:
    """
    Модифікований алгоритм ABC з "підтягуванням" до глобального best
    
    Параметри:
    - f: цільова функція для мінімізації
    - bounds: межі області пошуку, shape (d, 2)
    - cfg: конфігурація параметрів ABC
    - c: коефіцієнт "підтягування" до глобального best (типово 0.1-0.5)
    
    Модифікована формула оновлення:
    v_ij = x_ij + φ_ij(x_ij - x_kj) - c·r_ij(x_ij - g_j)
    де g - глобально найкраще знайдене джерело
    
    Повертає:
    - dict з результатами
    """
    np.random.seed(cfg.seed)
    d = len(bounds)
    
    # Ініціалізація джерел їжі
    food_sources = np.random.uniform(
        low=bounds[:, 0],
        high=bounds[:, 1],
        size=(cfg.colony_size, d)
    )
    
    # Обчислення значень цільової функції та fitness
    f_values = np.array([f(x) for x in food_sources])
    fitness = np.array([calculate_fitness(fv) for fv in f_values])
    
    # Лічильники спроб
    trials = np.zeros(cfg.colony_size, dtype=int)
    
    # Найкраще рішення
    best_idx = np.argmin(f_values)
    best_solution = food_sources[best_idx].copy()
    best_f_value = f_values[best_idx]
    
    # Історія збіжності
    history = [best_f_value]
    evaluations = cfg.colony_size
    
    # Основний цикл ABC
    for cycle in range(cfg.max_cycles):
        
        # ===== 1. EMPLOYED BEES PHASE (з модифікацією) =====
        for i in range(cfg.colony_size):
            # Вибір випадкової розмірності
            j = np.random.randint(d)
            
            # Вибір випадкового сусіда
            k = i
            while k == i:
                k = np.random.randint(cfg.colony_size)
            
            # Модифікована формула з підтягуванням до глобального best
            phi = np.random.uniform(-1, 1)
            r = np.random.uniform(0, 1)
            
            candidate = food_sources[i].copy()
            # v_ij = x_ij + φ(x_ij - x_kj) - c·r(x_ij - g_j)
            candidate[j] = (food_sources[i, j] + 
                          phi * (food_sources[i, j] - food_sources[k, j]) -
                          c * r * (food_sources[i, j] - best_solution[j]))
            
            # Обмеження меж
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            
            # Обчислення fitness
            candidate_f_value = f(candidate)
            candidate_fitness = calculate_fitness(candidate_f_value)
            evaluations += 1
            
            # Greedy selection
            if candidate_fitness > fitness[i]:
                food_sources[i] = candidate
                f_values[i] = candidate_f_value
                fitness[i] = candidate_fitness
                trials[i] = 0
            else:
                trials[i] += 1
        
        # ===== 2. ONLOOKER BEES PHASE (з модифікацією) =====
        probabilities = fitness / np.sum(fitness)
        
        onlooker_count = 0
        i = 0
        while onlooker_count < cfg.colony_size:
            if np.random.random() < probabilities[i]:
                # Вибір випадкової розмірності
                j = np.random.randint(d)
                
                # Вибір випадкового сусіда
                k = i
                while k == i:
                    k = np.random.randint(cfg.colony_size)
                
                # Модифікована формула
                phi = np.random.uniform(-1, 1)
                r = np.random.uniform(0, 1)
                
                candidate = food_sources[i].copy()
                candidate[j] = (food_sources[i, j] + 
                              phi * (food_sources[i, j] - food_sources[k, j]) -
                              c * r * (food_sources[i, j] - best_solution[j]))
                
                # Обмеження меж
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                
                # Обчислення fitness
                candidate_f_value = f(candidate)
                candidate_fitness = calculate_fitness(candidate_f_value)
                evaluations += 1
                
                # Greedy selection
                if candidate_fitness > fitness[i]:
                    food_sources[i] = candidate
                    f_values[i] = candidate_f_value
                    fitness[i] = candidate_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
                
                onlooker_count += 1
            
            i = (i + 1) % cfg.colony_size
        
        # ===== 3. SCOUT BEES PHASE =====
        for i in range(cfg.colony_size):
            if trials[i] >= cfg.limit:
                food_sources[i] = np.random.uniform(
                    low=bounds[:, 0],
                    high=bounds[:, 1],
                    size=d
                )
                f_values[i] = f(food_sources[i])
                fitness[i] = calculate_fitness(f_values[i])
                evaluations += 1
                trials[i] = 0
        
        # Оновлення найкращого рішення
        current_best_idx = np.argmin(f_values)
        if f_values[current_best_idx] < best_f_value:
            best_f_value = f_values[current_best_idx]
            best_solution = food_sources[current_best_idx].copy()
        
        history.append(best_f_value)
    
    return {
        'best_solution': best_solution,
        'best_fitness': best_f_value,
        'history': history,
        'evaluations': evaluations
    }

