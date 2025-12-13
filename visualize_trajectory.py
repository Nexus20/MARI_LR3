"""
Окремий модуль для візуалізації траєкторії ABC на функції Griewank
Може бути запущений незалежно від основних експериментів
"""
import numpy as np
from abc_config import ABCConfig
from abc_algorithms import abc_with_trajectory
from abc_visualization import plot_function_surface_with_trajectory


def griewank(x: np.ndarray) -> float:
    """
    Функція Griewank - багатомодальна тестова функція для оптимізації.
    Глобальний мінімум: f(0,...,0) = 0
    Область пошуку: x_i ∈ [-600, 600]
    """
    x = np.asarray(x)
    d = len(x)
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d+1))))
    return 1.0 + sum_term - prod_term


def visualize_classic_abc(
    bounds: np.ndarray,
    cfg: ABCConfig,
    save_every: int = 10
):
    """
    Візуалізація траєкторії класичного ABC
    
    Параметри:
    - bounds: межі області пошуку
    - cfg: конфігурація ABC
    - save_every: зберігати позицію кожні N циклів
    """
    print("=" * 80)
    print("ВІЗУАЛІЗАЦІЯ: КЛАСИЧНИЙ ABC")
    print("=" * 80)
    
    print("\nЗапускаємо класичний ABC для отримання траєкторії...")
    result = abc_with_trajectory(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        save_every=save_every,
        use_modified=False
    )
    
    print(f"\nРезультати:")
    print(f"  Фінальне значення: {result['best_fitness']:.6e}")
    print(f"  Фінальна позиція: {result['best_solution']}")
    print(f"  Збережено {len(result['trajectory'])} точок траєкторії")
    
    print("\nГенеруємо візуалізацію...")
    plot_function_surface_with_trajectory(
        f=griewank,
        bounds=bounds,
        trajectory=result['trajectory'],
        title="Класичний ABC на функції Griewank"
    )
    
    return result


def visualize_modified_abc(
    bounds: np.ndarray,
    cfg: ABCConfig,
    c: float = 0.2,
    save_every: int = 10
):
    """
    Візуалізація траєкторії модифікованого ABC
    
    Параметри:
    - bounds: межі області пошуку
    - cfg: конфігурація ABC
    - c: коефіцієнт підтягування до глобального best
    - save_every: зберігати позицію кожні N циклів
    """
    print("=" * 80)
    print(f"ВІЗУАЛІЗАЦІЯ: МОДИФІКОВАНИЙ ABC (c={c})")
    print("=" * 80)
    
    print(f"\nЗапускаємо модифікований ABC (c={c}) для отримання траєкторії...")
    result = abc_with_trajectory(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        save_every=save_every,
        use_modified=True,
        c=c
    )
    
    print(f"\nРезультати:")
    print(f"  Фінальне значення: {result['best_fitness']:.6e}")
    print(f"  Фінальна позиція: {result['best_solution']}")
    print(f"  Збережено {len(result['trajectory'])} точок траєкторії")
    
    print("\nГенеруємо візуалізацію...")
    plot_function_surface_with_trajectory(
        f=griewank,
        bounds=bounds,
        trajectory=result['trajectory'],
        title=f"Модифікований ABC (c={c}) на функції Griewank"
    )
    
    return result


def visualize_both(
    bounds: np.ndarray,
    cfg: ABCConfig,
    c: float = 0.2,
    save_every: int = 10
):
    """
    Візуалізація обох варіантів ABC (класичного та модифікованого)
    
    Параметри:
    - bounds: межі області пошуку
    - cfg: конфігурація ABC
    - c: коефіцієнт для модифікованого ABC
    - save_every: зберігати позицію кожні N циклів
    """
    print("\n" + "=" * 80)
    print("ВІЗУАЛІЗАЦІЯ ТРАЄКТОРІЙ: КЛАСИЧНИЙ ТА МОДИФІКОВАНИЙ ABC")
    print("=" * 80)
    
    # Класичний ABC
    result_classic = visualize_classic_abc(bounds, cfg, save_every)
    
    print("\n")
    
    # Модифікований ABC
    result_modified = visualize_modified_abc(bounds, cfg, c, save_every)
    
    # Порівняння результатів
    print("\n" + "=" * 80)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
    print("=" * 80)
    print(f"\nКласичний ABC:")
    print(f"  Фінальне значення: {result_classic['best_fitness']:.6e}")
    
    print(f"\nМодифікований ABC (c={c}):")
    print(f"  Фінальне значення: {result_modified['best_fitness']:.6e}")
    
    return result_classic, result_modified


if __name__ == "__main__":
    # Параметри задачі
    d = 2  # Розмірність
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація ABC
    cfg = ABCConfig(
        colony_size=30,
        limit=100,
        max_cycles=500,
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("МОДУЛЬ ВІЗУАЛІЗАЦІЇ ТРАЄКТОРІЙ ABC")
    print("=" * 80)
    print("\nОберіть режим візуалізації:")
    print("1 - Класичний ABC")
    print("2 - Модифікований ABC (c=0.2)")
    print("3 - Обидва варіанти")
    
    choice = input("\nВаш вибір (1/2/3): ").strip()
    
    if choice == "1":
        visualize_classic_abc(bounds, cfg, save_every=10)
    elif choice == "2":
        c_value = input("Введіть значення c (default=0.2): ").strip()
        c = float(c_value) if c_value else 0.2
        visualize_modified_abc(bounds, cfg, c=c, save_every=10)
    elif choice == "3":
        visualize_both(bounds, cfg, c=0.2, save_every=10)
    else:
        print("Невірний вибір. Показуємо обидва варіанти...")
        visualize_both(bounds, cfg, c=0.2, save_every=10)
    
    print("\n" + "=" * 80)
    print("ВІЗУАЛІЗАЦІЮ ЗАВЕРШЕНО!")
    print("=" * 80)
