"""
Лабораторна робота №3: Artificial Bee Colony (ABC)
Функція: Griewank
Мінімальний рівень: базова реалізація ABC з 10 прогонами та графіками
"""
import numpy as np
from abc_config import ABCConfig
from abc_algorithms import abc_classic, abc_modified
from abc_experiments import (
    run_multiple_experiments, 
    print_results,
    print_comparison_table,
    analyze_results,
    experiment_coefficient_c,
    select_best_config
)
from abc_visualization import plot_convergence, plot_comparison
from visualize_trajectory import visualize_both


# --- Griewank function ---
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


if __name__ == "__main__":
    # Параметри задачі
    d = 2  # Розмірність
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація ABC
    cfg = ABCConfig(
        colony_size=30,    # Розмір колонії (кількість джерел їжі)
        limit=100,         # Ліміт спроб для scout bees
        max_cycles=500,    # Кількість циклів
        seed=42
    )
    
    print("="*80)
    print("ЛАБОРАТОРНА РОБОТА №3: ARTIFICIAL BEE COLONY (ABC)")
    print("="*80)
    print("Функція: Griewank")
    print(f"Область: x_i ∈ [-600, 600], розмірність: {d}D")
    print(f"Глобальний мінімум: f(0, 0) = 0")
    print("="*80)
    print()
    
    # Запуск 10 незалежних експериментів
    experiments = run_multiple_experiments(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        n_runs=10
    )
    
    # Виведення статистики
    print_results(experiments, d)
    
    # Візуалізація збіжності
    plot_convergence(
        experiments["histories"],
        title=f"ABC Convergence on Griewank ({d}D)"
    )
    
    print("\nМІНІМАЛЬНИЙ РІВЕНЬ ЗАВЕРШЕНО!")
    print("✓ Реалізовано класичний ABC")
    print("✓ Виконано 10 прогонів з різними seed")
    print("✓ Обчислено статистику (mean, std, min, max)")
    print("✓ Побудовано графіки збіжності")
    
    # ========================================================================
    # РІВЕНЬ "ВІДМІННО B": ПОРІВНЯННЯ КЛАСИЧНОГО ТА МОДИФІКОВАНОГО ABC
    # ========================================================================
    
    print("\n" + "="*80)
    print("РІВЕНЬ 'ВІДМІННО B': ПОРІВНЯННЯ КЛАСИЧНОГО ТА МОДИФІКОВАНОГО ABC")
    print("="*80)
    
    # Експеримент 1: Класичний ABC (вже виконано вище)
    print("\n--- КЛАСИЧНИЙ ABC ---")
    print("(Результати вже отримано)")
    
    # Експеримент 2: Модифікований ABC з різними значеннями коефіцієнта c
    print("\n" + "="*80)
    print("ЕКСПЕРИМЕНТ: ВПЛИВ КОЕФІЦІЄНТА 'c' НА МОДИФІКОВАНИЙ ABC")
    print("="*80)
    
    summaries_c, histories_c = experiment_coefficient_c(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        c_values=(0.1, 0.2, 0.3, 0.4, 0.5)
    )
    
    # Порівняльна таблиця для різних c
    print("\n" + "="*80)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ: КОЕФІЦІЄНТ 'c' В МОДИФІКОВАНОМУ ABC")
    print("="*80)
    print_comparison_table(summaries_c, "Коефіцієнт c")
    analyze_results(summaries_c, "Коефіцієнт c")
    
    # Графік порівняння для різних c (з додаванням Classic ABC)
    histories_with_classic = {
        "Classic ABC": experiments["histories"],
        **histories_c
    }
    plot_comparison(
        histories_with_classic,
        title="Modified ABC: Effect of coefficient 'c' on convergence"
    )
    
    # Експеримент 3: Порівняння класичного та найкращого модифікованого
    print("\n" + "="*80)
    print("ФІНАЛЬНЕ ПОРІВНЯННЯ: КЛАСИЧНИЙ VS МОДИФІКОВАНИЙ ABC")
    print("="*80)
    
    # Знаходимо найкраще значення c з багатокритеріальним підходом
    best_c_name, best_c_data, selection_reason = select_best_config(summaries_c)
    best_c = float(best_c_name.split('=')[1])
    
    print(f"\nНайкращий коефіцієнт c = {best_c}")
    print(f"Критерій вибору: {selection_reason}")
    print(f"Mean: {best_c_data['statistics']['mean']:.6e}")
    
    # Запускаємо модифікований ABC з найкращим c
    print(f"\n--- МОДИФІКОВАНИЙ ABC (c={best_c}) ---")
    exp_modified = run_multiple_experiments(
        f=griewank,
        bounds=bounds,
        cfg=cfg,
        n_runs=10,
        use_modified=True,
        c=best_c
    )
    
    # Порівняльна таблиця
    comparison = {
        "Classic ABC": experiments,
        f"Modified ABC (c={best_c})": exp_modified
    }
    
    print("\n" + "="*80)
    print("ПІДСУМКОВА ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print("="*80)
    print_comparison_table(comparison, "Метод")
    analyze_results(comparison, "Метод")
    
    # Порівняльний графік
    comparison_histories = {
        "Classic ABC": experiments["histories"],
        f"Modified ABC (c={best_c})": exp_modified["histories"]
    }
    plot_comparison(
        comparison_histories,
        title="Classic ABC vs Modified ABC: Convergence Comparison"
    )
    
    # Аналіз результатів
    print("\n" + "="*80)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("="*80)
    
    classic_mean = experiments["statistics"]["mean"]
    classic_std = experiments["statistics"]["std"]
    modified_mean = exp_modified["statistics"]["mean"]
    modified_std = exp_modified["statistics"]["std"]
    
    print("\n1. ШВИДКІСТЬ ЗБІЖНОСТІ:")
    if abs(modified_mean - classic_mean) < 1e-15:
        print(f"   ≈ Обидва методи показують однакові результати (обидва досягли глобального мінімуму)")
    elif modified_mean < classic_mean:
        if classic_mean > 0:
            improvement = ((classic_mean - modified_mean) / classic_mean) * 100
            print(f"   ✓ Модифікований ABC показує кращий результат на {improvement:.2f}%")
        else:
            print(f"   ✓ Модифікований ABC показує кращий результат ({modified_mean:.6e} vs {classic_mean:.6e})")
    else:
        if classic_mean > 0:
            degradation = ((modified_mean - classic_mean) / classic_mean) * 100
            print(f"   ✗ Класичний ABC показує кращий результат на {degradation:.2f}%")
        else:
            print(f"   ✗ Класичний ABC показує кращий результат ({classic_mean:.6e} vs {modified_mean:.6e})")
    
    print("\n2. СТАБІЛЬНІСТЬ РОЗВ'ЯЗКУ:")
    if abs(modified_std - classic_std) < 1e-15:
        print(f"   ≈ Обидва методи показують однакову стабільність (std ≈ 0)")
    elif modified_std < classic_std:
        print(f"   ✓ Модифікований ABC більш стабільний (std: {modified_std:.6e} vs {classic_std:.6e})")
    else:
        print(f"   ✗ Класичний ABC більш стабільний (std: {classic_std:.6e} vs {modified_std:.6e})")
    
    print("\n3. РИЗИК ЛОКАЛЬНИХ МІНІМУМІВ:")
    classic_max = experiments["statistics"]["max"]
    modified_max = exp_modified["statistics"]["max"]
    
    if abs(modified_max - classic_max) < 1e-15:
        print(f"   ≈ Обидва методи мають однаковий ризик (обидва знайшли глобальний мінімум у всіх прогонах)")
    elif modified_max < classic_max:
        print(f"   ✓ Модифікований ABC має менший ризик застрягання")
        print(f"     (worst case: {modified_max:.6e} vs {classic_max:.6e})")
    else:
        print(f"   ✗ Класичний ABC має менший ризик застрягання")
        print(f"     (worst case: {classic_max:.6e} vs {modified_max:.6e})")
    
    print("\n" + "="*80)
    print("РІВЕНЬ 'ВІДМІННО B' ЗАВЕРШЕНО!")
    print("="*80)
    print("✓ Реалізовано модифікований ABC з підтягуванням до глобального best")
    print("✓ Протестовано різні значення коефіцієнта c (0.1-0.5)")
    print("✓ Проведено порівняльний аналіз класичного та модифікованого ABC")
    print("✓ Проаналізовано швидкість збіжності, стабільність та ризик локальних мінімумів")

    # =========================================================================
    # РІВЕНЬ 'ВІДМІННО A': ВІЗУАЛІЗАЦІЯ ПОВЕРХНІ З ТРАЄКТОРІЄЮ
    # =========================================================================
    
    print("\n" + "="*80)
    print("ДОДАТКОВА ВІЗУАЛІЗАЦІЯ: ПОВЕРХНЯ ФУНКЦІЇ З ТРАЄКТОРІЄЮ")
    print("="*80)
    print("Запускаємо окремий модуль візуалізації...")
    
    # Викликаємо модуль для візуалізації обох варіантів
    visualize_both(bounds=bounds, cfg=cfg, c=0.2, save_every=10)
    
    print("\n" + "="*80)
    print("УСІХ РІВНІВ ЗАВЕРШЕНО!")
    print("="*80)
    print("\nПРИМІТКА: Для окремого запуску візуалізації використовуйте:")
    print("  python visualize_trajectory.py")

