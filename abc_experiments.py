"""
Функції для проведення експериментів з ABC
"""
import numpy as np
import time
from typing import Callable, Dict, List
from abc_config import ABCConfig
from abc_algorithms import abc_classic, abc_modified


def run_multiple_experiments(f: Callable[[np.ndarray], float],
                            bounds: np.ndarray,
                            cfg: ABCConfig,
                            n_runs: int = 10,
                            use_modified: bool = False,
                            c: float = 0.3) -> Dict:
    """
    Запускає ABC n_runs razів з різними випадковими seed
    
    Параметри:
    - use_modified: якщо True, використовує abc_modified(), інакше abc_classic()
    - c: коефіцієнт підтягування для modified ABC
    
    Повертає:
    - dict з результатами всіх прогонів та статистикою
    """
    results = []
    histories = []
    times = []
    evaluations_list = []
    seeds_used = []
    
    algo_name = f"Modified ABC (c={c})" if use_modified else "Classic ABC"
    print(f"Запуск {n_runs} експериментів {algo_name}...")
    print(f"Параметри: colony_size={cfg.colony_size}, limit={cfg.limit}, max_cycles={cfg.max_cycles}")
    print()
    
    # Генеруємо випадкові seeds для кожного прогону
    np.random.seed(cfg.seed)
    random_seeds = np.random.randint(0, 100000, size=n_runs)
    
    for run in range(n_runs):
        # Використовуємо випадковий seed для кожного прогону
        current_seed = int(random_seeds[run])
        seeds_used.append(current_seed)
        
        cfg_run = ABCConfig(
            colony_size=cfg.colony_size,
            limit=cfg.limit,
            max_cycles=cfg.max_cycles,
            seed=current_seed
        )
        
        start_time = time.time()
        if use_modified:
            result = abc_modified(f, bounds, cfg_run, c=c)
        else:
            result = abc_classic(f, bounds, cfg_run)
        elapsed_time = time.time() - start_time
        
        results.append(result['best_fitness'])
        histories.append(result['history'])
        times.append(elapsed_time)
        evaluations_list.append(result['evaluations'])
        
        print(f"Run {run+1:2d}/{n_runs} [seed={current_seed:5d}]: f = {result['best_fitness']:.6e}, "
              f"evaluations = {result['evaluations']}, time = {elapsed_time:.3f}s")
    
    # Статистика
    results_array = np.array(results)
    statistics = {
        'mean': np.mean(results_array),
        'std': np.std(results_array),
        'min': np.min(results_array),
        'max': np.max(results_array),
        'median': np.median(results_array),
        'mean_time': np.mean(times),
        'mean_evaluations': np.mean(evaluations_list)
    }
    
    return {
        'results': results,
        'histories': histories,
        'times': times,
        'evaluations': evaluations_list,
        'seeds': seeds_used,
        'statistics': statistics
    }


def print_results(experiments: Dict, dimension: int):
    """
    Виводить статистику результатів експериментів
    """
    stats = experiments['statistics']
    
    print("\n" + "="*80)
    print("СТАТИСТИКА РЕЗУЛЬТАТІВ")
    print("="*80)
    print(f"Розмірність задачі:     {dimension}D")
    print(f"Кількість прогонів:     {len(experiments['results'])}")
    print()
    print(f"Найкраще значення:      {stats['min']:.6e}")
    print(f"Найгірше значення:      {stats['max']:.6e}")
    print(f"Середнє значення:       {stats['mean']:.6e}")
    print(f"Стандартне відхилення:  {stats['std']:.6e}")
    print(f"Медіана:                {stats['median']:.6e}")
    print()
    print(f"Середній час виконання: {stats['mean_time']:.3f} с")
    print(f"Середня к-ть обчислень: {stats['mean_evaluations']:.0f}")
    print("="*80)


def print_comparison_table(summaries: Dict, parameter_name: str):
    """
    Виводить порівняльну таблицю для різних конфігурацій
    
    Параметри:
    - summaries: dict вигляду {config_value: experiment_results}
    - parameter_name: назва параметра що варіюється
    """
    print(f"\n{'='*100}")
    print(f"ПОРІВНЯЛЬНА ТАБЛИЦЯ: {parameter_name}")
    print(f"{'='*100}")
    print(f"{'Параметр':<20} {'Середнє':<15} {'Std':<15} {'Min':<15} {'Max':<15} {'Час (с)':<10}")
    print(f"{'-'*100}")
    
    for config_value, exp_data in summaries.items():
        stats = exp_data["statistics"]
        print(f"{str(config_value):<20} "
              f"{stats['mean']:<15.6e} "
              f"{stats['std']:<15.6e} "
              f"{stats['min']:<15.6e} "
              f"{stats['max']:<15.6e} "
              f"{stats['mean_time']:<10.3f}")
    print(f"{'='*100}\n")


def analyze_results(summaries: Dict, parameter_name: str):
    """
    Аналізує результати та виводить висновки
    """
    print(f"\nАНАЛІЗ: {parameter_name}")
    print("-" * 80)
    
    # Знаходимо найкращу конфігурацію за середнім значенням
    best_config = min(summaries.items(), 
                     key=lambda x: x[1]["statistics"]["mean"])
    
    # Знаходимо найстабільнішу конфігурацію (найменше std)
    most_stable = min(summaries.items(),
                     key=lambda x: x[1]["statistics"]["std"])
    
    print(f"Найкраще середнє значення: {parameter_name} = {best_config[0]}")
    print(f"  Mean: {best_config[1]['statistics']['mean']:.6e}")
    print(f"  Std:  {best_config[1]['statistics']['std']:.6e}")
    print()
    print(f"Найстабільніша конфігурація: {parameter_name} = {most_stable[0]}")
    print(f"  Std:  {most_stable[1]['statistics']['std']:.6e}")
    print(f"  Mean: {most_stable[1]['statistics']['mean']:.6e}")
    print("-" * 80)


def calculate_cycles_to_threshold(history: List[float], threshold: float = 1e-10) -> int:
    """
    Підраховує кількість циклів, потрібних для досягнення порогового значення
    
    Параметри:
    - history: список значень функції на кожній ітерації
    - threshold: порогове значення
    
    Повертає:
    - кількість циклів до досягнення порогу (або len(history), якщо не досягнуто)
    """
    for i, value in enumerate(history):
        if value < threshold:
            return i
    return len(history)


def select_best_config(summaries: Dict, threshold: float = 1e-10) -> tuple:
    """
    Вибирає найкращу конфігурацію за багатокритеріальним підходом:
    1. Якщо різниця в mean < 1e-15 (всі досягли глобального мінімуму),
       то порівнюємо за швидкістю збіжності
    2. Якщо швидкість однакова, порівнюємо за стабільністю (std)
    3. Інакше використовуємо mean
    
    Параметри:
    - summaries: dict {config_name: experiment_results}
    - threshold: поріг для визначення досягнення мінімуму
    
    Повертає:
    - (config_name, experiment_results, selection_reason)
    """
    # Перевіряємо, чи всі mean значення практично однакові (близькі до 0)
    means = {k: v["statistics"]["mean"] for k, v in summaries.items()}
    mean_values = list(means.values())
    
    # Якщо всі mean дуже малі та приблизно рівні
    if all(m < 1e-10 for m in mean_values) and (max(mean_values) - min(mean_values)) < 1e-15:
        print("\nВСІ КОНФІГУРАЦІЇ ДОСЯГЛИ ГЛОБАЛЬНОГО МІНІМУМУ!")
        print("Вибір за швидкістю збіжності (кількість циклів до порогу на усередненій кривій)...")
        
        # Підраховуємо цикли до порогу на УСЕРЕДНЕНІЙ кривій для кожної конфігурації
        cycles_to_threshold = {}
        for config_name, exp_data in summaries.items():
            # Обчислюємо усереднену криву збіжності
            histories = exp_data["histories"]
            max_len = max(len(h) for h in histories)
            H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
            mean_curve = H.mean(axis=0)
            
            # Рахуємо цикли до порогу на усередненій кривій
            cycles = calculate_cycles_to_threshold(mean_curve.tolist(), threshold)
            cycles_to_threshold[config_name] = cycles
            print(f"  {config_name}: {cycles} циклів до порогу 1e-10")
        
        # Вибираємо конфігурацію з найменшою кількістю циклів
        best_config_name = min(cycles_to_threshold.items(), key=lambda x: x[1])[0]
        
        # Перевіряємо, чи є інші з такою ж швидкістю
        best_cycles = cycles_to_threshold[best_config_name]
        tied_configs = [k for k, v in cycles_to_threshold.items() if abs(v - best_cycles) < 5]
        
        if len(tied_configs) > 1:
            print(f"\nКонфігурації {tied_configs} мають близьку швидкість збіжності.")
            print("Вибір за стабільністю (std)...")
            best_config_name = min(tied_configs, 
                                  key=lambda k: summaries[k]["statistics"]["std"])
            reason = f"convergence_speed={best_cycles} cycles (tied, selected by stability)"
        else:
            reason = f"convergence_speed={best_cycles} cycles"
        
        return (best_config_name, summaries[best_config_name], reason)
    
    else:
        # Стандартний вибір за найменшим mean
        best_config_name = min(summaries.items(), 
                              key=lambda x: x[1]["statistics"]["mean"])[0]
        return (best_config_name, summaries[best_config_name], "best_mean")


def experiment_coefficient_c(f: Callable[[np.ndarray], float],
                             bounds: np.ndarray,
                             cfg: ABCConfig,
                             c_values: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)) -> tuple:
    """
    Експеримент: вплив коефіцієнта c на модифікований ABC
    
    Повертає:
    - summaries: dict {c_value: experiment_results}
    - histories: dict {c_value: list_of_histories}
    """
    summaries = {}
    histories_dict = {}
    
    for c in c_values:
        print(f"\n{'='*80}")
        print(f"Тестування Modified ABC з c = {c}")
        print(f"{'='*80}")
        
        exp = run_multiple_experiments(
            f=f,
            bounds=bounds,
            cfg=cfg,
            n_runs=10,
            use_modified=True,
            c=c
        )
        
        summaries[f"c={c}"] = exp
        histories_dict[f"c={c}"] = exp["histories"]
    
    return summaries, histories_dict

