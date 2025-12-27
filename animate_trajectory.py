"""
Модуль для анімації траєкторії ABC на функції Griewank
Створює анімацію руху найкращого розв'язку по поверхні функції
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from abc_config import ABCConfig
from abc_algorithms import abc_with_trajectory


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


def create_animation(
    bounds: np.ndarray,
    trajectory: list,
    title: str = "ABC Animation",
    interval: int = 100,
    save_path: str = None
):
    """
    Створює анімацію траєкторії на контурній карті
    
    Параметри:
    - bounds: межі області пошуку
    - trajectory: список позицій найкращого розв'язку
    - title: заголовок анімації
    - interval: інтервал між кадрами в мілісекундах
    - save_path: шлях для збереження GIF (якщо None, то тільки показує)
    """
    if bounds.shape[0] != 2:
        print("Анімація доступна тільки для 2D функцій")
        return
    
    # Створюємо сітку для контурів
    n_points = 200
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Обчислюємо значення функції
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = griewank(np.array([X1[i, j], X2[i, j]]))
    
    # Створюємо фігуру
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Контурний графік
    levels = np.logspace(-2, 2, 20)
    contourf = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(X1, X2, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contourf, ax=ax, label='f(x)')
    
    # Глобальний оптимум
    ax.plot(0, 0, 'y*', markersize=20, label='Global optimum (0,0)', zorder=5)
    
    # Елементи для анімації
    trajectory_array = np.array(trajectory)
    line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Trajectory')
    point, = ax.plot([], [], 'go', markersize=10, label='Current position', zorder=6)
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Текст з інформацією про поточний крок
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Ініціалізація анімації"""
        line.set_data([], [])
        point.set_data([], [])
        info_text.set_text('')
        return line, point, info_text
    
    def animate(frame):
        """Оновлення кадру анімації"""
        # Траєкторія до поточного кадру
        x_data = trajectory_array[:frame+1, 0]
        y_data = trajectory_array[:frame+1, 1]
        line.set_data(x_data, y_data)
        
        # Поточна позиція
        point.set_data([trajectory_array[frame, 0]], [trajectory_array[frame, 1]])
        
        # Інформаційний текст
        current_value = griewank(trajectory_array[frame])
        info_text.set_text(f'Step: {frame}/{len(trajectory)-1}\n'
                          f'Position: [{trajectory_array[frame, 0]:.2f}, {trajectory_array[frame, 1]:.2f}]\n'
                          f'Value: {current_value:.6e}')
        
        return line, point, info_text
    
    # Створюємо анімацію
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(trajectory), interval=interval,
                        blit=True, repeat=True)
    
    # Зберігаємо або показуємо
    if save_path:
        print(f"Збереження анімації в {save_path}...")
        writer = PillowWriter(fps=1000//interval)
        anim.save(save_path, writer=writer)
        print(f"Анімацію збережено!")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def animate_classic_abc(
    bounds: np.ndarray,
    cfg: ABCConfig,
    save_every: int = 20,
    interval: int = 100,
    save_gif: bool = False
):
    """
    Анімація класичного ABC
    
    Параметри:
    - bounds: межі області пошуку
    - cfg: конфігурація ABC
    - save_every: зберігати позицію кожні N циклів
    - interval: інтервал між кадрами (мс)
    - save_gif: зберігати GIF файл
    """
    print("=" * 80)
    print("АНІМАЦІЯ: КЛАСИЧНИЙ ABC")
    print("=" * 80)
    
    print("\nЗапускаємо класичний ABC...")
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
    print(f"  Збережено {len(result['trajectory'])} кадрів")
    
    save_path = "abc_classic_animation.gif" if save_gif else None
    
    print("\nСтворюємо анімацію...")
    anim = create_animation(
        bounds=bounds,
        trajectory=result['trajectory'],
        title="Анімація класичного ABC на функції Griewank",
        interval=interval,
        save_path=save_path
    )
    
    return result, anim


def animate_modified_abc(
    bounds: np.ndarray,
    cfg: ABCConfig,
    c: float = 0.2,
    save_every: int = 5,
    interval: int = 100,
    save_gif: bool = False
):
    """
    Анімація модифікованого ABC
    
    Параметри:
    - bounds: межі області пошуку
    - cfg: конфігурація ABC
    - c: коефіцієнт підтягування до глобального best
    - save_every: зберігати позицію кожні N циклів
    - interval: інтервал між кадрами (мс)
    - save_gif: зберігати GIF файл
    """
    print("=" * 80)
    print(f"АНІМАЦІЯ: МОДИФІКОВАНИЙ ABC (c={c})")
    print("=" * 80)
    
    print(f"\nЗапускаємо модифікований ABC (c={c})...")
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
    print(f"  Збережено {len(result['trajectory'])} кадрів")
    
    save_path = f"abc_modified_c{c}_animation.gif" if save_gif else None
    
    print("\nСтворюємо анімацію...")
    anim = create_animation(
        bounds=bounds,
        trajectory=result['trajectory'],
        title=f"Анімація модифікованого ABC (c={c}) на функції Griewank",
        interval=interval,
        save_path=save_path
    )
    
    return result, anim


if __name__ == "__main__":
    # Параметри задачі
    d = 2
    bounds = np.array([[-600, 600]] * d)
    
    # Конфігурація ABC
    cfg = ABCConfig(
        colony_size=30,
        limit=100,
        max_cycles=500,
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("МОДУЛЬ АНІМАЦІЇ ТРАЄКТОРІЙ ABC")
    print("=" * 80)
    print("\nОберіть режим анімації:")
    print("1 - Класичний ABC")
    print("2 - Модифікований ABC (c=0.2)")
    print("3 - Порівняння: спочатку класичний, потім модифікований")
    
    choice = input("\nВаш вибір (1/2/3): ").strip()
    
    save_gif_input = input("Зберегти як GIF? (y/n, default=n): ").strip().lower()
    save_gif = save_gif_input == 'y'
    
    interval_input = input("Інтервал між кадрами в мс (default=200): ").strip()
    interval = int(interval_input) if interval_input else 200
    
    save_every_input = input("Зберігати кожні N циклів (default=20): ").strip()
    save_every = int(save_every_input) if save_every_input else 20
    
    if choice == "1":
        animate_classic_abc(bounds, cfg, save_every, interval, save_gif)
    elif choice == "2":
        c_value = input("Введіть значення c (default=0.2): ").strip()
        c = float(c_value) if c_value else 0.2
        animate_modified_abc(bounds, cfg, c, save_every, interval, save_gif)
    elif choice == "3":
        print("\n" + "=" * 80)
        print("ПОРІВНЯЛЬНА АНІМАЦІЯ")
        print("=" * 80)
        
        # Класичний ABC
        animate_classic_abc(bounds, cfg, save_every, interval, save_gif)
        
        print("\n\nНатисніть Enter для продовження до модифікованого ABC...")
        input()
        
        # Модифікований ABC
        animate_modified_abc(bounds, cfg, 0.2, save_every, interval, save_gif)
    else:
        print("Невірний вибір. Показуємо класичний ABC...")
        animate_classic_abc(bounds, cfg, save_every, interval, save_gif)
    
    print("\n" + "=" * 80)
    print("АНІМАЦІЮ ЗАВЕРШЕНО!")
    print("=" * 80)
