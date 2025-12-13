"""
Візуалізація результатів ABC
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable


def plot_convergence(histories: List[List[float]], title: str = "ABC Convergence"):
    """
    Малює криві збіжності best-so-far для всіх прогонів + середню
    
    Параметри:
    - histories: список історій збіжності з кожного прогону
    - title: заголовок графіку
    """
    # Вирівнювання довжин історій
    max_len = max(len(h) for h in histories)
    H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
    mean_curve = H.mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    
    # Малюємо всі окремі прогони
    for h in histories:
        plt.plot(h, alpha=0.25, color='blue')
    
    # Малюємо середню криву
    plt.plot(mean_curve, linewidth=2.5, color='red', label="Mean best-so-far")
    
    plt.yscale("log")
    plt.xlabel("Cycle", fontsize=12)
    plt.ylabel("Best-so-far f(x)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_comparison(histories_dict: dict, title: str = "ABC Comparison"):
    """
    Порівняння кількох конфігурацій ABC на одному графіку
    
    Параметри:
    - histories_dict: словник {назва_конфігурації: список_історій}
    - title: заголовок графіку
    """
    plt.figure(figsize=(12, 6))
    
    # Визначаємо кольори для кожної конфігурації
    color_map = {
        'Classic ABC': 'blue',
        'c=0.1': 'blue',
        'c=0.2': 'red',
        'c=0.3': 'green',
        'c=0.4': 'orange',
        'c=0.5': 'purple'
    }
    default_colors = ['brown', 'pink', 'gray', 'cyan', 'magenta']
    color_idx = 0
    
    for config_name, histories in histories_dict.items():
        max_len = max(len(h) for h in histories)
        H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
        mean_curve = H.mean(axis=0)
        
        # Вибираємо колір
        if config_name in color_map:
            color = color_map[config_name]
        else:
            color = default_colors[color_idx % len(default_colors)]
            color_idx += 1
        
        # Особливе форматування для Classic ABC
        if config_name == 'Classic ABC':
            plt.plot(mean_curve, linewidth=2.5, color=color, 
                    label=config_name, linestyle='--', alpha=0.8)
        else:
            plt.plot(mean_curve, linewidth=2, color=color, label=config_name)
    
    plt.yscale("log")
    plt.xlabel("Cycle", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_function_surface_with_trajectory(
    f: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    trajectory: List[np.ndarray],
    title: str = "ABC Trajectory on Griewank Function"
):
    """
    Візуалізація поверхні/контурів функції та траєкторії найкращого розв'язку
    
    Параметри:
    - f: цільова функція
    - bounds: межі області, shape (2, 2) для 2D
    - trajectory: список позицій найкращого розв'язку по циклах
    - title: заголовок
    """
    if bounds.shape[0] != 2:
        print("Візуалізація доступна тільки для 2D функцій")
        return
    
    # Створюємо сітку для обчислення значень функції
    n_points = 200
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Обчислюємо значення функції на сітці
    Z = np.zeros_like(X1)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))
    
    # Створюємо фігуру з двома підграфіками
    fig = plt.figure(figsize=(16, 6))
    
    # Лівий графік: контури функції з траєкторією
    ax1 = fig.add_subplot(121)
    
    # Контурний графік
    levels = np.logspace(-2, 2, 20)  # логарифмічні рівні для кращої візуалізації
    contour = ax1.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contour(X1, X2, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='f(x)')
    
    # Траєкторія найкращого розв'язку
    trajectory_array = np.array(trajectory)
    ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], 
            'r-', linewidth=2, alpha=0.7, label='Best solution trajectory')
    ax1.plot(trajectory_array[0, 0], trajectory_array[0, 1], 
            'go', markersize=10, label='Start', zorder=5)
    ax1.plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 
            'r*', markersize=15, label='Final', zorder=5)
    ax1.plot(0, 0, 'y*', markersize=20, label='Global optimum (0,0)', zorder=5)
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('Contour Map with Trajectory', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Правий графік: 3D поверхня
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 3D поверхня (зменшуємо точки для швидкості)
    stride = 5
    surf = ax2.plot_surface(X1[::stride, ::stride], X2[::stride, ::stride], 
                           Z[::stride, ::stride], cmap='viridis', 
                           alpha=0.6, antialiased=True)
    
    # Траєкторія на 3D графіку
    trajectory_z = [f(pos) for pos in trajectory]
    ax2.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_z,
            'r-', linewidth=2, label='Trajectory', zorder=10)
    ax2.scatter(trajectory_array[0, 0], trajectory_array[0, 1], trajectory_z[0],
               c='green', s=100, label='Start', zorder=11)
    ax2.scatter(trajectory_array[-1, 0], trajectory_array[-1, 1], trajectory_z[-1],
               c='red', s=150, marker='*', label='Final', zorder=11)
    ax2.scatter(0, 0, f(np.array([0, 0])), c='yellow', s=200, marker='*', 
               label='Global optimum', zorder=11)
    
    ax2.set_xlabel('x₁', fontsize=10)
    ax2.set_ylabel('x₂', fontsize=10)
    ax2.set_zlabel('f(x)', fontsize=10)
    ax2.set_title('3D Surface with Trajectory', fontsize=13)
    ax2.legend(fontsize=8)
    
    plt.suptitle(title, fontsize=15, y=0.98)
    plt.tight_layout()
    plt.show()
