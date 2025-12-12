"""
Візуалізація результатів ABC
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List


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
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (config_name, histories) in enumerate(histories_dict.items()):
        max_len = max(len(h) for h in histories)
        H = np.array([h + [h[-1]]*(max_len-len(h)) for h in histories])
        mean_curve = H.mean(axis=0)
        
        color = colors[idx % len(colors)]
        plt.plot(mean_curve, linewidth=2, color=color, label=config_name)
    
    plt.yscale("log")
    plt.xlabel("Cycle", fontsize=12)
    plt.ylabel("Mean best-so-far f(x)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
