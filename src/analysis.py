import numpy as np
import matplotlib.pyplot as plt
from preprocessing.dataset import load_data
from instances import get_cost_distribution

def plot_density(filename, titulo="Distribución de Densidad", color="skyblue"):
    data = load_data(filename)
    data = data['C']

    plt.figure(figsize=(10, 6))
    
    # 1. Definir bins que rodeen a cada número entero
    # Esto elimina los huecos artificiales
    bins = np.arange(int(min(data)), int(max(data)) + 2) - 0.5
    
    # 2. Crear el histograma con los bins correctos
    count, bins_hist, ignored = plt.hist(data, bins=bins, density=True, 
                                        alpha=0.6, color=color, edgecolor='white', 
                                        label='Histograma (Normalizado)')
    
    # 3. Para la línea roja, usamos un rango continuo (smooth) en lugar de los bins
    # Esto hace que la curva se vea perfecta y no "poligonal"
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (x - mu))**2))
    
    plt.plot(x, y, color='red', linewidth=2, label=rf'Normal ($\mu$={mu:.2f}, $\sigma$={sigma:.2f})')
    
    # Estética
    plt.title(titulo, fontsize=15)
    plt.xlabel('Número de pasos (costo)', fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_counts(filename, titulo="Distribución de Frecuencias", color="skyblue"):
    data = load_data(filename)
    data = data['C']

    plt.figure(figsize=(10, 6))
    
    # 1. Definir bins centrados en los enteros
    # El uso de - 0.5 asegura que el número entero quede en el centro de la barra
    bins = np.arange(int(min(data)), int(max(data)) + 2) - 0.5
    
    # 2. Crear el histograma con frecuencia real
    # density=False (valor por defecto) muestra el conteo exacto en el eje Y
    plt.hist(data, bins=bins, density=False, 
             alpha=0.7, color=color, edgecolor='white', 
             label='Frecuencia absoluta')
    
    # Estética
    plt.title(titulo, fontsize=15)
    plt.xlabel('Número de pasos (costo)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    
    # Opcional: Mostrar una rejilla para facilitar la lectura de cantidades
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Ajustar los ticks del eje X para que coincidan con los enteros si el rango no es excesivo
    if (max(data) - min(data)) < 30:
        plt.xticks(np.arange(int(min(data)), int(max(data)) + 1))

    plt.legend()
    plt.show()

def plot_cost_distribution(folder, H):
    data = get_cost_distribution(folder, H)
    plot(data)

def plot(data, titulo="Distribución de Frecuencias", color="skyblue"):
    plt.figure(figsize=(10, 6))
    
    # 1. Definir bins centrados en los enteros
    # El uso de - 0.5 asegura que el número entero quede en el centro de la barra
    bins = np.arange(int(min(data)), int(max(data)) + 2) - 0.5
    
    # 2. Crear el histograma con frecuencia real
    # density=False (valor por defecto) muestra el conteo exacto en el eje Y
    plt.hist(data, bins=bins, density=False, 
             alpha=0.7, color=color, edgecolor='white', 
             label='Frecuencia absoluta')
    
    # Estética
    plt.title(titulo, fontsize=15)
    plt.xlabel('Número de pasos (costo)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    
    # Opcional: Mostrar una rejilla para facilitar la lectura de cantidades
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Ajustar los ticks del eje X para que coincidan con los enteros si el rango no es excesivo
    if (max(data) - min(data)) < 30:
        plt.xticks(np.arange(int(min(data)), int(max(data)) + 1))

    plt.legend()
    plt.show()