# --- Importar librerias ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Aquí se almacenan datos para la animación ---
coefficients = []
cost_history = []

# --- Definir la función costo ---
def cost(A, B, C, m, x, y, z):
    sums = np.sum(((A*x + B*y + C)-z)**2)
    total = 1/m * sums
    return total

# --- Aplicar el descenso de gradiente y optimizar los parámetros (solo 500 iteraciones)---
def gradient_descent(A, B, C, m, x, y, z, ALPHA=0.001):
    for i in range(1000):
        dJ_dA = (2/m) * np.sum(x * ((A*x + B*y + C)-z))
        dJ_dB = (2/m) * np.sum(y * ((A*x + B*y + C)-z))
        dJ_dC = (2/m) * np.sum((A*x + B*y + C)-z)
        A = A - ALPHA*dJ_dA
        B = B - ALPHA*dJ_dB
        C = C - ALPHA*dJ_dC
        coefficients.append(np.array([A, B, C]))
        cost_history.append([cost(A, B, C, m, x, y, z), i])
        print(cost(A, B, C, m, x, y, z))
    
    return A, B, C

# --- Graficar la función costo para cada iteración del descenso de gradiente ---
def plot_cost():
    x_v = [p[1] for p in cost_history]
    y_v = [p[0] for p in cost_history]

    plt.plot(x_v, y_v, marker='o', markersize=1) 
    plt.xlabel("Número de iteración")
    plt.ylabel("Costo")
    plt.title("Gráfica del costo en función de la iteración")
    plt.grid(True)
    plt.show()
    

# --- Animar la evolución del modelo de regresión a lo largo de las iteraciones del descenso de gradiente ---
def animation(x_points, y_points, z_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    x = np.linspace(0, 15, 20)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y)

    plane = [None]  
    def update(i):
        ax.clear()

        ax.scatter(x_points, y_points, z_points, alpha=0.9, color='black')

        A, B, C = coefficients[i]
        Z = A*X + B*Y + C

        norm = plt.Normalize(Z.min(), Z.max())
        colors = plt.cm.viridis(norm(Z))

        plane[0] = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7, rstride=1, cstride=1, edgecolor='none')

        ax.set_xlim(0, 15)
        ax.set_ylim(0, 10)
        ax.set_zlim(-15, 15)
        ax.set_xlabel('Tasa de desempleo estadounidense (x)')
        ax.set_ylabel('Tasa de inflación estadounidense (y)')
        ax.set_zlabel('Cambio porcentual (z)')
        ax.set_title(f"Iteración {i}: z = {A:.2f}x + {B:.2f}y + {C:.2f}")

    ani = FuncAnimation(fig, update, frames=len(coefficients), interval=20)
    plt.show()


# --- Función principal del programa ---

def main():
    A, B, C = np.random.rand(3)
    A = A*-3
    B = B*-1

    x = [5.0, 5.1, 4.8, 4.9, 4.8, 4.9, 5.0, 4.9, 4.7, 4.7, 4.7, 4.6, 4.4, 4.4, 4.4, 4.2, 4.2, 4.4, 4.3, 4.2, 4.2, 4.1, 4.0, 4.1, 4.0, 4.0, 3.8, 4.0, 3.8, 3.8, 3.7, 3.8, 3.8, 3.9, 4.0, 3.8, 3.8, 3.7, 3.6, 3.6, 3.7, 3.6, 3.5, 3.6, 3.6, 3.6, 3.6, 3.5, 4.4, 14.8, 13.2, 11.0, 10.2, 8.4, 7.8, 6.9, 6.7, 6.7, 6.4, 6.2, 6.1, 6.1, 5.8, 5.9, 5.4, 5.1, 4.7, 4.5, 4.1, 3.9, 4.0, 3.9, 3.7, 3.7, 3.6, 3.6, 3.5, 3.6, 3.5, 3.6, 3.6, 3.5, 3.5, 3.6, 3.5, 3.5, 3.6, 3.6, 3.5, 3.7, 3.7, 3.9, 3.7, 3.8, 3.7, 3.9, 3.9, 3.9, 3.9, 4.1, 4.2, 4.2, 4.1, 4.1, 4.2, 4.1, 4.0, 4.2, 4.2, 4.2, 4.3, 4.1, 4.3, 4.3, 4.4, 4.5, 4.4, 4.3, 4.4]
    y = [0.9, 1.1, 1.0, 1.0, 0.8, 1.1, 1.5, 1.6, 1.7, 2.1, 2.5, 2.7, 2.4, 2.2, 1.9, 1.6, 1.7, 1.9, 2.2, 2.0, 2.2, 2.1, 2.1, 2.2, 2.4, 2.5, 2.8, 2.9, 2.9, 2.7, 2.3, 2.5, 2.2, 1.9, 1.6, 1.5, 1.9, 2.0, 1.8, 1.6, 1.8, 1.7, 1.7, 1.8, 2.1, 2.3, 2.5, 2.3, 1.5, 0.3, 0.1, 0.6, 1.0, 1.3, 1.4, 1.2, 1.2, 1.4, 1.4, 1.7, 2.6, 4.2, 5.0, 5.4, 5.4, 5.3, 5.4, 6.2, 6.8, 7.0, 7.5, 7.9, 8.5, 8.3, 8.6, 9.1, 8.5, 8.3, 8.2, 7.7, 7.1, 6.5, 6.4, 6.0, 5.0, 4.9, 4.1, 3.0, 3.2, 3.7, 3.7, 3.2, 3.1, 3.4, 3.1, 3.2, 3.5, 3.4, 3.3, 3.0, 2.9, 2.5, 2.4, 2.6, 2.7, 2.9, 3.0, 2.8, 2.4, 2.3, 2.4, 2.7, 2.7, 2.9, 3.0, 2.7, 2.7, 2.4, 2.4]
    z = [6.60, 0.27, 1.53, 0.09, 3.56, -0.12, -0.12, -1.94, 3.42, 1.82, 1.79, 3.72, -0.04, 0.91, 1.16, 0.48, 1.93, 0.05, 1.93, 2.22, 2.81, 0.98, 5.62, -3.89, -2.69, 0.27, 2.16, 0.48, 3.60, 3.03, 0.43, -6.94, 1.79, -9.18, 7.87, 2.97, 1.79, 3.93, -6.58, 6.89, 1.31, -1.81, 1.72, 2.04, 3.40, 2.86, -0.16, -8.41, -12.51, 12.68, 4.53, 1.84, 5.51, 7.01, -3.92, -2.77, 10.75, 3.71, -1.11, 2.61, 4.24, 5.24, 0.55, 2.22, 2.27, 2.90, -4.76, 6.91, -0.83, 4.36, -5.26, -3.14, 3.58, -8.80, 0.01, -8.39, 9.11, -4.24, -9.34, 7.99, 5.38, -5.90, 6.18, -2.61, 3.51, 1.46, 0.25, 6.47, 3.11, -1.77, -4.87, -2.20, 8.92, 4.42, 1.59, 5.17, 3.10, -4.16, 4.80, 3.47, 1.13, 2.28, 2.02, -0.99, 5.73, -2.50, 2.70, -1.42, -5.75, -0.76, 6.15, 4.96, 2.17, 1.91, 3.53, 0.13, -0.05, 1.37, -0.87]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    m = len(z)

    A, B, C = gradient_descent(A, B, C, m, x, y, z)

    plot_cost()
    animation(x, y, z)


main()
