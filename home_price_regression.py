import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- Aquí se almacenan datos para la animación ---
coefficients = []
cost_history = []

# --- Definir la función costo ---
def cost(A, B, C, n, x, y, z):
    sums = np.sum((z - (A*x + B*y + C))**2)
    total = 1/n * sums
    return total

# --- Aplicar el descenso de gradiente y optimizar los parámetros ---
def gradient_descent(A, B, C, n, x, y, z, ALPHA=0.000005):
    for i in range(500):
        dJ_dA = (-2/n) * np.sum(x * (z - (A*x + B*y + C)))
        dJ_dB = (-2/n) * np.sum(y * (z - (A*x + B*y + C)))
        dJ_dC = (-2/n) * np.sum(z - (A*x + B*y + C))
        A = A - ALPHA*dJ_dA
        B = B - ALPHA*dJ_dB
        C = C - ALPHA*dJ_dC
        coefficients.append(np.array([A, B, C]))
        cost_history.append([cost(A, B, C, n, x, y, z), i])
        print(cost(A, B, C, n, x, y, z))
    
    return A, B, C

# --- Dibujar la función costo para cada paso del descenso de gradiente ---
def plot_cost():
    x_v = [p[1] for p in cost_history]
    y_v = [p[0] for p in cost_history]

    plt.plot(x_v, y_v, marker='o', markersize=1) 
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Costo en función de la iteración del descenso de gradiente")
    plt.grid(True)
    plt.show()
    

# --- Animar la evolución del modelo de regresión ---
def animation(x_points, y_points, z_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    x_min, x_max = np.min(x_points), np.max(x_points)
    y_min, y_max = np.min(y_points), np.max(y_points)
    z_min, z_max = np.min(z_points), np.max(z_points)

 
    x_plot_range = np.linspace(0, 75, 10)
    y_plot_range = np.linspace(0, 10, 10)  
    z_plot_range = np.linspace(0, 15, 10)  


    X, Y = np.meshgrid(x_plot_range, y_plot_range)

    plane = [None]  
    def update(i):
        ax.clear()

        ax.scatter(x_points, y_points, z_points, alpha=0.9, color='black')

        A, B, C = coefficients[i]
        Z = A*X + B*Y + C

        norm = plt.Normalize(Z.min(), Z.max())
        colors = plt.cm.viridis(norm(Z))

        plane[0] = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7, rstride=1, cstride=1, edgecolor='none')

        ax.set_xlim(np.min(x_plot_range), np.max(x_plot_range))
        ax.set_ylim(np.min(y_plot_range), np.max(y_plot_range))
        ax.set_zlim(np.min(z_plot_range), np.max(z_plot_range))
        
        ax.set_xlabel("Tamaño de la casa")
        ax.set_ylabel("Número de habitaciones")
        ax.set_zlabel("Precio")
        ax.set_title(f"Step {i}: z = {A:.2f}x + {B:.2f}y + {C:.2f}")

    ani = FuncAnimation(fig, update, frames=len(coefficients), interval=20)
    plt.show()


# --- Función principal del programa ---

def main():
    A, B, C = np.random.rand(3)
    B = B*-2
    C = C+4
    x = [18.3, 30, 17, 25, 3.7, 4, 15, 52.2, 65, 70]
    y = [3, 3, 3, 3, 1, 1, 4, 4, 6, 5]
    z = [2.25, 4.85, 1.56, 2.31, 1.35, 1.29, 1.37, 4.3, 6.8, 10.3]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    n = len(z)

    print(A, B, C)
    print(cost(A, B, C, n, x, y, z))

    A, B, C = gradient_descent(A, B, C, n, x, y, z)
    print(A, B, C)
    print(cost(A, B, C, n, x, y, z))

    plot_cost()
    animation(x, y, z)


main()
