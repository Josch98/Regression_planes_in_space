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
def gradient_descent(A, B, C, n, x, y, z, ALPHA=0.001):
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
    plt.title("Plot of [y, x] points")
    plt.grid(True)
    plt.show()
    

# --- Animar la evolución del modelo de regresión ---
def animation(x_points, y_points, z_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
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

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
        ax.set_title(f"Step {i}: z = {A:.2f}x + {B:.2f}y + {C:.2f}")

    ani = FuncAnimation(fig, update, frames=len(coefficients), interval=20)
    plt.show()


# --- Función principal del programa ---

def main():
    # z(x, y) = Ax + By + C PLANE EQUATION
    A, B, C = np.random.rand(3)
    A = A*-3
    B = B*-1
    x = [1, 1.2, 1.6, 2, 2.3, 2.8, 3, 3.5, 4, 4.4, 5, 5.2, 5.5, 6]
    y = [1, 3, 7, 3, 6, 4, 5, 5, 6, 5, 6, 3, 5, 3]
    z = [1, 5, 6, 4, 4, 3, 5, 4, 8, 8, 7, 4, 7, 6]

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
