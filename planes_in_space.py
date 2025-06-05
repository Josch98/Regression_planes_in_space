import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Coeficientes del la ecuacion del plano (Ax+By+C=Z) ---
A = 1   
B = 2   
C = 3   

# --- Generar el espacio ---
x = np.linspace(-10, 10, 30)
y = np.linspace(-10, 10, 30)
x, y = np.meshgrid(x, y)

# --- Hacer el modelo z=Ax+By+C ---
z = A * x + B * y + C

# --- Dibujar el plano ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, alpha=0.7, cmap='viridis')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.set_title(f"Plane: z = {A}x + {B}y + {C}")

plt.show()
