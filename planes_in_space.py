import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the plane equation Ax + By + Cz + D = 0
A, B, C, D = 1, 2, -1, -3  # Example values

# Create a meshgrid for X and Y
x_vals = np.linspace(-10, 10, 10)
y_vals = np.linspace(-10, 10, 10)
X, Y = np.meshgrid(x_vals, y_vals)

# Solve for Z if C is not zero
if C != 0:
    Z = (-A * X - B * Y - D) / C
else:
    raise ValueError("C cannot be zero; choose a different normal vector.")

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the plane
ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', edgecolor='k')

# Labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Plane in 3D Space")

plt.show()
