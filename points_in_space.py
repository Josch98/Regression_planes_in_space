import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Example lists of coordinates ---
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
z = [3, 1, 4, 2, 5]

# --- 3D scatter plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='red', marker='o')  # You can change color/marker

# --- Labels ---
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Scatter Plot of Points')

plt.show()
