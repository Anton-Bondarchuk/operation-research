import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function
def f(x, y):
    """
    Calculate the function value:
    7x^2 + 3xy + 9y^2 + 8exp[- ((x^2)/3 - ((y + 3)^2)/5)] + 9exp[- (((x - 1)^2)/5 + ((y - 2)^2)/2)]
    """
    term1 = 7 * x**2 + 3 * x * y + 9 * y**2
    term2 = 8 * np.exp(-(x**2 / 3 - (y + 3)**2 / 5))
    term3 = 9 * np.exp(-((x - 1)**2 / 5 + (y - 2)**2 / 2))
    
    return term1 + term2 + term3

# Create a grid of x and y values
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                       linewidth=0, antialiased=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x,y)')
ax.set_title('3D Surface Plot of the Function')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Function Value')
plt.savefig('surface_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
plt.colorbar(label='Function Value')
plt.title('Contour Plot of the Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('contour_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a zoomed-in contour plot to show more detail around the minimum
# Determine reasonable bounds by focusing on areas with lower values
x_zoom = np.linspace(-2, 2, 200)
y_zoom = np.linspace(-1, 3, 200)
X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
Z_zoom = f(X_zoom, Y_zoom)

plt.figure(figsize=(10, 8))
contour_zoom = plt.contourf(X_zoom, Y_zoom, Z_zoom, 50, cmap='viridis')
plt.colorbar(label='Function Value')
plt.title('Zoomed Contour Plot (Detail View)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.7)

# Add contour lines for better visibility
plt.contour(X_zoom, Y_zoom, Z_zoom, 20, colors='black', alpha=0.5, linewidths=0.5)
plt.savefig('contour_plot_zoomed.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate the minimum value and its location
def find_minimum():
    min_val = np.min(Z)
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    min_x = X[min_idx]
    min_y = Y[min_idx]
    return min_x, min_y, min_val

min_x, min_y, min_val = find_minimum()
print(f"Approximate minimum location: x = {min_x}, y = {min_y}")
print(f"Approximate minimum value: {min_val}")