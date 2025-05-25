import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# Define the objective function (assuming a simple quadratic function)
def f(x):
    return x[0]**2 + x[1]**2

# Define the gradient of the objective function
def f_grad(x):
    return np.array([2*x[0], 2*x[1]])

# Define the inequality constraints: g(x) <= 0
def constr_ineq(x):
    g1 = (x[0]**2)/10 + (x[1]**2)/3 - (x[0]*x[1])/5 - 6
    g2 = (x[0]**2)/3 + (x[1]**2)/104 + (x[0]*x[1])/5 - 3
    return np.array([g1, g2])

# Define the Jacobian of the constraints
def constr_ineq_jac(x):
    # Jacobian of g1
    dg1_dx1 = x[0]/5 - x[1]/5
    dg1_dx2 = 2*x[1]/3 - x[0]/5
    
    # Jacobian of g2
    dg2_dx1 = 2*x[0]/3 + x[1]/5
    dg2_dx2 = x[1]/52 + x[0]/5
    
    return np.array([[dg1_dx1, dg1_dx2], [dg2_dx1, dg2_dx2]])

# Define the penalty term
def penalty_term(x, constr_func):
    g_values = constr_func(x)
    penalties = np.sum(np.maximum(0, g_values)**2)
    return penalties

# Define the gradient of the penalty term
def grad_penalty_term(x, constr_func, constr_jac_func):
    g_values = constr_func(x)
    jac_g_values = constr_jac_func(x)
    
    grad_pen = np.zeros(len(x))
    
    for j in range(len(g_values)):
        if g_values[j] > 0:
            grad_pen += 2 * g_values[j] * jac_g_values[j]
    
    return grad_pen

# Define the penalized objective function
def f_penalty(x, rho, base_f, constr_func):
    return base_f(x) + rho * penalty_term(x, constr_func)

# Define the gradient of the penalized objective function
def f_grad_penalty(x, rho, base_f_grad, constr_func, constr_jac_func):
    return base_f_grad(x) + rho * grad_penalty_term(x, constr_func, constr_jac_func)

# Initial parameters for the penalty method
rho_initial = 1
rho_multiplier = 10
max_penalty_iter = 5
x0 = np.array([0, 0])  # Initial guess

# Initialize lists to store results
penalty_solutions = []
penalty_objectives = []
all_paths_penalty = []

# Run the penalty method
current_rho = rho_initial
current_x0 = x0.copy()

for iter in range(1, max_penalty_iter + 1):
    print(f"\nПенальти метод итерация: {iter} с rho = {current_rho}")
    
    # Define the current penalized function and its gradient
    def current_f_penalty(x):
        return f_penalty(x, current_rho, f, constr_ineq)
    
    def current_f_grad_penalty(x):
        return f_grad_penalty(x, current_rho, f_grad, constr_ineq, constr_ineq_jac)
    
    # Solve the unconstrained optimization problem
    result = minimize(
        current_f_penalty,
        current_x0,
        method='L-BFGS-B',
        jac=current_f_grad_penalty,
        options={'gtol': 1e-6}
    )
    
    current_x0 = result.x
    penalty_solutions.append(current_x0)
    
    # Store the results
    penalty_objectives.append({
        'penalized_obj': result.fun,
        'original_obj': f(current_x0),
        'constraints': constr_ineq(current_x0)
    })
    
    print(f"Решение на итерации: {current_x0}")
    print(f"Значение исходной функции: {f(current_x0)}")
    print(f"Значения ограничений: {constr_ineq(current_x0)}")
    
    all_paths_penalty.append(current_x0)
    
    # Increase the penalty coefficient
    current_rho *= rho_multiplier

# Final solution
final_solution_penalty = current_x0
final_objective_penalty = f(final_solution_penalty)
final_constraints_penalty = constr_ineq(final_solution_penalty)

print("\n--- Итоговое решение методом штрафных функций ---")
print(f"Решение x: {final_solution_penalty}")
print(f"Значение целевой функции f(x): {final_objective_penalty}")
print(f"Значения ограничений g(x): {final_constraints_penalty}")

# Create a results table
results_df = pd.DataFrame({
    'Iteration': range(1, max_penalty_iter + 1),
    'x1': [sol[0] for sol in penalty_solutions],
    'x2': [sol[1] for sol in penalty_solutions],
    'Original_Objective': [obj['original_obj'] for obj in penalty_objectives],
    'Constraint1_g1': [obj['constraints'][0] for obj in penalty_objectives],
    'Constraint2_g2': [obj['constraints'][1] for obj in penalty_objectives],
    'Penalized_Objective': [obj['penalized_obj'] for obj in penalty_objectives]
})

print("\nРезультаты итераций метода штрафных функций:")
print(results_df.round(6))

# Create a contour plot of the objective function
def create_contour_plot(title, xlim=(-10, 10), ylim=(-10, 10), resolution=100):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Function Value')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    return plt

# Add constraint contours to the plot
def add_constraint_contours(plt):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # First constraint: (x^2)/10 + (y^2)/3 - (xy)/5 <= 6
    Z1 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = (X[i, j]**2)/10 + (Y[i, j]**2)/3 - (X[i, j]*Y[i, j])/5 - 6
    
    # Second constraint: (x^2)/3 + (y^2)/104 + (xy)/5 <= 3
    Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z2[i, j] = (X[i, j]**2)/3 + (Y[i, j]**2)/104 + (X[i, j]*Y[i, j])/5 - 3
    
    # Plot the constraint boundaries (where g(x) = 0)
    plt.contour(X, Y, Z1, levels=[0], colors='red', linewidths=2, linestyles='dashed')
    plt.contour(X, Y, Z2, levels=[0], colors='blue', linewidths=2, linestyles='dashed')
    
    # Add legends for constraints
    plt.plot([], [], color='red', linestyle='dashed', linewidth=2, label='Constraint 1')
    plt.plot([], [], color='blue', linestyle='dashed', linewidth=2, label='Constraint 2')
    
    return plt

# Create the visualization
plt_penalty = create_contour_plot("Путь метода штрафных функций")
plt_penalty = add_constraint_contours(plt_penalty)

# Plot the optimization path
path_array = np.array(penalty_solutions)
plt_penalty.plot(path_array[:, 0], path_array[:, 1], 'r-o', markersize=8, linewidth=2, label='Optimization Path')

# Mark start and end points
plt_penalty.plot(x0[0], x0[1], 'go', markersize=10, label='Start Point')
plt_penalty.plot(final_solution_penalty[0], final_solution_penalty[1], 'bo', markersize=10, label='Final Solution')

plt_penalty.legend()
plt.savefig('penalty_method_path.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a 3D surface plot to better visualize the objective function and constraints
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Generate data for the 3D plot
x = np.linspace(-6, 6, 50)
y = np.linspace(-6, 6, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f([X[i, j], Y[i, j]])

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

# Plot the optimization path in 3D
path_array = np.array(penalty_solutions)
path_z = np.array([f(point) for point in penalty_solutions])
ax.plot(path_array[:, 0], path_array[:, 1], path_z, 'r-o', markersize=8, linewidth=2, label='Optimization Path')

# Mark start and end points
ax.scatter([x0[0]], [x0[1]], [f(x0)], color='green', s=100, label='Start Point')
ax.scatter([final_solution_penalty[0]], [final_solution_penalty[1]], [f(final_solution_penalty)], color='blue', s=100, label='Final Solution')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('3D Visualization of Penalty Method Optimization')

# Add a colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.legend()

plt.savefig('penalty_method_3d.png', dpi=300, bbox_inches='tight')
plt.show()