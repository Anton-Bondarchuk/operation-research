import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
import pandas as pd

# Define the "Ravine" function
def ravine_function(vars):
    x, y = vars
    term1 = (9 * x + 8 * y - 5)**4
    term2 = (5 * x + 2 * y - 1)**4
    return term1 + term2

# Define the gradient of the "Ravine" function
def ravine_gradient(vars):
    x, y = vars
    
    # df/dx
    df_dx = 4 * (9 * x + 8 * y - 5)**3 * 9 + 4 * (5 * x + 2 * y - 1)**3 * 5
    
    # df/dy
    df_dy = 4 * (9 * x + 8 * y - 5)**3 * 8 + 4 * (5 * x + 2 * y - 1)**3 * 2
    
    return np.array([df_dx, df_dy])

# Create contour plot of the objective function
def create_contour_plot(title, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ravine_function([X[i, j], Y[i, j]])
    
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    return plt, X, Y, Z

# Sample points uniformly for visualization
def uni_sample(points, max_samples=100):
    n = len(points)
    if n <= max_samples:
        return points
    indices = np.linspace(0, n-1, max_samples, dtype=int)
    return [points[i] for i in indices]

# Function to trace optimization steps
def trace_optimization(method, x0, options=None, jac=None, bounds=None):
    # Set default options
    if options is None:
        if method == 'Nelder-Mead':
            options = {'xatol': 1e-8, 'maxiter': 500}
        else:
            options = {'xtol': 1e-8}
    
    # Initialize tracking variables
    history = {'x': [], 'fun': [], 'grad': []}
    function_calls = 0
    gradient_calls = 0
    
    # Wrapper functions to track calls
    def objective_wrapper(x):
        nonlocal function_calls
        function_calls += 1
        value = ravine_function(x)
        history['x'].append(x.copy())
        history['fun'].append(value)
        return value
    
    def gradient_wrapper(x):
        nonlocal gradient_calls
        gradient_calls += 1
        grad = ravine_gradient(x)
        history['grad'].append(grad.copy())
        return grad
    
    # Run optimization
    if jac is not None:
        result = optimize.minimize(
            objective_wrapper, 
            x0, 
            method=method,
            jac=gradient_wrapper,
            options=options,
            bounds=bounds
        )
    else:
        result = optimize.minimize(
            objective_wrapper, 
            x0, 
            method=method,
            options=options,
            bounds=bounds
        )
    
    # Prepare path points from history
    path_points = np.array(history['x'])
    
    # Prepare evaluation data
    eval_data = {
        'eval_f': [(x, f) for x, f in zip(history['x'], history['fun'])]
    }
    
    if jac is not None:
        eval_data['eval_grad_f'] = [(x, g) for x, g in zip(history['x'], history['grad'])]
    
    return {
        'result': result,
        'path': path_points,
        'eval': eval_data,
        'count': {
            'eval_f': function_calls,
            'eval_grad_f': gradient_calls if jac is not None else 0
        }
    }

# Plot evaluation points on contour
def plot_eval_points(plt_obj, eval_points, sample_size=100):
    # Sample evaluation points if there are too many
    sampled_points = uni_sample(eval_points, sample_size)
    
    # Extract x, y coordinates and values
    x_vals = [p[0][0] for p in sampled_points]
    y_vals = [p[0][1] for p in sampled_points]
    
    # Plot the points
    plt_obj.scatter(x_vals, y_vals, c='red', s=30, alpha=0.7, label='Evaluation Points')
    
    return plt_obj

# Plot start and stop points
def plot_start_stop_points(plt_obj, eval_points):
    if len(eval_points) == 0:
        return plt_obj
    
    # Get start point
    start_x, start_y = eval_points[0][0]
    
    # Get end point
    end_x, end_y = eval_points[-1][0]
    
    # Plot start point
    plt_obj.plot(start_x, start_y, 'go', markersize=10, label='Start Point')
    
    # Plot end point
    plt_obj.plot(end_x, end_y, 'bo', markersize=10, label='End Point')
    
    return plt_obj

# Plot optimization path
def plot_path_points(plt_obj, path_points):
    # Extract x and y coordinates
    x_vals = path_points[:, 0]
    y_vals = path_points[:, 1]
    
    # Plot the path
    plt_obj.plot(x_vals, y_vals, 'r-o', markersize=5, linewidth=2, label='Optimization Path')
    
    return plt_obj

# Main execution
if __name__ == "__main__":
    # Initial point
    x0_ravine = np.array([0, 0])
    
    # Bounds for the domain
    bounds = [(-5, 5), (-5, 5)]
    
    # 1. Nelder-Mead method
    print("\n## Method: Nelder-Mead (Ravine)")
    trace_nm_ravine = trace_optimization(
        method='Nelder-Mead',
        x0=x0_ravine,
        options={'xatol': 1e-8, 'maxiter': 500}
    )
    
    print("Optimization result (Nelder-Mead):")
    print(f"Success: {trace_nm_ravine['result'].success}")
    print(f"Status: {trace_nm_ravine['result'].status}")
    print(f"Message: {trace_nm_ravine['result'].message}")
    print(f"Function evaluations: {trace_nm_ravine['count']['eval_f']}")
    print(f"Solution: x = {trace_nm_ravine['result'].x[0]:.8f}, y = {trace_nm_ravine['result'].x[1]:.8f}")
    print(f"Function value: {trace_nm_ravine['result'].fun:.8f}")
    
    # Create contour plot for Nelder-Mead
    plt_nm, X_nm, Y_nm, Z_nm = create_contour_plot("Contour Plot of Ravine Function (Nelder-Mead)")
    
    # Plot evaluation points
    plt_nm = plot_eval_points(plt_nm, trace_nm_ravine['eval']['eval_f'])
    plt_nm = plot_start_stop_points(plt_nm, trace_nm_ravine['eval']['eval_f'])
    plt_nm.legend()
    plt_nm.savefig('nelder_mead_ravine_eval_points.png', dpi=300, bbox_inches='tight')
    plt_nm.show()
    
    # Plot optimization path
    plt_path_nm, _, _, _ = create_contour_plot("Optimization Path (Nelder-Mead)")
    plt_path_nm = plot_path_points(plt_path_nm, trace_nm_ravine['path'])
    plt_path_nm = plot_start_stop_points(plt_path_nm, trace_nm_ravine['eval']['eval_f'])
    plt_path_nm.legend()
    plt_path_nm.savefig('nelder_mead_ravine_path.png', dpi=300, bbox_inches='tight')
    plt_path_nm.show()
    
    # Display function call counts
    print("\nFunction Calls (Nelder-Mead):")
    nm_calls_df = pd.DataFrame({
        'Function': ['eval_f'],
        'Count': [trace_nm_ravine['count']['eval_f']]
    })
    print(nm_calls_df)
    
    # 2. Newton's method (Newton-CG in scipy)
    print("\n## Method: Newton (Ravine)")
    trace_newton_ravine = trace_optimization(
        method='Newton-CG',
        x0=x0_ravine,
        jac=True,
        options={'xtol': 1e-8}
    )
    
    print("Optimization result (Newton):")
    print(f"Success: {trace_newton_ravine['result'].success}")
    print(f"Status: {trace_newton_ravine['result'].status}")
    print(f"Message: {trace_newton_ravine['result'].message}")
    print(f"Function evaluations: {trace_newton_ravine['count']['eval_f']}")
    print(f"Gradient evaluations: {trace_newton_ravine['count']['eval_grad_f']}")
    print(f"Solution: x = {trace_newton_ravine['result'].x[0]:.8f}, y = {trace_newton_ravine['result'].x[1]:.8f}")
    print(f"Function value: {trace_newton_ravine['result'].fun:.8f}")
    
    # Create contour plot for Newton's method - function evaluations
    plt_newton_func, X_n, Y_n, Z_n = create_contour_plot("Function Evaluations (Newton)")
    plt_newton_func = plot_eval_points(plt_newton_func, trace_newton_ravine['eval']['eval_f'])
    plt_newton_func = plot_start_stop_points(plt_newton_func, trace_newton_ravine['eval']['eval_f'])
    plt_newton_func.legend()
    plt_newton_func.savefig('newton_ravine_func_eval.png', dpi=300, bbox_inches='tight')
    plt_newton_func.show()
    
    # Create contour plot for Newton's method - gradient evaluations
    plt_newton_grad, _, _, _ = create_contour_plot("Gradient Evaluations and Optimization Path (Newton)")
    plt_newton_grad = plot_eval_points(plt_newton_grad, trace_newton_ravine['eval']['eval_grad_f'])
    plt_newton_grad = plot_path_points(plt_newton_grad, trace_newton_ravine['path'])
    plt_newton_grad = plot_start_stop_points(plt_newton_grad, trace_newton_ravine['eval']['eval_grad_f'])
    plt_newton_grad.legend()
    plt_newton_grad.savefig('newton_ravine_grad_eval.png', dpi=300, bbox_inches='tight')
    plt_newton_grad.show()
    
    # Create contour plot for Newton's method - optimization path
    plt_newton_path, _, _, _ = create_contour_plot("Optimization Path on Target Function (Newton)")
    plt_newton_path = plot_path_points(plt_newton_path, trace_newton_ravine['path'])
    plt_newton_path = plot_start_stop_points(plt_newton_path, trace_newton_ravine['eval']['eval_f'])
    plt_newton_path.legend()
    plt_newton_path.savefig('newton_ravine_path.png', dpi=300, bbox_inches='tight')
    plt_newton_path.show()
    
    # Display function call counts
    print("\nFunction Calls (Newton):")
    newton_calls_df = pd.DataFrame({
        'Function': ['eval_f', 'eval_grad_f'],
        'Count': [
            trace_newton_ravine['count']['eval_f'],
            trace_newton_ravine['count']['eval_grad_f']
        ]
    })
    print(newton_calls_df)
    
    # 3D visualization of the ravine function
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate data for 3D plot
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = ravine_function([X[i, j], Y[i, j]])
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Function Value')
    ax.set_title('3D Surface of Ravine Function')
    
    plt.savefig('ravine_3d_surface.png', dpi=300, bbox_inches='tight')
    plt.show()