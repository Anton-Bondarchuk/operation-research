import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
from matplotlib.colors import LogNorm
import pandas as pd

# Define the objective function
def objective_function(x):
    x1, x2 = x
    return (7 * x1**2 + 3 * x1 * x2 + 9 * x2**2 +
            8 * np.exp(-(x1**2 / 3 - (x2 + 3)**2 / 5)) +
            9 * np.exp(-((x1 - 1)**2 / 5 + (x2 - 2)**2 / 2)))

# Define the gradient function
def gradient_function(x):
    x1, x2 = x
    
    term_A_exponent = -(x1**2 / 3 - (x2 + 3)**2 / 5)
    term_B_exponent = -((x1 - 1)**2 / 5 + (x2 - 2)**2 / 2)
    
    exp_A = np.exp(term_A_exponent)
    exp_B = np.exp(term_B_exponent)
    
    # df/dx1
    df_dx1 = (14 * x1 + 3 * x2 + 
              8 * exp_A * (-2 * x1 / 3) + 
              9 * exp_B * (-2 * (x1 - 1) / 5))
              
    # df/dx2
    df_dx2 = (3 * x1 + 18 * x2 + 
              8 * exp_A * (2 * (x2 + 3) / 5) + 
              9 * exp_B * (-(x2 - 2)))
    
    return np.array([df_dx1, df_dx2])

# Create contour plot of the objective function
def create_contour_plot(title):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function([X[i, j], Y[i, j]])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    return plt, X, Y, Z

# Function to trace optimization steps
def trace_optimization(method, x0, options=None, jac=None):
    iterations = []
    function_values = []
    points = []
    function_calls = 0
    gradient_calls = 0
    
    def callback(xk):
        nonlocal function_calls
        function_calls += 1
        iterations.append(len(iterations))
        function_values.append(objective_function(xk))
        points.append(xk.copy())
    
    # Default options if none provided
    if options is None:
        options = {'xtol': 1e-8, 'disp': True}
    
    # Perform optimization
    if jac is not None:
        result = optimize.minimize(
            objective_function, 
            x0, 
            method=method,
            jac=jac,
            callback=callback,
            options=options
        )
        gradient_calls = result.njev if hasattr(result, 'njev') else 'N/A'
    else:
        result = optimize.minimize(
            objective_function, 
            x0, 
            method=method,
            callback=callback,
            options=options
        )
    
    # Update function_calls with actual count from optimization result
    if hasattr(result, 'nfev'):
        function_calls = result.nfev
    
    return {
        'result': result,
        'iterations': iterations,
        'function_values': function_values,
        'points': np.array(points),
        'function_calls': function_calls,
        'gradient_calls': gradient_calls
    }

# Plot optimization path on contour
def plot_optimization_path(trace, title):
    plt, X, Y, Z = create_contour_plot(f"{title} Optimization Path")
    
    # Plot optimization path
    path = trace['points']
    plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=5, linewidth=2, label='Optimization Path')
    
    # Mark start and end points
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'bo', markersize=10, label='End')
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot evaluation points
def plot_evaluation_points(trace, title):
    plt, X, Y, Z = create_contour_plot(f"{title} Evaluation Points")
    
    # Plot evaluation points
    path = trace['points']
    plt.scatter(path[:, 0], path[:, 1], c='red', s=30, label='Evaluation Points')
    
    # Mark start and end points
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'bo', markersize=10, label='End')
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Display optimization stats
def display_stats(trace, method_name):
    result = trace['result']
    
    print(f"\n{method_name} Optimization Results:")
    print(f"Success: {result.success}")
    print(f"Status: {result.status}, {result.message}")
    print(f"Number of iterations: {len(trace['iterations'])}")
    print(f"Function evaluations: {trace['function_calls']}")
    
    if trace['gradient_calls'] != 'N/A':
        print(f"Gradient evaluations: {trace['gradient_calls']}")
    
    print(f"Final function value: {result.fun}")
    print(f"Solution: x1 = {result.x[0]}, x2 = {result.x[1]}")
    
    # Create a DataFrame for function call counts
    data = {'Function': ['eval_f']}
    if trace['gradient_calls'] != 'N/A':
        data['Function'].append('eval_grad_f')
    
    data['Count'] = [trace['function_calls']]
    if trace['gradient_calls'] != 'N/A':
        data['Count'].append(trace['gradient_calls'])
    
    df = pd.DataFrame(data)
    print("\nFunction Calls:")
    print(df)

# Main execution
if __name__ == "__main__":
    # Initial point
    x0 = np.array([0, 0])
    
    # 1. Nelder-Mead method
    print("\n## Method: Nelder-Mead")
    trace_nm = trace_optimization(
        method='Nelder-Mead',
        x0=x0,
        options={'xatol': 1e-8, 'maxiter': 200}
    )
    display_stats(trace_nm, "Nelder-Mead")
    plot_evaluation_points(trace_nm, "Nelder-Mead")
    plot_optimization_path(trace_nm, "Nelder-Mead")
    
    # 2. Powell method (similar to PRAXIS in R)
    print("\n## Method: Powell (similar to PRAXIS)")
    trace_powell = trace_optimization(
        method='Powell',
        x0=x0,
        options={'xtol': 1e-8, 'maxiter': 200}
    )
    display_stats(trace_powell, "Powell")
    plot_evaluation_points(trace_powell, "Powell")
    plot_optimization_path(trace_powell, "Powell")
    
    # 3. BFGS method
    print("\n## Method: BFGS")
    trace_bfgs = trace_optimization(
        method='BFGS',
        x0=x0,
        jac=gradient_function,
        options={'gtol': 1e-8}
    )
    display_stats(trace_bfgs, "BFGS")
    plot_evaluation_points(trace_bfgs, "BFGS")
    plot_optimization_path(trace_bfgs, "BFGS")
    
    # 4. Newton-CG method (similar to truncated Newton in R)
    print("\n## Method: Newton-CG")
    trace_newton = trace_optimization(
        method='Newton-CG',
        x0=x0,
        jac=gradient_function,
        options={'xtol': 1e-8}
    )
    display_stats(trace_newton, "Newton-CG")
    plot_evaluation_points(trace_newton, "Newton-CG")
    plot_optimization_path(trace_newton, "Newton-CG")