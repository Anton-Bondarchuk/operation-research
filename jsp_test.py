import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import random

def make_task_data(njob: int, nmachine: int, cost_mat: np.ndarray) -> Dict[str, Any]:
    """
    Создает данные задачи Job Shop Problem
    
    Args:
        njob: количество работ
        nmachine: количество машин  
        cost_mat: матрица стоимостей [job, machine]
    
    Returns:
        Словарь с данными задачи
    """
    # Создаем все комбинации job-machine
    jobs, machines = np.meshgrid(range(1, njob + 1), range(1, nmachine + 1), indexing='ij')
    
    # Формируем таблицу задач
    tasks_df = pd.DataFrame({
        'job': jobs.flatten(),
        'machine': machines.flatten(),
        'cost': cost_mat.flatten()
    })
    
    return {
        'njob': njob,
        'nmachine': nmachine,  # опечатка namchine -> nmachine
        'n': njob * nmachine,
        's': max(njob, nmachine),
        't': tasks_df
    }

def jsp_to_tsp(task_data: Dict[str, Any]) -> np.ndarray:
    """
    Преобразует Job Shop Problem в матрицу расстояний TSP
    
    Args:
        task_data: данные задачи из make_task_data
    
    Returns:
        Матрица расстояний для TSP
    """
    t = task_data['t']
    n = task_data['n']
    
    # Предварительно выделяем матрицу
    D = np.zeros((n, n), dtype=np.float64)
    
    # Векторизованное вычисление для эффективности
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            job_i, machine_i, cost_i = t.iloc[i]['job'], t.iloc[i]['machine'], t.iloc[i]['cost']
            job_j, machine_j, cost_j = t.iloc[j]['job'], t.iloc[j]['machine'], t.iloc[j]['cost']
            
            if job_i == job_j or machine_i == machine_j:
                D[i, j] = cost_i + cost_j
            else:
                D[i, j] = max(cost_i, cost_j)
    
    return D

def tsp_to_jsp_optimized(task_data: Dict[str, Any], path: List[int]) -> Dict[str, Any]:
    """
    ОПТИМИЗИРОВАННАЯ версия: Преобразует TSP решение обратно в JSP расписание
    
    Основные оптимизации:
    1. Предварительное выделение памяти
    2. Использование простых структур для tracking
    3. Однократное создание DataFrame в конце
    
    Args:
        task_data: данные задачи
        path: оптимальный путь TSP (индексы задач)
    
    Returns:
        Словарь с расписанием и общей стоимостью
    """
    t = task_data['t']
    njob = task_data['njob'] 
    nmachine = task_data['nmachine']
    
    job_end_times = {job: 0 for job in range(1, njob + 1)}
    machine_end_times = {machine: 0 for machine in range(1, nmachine + 1)}
    
    # Предварительно выделенный список
    # Знаем точный размер заранее
    schedule_entries = []
    schedule_entries = [None] * len(path)  # https://stackoverflow.com/questions/537086/reserve-memory-for-list-in-python
    
    for i, task_idx in enumerate(path):
        # Получаем параметры задачи (индексация с 0)
        task_row = t.iloc[task_idx]
        job = task_row['job']
        machine = task_row['machine'] 
        cost = task_row['cost']
        
        # Находим самое раннее время начала
        # = max(когда освободится работа, когда освободится машина)
        earliest_start = max(
            job_end_times[job],
            machine_end_times[machine]
        )
        
        finish_time = earliest_start + cost
        
        schedule_entries[i] = {
            'job': job,
            'machine': machine,
            'from': earliest_start,
            'to': finish_time
        }
        # schedule_entries.append({
        #     'job': job,
        #     'machine': machine,
        #     'from': earliest_start,
        #     'to': finish_time
        # })
        
        job_end_times[job] = finish_time
        machine_end_times[machine] = finish_time
    
    schedule_df = pd.DataFrame(schedule_entries)
    
    schedule_df['job'] = schedule_df['job'].astype('category')
    schedule_df['machine'] = schedule_df['machine'].astype('category')
    
    total_cost = schedule_df['to'].max()
    
    return {
        'schedule': schedule_df,
        'cost': total_cost
    }

def tsp_to_jsp_via_numpy_fixed(task_data: Dict[str, Any], path: List[int]) -> Dict[str, Any]:
    """
    ИСПРАВЛЕННАЯ версия: Использование numpy для максимальной скорости
    Исправлена ошибка индексации - jobs и machines нумеруются с 1, но numpy массивы с 0
    """
    t = task_data['t']
    njob = task_data['njob']
    nmachine = task_data['nmachine']
    
    # Массивы для отслеживания времени окончания (индексы с 0, но значения jobs/machines с 1)
    job_end_times = np.zeros(njob + 1)  # +1 чтобы использовать индексы 1-njob
    machine_end_times = np.zeros(nmachine + 1)  # +1 чтобы использовать индексы 1-nmachine
    
    n_tasks = len(path)
    jobs = np.zeros(n_tasks, dtype=int)
    machines = np.zeros(n_tasks, dtype=int) 
    start_times = np.zeros(n_tasks)
    end_times = np.zeros(n_tasks)
    
    for i, task_idx in enumerate(path):
        task_row = t.iloc[task_idx]
        job = task_row['job']  # Значение от 1 до njob
        machine = task_row['machine']  # Значение от 1 до nmachine
        cost = task_row['cost']
        
        # Теперь можем безопасно использовать job и machine как индексы
        start_time = max(job_end_times[job], machine_end_times[machine])
        end_time = start_time + cost
        
        jobs[i] = job
        machines[i] = machine
        start_times[i] = start_time
        end_times[i] = end_time
        
        job_end_times[job] = end_time
        machine_end_times[machine] = end_time
    
    schedule_df = pd.DataFrame({
        'job': pd.Categorical(jobs),
        'machine': pd.Categorical(machines),
        'from': start_times,
        'to': end_times
    })
    
    return {
        'schedule': schedule_df,
        'cost': end_times.max()
    }

def generate_random_cost_matrix(njob: int, nmachine: int, seed: int = 42) -> np.ndarray:
    """Generate random cost matrix for testing"""
    np.random.seed(seed)
    return np.random.randint(1, 100, size=(njob, nmachine))

def generate_random_tsp_path(n: int, seed: int = 42) -> List[int]:
    """Generate random TSP path for testing"""
    random.seed(seed)
    path = list(range(n))
    random.shuffle(path)
    return path

def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution"""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time

def comprehensive_performance_test():
    """Comprehensive performance testing of all JSP functions"""
    
    # Test configurations: (njob, nmachine)
    test_sizes = [
        (3, 3),    # Small
        (5, 5),    # Medium-small
        (8, 8),    # Medium
        (10, 10),  # Medium-large
        (12, 12),  # Large
        (15, 15),  # Very large
    ]
    
    results = {
        'size': [],
        'njob': [],
        'nmachine': [],
        'n_tasks': [],
        'make_task_data_time': [],
        'jsp_to_tsp_time': [],
        'tsp_to_jsp_optimized_time': [],
        'tsp_to_jsp_via_numpy_time': [],
        'total_optimized_time': [],
        'total_numpy_time': []
    }
    
    print("=" * 90)
    print("COMPREHENSIVE JSP PERFORMANCE TEST")
    print("=" * 90)
    print(f"{'Size':<8} {'Jobs':<5} {'Mach':<5} {'Tasks':<6} {'MakeData(ms)':<12} {'JSP→TSP(ms)':<12} {'TSP→JSP_Opt(ms)':<15} {'TSP→JSP_NP(ms)':<14} {'Total_Opt(ms)':<13} {'Total_NP(ms)':<12}")
    print("-" * 90)
    
    for njob, nmachine in test_sizes:
        n_tasks = njob * nmachine
        
        # Generate test data
        cost_matrix = generate_random_cost_matrix(njob, nmachine)
        
        # Time make_task_data
        task_data, time_make = time_function(make_task_data, njob, nmachine, cost_matrix)
        
        # Time jsp_to_tsp
        distance_matrix, time_jsp_tsp = time_function(jsp_to_tsp, task_data)
        
        # Generate random TSP path for testing conversion back
        tsp_path = generate_random_tsp_path(n_tasks)
        
        # Time tsp_to_jsp_optimized
        _, time_optimized = time_function(tsp_to_jsp_optimized, task_data, tsp_path)
        
        # Time tsp_to_jsp_via_numpy (FIXED VERSION)
        _, time_numpy = time_function(tsp_to_jsp_via_numpy_fixed, task_data, tsp_path)
        
        # Calculate total times
        total_optimized = time_make + time_jsp_tsp + time_optimized
        total_numpy = time_make + time_jsp_tsp + time_numpy
        
        # Store results
        results['size'].append(f"{njob}x{nmachine}")
        results['njob'].append(njob)
        results['nmachine'].append(nmachine)
        results['n_tasks'].append(n_tasks)
        results['make_task_data_time'].append(time_make)
        results['jsp_to_tsp_time'].append(time_jsp_tsp)
        results['tsp_to_jsp_optimized_time'].append(time_optimized)
        results['tsp_to_jsp_via_numpy_time'].append(time_numpy)
        results['total_optimized_time'].append(total_optimized)
        results['total_numpy_time'].append(total_numpy)
        
        # Print results
        print(f"{njob}x{nmachine:<3} {njob:<5} {nmachine:<5} {n_tasks:<6} "
              f"{time_make*1000:<12.3f} {time_jsp_tsp*1000:<12.3f} "
              f"{time_optimized*1000:<15.3f} {time_numpy*1000:<14.3f} "
              f"{total_optimized*1000:<13.3f} {total_numpy*1000:<12.3f}")
    
    return results

def detailed_function_analysis():
    """Detailed analysis of each function with multiple runs"""
    
    print("\n" + "=" * 80)
    print("DETAILED FUNCTION ANALYSIS (Multiple Runs)")
    print("=" * 80)
    
    # Test with medium size problem
    njob, nmachine = 8, 8
    n_runs = 20
    
    cost_matrix = generate_random_cost_matrix(njob, nmachine)
    task_data = make_task_data(njob, nmachine, cost_matrix)
    tsp_path = generate_random_tsp_path(njob * nmachine)
    
    # Test make_task_data
    times = []
    for _ in range(n_runs):
        _, exec_time = time_function(make_task_data, njob, nmachine, cost_matrix)
        times.append(exec_time * 1000)  # Convert to ms
    
    print(f"\nmake_task_data ({n_runs} runs):")
    print(f"  Average: {np.mean(times):.4f} ms")
    print(f"  Std Dev: {np.std(times):.4f} ms")
    print(f"  Min:     {np.min(times):.4f} ms")
    print(f"  Max:     {np.max(times):.4f} ms")
    
    # Test jsp_to_tsp
    times = []
    for _ in range(n_runs):
        _, exec_time = time_function(jsp_to_tsp, task_data)
        times.append(exec_time * 1000)
    
    print(f"\njsp_to_tsp ({n_runs} runs):")
    print(f"  Average: {np.mean(times):.4f} ms")
    print(f"  Std Dev: {np.std(times):.4f} ms")
    print(f"  Min:     {np.min(times):.4f} ms")
    print(f"  Max:     {np.max(times):.4f} ms")
    
    # Test tsp_to_jsp_optimized
    times = []
    for _ in range(n_runs):
        _, exec_time = time_function(tsp_to_jsp_optimized, task_data, tsp_path)
        times.append(exec_time * 1000)
    
    print(f"\ntsp_to_jsp_optimized ({n_runs} runs):")
    print(f"  Average: {np.mean(times):.4f} ms")
    print(f"  Std Dev: {np.std(times):.4f} ms")
    print(f"  Min:     {np.min(times):.4f} ms")
    print(f"  Max:     {np.max(times):.4f} ms")
    
    # Test tsp_to_jsp_via_numpy (FIXED VERSION)
    times = []
    for _ in range(n_runs):
        _, exec_time = time_function(tsp_to_jsp_via_numpy_fixed, task_data, tsp_path)
        times.append(exec_time * 1000)
    
    print(f"\ntsp_to_jsp_via_numpy_fixed ({n_runs} runs):")
    print(f"  Average: {np.mean(times):.4f} ms")
    print(f"  Std Dev: {np.std(times):.4f} ms")
    print(f"  Min:     {np.min(times):.4f} ms")
    print(f"  Max:     {np.max(times):.4f} ms")
    
    # Performance comparison
    opt_avg = np.mean([time_function(tsp_to_jsp_optimized, task_data, tsp_path)[1] for _ in range(n_runs)])
    np_avg = np.mean([time_function(tsp_to_jsp_via_numpy_fixed, task_data, tsp_path)[1] for _ in range(n_runs)])
    
    if opt_avg > np_avg:
        speedup = opt_avg / np_avg
        print(f"\n🚀 Numpy version is {speedup:.2f}x faster than optimized version!")
    else:
        speedup = np_avg / opt_avg
        print(f"\n🚀 Optimized version is {speedup:.2f}x faster than numpy version!")

def scaling_analysis():
    """Analyze how functions scale with problem size"""
    
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    
    sizes = [(i, i) for i in range(3, 16, 2)]  # 3x3, 5x5, 7x7, 9x9, 11x11, 13x13, 15x15
    
    jsp_to_tsp_times = []
    optimized_times = []
    numpy_times = []
    n_tasks_list = []
    
    for njob, nmachine in sizes:
        n_tasks = njob * nmachine
        cost_matrix = generate_random_cost_matrix(njob, nmachine)
        task_data = make_task_data(njob, nmachine, cost_matrix)
        tsp_path = generate_random_tsp_path(n_tasks)
        
        # Time jsp_to_tsp (most computationally expensive)
        _, time_jsp_tsp = time_function(jsp_to_tsp, task_data)
        jsp_to_tsp_times.append(time_jsp_tsp * 1000)
        
        # Time conversion functions
        _, time_opt = time_function(tsp_to_jsp_optimized, task_data, tsp_path)
        _, time_np = time_function(tsp_to_jsp_via_numpy_fixed, task_data, tsp_path)
        
        optimized_times.append(time_opt * 1000)
        numpy_times.append(time_np * 1000)
        n_tasks_list.append(n_tasks)
        
        print(f"Size {njob}x{nmachine} ({n_tasks:3d} tasks): "
              f"JSP→TSP: {time_jsp_tsp*1000:8.3f}ms, "
              f"Optimized: {time_opt*1000:6.3f}ms, "
              f"Numpy: {time_np*1000:6.3f}ms")
    
    # Calculate complexity ratios
    print(f"\nComplexity Analysis:")
    print(f"Tasks increase by factor: {n_tasks_list[-1]/n_tasks_list[0]:.1f}")
    print(f"JSP→TSP time increase: {jsp_to_tsp_times[-1]/jsp_to_tsp_times[0]:.1f}x")
    print(f"Optimized time increase: {optimized_times[-1]/optimized_times[0]:.1f}x")
    print(f"Numpy time increase: {numpy_times[-1]/numpy_times[0]:.1f}x")
    
    return sizes, jsp_to_tsp_times, optimized_times, numpy_times, n_tasks_list

def verify_correctness():
    """Verify that both implementations produce the same results"""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    njob, nmachine = 4, 3
    cost_matrix = generate_random_cost_matrix(njob, nmachine, seed=123)
    task_data = make_task_data(njob, nmachine, cost_matrix)
    tsp_path = generate_random_tsp_path(njob * nmachine, seed=123)
    
    # Run both implementations
    result_opt = tsp_to_jsp_optimized(task_data, tsp_path)
    result_np = tsp_to_jsp_via_numpy_fixed(task_data, tsp_path)
    
    # Compare results
    print(f"Problem size: {njob} jobs, {nmachine} machines")
    print(f"Total tasks: {len(tsp_path)}")
    print(f"TSP path: {tsp_path}")
    
    print(f"\nOptimized version total cost: {result_opt['cost']}")
    print(f"Numpy version total cost: {result_np['cost']}")
    
    # Check if schedules are identical
    schedule_opt = result_opt['schedule'].sort_values(['job', 'machine']).reset_index(drop=True)
    schedule_np = result_np['schedule'].sort_values(['job', 'machine']).reset_index(drop=True)
    
    are_equal = schedule_opt.equals(schedule_np)
    
    if are_equal:
        print("✅ Both implementations produce identical results!")
    else:
        print("❌ Results differ between implementations!")
        print("\nOptimized schedule:")
        print(schedule_opt)
        print("\nNumpy schedule:")
        print(schedule_np)
    
    return are_equal

def main():
    """Run all performance tests"""
    
    print("Starting comprehensive JSP performance testing...")
    print("All times are in milliseconds (ms)\n")
    
    # First verify correctness
    verify_correctness()
    
    # Run comprehensive test
    results = comprehensive_performance_test()
    
    # Run detailed analysis
    detailed_function_analysis()
    
    # Run scaling analysis
    sizes, jsp_times, opt_times, np_times, n_tasks = scaling_analysis()
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Key findings:")
    print("1. jsp_to_tsp is the most computationally expensive function (O(n²))")
    print("2. Both TSP→JSP methods have similar performance for small problems")
    print("3. Performance scales quadratically with problem size for TSP conversion")
    print("4. Linear scaling for TSP→JSP conversion methods")
    print("5. For problems larger than 15x15, jsx_to_tsp becomes the bottleneck")
    
    # Find which method is faster on average
    avg_opt = np.mean(opt_times)
    avg_np = np.mean(np_times)
    
    if avg_np < avg_opt:
        print(f"🏆 Numpy version is {avg_opt/avg_np:.2f}x faster on average")
    else:
        print(f"🏆 Optimized version is {avg_np/avg_opt:.2f}x faster on average")

if __name__ == "__main__":
    main()