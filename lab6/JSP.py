import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

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

def tsp_to_jsp_via_numpy(task_data: Dict[str, Any], path: List[int]) -> Dict[str, Any]:
    """
    Использование numpy для максимальной скорости
    """
    t = task_data['t']
    njob = task_data['njob']
    nmachine = task_data['nmachine']
    
    job_end_times = np.zeros(njob)
    machine_end_times = np.zeros(nmachine)
    
    n_tasks = len(path)
    jobs = np.zeros(n_tasks, dtype=int)
    machines = np.zeros(n_tasks, dtype=int) 
    start_times = np.zeros(n_tasks)
    end_times = np.zeros(n_tasks)
    
    for i, task_idx in enumerate(path):
        task_row = t.iloc[task_idx]
        job = task_row['job']
        machine = task_row['machine']
        cost = task_row['cost']
        
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

def plot_jsp_schedule(schedule_df: pd.DataFrame) -> None:
    """
    Визуализирует расписание Job Shop
    
    Args:
        schedule_df: DataFrame с расписанием
    """
    plt.figure(figsize=(12, 8))
    
    # Уникальные работы для цветовой карты
    unique_jobs = schedule_df['job'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_jobs)))
    job_colors = dict(zip(unique_jobs, colors))
    
    # Рисуем каждую задачу как горизонтальный отрезок
    for _, task in schedule_df.iterrows():
        y = task['machine']
        x_start = task['from'] 
        x_end = task['to']
        color = job_colors[task['job']]
        
        plt.barh(y, x_end - x_start, left=x_start, height=0.6, 
                color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Подписываем номер работы
        plt.text((x_start + x_end) / 2, y, f'J{task["job"]}', 
                ha='center', va='center', fontweight='bold', fontsize=8)
    
    plt.xlabel('Время')
    plt.ylabel('Машина')
    plt.title('Расписание Job Shop Problem')
    plt.grid(True, alpha=0.3)
    
    # Легенда для работ
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=job_colors[job], 
                                   label=f'Работа {job}') for job in unique_jobs]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()