import yaml
import xml.etree.ElementTree as ET
import networkx as nx
import random
import math
from deap import base, creator, tools, algorithms
from itertools import combinations

def load_database_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_queries_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_graph(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    G = nx.DiGraph()
    
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    
    for node in root.findall('.//graphml:node', ns):
        node_id = node.get('id')
        name = node.find(".//graphml:data[@key='v_name']", ns).text
        fun = node.find(".//graphml:data[@key='v_fun']", ns).text
        G.add_node(node_id, name=name, fun=fun)
    
    for edge in root.findall('.//graphml:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        label_elem = edge.find(".//graphml:data[@key='e_label']", ns)
        label = label_elem.text if label_elem is not None else ""
        G.add_edge(source, target, label=label)
    
    return G

def get_column_type_size(column_type):
    type_sizes = {
        'INTEGER': 4,    # 4 байта
        'REAL': 8,       # 8 байт (double)
        'DATE': 4,       # 4 байта (timestamp)
        'TEXT': 100      # 100 байт (средний размер строки)
    }
    return type_sizes.get(column_type, 4)


def calculate_selectivity(where_conditions):
    '''
    общую селективность для набора WHERE условий    
    
    Условие 1: sample: 0.3 (30% строк проходят фильтр)
    Условие 2: sample: 0.4 (40% строк проходят фильтр)
    Общая селективность: 0.3 × 0.4 = 0.12 (12% строк)
    '''
    total_selectivity = 1.0
    for condition in where_conditions:
        sample = condition.get('sample', 1.0)
        total_selectivity *= sample
    return total_selectivity

# Получение количества строк и частоты обновлений таблицы
def get_table_metadata(table_name, database_metadata):
    for table in database_metadata:
        if table['name'] == table_name:
            return table
    return None


def calculate_join_result_size(table1_size, table2_size, selectivity):
    '''
    Декартово произведение: |A| × |B|
    С учетом JOIN условий: |A| × |B| × selectivity
    Пример: 1000 × 2000 × 0.1 = 200,000 строк
    '''
    return int(table1_size * table2_size * selectivity)

def calculate_where_result_size(input_size, selectivity):
    return int(input_size * selectivity)

def get_operation_cost(operation_type, input_size, num_conditions=1):
    if operation_type == 'JOIN':
        if input_size <= 0:
            return 0
        # JOIN: n log n - сложность хеш-соединения с сортировкой
        return input_size * math.log(input_size) 
    elif operation_type == 'WHERE':
        return input_size * num_conditions
    elif operation_type == 'SELECT':
        if input_size <= 0:
            return 0
        return math.log(input_size)
    elif operation_type == 'AGGREGATE':
        return input_size
    else:
        return 0

def get_aggregate_cost(aggregate_type, input_size):
    costs = {
        'IDENTITY': 0,
        'COUNT': 1,
        'SUM': input_size,
        'MAX': input_size,
        'AVG': input_size,
        'MIN': input_size,
        'DISTINCT': input_size * math.log(input_size) if input_size > 0 else 0
    }
    return costs.get(aggregate_type, input_size)

# Материализовать можно только промежуточные результаты операций,
# не базовые таблицы или колонки
def identify_candidate_nodes(graph):
    candidates = []
    for node_id, data in graph.nodes(data=True):
        fun = data.get('fun', '')
        if fun in ['JOIN', 'WHERE', 'SELECT', 'AGGREGATE']:
            candidates.append(node_id)
    return candidates

def calculate_node_result_size(node_id, graph, database_metadata, materialized_views, memo=None):
    if memo is None:
        memo = {}
    
    if node_id in memo:
        return memo[node_id]
    
    if node_id in materialized_views:
        return materialized_views[node_id]
    
    node_data = graph.nodes[node_id]
    fun = node_data.get('fun', '')
    name = node_data.get('name', '')
    
    if fun == 'table':
        table_meta = get_table_metadata(name, database_metadata)
        if table_meta:
            result = table_meta['lines']
            memo[node_id] = result
            return result
        return 100
    
    if fun == 'column' or fun == 'condition':
        memo[node_id] = 0
        return 0
    
    predecessors = list(graph.predecessors(node_id))
    
    if fun == 'JOIN':
        if len(predecessors) >= 2:
            size1 = calculate_node_result_size(predecessors[0], graph, database_metadata, materialized_views, memo)
            size2 = calculate_node_result_size(predecessors[1], graph, database_metadata, materialized_views, memo)
            result = max(size1 * size2 * 0.1, 1)
        else:
            result = 100
    elif fun == 'WHERE':
        if predecessors:
            input_size = calculate_node_result_size(predecessors[0], graph, database_metadata, materialized_views, memo)
            result = max(int(input_size * 0.3), 1)
        else:
            result = 100
    elif fun == 'SELECT':
        if predecessors:
            result = calculate_node_result_size(predecessors[0], graph, database_metadata, materialized_views, memo)
        else:
            result = 100
    elif fun == 'AGGREGATE':
        if predecessors:
            result = max(int(calculate_node_result_size(predecessors[0], graph, database_metadata, materialized_views, memo) * 0.1), 1)
        else:
            result = 10
    else:
        result = 100
    
    memo[node_id] = result
    return result

# Если узел материализован, не нужно вычислять его предшественников
def calculate_node_computation_cost(node_id, graph, database_metadata, materialized_views, memo=None):
    if memo is None:
        memo = {}
    
    if node_id in memo:
        return memo[node_id]
    
    if node_id in materialized_views:
        memo[node_id] = 1
        return 1
    
    node_data = graph.nodes[node_id]
    fun = node_data.get('fun', '')
    
    if fun in ['table', 'column', 'condition']:
        memo[node_id] = 0
        return 0
    
    predecessors = list(graph.predecessors(node_id))
    total_cost = 0
    
    for pred in predecessors:
        total_cost += calculate_node_computation_cost(pred, graph, database_metadata, materialized_views, memo)
    
    input_size = calculate_node_result_size(node_id, graph, database_metadata, materialized_views)
    
    if fun == 'JOIN':
        cost = get_operation_cost('JOIN', input_size)
    elif fun == 'WHERE':
        cost = get_operation_cost('WHERE', input_size, len(predecessors))
    elif fun == 'SELECT':
        cost = get_operation_cost('SELECT', input_size)
    elif fun == 'AGGREGATE':
        cost = get_aggregate_cost('AVG', input_size)
    else:
        cost = 0
    
    total_cost += cost
    memo[node_id] = total_cost
    
    return total_cost

def get_dependent_tables(node_id, graph, memo=None):
    if memo is None:
        memo = {}
    
    if node_id in memo:
        return memo[node_id]
    
    node_data = graph.nodes[node_id]
    fun = node_data.get('fun', '')
    name = node_data.get('name', '')
    
    if fun == 'table':
        result = {name}
        memo[node_id] = result
    
        return result
    
    predecessors = list(graph.predecessors(node_id))
    dependent_tables = set()
    
    for pred in predecessors:
        dependent_tables.update(get_dependent_tables(pred, graph, memo))
    
    memo[node_id] = dependent_tables
    
    return dependent_tables

# Для расчета частоты обновлений материализованного представления
def calculate_storage_cost(node_id, graph, database_metadata):
    node_size = calculate_node_result_size(node_id, graph, database_metadata, {})
    avg_column_size = 50
    storage_cost = node_size * avg_column_size
    
    return storage_cost

def calculate_maintenance_cost(node_id, graph, database_metadata):
    dependent_tables = get_dependent_tables(node_id, graph)
    storage_cost = calculate_storage_cost(node_id, graph, database_metadata)
    
    total_update_freq = 0
    for table_name in dependent_tables:
        table_meta = get_table_metadata(table_name, database_metadata)
        if table_meta and 'freq' in table_meta:
            total_update_freq += table_meta['freq']
    
    maintenance_cost = storage_cost * total_update_freq
    return maintenance_cost

def find_query_nodes(graph, query_name):
    query_nodes = []
    for node_id, data in graph.nodes(data=True):
        if data.get('name') == query_name:
            query_nodes.append(node_id)
    return query_nodes

def calculate_query_cost(query, graph, database_metadata, materialized_views):
    query_name = query['name']
    query_freq = query.get('freq', 1)
    
    query_nodes = find_query_nodes(graph, query_name)
    if not query_nodes:
        return 0
    
    total_cost = 0
    for query_node in query_nodes:
        cost = calculate_node_computation_cost(query_node, graph, database_metadata, materialized_views)
        total_cost += cost
    
    return total_cost * query_freq

def fitness_function(individual, candidate_nodes, graph, database_metadata, queries_metadata):
    '''
    
    Декодирование - преобразование бинарной строки в множество материализованных узлов
    Расчет стоимости запросов - суммирование по всем запросам с учетом частот
    Расчет стоимости обслуживания - учет хранения и обновлений
    Агрегация - получение единой метрики для оптимизации

    '''
    
    # Декодируем хромосому в множество материализованных узлов
    materialized_views = {}
    for i, is_materialized in enumerate(individual):
        if is_materialized:
            node_id = candidate_nodes[i]
            materialized_views[node_id] = calculate_node_result_size(node_id, graph, database_metadata, {})
    
    # Вычисляем стоимость всех запросов
    total_query_cost = 0
    all_queries = queries_metadata.get('queries', []) + queries_metadata.get('subqueries', [])
    
    # Вычисляем стоимость обслуживания
    for query in all_queries:
        query_cost = calculate_query_cost(query, graph, database_metadata, materialized_views)
        total_query_cost += query_cost
    
    total_maintenance_cost = 0
    for node_id in materialized_views:
        maintenance_cost = calculate_maintenance_cost(node_id, graph, database_metadata)
        total_maintenance_cost += maintenance_cost
    
    total_cost = total_query_cost + total_maintenance_cost
    
    return (total_cost,)

def main():
    database_metadata = load_database_metadata('database.yaml')
    queries_metadata = load_queries_metadata('queries.yaml')
    graph = load_graph('plan.graphml')
    
    candidate_nodes = identify_candidate_nodes(graph)
    print(f"Количество кандидатов для материализации: {len(candidate_nodes)}")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Минимизация
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=len(candidate_nodes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", fitness_function, candidate_nodes=candidate_nodes,
                     graph=graph, database_metadata=database_metadata, 
                     queries_metadata=queries_metadata)
    toolbox.register("mate", tools.cxTwoPoint)  # Скрещивание
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Мутация
    toolbox.register("select", tools.selTournament, tournsize=3) # Селекция
    
    population_size = 50
    generations = 30
    crossover_prob = 0.7
    mutation_prob = 0.2
    
    population = toolbox.population(n=population_size)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
    stats.register("min", min)
    stats.register("max", max)
    
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    for generation in range(generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
        hof.update(population)
        
        record = stats.compile(population)
        print(f"Поколение {generation+1}: Мин={record['min']:.0f}, Сред={record['avg']:.0f}, Макс={record['max']:.0f}")
    
    best_individual = hof[0]
    best_cost = best_individual.fitness.values[0]
    
    print(f"\nЛучшее решение:")
    print(f"Общая стоимость: {best_cost:.0f}")
    
    materialized_nodes = []
    for i, is_materialized in enumerate(best_individual):
        if is_materialized:
            node_id = candidate_nodes[i]
            node_name = graph.nodes[node_id].get('name', node_id)
            materialized_nodes.append(f"{node_id}({node_name[:50]})")
    
    print(f"Материализованные узлы ({len(materialized_nodes)}):")
    for node in materialized_nodes:
        print(f"  - {node}")
    
    print(f"\nБитовая строка: {best_individual}")
    
    materialized_views = {}
    for i, is_materialized in enumerate(best_individual):
        if is_materialized:
            node_id = candidate_nodes[i]
            materialized_views[node_id] = calculate_node_result_size(node_id, graph, database_metadata, {})
    
    total_query_cost = 0
    all_queries = queries_metadata.get('queries', []) + queries_metadata.get('subqueries', [])
    
    for query in all_queries:
        query_cost = calculate_query_cost(query, graph, database_metadata, materialized_views)
        total_query_cost += query_cost
    
    total_maintenance_cost = 0
    for node_id in materialized_views:
        maintenance_cost = calculate_maintenance_cost(node_id, graph, database_metadata)
        total_maintenance_cost += maintenance_cost
    
    print(f"\nДетализация стоимости:")
    print(f"Стоимость обработки запросов: {total_query_cost:.0f}")
    print(f"Стоимость обслуживания представлений: {total_maintenance_cost:.0f}")
    print(f"Общая стоимость: {total_query_cost + total_maintenance_cost:.0f}")


main()