import time
from collections import deque
import heapq
import tkinter as tk
from tkinter import messagebox
import threading
import random
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import copy

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_NAMES = {(-1, 0): "Down", (1, 0): "Up", (0, -1): "Right", (0, 1): "Left"}
GOAL_POSITIONS = {}


def initialize_goal_positions(goal):
    for i in range(3):
        for j in range(3):
            val = goal[i][j]
            GOAL_POSITIONS[val] = (i, j)

def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def find_empty(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j
    raise ValueError("Không tìm thấy ô trống trên bảng!")

def move_tile(board, empty_pos, direction):
    x, y = empty_pos
    dx, dy = direction
    new_x, new_y = x + dx, y + dy
    
    if 0 <= new_x < 3 and 0 <= new_y < 3:
        new_board = [list(row) for row in board]
        new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
        return new_board
    return None

def heuristic(board, goal):
    manhattan = 0
    conflict = 0
    for i in range(3):
        for j in range(3):
            val = board[i][j]
            if val != 0:
                goal_pos = GOAL_POSITIONS[val]
                manhattan += abs(i - goal_pos[0]) + abs(j - goal_pos[1])
        
        row = [board[i][j] for j in range(3) if board[i][j] != 0]
        goal_row = [goal[i][j] for j in range(3) if goal[i][j] != 0]
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                if row[j] in goal_row and row[k] in goal_row:
                    if goal_row.index(row[j]) > goal_row.index(row[k]) and row[j] > row[k]:
                        conflict += 2
    return manhattan + conflict

def get_direction(before, after):
    empty_before = find_empty(before)
    for d in DIRECTIONS:
        candidate = move_tile(before, empty_before, d)
        if candidate and candidate == after:
            return d
    return None

def is_solvable(puzzle):
    flat = [tile for row in puzzle for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    empty_pos = find_empty(puzzle)
    taxicab_distance = abs(empty_pos[0] - 2) + abs(empty_pos[1] - 2)
    return (inversions + taxicab_distance) % 2 == 0

def generate_random_puzzle():
    while True:
        tiles = list(range(9))
        random.shuffle(tiles)
        puzzle = [
            tiles[0:3],
            tiles[3:6],
            tiles[6:9]
        ]
        if is_solvable(puzzle):
            return puzzle

def bfs_8_puzzle(start, goal, parallel=False):
    initialize_goal_positions(goal)
    visited = set([board_to_tuple(start)])
    
    if not parallel:
        queue = deque([(start, [])])
        while queue:
            current_board, path = queue.popleft()
            if current_board == goal:
                return path
            empty_pos = find_empty(current_board)
            for direction in DIRECTIONS:
                new_board = move_tile(current_board, empty_pos, direction)
                if new_board and board_to_tuple(new_board) not in visited:
                    queue.append((new_board, path + [new_board]))
                    visited.add(board_to_tuple(new_board))
    else:
        def explore_branch(start_board, goal, visited_set):
            queue = deque([(start_board, [])])
            while queue:
                current_board, path = queue.popleft()
                if current_board == goal:
                    return path
                empty_pos = find_empty(current_board)
                for direction in DIRECTIONS:
                    new_board = move_tile(current_board, empty_pos, direction)
                    if new_board and board_to_tuple(new_board) not in visited_set:
                        queue.append((new_board, path + [new_board]))
                        visited_set.add(board_to_tuple(new_board))
            return None
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(explore_branch, start, goal, visited) for _ in range(4)]
            for future in futures:
                result = future.result()
                if result:
                    return result
    return None

def dfs_8_puzzle(start, goal, max_depth=100, time_limit=5):
    start_time = time.time()
    best_solution = None
    best_heuristic = float('inf')
    visited = {}
    
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            goal_positions[goal[i][j]] = (i, j)
    
    def optimized_heuristic(board):
        distance = 0
        for i in range(3):
            for j in range(3):
                val = board[i][j]
                if val != 0:
                    goal_i, goal_j = goal_positions[val]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def evaluate_branch(board, path_length):
        h = optimized_heuristic(board)
        return h * 2 + path_length
    
    stack = [(start, [], set([board_to_tuple(start)]))]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        current_board, path, path_visited = stack.pop()
        current_h = optimized_heuristic(current_board)
        
        if current_board == goal:
            if best_solution is None or len(path) < len(best_solution):
                best_solution = path
                if len(path) == current_h:
                    break
            continue
            
        if (len(path) >= max_depth or 
            len(path) + current_h > max_depth or
            (best_solution and len(path) >= len(best_solution))):
            continue
            
        board_key = board_to_tuple(current_board)
        if board_key in visited:
            if len(path) >= visited[board_key]:
                continue
        visited[board_key] = len(path)
        
        empty_pos = find_empty(current_board)
        neighbors = []
        
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                new_board_key = board_to_tuple(new_board)
                if new_board_key not in path_visited:
                    neighbors.append(new_board)
        
        neighbors.sort(key=lambda x: optimized_heuristic(x))
        
        for neighbor in neighbors:
            new_path_visited = path_visited.copy()
            new_path_visited.add(board_to_tuple(neighbor))
            stack.append((neighbor, path + [neighbor], new_path_visited))
    
    return best_solution

def iterative_deepening_8_puzzle(start, goal, max_depth=50):
    initialize_goal_positions(goal)
    def dls(node, depth, path, visited):
        if node == goal:
            return path
        if depth <= 0 or heuristic(node, goal) > depth:
            return None
        empty_pos = find_empty(node)
        visited.add(board_to_tuple(node))
        for direction in DIRECTIONS:
            new_node = move_tile(node, empty_pos, direction)
            if new_node and board_to_tuple(new_node) not in visited:
                result = dls(new_node, depth - 1, path + [new_node], visited.copy())
                if result:
                    return result
        return None
    
    for depth in range(max_depth + 1):
        visited = set()
        result = dls(start, depth, [start], visited)
        if result:
            return result
    return None

def greedy_8_puzzle(start, goal):
    initialize_goal_positions(goal)
    pq = [(heuristic(start, goal), start, [])]
    visited = set()
    visited.add(board_to_tuple(start))
    
    while pq:
        _, current_board, path = heapq.heappop(pq)
        if current_board == goal:
            return path
        empty_pos = find_empty(current_board)
        
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board and board_to_tuple(new_board) not in visited:
                heapq.heappush(pq, (heuristic(new_board, goal), new_board, path + [new_board]))
                visited.add(board_to_tuple(new_board))
    return None

def ucs_8_puzzle(start, goal):
    initialize_goal_positions(goal)
    pq = [(0, start, [])]
    visited = set()
    visited.add(board_to_tuple(start))
    
    while pq:
        cost, current_board, path = heapq.heappop(pq)
        if current_board == goal:
            return path
        empty_pos = find_empty(current_board)
        
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board and board_to_tuple(new_board) not in visited:
                heapq.heappush(pq, (cost + 1, new_board, path + [new_board]))
                visited.add(board_to_tuple(new_board))
    return None

def a_star_8_puzzle(start, goal):
    initialize_goal_positions(goal)
    pq = [(heuristic(start, goal), 0, start, [])]
    visited = set([board_to_tuple(start)])
    
    while pq:
        f, g, current_board, path = heapq.heappop(pq)
        if current_board == goal:
            return path
        empty_pos = find_empty(current_board)
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board and board_to_tuple(new_board) not in visited:
                new_g = g + 1
                h = heuristic(new_board, goal)
                heapq.heappush(pq, (new_g + h, new_g, new_board, path + [new_board]))
                visited.add(board_to_tuple(new_board))
    return None

def simultaneous_a_star_8_puzzle(start, goal, beam_width=3):
    initialize_goal_positions(goal)
    def f(board, g):
        return g + heuristic(board, goal)

    frontier = [(f(start, 0), 0, start, [start])]
    visited = set()
    visited.add(board_to_tuple(start))

    while frontier:
        frontier.sort(key=lambda x: x[0])
        next_frontier = []

        for _, g, board, path in frontier[:beam_width]:
            if board == goal:
                return path

            empty_pos = find_empty(board)
            for direction in DIRECTIONS:
                new_board = move_tile(board, empty_pos, direction)
                if new_board:
                    key = board_to_tuple(new_board)
                    if key not in visited:
                        visited.add(key)
                        new_g = g + 1
                        new_path = path + [new_board]
                        next_frontier.append((f(new_board, new_g), new_g, new_board, new_path))

        frontier = next_frontier

    return None

def belief_propagation_8_puzzle(start, goal, max_iter=100):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    path = [current_board]
    visited = set()
    visited.add(board_to_tuple(current_board))

    for _ in range(max_iter):
        if current_board == goal:
            return path

        empty_pos = find_empty(current_board)
        best_board = None
        best_score = float('inf')

        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                board_key = board_to_tuple(new_board)
                if board_key not in visited:
                    h = heuristic(new_board, goal)
                    if h < best_score:
                        best_score = h
                        best_board = new_board

        if best_board:
            current_board = best_board
            path.append(current_board)
            visited.add(board_to_tuple(current_board))
        else:
            break 

    return path if current_board == goal else None



def dfs_ida_star(board, goal, empty_pos, path, depth, threshold, visited):
    initialize_goal_positions(goal)
    board_tuple = board_to_tuple(board)
    
    if board == goal:
        return path
    
    current_h = heuristic(board, goal)
    f = depth + current_h
    
    if f > threshold:
        return f
    
    visited.add(board_tuple)
    
    min_threshold = float('inf')
    for direction in DIRECTIONS:
        new_board = move_tile(board, empty_pos, direction)
        if new_board:
            new_board_tuple = board_to_tuple(new_board)
            if new_board_tuple not in visited:
                new_empty_pos = (empty_pos[0] + direction[0], empty_pos[1] + direction[1])
                result = dfs_ida_star(new_board, goal, new_empty_pos, path + [new_board], depth + 1, threshold, visited)
                if isinstance(result, list):
                    return result
                if result < min_threshold:
                    min_threshold = result
    
    visited.remove(board_tuple)
    return min_threshold

def ida_star_8_puzzle(start, goal):
    initialize_goal_positions(goal)
    threshold = heuristic(start, goal)
    empty_pos = find_empty(start)
    path = []
    visited = set()
    
    while True:
        result = dfs_ida_star(start, goal, empty_pos, path, 0, threshold, visited)
        if isinstance(result, list):
            return result
        if result == float('inf'):
            return None
        threshold = result

def simple_hill_climbing_8_puzzle(start, goal, max_steps=1000):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    path = [current_board]
    visited = set()
    steps = 0
    
    while steps < max_steps:
        if current_board == goal:
            return path
        
        empty_pos = find_empty(current_board)
        current_h = heuristic(current_board, goal)
        best_move = None
        best_h = current_h
        
        directions = random.sample(DIRECTIONS, len(DIRECTIONS))
        
        for direction in directions:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                board_key = board_to_tuple(new_board)
                if board_key not in visited:
                    new_h = heuristic(new_board, goal)
                    if new_h < best_h:
                        best_move = new_board
                        best_h = new_h
        
        if best_move and best_h < current_h:
            current_board = [row[:] for row in best_move]
            path.append(current_board)
            visited.add(board_to_tuple(current_board))
            steps += 1
        else:
            possible_moves = [move_tile(current_board, empty_pos, d) for d in DIRECTIONS]
            valid_moves = [m for m in possible_moves if m and board_to_tuple(m) not in visited]
            if not valid_moves:
                break
                
            current_board = random.choice(valid_moves)
            path.append(current_board)
            visited.add(board_to_tuple(current_board))
            steps += 1
    return path if current_board == goal else None

def steepest_ascent_hill_climbing_8_puzzle(start, goal, max_steps=1000):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    path = [current_board]
    visited = set()
    steps = 0
    
    while steps < max_steps:
        if current_board == goal:
            return path
        
        empty_pos = find_empty(current_board)
        current_h = heuristic(current_board, goal)
        best_move = None
        best_h = current_h
        
        neighbors = []
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                neighbors.append(new_board)
        
        if not neighbors:
            break
        
        for neighbor in neighbors:
            neighbor_h = heuristic(neighbor, goal)
            if neighbor_h < best_h:
                best_h = neighbor_h
                best_move = neighbor
        
        if best_move and best_h < current_h:
            current_board = [row[:] for row in best_move]
            path.append(current_board)
            steps += 1
        else:
            possible_moves = [move_tile(current_board, empty_pos, d) for d in DIRECTIONS]
            valid_moves = [m for m in possible_moves if m]
            
            if not valid_moves:
                break
                
            current_board = random.choice(valid_moves)
            path.append(current_board)
            steps += 1
    return path if current_board == goal else None

def stochastic_hill_climbing_8_puzzle(start, goal, max_steps=1000):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    path = [current_board]
    steps = 0
    
    while steps < max_steps:
        if current_board == goal:
            return path
        
        empty_pos = find_empty(current_board)
        current_h = heuristic(current_board, goal)
        
        improving_neighbors = []
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                new_h = heuristic(new_board, goal)
                if new_h < current_h:
                    improving_neighbors.append((new_h, new_board))
        
        if improving_neighbors:
            improving_neighbors.sort()
            total = sum(1/(h+1) for h, _ in improving_neighbors)  
            r = random.uniform(0, total)
            upto = 0
            for h, neighbor in improving_neighbors:
                if upto + 1/(h+1) >= r:
                    current_board = [row[:] for row in neighbor]
                    path.append(current_board)
                    steps += 1
                    break
                upto += 1/(h+1)
        else:
            break
    return path if current_board == goal else None

def simulated_annealing_hill_climbing_8_puzzle(start, goal, initial_temp=None, cooling_rate=0.95, max_steps=50000, time_limit=5.0):
    start_time = time.time()
    current_board = [row[:] for row in start]
    current_h = heuristic(current_board, goal)
    best_board = [row[:] for row in current_board]
    best_h = current_h
    path = [current_board]
    
    if initial_temp is None:
        initial_temp = current_h * 2 if current_h > 0 else 100
    temp = initial_temp
    
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            goal_positions[goal[i][j]] = (i, j)
    
    def optimized_heuristic(board):
        distance = 0
        for i in range(3):
            for j in range(3):
                val = board[i][j]
                if val != 0:
                    goal_i, goal_j = goal_positions[val]
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        if current_h == 0:
            break
            
        empty_pos = find_empty(current_board)
        neighbors = []
        
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                neighbors.append(new_board)
        
        if not neighbors:
            break
            
        if random.random() < 0.7:  
            neighbors.sort(key=lambda x: optimized_heuristic(x))
            next_board = neighbors[0]
        else:  
            next_board = random.choice(neighbors)
        
        next_h = optimized_heuristic(next_board)
        delta = next_h - current_h
        
        if delta <= 0 or math.exp(-delta / temp) > random.random():
            current_board = [row[:] for row in next_board]
            current_h = next_h
            path.append(current_board)
            
            if current_h < best_h:
                best_board = [row[:] for row in current_board]
                best_h = current_h
        
        if step % 100 == 0:
            temp = initial_temp / (1 + math.log(1 + step)) 
        
        temp = max(temp * cooling_rate, 1e-3)
    
    return path if current_h == 0 else (path if best_h < optimized_heuristic(start) else None)



def randomized_hill_climbing_8_puzzle(start, goal, max_steps=1000, restarts=10):
    def board_to_tuple(board):
        return tuple(tuple(row) for row in board)

    initialize_goal_positions(goal)
    best_path = None

    for _ in range(restarts):
        current_board = copy.deepcopy(start)
        current_h = heuristic(current_board, goal)
        visited = set()
        visited.add(board_to_tuple(current_board))
        path = [current_board]

        for _ in range(max_steps):
            if current_board == goal:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                break

            empty_pos = find_empty(current_board)
            directions = random.sample(DIRECTIONS, len(DIRECTIONS))
            neighbors = []

            for direction in directions:
                neighbor = move_tile(current_board, empty_pos, direction)
                if neighbor:
                    h = heuristic(neighbor, goal)
                    neighbors.append((h, neighbor))

            neighbors.sort(key=lambda x: x[0])

            moved = False
            for h, neighbor in neighbors:
                neighbor_tuple = board_to_tuple(neighbor)
                if h < current_h and neighbor_tuple not in visited:
                    current_board = neighbor
                    current_h = h
                    path.append(neighbor)
                    visited.add(neighbor_tuple)
                    moved = True
                    break

            if not moved:
                break 

    return best_path

def no_observation_search_8_puzzle(start, goal, max_steps=500):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    path = [current_board]
    
    for _ in range(max_steps):
        if current_board == goal:
            return path
        
        empty_pos = find_empty(current_board)
        random.shuffle(DIRECTIONS) 
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                current_board = new_board
                path.append(current_board)
                break  
    return path if current_board == goal else None

def partial_observation_search_8_puzzle(start, goal, max_depth=30):
    initialize_goal_positions(goal)
    visited = set()
    best_path = None

    def limited_dfs(board, path, depth):
        nonlocal best_path

        if depth > max_depth:
            return
        if board == goal:
            if best_path is None or len(path) < len(best_path):
                best_path = path[:]
            return

        board_key = board_to_tuple(board)
        visited.add(board_key)

        empty_pos = find_empty(board)
        for direction in DIRECTIONS:
            new_board = move_tile(board, empty_pos, direction)
            if new_board:
                new_key = board_to_tuple(new_board)
                if new_key not in visited:
                    path.append(new_board)
                    limited_dfs(new_board, path, depth + 1)
                    path.pop()
        visited.remove(board_key)

    limited_dfs(start, [start], 0)
    return best_path

def genetic_search_8_puzzle(start, goal, population_size=50, generations=200, mutation_rate=0.1, move_length=30, elite_ratio=0.2):
    initialize_goal_positions(goal)

    def random_individual():
        return [random.choice(list(DIRECTION_NAMES.values())) for _ in range(move_length)]

    def apply_moves(board, moves):
        current = [row[:] for row in board]
        for move in moves:
            dxdy = next((k for k, v in DIRECTION_NAMES.items() if v == move), None)
            if dxdy:
                new_board = move_tile(current, find_empty(current), dxdy)
                if new_board:
                    current = new_board
        return current

    def fitness(individual):
        result = apply_moves(start, individual)
        h = heuristic(result, goal)
        return h + 0.1 * sum(1 for m in individual if m)

    def mutate(individual):
        new_ind = individual[:]
        for i in range(len(new_ind)):
            if random.random() < mutation_rate:
                new_ind[i] = random.choice(list(DIRECTION_NAMES.values()))
        return new_ind

    def crossover(parent1, parent2):
        idx = random.randint(1, len(parent1) - 2)
        return parent1[:idx] + parent2[idx:]

    def tournament_selection(population, k=3):
        return min(random.sample(population, k), key=fitness)

    population = [random_individual() for _ in range(population_size)]
    best_fitness = float('inf')
    stagnant = 0  

    for gen in range(generations):
        population.sort(key=fitness)
        current_best = fitness(population[0])

        if current_best < best_fitness:
            best_fitness = current_best
            stagnant = 0
        else:
            stagnant += 1
        if best_fitness == 0:
            break
        if stagnant >= 30:
            break  

        next_gen = population[:int(population_size * elite_ratio)]  
        while len(next_gen) < population_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    best = population[0]
    final_board = apply_moves(start, best)
    if heuristic(final_board, goal) == 0:
        path = [start]
        current_board = [row[:] for row in start]
        for move in best:
            dxdy = next((k for k, v in DIRECTION_NAMES.items() if v == move), None)
            if dxdy:
                new_board = move_tile(current_board, find_empty(current_board), dxdy)
                if new_board:
                    current_board = new_board
                    path.append(current_board)
                    if current_board == goal:
                        break
        return path
    return None



def simulated_annealing_8_puzzle(start, goal, initial_temp=1000, cooling_rate=0.99, max_iterations=10000):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    current_energy = heuristic(current_board, goal)
    best_board = [row[:] for row in current_board]
    best_energy = current_energy
    
    temp = initial_temp
    path = [current_board]
    
    for i in range(max_iterations):
        if current_energy == 0:
            break
            
        temp *= cooling_rate
        empty_pos = find_empty(current_board)
        neighbors = []
        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                neighbors.append(new_board)
        
        if not neighbors:
            break
            
        new_board = random.choice(neighbors)
        new_energy = heuristic(new_board, goal)
        
        delta_energy = new_energy - current_energy
        
        if delta_energy < 0 or (temp > 0 and random.random() < math.exp(-delta_energy / temp)):
            current_board = [row[:] for row in new_board]
            current_energy = new_energy
            path.append(current_board)
            
            if current_energy < best_energy:
                best_board = [row[:] for row in current_board]
                best_energy = current_energy
    
    return path if best_energy == 0 else None

def beam_search_8_puzzle(start, goal, beam_width=3):
    initialize_goal_positions(goal)
    def evaluate_node(board):
        return heuristic(board, goal)
    
    beam = [(start, [], evaluate_node(start))]
    visited = set()
    visited.add(board_to_tuple(start))
    
    while beam:
        next_beam = []
        
        for current_board, path, score in beam:
            if current_board == goal:
                return path
            
            empty_pos = find_empty(current_board)
            
            for direction in DIRECTIONS:
                new_board = move_tile(current_board, empty_pos, direction)
                if new_board:
                    new_board_tuple = board_to_tuple(new_board)
                    if new_board_tuple not in visited:
                        new_path = path + [new_board]
                        new_score = evaluate_node(new_board)
                        next_beam.append((new_board, new_path, new_score))
                        visited.add(new_board_tuple)
        
        if not next_beam:
            return None
        
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:beam_width]
    
    return None

def and_or_search_8_puzzle(start, goal, max_depth=25):
    initialize_goal_positions(goal)
    visited = set()  
    
    def recursive_search(state, path, depth):
        state_tuple = board_to_tuple(state)
        
        if depth > max_depth:
            return None
        if state_tuple in visited:
            return None
        if state == goal:
            return path
            
        visited.add(state_tuple)
        empty_pos = find_empty(state)
        solutions = []
        
        for direction in DIRECTIONS:
            new_state = move_tile(state, empty_pos, direction)
            if new_state:
                solution = recursive_search(new_state, path + [new_state], depth + 1)
                if solution:
                    solutions.append(solution)
        
        return min(solutions, key=len) if solutions else None
    
    return recursive_search(start, [start], 0)

def backtracking_8_puzzle(start, goal, max_depth=30):
    initialize_goal_positions(goal)
    best_solution = None
    best_cost = float('inf')
    visited = {}

    def board_to_tuple(board):
        return tuple(tuple(row) for row in board)

    def recursive_search(state, path, depth):
        nonlocal best_solution, best_cost

        if depth > max_depth:
            return

        state_tuple = board_to_tuple(state)
        if state_tuple in visited and visited[state_tuple] <= depth:
            return
        visited[state_tuple] = depth

        if state == goal:
            if depth < best_cost:
                best_cost = depth
                best_solution = path.copy()
            return

        empty_pos = find_empty(state)
        neighbors = []

        for direction in DIRECTIONS:
            new_state = move_tile(state, empty_pos, direction)
            if new_state:
                h = heuristic(new_state, goal)
                neighbors.append((h, new_state))

        neighbors.sort(key=lambda x: x[0])

        for h, new_state in neighbors:
            if depth + h >= best_cost:  
                continue
            path.append(new_state)
            recursive_search(new_state, path, depth + 1)
            path.pop()

    recursive_search(start, [start], 0)
    return best_solution


def forward_tracking_8_puzzle(start, goal, max_depth=30, allowed_h_delta=1):
    initialize_goal_positions(goal)
    best_solution = None
    best_cost = float('inf')
    visited = {}

    def board_to_tuple(board):
        return tuple(tuple(row) for row in board)

    def dfs(state, path, g, depth):
        nonlocal best_solution, best_cost

        if depth > max_depth:
            return
        h = heuristic(state, goal)
        f = g + h
        if f >= best_cost:
            return

        if state == goal:
            if g < best_cost:
                best_cost = g
                best_solution = path.copy()
            return

        state_key = board_to_tuple(state)
        if state_key in visited and visited[state_key] <= g:
            return
        visited[state_key] = g

        empty_pos = find_empty(state)
        neighbors = []

        for direction in DIRECTIONS:
            new_state = move_tile(state, empty_pos, direction)
            if new_state:
                new_h = heuristic(new_state, goal)
                if new_h <= h + allowed_h_delta: 
                    neighbors.append((new_h, new_state))

        neighbors.sort(key=lambda x: x[0])

        for new_h, neighbor in neighbors:
            path.append(neighbor)
            dfs(neighbor, path, g + 1, depth + 1)
            path.pop()

    dfs(start, [start], 0, 0)
    return best_solution


def partial_order_search_8_puzzle(start, goal, max_depth=30, beam_width=3):
    initialize_goal_positions(goal)
    best_path = None
    visited = {}

    def board_to_tuple(board):
        return tuple(tuple(row) for row in board)

    def dfs(board, path, depth):
        nonlocal best_path

        if depth > max_depth:
            return
        if board == goal:
            if best_path is None or len(path) < len(best_path):
                best_path = path[:]
            return

        board_key = board_to_tuple(board)

        if board_key in visited and visited[board_key] <= depth:
            return
        visited[board_key] = depth

        empty_pos = find_empty(board)
        prev_board = path[-2] if len(path) >= 2 else None

        neighbors = []
        for direction in DIRECTIONS:
            new_board = move_tile(board, empty_pos, direction)
            if new_board:
                if new_board != prev_board:
                    h = heuristic(new_board, goal)
                    neighbors.append((h, new_board))

        neighbors.sort(key=lambda x: x[0])
        for _, neighbor in neighbors[:beam_width]:
            path.append(neighbor)
            dfs(neighbor, path, depth + 1)
            path.pop()

    dfs(start, [start], 0)
    return best_path


def minimum_conflicts_8_puzzle(start, goal, max_steps=1000, stuck_limit=50, epsilon=0.1):
    initialize_goal_positions(goal)
    current_board = [row[:] for row in start]
    best_board = current_board
    best_h = heuristic(current_board, goal)
    path = [current_board]
    visited = set()
    visited.add(board_to_tuple(current_board))
    stuck_counter = 0

    for steps in range(max_steps):
        if current_board == goal:
            return path

        empty_pos = find_empty(current_board)
        neighbors = []

        for direction in DIRECTIONS:
            new_board = move_tile(current_board, empty_pos, direction)
            if new_board:
                h = heuristic(new_board, goal)
                board_key = board_to_tuple(new_board)
                if board_key not in visited:
                    neighbors.append((h, new_board))

        if not neighbors:
            break

        neighbors.sort(key=lambda x: x[0])
        min_h = neighbors[0][0]

        candidates = [b for h, b in neighbors if h <= min_h + 1]
        next_board = random.choice(candidates) if random.random() > epsilon else random.choice([b for _, b in neighbors])

        next_h = heuristic(next_board, goal)
        board_key = board_to_tuple(next_board)

        if next_h < best_h:
            best_board = next_board
            best_h = next_h
            stuck_counter = 0
        else:
            stuck_counter += 1

        current_board = [row[:] for row in next_board]
        visited.add(board_key)
        path.append(current_board)

        if stuck_counter >= stuck_limit:
            current_board = [row[:] for row in start]
            path = [current_board]
            stuck_counter = 0
            visited.clear()
            visited.add(board_to_tuple(current_board))

    return path if current_board == goal else None


def q_learning_8_puzzle(start, goal, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps=100):
    initialize_goal_positions(goal)
    q_table = {}  
    actions = DIRECTIONS
    goal_tuple = board_to_tuple(goal)

    try:
        path_from_astar = a_star_8_puzzle(start, goal)
        if path_from_astar is not None:
            for i in range(len(path_from_astar) - 1):
                state = board_to_tuple(path_from_astar[i])
                next_state = board_to_tuple(path_from_astar[i + 1])
                empty_pos = find_empty(path_from_astar[i])
                action = None
                for dir in actions:
                    possible_next = move_tile(path_from_astar[i], empty_pos, dir)
                    if possible_next and board_to_tuple(possible_next) == next_state:
                        action = dir
                        break
                if action is not None:
                    q_table[(state, action)] = 100.0
    except Exception as e:
        pass

    def get_q(state, action):
        return q_table.get((state, action), 0.0)

    def choose_action(state):
        if random.random() < epsilon:
            return random.choice(actions)
        qs = [(get_q(state, a), a) for a in actions]
        return max(qs, key=lambda x: x[0])[1]

    for _ in range(episodes):
        state = [row[:] for row in generate_random_puzzle()]
        state_tuple = board_to_tuple(state)

        for _ in range(max_steps):
            action = choose_action(state_tuple)
            empty_pos = find_empty(state)
            next_state = move_tile(state, empty_pos, action)

            if next_state:
                next_state_tuple = board_to_tuple(next_state)
                reward = 100 if next_state_tuple == goal_tuple else -1

                max_q_next = max([get_q(next_state_tuple, a) for a in actions])
                old_q = get_q(state_tuple, action)
                new_q = old_q + alpha * (reward + gamma * max_q_next - old_q)

                q_table[(state_tuple, action)] = new_q

                if next_state_tuple == goal_tuple:
                    break

                state = next_state
                state_tuple = next_state_tuple
            else:
                q_table[(state_tuple, action)] = get_q(state_tuple, action) - 5 

    state = [row[:] for row in start]
    path = [state]
    visited = set()
    for _ in range(100):
        if board_to_tuple(state) == goal_tuple:
            return path
        state_tuple = board_to_tuple(state)
        action = choose_action(state_tuple)
        next_state = move_tile(state, find_empty(state), action)
        if not next_state or board_to_tuple(next_state) in visited:
            break
        path.append(next_state)
        visited.add(board_to_tuple(next_state))
        state = next_state

    return path if board_to_tuple(state) == goal_tuple else None




class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#F0F2F5")
        self.frame = tk.Frame(self.canvas, background="#F0F2F5")
        
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hsb = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)  

        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)  

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")  
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.configure(bg="#F0F2F5")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.resizable(True, True)

        self.colors = {
            "primary": "#2A9DF4",
            "secondary": "#187BCD",
            "background": "#F0F2F5",
            "tile": "#FFFFFF",
            "text": "#2D3436",
            "accent": "#FF7675",
            "zero_tile": "#FFD700"
        }

        self.original_random_state = generate_random_puzzle()
        self.start_state = [row[:] for row in self.original_random_state]
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.solution = []
        self.current_step = 0
        self.step_count = 0
        self.move_directions = []
        self.stop_requested = False
        self.stats = []

        solution = q_learning_8_puzzle(self.start_state, self.goal_state)
        if solution:
            print("Tìm được lời giải với Q-learning + A* hỗ trợ!")
        else:
            print("Q-learning không tìm được lời giải.")

        
        self.main_frame = tk.Frame(root, bg=self.colors["background"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.left_frame = tk.Frame(self.main_frame, bg=self.colors["background"])
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.main_frame, bg=self.colors["background"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.board_frame = tk.Frame(self.left_frame, bg=self.colors["background"])
        self.board_frame.pack(pady=10)

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(self.board_frame, text="", font=("Segoe UI", 36, "bold"), 
                                width=6, height=3,  
                                bg=self.colors["tile"], fg=self.colors["text"],
                                relief="ridge", borderwidth=4)
                btn.grid(row=i, column=j, padx=10, pady=10)  
                row.append(btn)
            self.buttons.append(row)

        self.control_frame = tk.Frame(self.right_frame, bg=self.colors["background"])
        self.control_frame.pack(fill=tk.X, pady=10)

        self.algo_scroll_frame = ScrollableFrame(self.control_frame)
        self.algo_scroll_frame.pack(fill=tk.BOTH, expand=True)

        categories = {
            "Uninformed Search Algorithms": [
                ("BFS", bfs_8_puzzle),
                ("DFS", dfs_8_puzzle),
                ("IDS", iterative_deepening_8_puzzle),
                ("UCS", ucs_8_puzzle)
            ],
            "Informed Search Algorithms": [
                ("Greedy", greedy_8_puzzle),
                ("A*", a_star_8_puzzle),
                ("IDA*", ida_star_8_puzzle),
                ("Simultaneous A*", simultaneous_a_star_8_puzzle)
            ],
            "Local Optimization Algorithms": [
                ("Simple Hill Climbing", simple_hill_climbing_8_puzzle),
                ("Steepest Hill Climbing", steepest_ascent_hill_climbing_8_puzzle),
                ("Stochastic Hill Climbing", stochastic_hill_climbing_8_puzzle),
                ("SA Hill Climbing", simulated_annealing_hill_climbing_8_puzzle),
                ("Randomized Hill Climbing", randomized_hill_climbing_8_puzzle),
                ("Simulated Annealing (SA)", simulated_annealing_8_puzzle),
                ("Beam Search", lambda s, g: beam_search_8_puzzle(s, g, beam_width=3)),
                ("Genetic Search", genetic_search_8_puzzle)
            ],
            "Search in Complex Environments": [
                ("Belief Propagation", belief_propagation_8_puzzle),
                
                ("No Observation Search", no_observation_search_8_puzzle),
                ("Partial Observation Search", partial_observation_search_8_puzzle),
                ("AND-OR Search", and_or_search_8_puzzle),               
                ("Partial Order Search", partial_order_search_8_puzzle),
                
            ],
            "Constraint Satisfaction Problem (CSP)": [
                ("Backtracking", backtracking_8_puzzle),
                ("Forward Tracking", forward_tracking_8_puzzle),
                ("Minimum Conflicts", minimum_conflicts_8_puzzle)
            ],
            "Machine Learning-Based Algorithms": [
                ("Q-Learning", q_learning_8_puzzle)
            ]
        }

        for category, algos in categories.items():
            cat_frame = tk.LabelFrame(self.algo_scroll_frame.frame, text=category, bg=self.colors["background"], fg=self.colors["text"])
            cat_frame.pack(fill=tk.X, pady=5)
            for i, (name, func) in enumerate(algos):
                row = i // 3
                col = i % 3
                btn = tk.Button(cat_frame, text=name, width=20, bg=self.colors["primary"], fg="white",
                                command=lambda f=func, n=name: self.solve_puzzle(f, n))
                btn.grid(row=row, column=col, padx=5, pady=5)

        self.utility_frame = tk.Frame(self.control_frame, bg=self.colors["background"])
        self.utility_frame.pack(fill=tk.X, pady=5)

        self.step_label = tk.Label(self.utility_frame, text="Steps: 0", font=("Segoe UI", 12, "bold"),
                                   bg=self.colors["background"], fg=self.colors["text"])
        self.step_label.pack(pady=5)

        self.reset_btn = tk.Button(self.utility_frame, text="Reset Puzzle", command=self.reset_board,
                                   bg=self.colors["accent"], fg="white", width=20)
        self.reset_btn.pack(pady=3)

        self.random_btn = tk.Button(self.utility_frame, text="Random Puzzle", command=self.randomize_board,
                                    bg=self.colors["accent"], fg="white", width=20)
        self.random_btn.pack(pady=3)

        self.stop_btn = tk.Button(self.utility_frame, text="Stop", command=self.stop_solving,
                                  bg="#e17055", fg="white", width=20)
        self.stop_btn.pack(pady=3)

        self.stats_btn = tk.Button(self.utility_frame, text="Thống kê", command=self.show_statistics,
                                   bg=self.colors["primary"], fg="white", width=20)
        self.stats_btn.pack(pady=3)

        self.moves_label = tk.Label(self.right_frame, text="Move Directions", font=("Segoe UI", 14, "bold"),
                                    bg=self.colors["background"], fg=self.colors["text"])
        self.moves_label.pack(pady=(10, 5))

        self.moves_text = tk.Text(self.right_frame, height=20, width=30, font=("Segoe UI", 12),
                                  bg="#FFFFFF", fg=self.colors["text"], relief="flat", bd=2)
        self.moves_text.pack(fill=tk.BOTH, expand=True, padx=10)
        self.moves_text.config(state=tk.DISABLED)

        self.update_board(self.start_state)

    def update_board(self, state):
        for i in range(3):
            for j in range(3):
                value = state[i][j]
                self.buttons[i][j].config(text=str(value) if value != 0 else "",
                                          bg=self.colors["zero_tile"] if value == 0 else self.colors["tile"])

    def reset_board(self):
        self.start_state = [row[:] for row in self.original_random_state]
        self.update_board(self.start_state)
        self.step_count = 0
        self.current_step = 0
        self.solution = []
        self.move_directions = []
        self.step_label.config(text="Steps: 0")
        self.update_moves_display()

    def randomize_board(self):
        self.original_random_state = generate_random_puzzle()
        self.reset_board()
        self.stats.clear()  

    def stop_solving(self):
        self.stop_requested = True

    def show_statistics(self):
        stat_window = tk.Toplevel(self.root)
        stat_window.title("Thống kê")
        stat_window.geometry("400x300")  
        stat_window.configure(bg="#F0F2F5")

        headers = ["Thuật toán", "Thời gian", "Số bước"]
        for col, header in enumerate(headers):
            tk.Label(stat_window, text=header, font=("Segoe UI", 10, "bold"), bg="#F0F2F5",
                     padx=10, pady=5).grid(row=0, column=col, sticky="nsew")

        for i, (name, time_str, steps) in enumerate(self.stats, start=1):
            tk.Label(stat_window, text=name, bg="#F0F2F5", padx=10, pady=5).grid(row=i, column=0, sticky="w")
            tk.Label(stat_window, text=time_str, bg="#F0F2F5", padx=10, pady=5).grid(row=i, column=1, sticky="w")
            tk.Label(stat_window, text=steps, bg="#F0F2F5", padx=10, pady=5).grid(row=i, column=2, sticky="w")

        for col in range(len(headers)):
            stat_window.grid_columnconfigure(col, weight=1)

    def solve_puzzle(self, algorithm, algo_name):
        def run_algorithm():
            try:
                self.stop_requested = False
                self.reset_btn.config(state=tk.DISABLED)
                self.random_btn.config(state=tk.DISABLED)
                self.step_count = 0
                self.move_directions = []
                self.update_moves_display()
                self.step_label.config(text="Đang giải...")

                loading = tk.Toplevel(self.root)
                loading.title("Đang giải")
                loading.geometry("200x100")
                loading_label = tk.Label(loading, text="Đang tìm lời giải...", font=("Segoe UI", 12))
                loading_label.pack(expand=True)
                loading.transient(self.root)
                loading.grab_set()
                self.root.update()

                if not is_solvable(self.start_state):
                    loading.destroy() 
                    messagebox.showerror("Unsolvable", f"{algo_name}: Trạng thái hiện tại không có lời giải.")
                    self.step_label.config(text="Steps: 0")
                    return

                start_time = time.time()
                solution = algorithm(self.start_state, self.goal_state)
                elapsed_time = time.time() - start_time

                if self.stop_requested:
                    return

                self.solution = solution
                if solution and isinstance(solution, list):
                    steps = len(solution) - 1
                    self.stats.append((algo_name, f"{elapsed_time:.2f}s", str(steps)))
                    loading.destroy() 
                    messagebox.showinfo("Solved", f"{algo_name} solved in {steps} steps.\nTime: {elapsed_time:.2f}s")
                    self.current_step = 0
                    self.step_label.config(text=f"Steps: {steps}")  
                    self.animate_solution()
                else:
                    self.stats.append((algo_name, f"{elapsed_time:.2f}s", "Không có lời giải"))
                    loading.destroy() 
                    messagebox.showerror("Error", f"{algo_name} không tìm được lời giải.")
                    self.step_label.config(text="Steps: 0") 
            finally:
                self.reset_btn.config(state=tk.NORMAL)
                self.random_btn.config(state=tk.NORMAL)

        threading.Thread(target=run_algorithm).start()

    def animate_solution(self):
        if self.stop_requested or not self.solution:
            return
        if self.current_step < len(self.solution):
            board = self.solution[self.current_step]
            self.update_board(board)
            if self.current_step > 0:
                direction = self.get_move_direction(self.solution[self.current_step-1], board)
                if direction:
                    self.move_directions.append(direction)
                    self.update_moves_display()
            self.step_count = len(self.move_directions)
            self.step_label.config(text=f"Steps: {self.step_count}")
            self.current_step += 1
            self.root.after(400, self.animate_solution)

    def get_move_direction(self, prev_board, curr_board):
        try:
            prev_empty = find_empty(prev_board)
            curr_empty = find_empty(curr_board)
            dx = curr_empty[0] - prev_empty[0]
            dy = curr_empty[1] - prev_empty[1]
            return DIRECTION_NAMES.get((dx, dy), "")
        except:
            return ""

    def update_moves_display(self):
        self.moves_text.config(state=tk.NORMAL)
        self.moves_text.delete(1.0, tk.END)
        for i, direction in enumerate(self.move_directions, 1):
            self.moves_text.insert(tk.END, f"{i}. {direction}\n")
        self.moves_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()