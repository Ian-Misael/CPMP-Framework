from settings import INSTANCE_FOLDER, DATA_FOLDER
from generation.instances import read_instance
import copy
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from solvers.FRG import FRGSolver
from solvers.model import ModelSolver
import torch
import random

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)

    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))

    return moves
    
def get_moves_costs(layout, H, max_steps):
    moves = get_feasible_moves(layout)

    lay_copies = []
    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        lay_copies.append(lay_copy)

    results = worker_solver.solve_from_layouts(lay_copies, H, max_steps)
    worker_solver.reset()

    all_moves_costs = []
    min_cost = float('inf')

    for move, (solved, cost) in zip(moves, results):
        if not solved: 
            continue

        if cost < min_cost:
            min_cost = cost + 1
            
        all_moves_costs.append((move, cost + 1))
            
    return all_moves_costs, min_cost

def generate_data_from_file(filepath):
    layout = read_instance(filepath, worker_H)
    if layout.is_sorted():
        return None

    input_vec = worker_la_adapter.input_2_vec(layout, worker_H)

    moves_costs, best_cost = get_moves_costs(layout, worker_H, worker_max_steps)
    if len(moves_costs) == 0:
        return None

    output_vec = worker_ma_adapter.output_2_vec(moves_costs)

    return input_vec, output_vec, best_cost

def generate_data(filepaths, input_adapter, output_adapter, init_worker, init_args, num_workers):
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=init_args
    ) as executor:
        results = list(executor.map(generate_data_from_file, filepaths))

    la_class, *la_args = input_adapter
    ma_class, *ma_args = output_adapter
    input_adapter = la_class(*la_args)
    output_adapter = ma_class(*ma_args)

    costs = []
    for result in results:
        if result is None:
            continue

        input_vec, output_vec, cost = result
        input_adapter.add(input_vec)
        output_adapter.add(output_vec)
        costs.append(cost)

    input_data = input_adapter.get()
    output_data = output_adapter.get()

    return input_data, output_data, costs

def save_data(input_data, output_data, costs, output_name):
    output_path = DATA_FOLDER / f"{output_name}"

    with h5py.File(output_path, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")

        input_keys = list(input_data.keys())
        for key in input_keys:
            g_input.create_dataset(key, data=input_data[key])
        g_input.attrs['key_order'] = [k for k in input_keys]

        output_keys = list(output_data.keys())
        for key in output_keys:
            g_output.create_dataset(key, data=output_data[key])
        g_output.attrs['key_order'] = [k for k in output_keys]

        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {len(output_data[key])})")

def init_worker(H, max_steps, input_adapter_config, output_adapter_config):
    global worker_la_adapter
    global worker_ma_adapter
    global worker_H
    global worker_max_steps

    la_class, *la_args = input_adapter_config
    ma_class, *ma_args = output_adapter_config
    worker_la_adapter = la_class(*la_args)
    worker_ma_adapter = ma_class(*ma_args)

    worker_H = H
    worker_max_steps = max_steps

def init_worker_sl(H, max_steps, input_adapter_config, output_adapter_config, solver_config):
    global worker_solver

    init_worker(H, max_steps, input_adapter_config, output_adapter_config)

    solver_class, *solver_args = solver_config
    worker_solver = solver_class(*solver_args)

def generate_data_sl(folder, H, max_steps, input_adapter_config, output_adapter_config, solver_config, num_workers, output_name_prefix=None):
    # Agrupamos los argumentos de inicialización
    init_args = (H, max_steps, input_adapter_config, output_adapter_config, solver_config)
    
    # Construimos las rutas de las instancias dentro de la carpeta seleccionada
    folder_path = INSTANCE_FOLDER / folder
    instance_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    # Definimos el nombre del archivo de salida
    output_name = f"{folder}.data"
    if output_name_prefix:
        output_name = f"{output_name_prefix}_{output_name}"
    
    # Ejecutamos la generación de datos para la carpeta específica
    input_data, output_data, costs = generate_data(
        instance_files, 
        input_adapter_config, 
        output_adapter_config, 
        init_worker_sl, 
        init_args, 
        num_workers
    )
    
    # Guardamos los resultados
    save_data(input_data, output_data, costs, output_name)
    
def init_worker_rl(H, max_steps, model_cls, model_params, weights, input_adapter_config, output_adapter_config, batch_size):
    global worker_solver

    torch.set_num_threads(1) 
    torch.set_num_interop_threads(1)

    init_worker(H, max_steps, input_adapter_config, output_adapter_config)
    model = model_cls(**model_params)
    model.load_state_dict(weights)
    model.eval()
    worker_solver = ModelSolver(model, worker_la_adapter, batch_size)

def generate_data_rl(instance_files, H, max_steps, input_adapter_config, output_adapter_config, model, batch_size, num_workers, output_name):
    model_cls = model.__class__
    model_params = model.hyperparams
    weights = model.state_dict()
    
    temp_inputs = {}
    temp_outputs = {}
    all_costs = []

    for files, H_file in zip(instance_files, H):
        init_args = (H_file, max_steps, model_cls, model_params, weights, input_adapter_config, output_adapter_config, batch_size)

        input_data, output_data, costs = generate_data(files, input_adapter_config, output_adapter_config, init_worker_rl, init_args, num_workers)
        
        # Agrupamos los diccionarios en listas de arrays
        for k, v in input_data.items():
            temp_inputs.setdefault(k, []).append(v)
        
        for k, v in output_data.items():
            temp_outputs.setdefault(k, []).append(v)
            
        all_costs.extend(costs)

    all_input_data = {k: np.concatenate(v) for k, v in temp_inputs.items()}
    all_output_data = {k: np.concatenate(v) for k, v in temp_outputs.items()}

    save_data(all_input_data, all_output_data, all_costs, output_name)

def split_instances(folder, p1, p2, seed):
    # 1. Preparación de archivos
    path = INSTANCE_FOLDER / folder
    instance_files = [os.path.join(folder, f) for f in os.listdir(path)]
    
    # 2. Mezcla aleatoria reproducible
    random.seed(seed)
    random.shuffle(instance_files)
    
    # 3. Normalización de p1 y p2
    total_p = p1 + p2
    p1_norm = p1 / total_p
    
    # 4. Cálculo del índice de división
    total_files = len(instance_files)
    limit = int(total_files * p1_norm)
    
    # 5. Segmentación (Slicing)
    # list1 toma desde el inicio hasta 'limit'
    # list2 toma desde 'limit' hasta el final (asegurando el uso de todos los archivos)
    list1 = instance_files[:limit]
    list2 = instance_files[limit:]
    
    return list1, list2

# Variable globales
worker_solver = None
worker_la_adapter = None
worker_ma_adapter = None
worker_H = None
worker_max_steps = None