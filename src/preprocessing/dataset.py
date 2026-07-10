import torch
import h5py
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, filepath, max_size=None):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.input_keys = list(f['input'].attrs['key_order'])
            self.output_keys = list(f['output'].attrs['key_order'])
            
            total_len = len(f['input'][self.input_keys[0]])
            self.dataset_len = total_len if max_size is None else min(total_len, max_size)

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.input_datasets = {k: self.file[f'input/{k}'] for k in self.input_keys}
        self.output_datasets = {k: self.file[f'output/{k}'] for k in self.output_keys}
        self.cost_dataset = self.file['C']
        
    def _to_tensor(self, val):
        """Helper para convertir datos a tensores de forma eficiente"""
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return torch.tensor(val)

    def __getitem__(self, idx):
        if self.file is None: 
            self._open_file()
            
        inputs = [self._to_tensor(self.input_datasets[k][idx]) for k in self.input_keys]
        outputs = [self._to_tensor(self.output_datasets[k][idx]) for k in self.output_keys]
        return tuple(inputs), tuple(outputs)
    
    def __len__(self):
        return self.dataset_len

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['input_datasets'] = None
        state['output_datasets'] = None
        state['cost_dataset'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None

def load_dataset(filepath, max_size=None, verbose=True):
    dataset = H5Dataset(DATA_FOLDER / filepath, max_size)
    if verbose:
        print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data_from_path(filepath):
    """Carga datos respetando la estructura de grupos input/output"""
    with h5py.File(filepath, "r") as f:
        # Extraemos keys de los grupos
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        data = {}
        # Cargamos los inputs
        for k in input_keys:
            data[f'input/{k}'] = f['input'][k][:]
            
        # Cargamos los outputs
        for k in output_keys:
            data[f'output/{k}'] = f['output'][k][:]
            
        data['C'] = f['C'][:]
        
        # Guardamos metadatos de orden para la reconstrucción
        data['_input_order'] = input_keys
        data['_output_order'] = output_keys
        return data
    
def load_data(filename):
    return load_data_from_path(DATA_FOLDER / filename)

def generate_dataset(data_files, output_name, min_cost=0, max_cost=999999, max_size=999999, balance_method=None, seed=42):
    """
    Genera un dataset combinado y opcionalmente balanceado.
    
    balance_method: None (sin balanceo), 'cost' (por costo), 'file' (por archivo de origen)
    """
    output_path = DATA_FOLDER / output_name
    all_data = {}
    input_order = []
    output_order = []
    file_indices = [] # Guardará el índice del archivo origen para cada fila
    
    for idx, data_file in enumerate(data_files):
        path = DATA_FOLDER / data_file
        if path.exists():
            data = load_data_from_path(path)
            if not all_data:
                all_data = {k: [] for k in data.keys()}
                input_order = data['_input_order']
                output_order = data['_output_order']
            
            # Asumimos que todas las keys tienen el mismo largo de filas en este archivo
            # Usamos una key común que no empiece con '_' para medir el largo
            any_key = [k for k in data if not k.startswith('_')][0]
            num_rows = len(data[any_key])
            
            # Guardamos a qué archivo pertenece cada fila de este bloque
            file_indices.append(np.full(num_rows, idx))
            
            for k in data:
                all_data[k].append(data[k])

    if not all_data: return

    combined_data = {k: np.concatenate(all_data[k], axis=0) for k in all_data if not k.startswith('_')}
    # Creamos el array global de origen de archivos
    file_origin = np.concatenate(file_indices, axis=0)
    
    # 1. Aplicamos el filtro de rango de costo inicial
    mask = (combined_data['C'] >= min_cost) & (combined_data['C'] <= max_cost)
    file_origin = file_origin[mask] # También filtramos el vector de origen
    for k in combined_data:
        combined_data[k] = combined_data[k][mask]

    # Usamos el nuevo generador de NumPy recomendado para evitar alterar el estado global
    rng = np.random.default_rng(seed)
    shuffle_indices = rng.permutation(len(combined_data['C']))
    
    # Mezclamos tanto el origen de archivos como todas las matrices de datos de forma alineada
    file_origin = file_origin[shuffle_indices]
    for k in combined_data:
        combined_data[k] = combined_data[k][shuffle_indices]

    # 2. Lógica de balanceo y tamaño máximo integrados
    if len(combined_data['C']) > 0:
        
        # --- OPCIÓN A: BALANCEO POR COSTO ---
        if balance_method == 'cost':
            unique_costs, counts = np.unique(combined_data['C'], return_counts=True)
            num_costs = len(unique_costs)
            
            ideal_samples_per_cost = max_size // num_costs
            limit = min(np.min(counts), ideal_samples_per_cost)
            
            balanced_indices = []
            for cost in unique_costs:
                indices = np.where(combined_data['C'] == cost)[0]
                balanced_indices.extend(indices[:limit])
            
            current_idx = 0
            while len(balanced_indices) < max_size and len(balanced_indices) < len(combined_data['C']):
                cost_to_fill = unique_costs[current_idx % num_costs]
                all_indices_for_cost = np.where(combined_data['C'] == cost_to_fill)[0]
                
                used_count_for_this_cost = limit + (current_idx // num_costs)
                if used_count_for_this_cost < len(all_indices_for_cost):
                    balanced_indices.append(all_indices_for_cost[used_count_for_this_cost])
                    current_idx += 1
                else:
                    break 

            balanced_indices.sort()
            for k in combined_data:
                combined_data[k] = combined_data[k][balanced_indices]
                
        # --- OPCIÓN B: BALANCEO POR ARCHIVO (NUEVA) ---
        elif balance_method == 'file':
            unique_files, counts = np.unique(file_origin, return_counts=True)
            num_files = len(unique_files)
            
            # Mismo principio: cuántas muestras ideales por archivo
            ideal_samples_per_file = max_size // num_files
            limit = min(np.min(counts), ideal_samples_per_file)
            
            balanced_indices = []
            for f_idx in unique_files:
                indices = np.where(file_origin == f_idx)[0]
                balanced_indices.extend(indices[:limit])
                
            # Rellenar uno a uno si sobra espacio por redondeo (igual que con el costo)
            current_idx = 0
            while len(balanced_indices) < max_size and len(balanced_indices) < len(combined_data['C']):
                file_to_fill = unique_files[current_idx % num_files]
                all_indices_for_file = np.where(file_origin == file_to_fill)[0]
                
                used_count_for_this_file = limit + (current_idx // num_files)
                if used_count_for_this_file < len(all_indices_for_file):
                    balanced_indices.append(all_indices_for_file[used_count_for_this_file])
                    current_idx += 1
                else:
                    break
                    
            balanced_indices.sort()
            for k in combined_data:
                combined_data[k] = combined_data[k][balanced_indices]
        
        # --- OPCIÓN C: SIN BALANCEO ---
        else:
            final_len = min(len(combined_data['C']), max_size)
            for k in combined_data:
                combined_data[k] = combined_data[k][:final_len]

    # 3. Aplicamos el límite final de tamaño (max_size)
    final_len = min(len(combined_data['C']), max_size)
    for k in combined_data:
        combined_data[k] = combined_data[k][:final_len]

    # Escritura del archivo H5
    with h5py.File(output_path, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")
        
        for k in input_order:
            full_key = f'input/{k}'
            g_input.create_dataset(k, data=combined_data[full_key])
        g_input.attrs['key_order'] = input_order
        
        for k in output_order:
            full_key = f'output/{k}'
            g_output.create_dataset(k, data=combined_data[full_key])
        g_output.attrs['key_order'] = output_order
        
        f.create_dataset("C", data=combined_data['C'])

    print(f"Dataset generado exitosamente en: {output_path} (Tamaño {final_len})")