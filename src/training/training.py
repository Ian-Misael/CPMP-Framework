from torch.utils.data import DataLoader, Subset
import torch
import os
import copy
import json
from settings import INSTANCE_FOLDER, MODELS_FOLDER, HYPERPARAMETERS_FOLDER
from torch.amp import GradScaler, autocast
from training.metrics import *
import random
from generation.data import generate_data_rl, split_instances
from preprocessing.dataset import load_dataset
import torch.multiprocessing as mp
import numpy as np
from utils.utils import distribuir_suma_exacta
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
from dataclasses import dataclass
    
class ModelScorer:
    def __init__(self, model):
        self.model = model
        self.best_models = {}

    def update_best_models(self, epoch, val_metrics: EpochMetrics):
        for metric in val_metrics.metrics:
            sign = 1 if metric.maximize else -1
            score = sign * val_metrics.metrics[metric][-1]

            if metric in self.best_models and score < self.best_models[metric]["score"]: continue

            if metric not in self.best_models:
                self.best_models[metric] = {}
                
            self.best_models[metric]["score"] = score
            self.best_models[metric]["weights"] = copy.deepcopy(self.model.state_dict())
            self.best_models[metric]["epoch"] = epoch

    def print_best_scores(self):
        print("Mejores modelos por métrica:")
        for metric in self.best_models:
            sign = 1 if metric.maximize else -1
            print(f"    {metric.name}: {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
        
    def print_best_score(self, metric):
        sign = 1 if metric.maximize else -1
        print(f"Mejor modelo ({metric.name}): {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
    
    def get_best_weights(self):
        return {metric.name: self.best_models[metric]["weights"] for metric in self.best_models}
    
    def get_best_weights_by_metric(self, metric):
        return self.best_models[metric]["weights"]
    
    def get_last_update_epoch(self, metric):
        return self.best_models[metric]["epoch"]
    
@dataclass
class LRConfig:
    start: float            # Tasa de aprendizaje inicial
    factor: float = 0.5     # Factor de reducción
    patience: int = 999999  # Épocas sin mejora antes de reducir el LR
    min: float = 0.0        # Tasa de aprendizaje mínima permitida
    
def train_epoch(model, train_loader, optimizer, loss_functions, metrics_list, device, scaler):
    """
    metrics_list: Lista de listas. metrics_list[i] son las métricas para la salida i.
    """
    model.train()

    for inputs_batch, y_batch in train_loader:
        inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
        targets = [t.to(device, non_blocking=True) for t in y_batch]

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type):
            logits_list = model(*inputs)
            if not isinstance(logits_list, (list, tuple)):
                logits_list = [logits_list]

            total_loss = 0
            # Iteramos por cada salida del modelo
            for i, (lf, logits, target) in enumerate(zip(loss_functions, logits_list, targets)):
                # 1. Pérdida
                total_loss += lf.step(logits, target)
                
                # 2. Métricas específicas de esta salida
                for metric in metrics_list[i]:
                    metric.step(logits, target)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Computar resultados finales de la época
    losses = [lf.compute() for lf in loss_functions]
    m_values = [[m.compute() for m in m_sublist] for m_sublist in metrics_list]
    
    return losses, m_values

def val_epoch(model, val_loader, loss_functions, metrics_list, device):
    model.eval()

    with torch.no_grad():
        for inputs_batch, y_batch in val_loader:
            inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
            targets = [t.to(device, non_blocking=True) for t in y_batch]

            logits_list = model(*inputs)
            if not isinstance(logits_list, (list, tuple)):
                logits_list = [logits_list]

            for i, (lf, logits, target) in enumerate(zip(loss_functions, logits_list, targets)):
                lf.step(logits, target)
                for metric in metrics_list[i]:
                    metric.step(logits, target)

    losses = [lf.compute() for lf in loss_functions]
    m_values = [[m.compute() for m in m_sublist] for m_sublist in metrics_list]
    
    return losses, m_values

def _train(model, epochs, train_set, test_set, batch_size, lr_config: LRConfig, weight_decay, loss_functions, print_epoch_results, model_scorer, patience, metrics_list, device): 
    num_workers = os.cpu_count()
    use_pin_memory = device.type in ['cuda', 'mps']

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory)
    
    # Inicializamos el optimizador usando el lr inicial (start)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_config.start, weight_decay=weight_decay)
    
    # Configuramos el scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_config.factor, patience=lr_config.patience, min_lr=lr_config.min)
    
    scaler = GradScaler(device.type)

    train_metrics, val_metrics = EpochMetrics(), EpochMetrics()
    primary_loss = loss_functions[0]

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        train_loss_vals, train_m_vals = train_epoch(model, train_loader, optimizer, loss_functions, metrics_list, device, scaler)
        
        for lf, val in zip(loss_functions, train_loss_vals): 
            train_metrics.add_value(lf, val)
        # Añadir métricas (aplanando la lista de listas)
        for i, sublist in enumerate(train_m_vals):
            for j, val in enumerate(sublist):
                train_metrics.add_value(metrics_list[i][j], val)

        # --- VAL ---
        val_loss_vals, val_m_vals = val_epoch(model, test_loader, loss_functions, metrics_list, device)
        
        for lf, val in zip(loss_functions, val_loss_vals): 
            val_metrics.add_value(lf, val)
        for i, sublist in enumerate(val_m_vals):
            for j, val in enumerate(sublist):
                val_metrics.add_value(metrics_list[i][j], val)

        print_epoch_results(epoch, train_metrics, val_metrics)
        model_scorer.update_best_models(epoch, val_metrics)

        # Usamos la pérdida primaria de validación para actualizar el scheduler
        primary_val_loss = val_loss_vals[0]
        scheduler.step(primary_val_loss)

        # Early stopping (nota: la paciencia aquí es la global de la función train, no la del LR)
        if epoch - model_scorer.get_last_update_epoch(primary_loss) > patience:
            print(f"Early stopping en época {epoch} (Pérdida primaria: {primary_loss.name})")
            break

    return train_metrics, val_metrics

def generate_sets(dataset, train_size, test_size, seed):
    with h5py.File(dataset.filepath, 'r') as f:
        costs = np.array(f['C'][:dataset.dataset_len])
    
    indices = np.arange(len(dataset))
    used_size = train_size + test_size
    
    def get_safe_labels(labels):
        """
        Función auxiliar que agrupa dinámicamente cualquier clase 
        con menos de 2 muestras justo antes de un split.
        """
        safe = labels.copy()
        unique, counts = np.unique(safe, return_counts=True)
        rare = unique[counts < 2]
        
        # 1. Agrupar todas las clases con 1 sola muestra bajo la etiqueta -1
        for r in rare:
            safe[safe == r] = -1
            
        # 2. Salvaguarda: Si el grupo '-1' en su totalidad solo suma 1 muestra,
        # lo fusionamos con la clase mayoritaria de este subconjunto.
        if 0 < np.sum(safe == -1) < 2:
            valid_mask = unique != -1
            if np.any(valid_mask):
                most_freq = unique[valid_mask][np.argmax(counts[valid_mask])]
                safe[safe == -1] = most_freq
                
        return safe

    # --- PASO A: Extraer la porción útil (train + test) y descartar el resto ---
    if used_size < len(dataset):
        safe_labels_A = get_safe_labels(costs)
        # Observa que aquí recuperamos "used_costs" intactos (sin el -1) 
        # para pasarlos al siguiente nivel
        used_idx, _, used_costs, _ = train_test_split(
            indices, costs, 
            train_size=used_size, 
            stratify=safe_labels_A, 
            random_state=seed
        )
    else:
        used_idx = indices
        used_costs = costs

    # --- PASO B: Dividir la porción útil en train y test ---
    # Al haber recortado los datos, used_costs puede tener nuevas clases con 1 muestra.
    # Volvemos a generar etiquetas seguras exclusivas para este nuevo subconjunto.
    safe_labels_B = get_safe_labels(used_costs)
    
    train_idx, test_idx = train_test_split(
        used_idx, 
        train_size=train_size, 
        stratify=safe_labels_B, 
        random_state=seed
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

def config_training(model, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)
    return device

def train(model, epochs, train_set, test_set, batch_size, lr_config: LRConfig, weight_decay, loss_functions, patience, metrics, device):
    model_scorer = ModelScorer(model)
    primary_loss = loss_functions[0]

    def print_epoch_results(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics):
        print(f"{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs}")
        
        train_loss_str = " | ".join([f"Train {lf.name}: {lf.format(train_metrics.get_last_value(lf))}" for lf in loss_functions])
        val_loss_str = " | ".join([f"Val {lf.name}: {lf.format(val_metrics.get_last_value(lf))}" for lf in loss_functions])
        
        print(f"    {train_loss_str}")
        print(f"    {val_loss_str}")

        for i, metric in enumerate(val_metrics.metrics):
            if metric in loss_functions: 
                continue
            value = val_metrics.get_last_value(metric)
            print(f"{' | ' if i > 0 else '    '}{metric.name}: {metric.format(value)}", end='')
        print()

    # Pasamos lr_config en lugar de learning_rate
    _train(model, epochs, train_set, test_set, batch_size, lr_config, weight_decay, loss_functions, print_epoch_results, model_scorer, patience, metrics, device)
    
    weights = model_scorer.get_best_weights_by_metric(primary_loss)
    model.load_state_dict(weights)
    model_scorer.print_best_score(primary_loss)

    return model

def sl_train(model, epochs, dataset, train_size, test_size, batch_size, lr_config, weight_decay, loss_functions, patience, metrics, seed=42):
    device = config_training(model, seed)
    train_set, test_set = generate_sets(dataset, train_size, test_size, seed)
    return train(model, epochs, train_set, test_set, batch_size, lr_config, weight_decay, loss_functions, patience, metrics, device)

class DataGenerationConfigRL():
    def __init__(self, instance_sets, H, max_steps, input_adapter_config, output_adapter_config, num_workers):
        self.instance_sets = instance_sets
        self.H = H
        self.max_steps = max_steps
        self.input_adapter_config = input_adapter_config
        self.output_adapter_config = output_adapter_config
        self.num_workers = num_workers

def split_instances(folders, train_size, test_size, seed):
    # Mezcla aleatoria reproducible
    random.seed(seed)
    
    instance_files = []
    for instance_set in folders:
        path = INSTANCE_FOLDER / instance_set
        set_files = [os.path.join(path, f) for f in os.listdir(path)]
        random.shuffle(set_files)
        instance_files.append(set_files)

    files_len = [len(files) for files in instance_files]
    if sum(files_len) < train_size + test_size:
        train_size, test_size = distribuir_suma_exacta([train_size, test_size], sum(files_len))

    train_sizes = distribuir_suma_exacta(files_len, train_size)
    test_sizes = distribuir_suma_exacta(files_len, test_size)

    train_instances = []
    test_instances = []
    for i in range(len(instance_files)):
        train_instances.append(instance_files[i][:train_sizes[i]])
        test_instances.append(instance_files[i][train_sizes[i]:train_sizes[i] + test_sizes[i]])

    return train_instances, test_instances

def rl_train(model, iterations, datagen_config, epochs, train_size, test_size, batch_size, lr_config, weight_decay, loss_functions, patience, metrics, seed=42):
    device = config_training(model, seed)
    train_set_file = "tmp_train.data"
    test_set_file = "tmp_test.data"
    last_avg_cost_test = None
    i = 0

    train_instances, test_instances = split_instances(datagen_config.instance_sets, train_size, test_size, seed)

    try:
        while True:
            if i > 0: print()

            mp.set_start_method('spawn', force=True)
            generate_data_rl(train_instances, 
                datagen_config.H,
                datagen_config.max_steps,
                datagen_config.input_adapter_config,
                datagen_config.output_adapter_config,
                model,
                batch_size,
                datagen_config.num_workers,
                output_name=train_set_file)
            
            generate_data_rl(test_instances, 
                datagen_config.H,
                datagen_config.max_steps,
                datagen_config.input_adapter_config,
                datagen_config.output_adapter_config,
                model,
                batch_size,
                datagen_config.num_workers,
                output_name=test_set_file)
            
            train_set = load_dataset(train_set_file, verbose=False)
            test_set = load_dataset(test_set_file, verbose=False)

            # EXTRAEMOS DATOS DE TRAIN
            train_set._open_file()
            # Leemos el dataset como arreglo a memoria con [:]
            real_costs_train = train_set.file['realCost'][:] 
            # np.nanmean calcula el promedio ignorando los NaN
            avg_cost_train = np.nanmean(real_costs_train)
            # Contamos cuántos elementos NO son NaN
            solved_train = np.count_nonzero(~np.isnan(real_costs_train))
            total_train = len(real_costs_train)
            train_set.close()

            # EXTRAEMOS DATOS DE TEST
            test_set._open_file()
            real_costs_test = test_set.file['realCost'][:]
            avg_cost_test = np.nanmean(real_costs_test)
            solved_test = np.count_nonzero(~np.isnan(real_costs_test))
            total_test = len(real_costs_test)
            test_set.close()

            print(f"Tamaño datasets | Train: {len(train_set)} | Test: {len(test_set)}")
            print(f"Instancias resueltas | Train: {solved_train}/{total_train} ({(solved_train/total_train)*100:.1f}%) | Test: {solved_test}/{total_test} ({(solved_test/total_test)*100:.1f}%)")
            print(f"Costo promedio | Train: {avg_cost_train:.2f} | Test: {avg_cost_test:.2f}")

            if last_avg_cost_test:
                current_cost_red = -(avg_cost_test - last_avg_cost_test)
                total_cost_red = -(avg_cost_test - start_avg_cost_test)
                current_gap = current_cost_red / last_avg_cost_test * 100
                total_gap = total_cost_red / start_avg_cost_test * 100

                print(f"Reducción del Costo: {current_cost_red:.2f} (acumulado {total_cost_red:.2f})")
                print(f"Reducción del Gap: {current_gap:.2f}% (acumulado {total_gap:.2f}%)")

                if avg_cost_test >= last_avg_cost_test:
                    print(f"Early stopping en iteración {i+1}")
                    break
            else:
                start_avg_cost_test = avg_cost_test

            last_avg_cost_test = avg_cost_test
            best_weights = model.state_dict()

            if i == iterations: break
            model = train(model, epochs, train_set, test_set, batch_size, lr_config, weight_decay, loss_functions, patience, metrics, device)
            i += 1

        model.load_state_dict(best_weights)
        return model
    
    finally:
        if os.path.exists(train_set_file):
            os.remove(train_set_file)
        if os.path.exists(test_set_file):
            os.remove(test_set_file)

def save_model(model, model_name):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    print(f"✅ Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model