import multiprocessing as mp
import copy
import random
import numpy as np
from scipy.optimize import nnls
from instances.generators.instance_generator import InstanceGenerator
from instances.generators.random_moves import random_moves
from solvers import FRGSolver, ModelSolver 
from cpmp.layout import Layout
from utils.utils import distribuir_suma_exacta
import torch

# =============================================================================
# FUNCIONES DE WORKER (Deben ser globales para que Multiprocessing pueda usarlas)
# =============================================================================

_worker_solver = None

def _set_worker_seed():
    """Desvincula las semillas aleatorias para que cada worker sea verdaderamente independiente."""
    import os
    seed = int.from_bytes(os.urandom(4), 'little')
    random.seed(seed)
    np.random.seed(seed)

def init_worker_frg():
    """Inicializador exclusivo para FRG."""
    global _worker_solver
    
    _worker_solver = FRGSolver()
    _set_worker_seed()

def init_worker_model(model_cls, model_params, weights, input_adapter_config, batch_size):
    """Inicializador exclusivo para el modelo neuronal."""
    global _worker_solver
    
    # Evitar colisiones de hilos entre PyTorch y Multiprocessing
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    # 1. Reconstruir el modelo desde cero
    model = model_cls(**model_params)
    model.load_state_dict(weights)
    model.eval()
    
    # 2. Desempaquetar e instanciar el adaptador (igual que en tu ejemplo)
    adapter_cls, *adapter_args = input_adapter_config
    adapter = adapter_cls(*adapter_args)
    
    # 3. Instanciar el solver en el espacio global del worker
    _worker_solver = ModelSolver(model, adapter, batch_size)
    _set_worker_seed()

def worker_task(payload):
    """Tarea unificada: funciona igual sin importar qué solver está cargado."""
    stacks, H, k = payload
    
    # Aplicar movimientos aleatorios
    stacks = random_moves(stacks, H, k)
    lay = Layout(stacks, H)
    
    # Aquí asumo que tu _solve_layout unificado llama a la API común del solver
    cost = _solve_layout(lay, H)
    
    # Limpiamos estados internos si es necesario
    _worker_solver.reset()
        
    return k, copy.deepcopy(lay.stacks), cost

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)
    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))
    return moves

def _solve_layout(layout, H):
    """Función de resolución unificada usando el solver global."""
    global _worker_solver
    moves = get_feasible_moves(layout) # Asegúrate de definir/importar esto

    lay_copies = []
    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        lay_copy.steps = 0
        lay_copies.append(lay_copy)

    # Ambos solvers deben soportar esta interfaz
    results = _worker_solver.solve_from_layouts(lay_copies, H, 1000)
    min_cost = float('inf')

    for solved, cost in results:
        if not solved: 
            continue
        if cost + 1 < min_cost:
            min_cost = cost + 1
            
    return min_cost

# =============================================================================
# GENERADOR PRINCIPAL
# =============================================================================

class UniformEmpiricalGenerator(InstanceGenerator):
    def __init__(self, H, S, seed):
        super().__init__(H=H, S=S, N=S*(H-2), seed=seed)
        self.empirical_counts = {}
        self.U = 0
        self.k_anchors = []
        self.pool = None

    def get_pool_initializer(self):
        """Debe ser implementado por las subclases. Retorna (func_init, args_init)"""
        raise NotImplementedError("Debe usarse una subclase específica de generador.")

    def generate_instances(self, amount, num_workers=None):
        if num_workers is None:
            num_workers = mp.cpu_count()

        # Obtenemos la función de inicialización y sus parámetros desde la subclase
        init_func, init_args = self.get_pool_initializer()

        print(f"Iniciando Pool con {num_workers} workers paralelos...")
        with mp.Pool(processes=num_workers, initializer=init_func, initargs=init_args) as pool:
            self.pool = pool
            
            print("Iniciando fase de Burn-in (Descubrimiento)...")
            self._burn_in()
            self.U = max(1, self.U) 
            print(f"Burn-in finalizado. Upper Bound (U) detectado: {self.U}")

            # 1. Definir los Bins exactamente uniformes
            cuotas_iniciales = distribuir_suma_exacta(np.ones(self.U), amount)
            bins_faltantes = {costo: int(cuotas_iniciales[costo - 1]) for costo in range(1, self.U + 1)}
            instancias_aceptadas = 0

            print(f"Iniciando recolección dirigida (NNLS) para {amount} instancias...")
            
            # Tamaño del lote a procesar en paralelo. Un múltiplo de los workers es óptimo.
            batch_size = num_workers * 4 
            
            while instancias_aceptadas < amount:
                k_probs = self._calcular_pesos_nnls(bins_faltantes)
                
                # Seleccionamos un lote entero de k's basado en las probabilidades NNLS
                k_samples = random.choices(self.k_anchors, weights=k_probs, k=batch_size)
                
                # Preparamos las cargas de trabajo (payloads)
                payloads = []
                for k in k_samples:
                    stacks = self.generate_stacks(self.H, self.S, self.N, sorted=True)
                    payloads.append((stacks, self.H, k))
                
                # Disparamos el lote en paralelo. imap_unordered devuelve los resultados
                # a medida que están listos, sin importar el orden original.
                for ret_k, instancia, costo_real in self.pool.imap_unordered(worker_task, payloads):
                    self._registrar_resultado(ret_k, costo_real)
                    
                    if costo_real in bins_faltantes and bins_faltantes[costo_real] > 0:
                        if self.add_instance(instancia):
                            bins_faltantes[costo_real] -= 1
                            instancias_aceptadas += 1
                            
                            if instancias_aceptadas % max(1, amount // 10) == 0:
                                print(f"Progreso: {instancias_aceptadas}/{amount} instancias.")
                    
                    # Salida temprana si alcanzamos la meta en medio de un lote
                    if instancias_aceptadas >= amount:
                        break

        return self.instances[:amount]

    def _burn_in(self, batch_size=100, stagnation_patience=3, threshold=0.5):
        k_seq = self._fibonacci_gen()
        historial_promedios = []
        
        for k in k_seq:
            self.k_anchors.append(k)
            self.empirical_counts[k] = {}
            
            # Preparamos el lote completo para este valor de k
            payloads = []
            for _ in range(batch_size):
                stacks = self.generate_stacks(self.H, self.S, self.N, sorted=True)
                payloads.append((stacks, self.H, k))
                
            costos_batch = []
            # Evaluamos el lote entero en paralelo
            for ret_k, _, costo in self.pool.imap_unordered(worker_task, payloads):
                self._registrar_resultado(ret_k, costo)
                costos_batch.append(costo)
                
            promedio_actual = np.mean(costos_batch)
            historial_promedios.append(promedio_actual)
            
            self.U = int(round(max(historial_promedios)))
            
            if len(historial_promedios) >= stagnation_patience + 2:
                promedio_reciente = np.mean(historial_promedios[-stagnation_patience:])
                mejor_promedio_historico = max(historial_promedios[:-stagnation_patience])
                crecimiento_del_promedio = promedio_reciente - mejor_promedio_historico
                
                print(f"[Burn-in] k={k:<4} | Promedio={promedio_actual:.2f} | U_actual={self.U} | Crec. Prom={crecimiento_del_promedio:.2f}")
                
                if crecimiento_del_promedio <= threshold:
                    break
            else:
                print(f"[Burn-in] k={k:<4} | Promedio={promedio_actual:.2f} | U_actual={self.U}")

    def _calcular_pesos_nnls(self, bins_faltantes):
        b = np.array([bins_faltantes.get(c, 0) for c in range(1, self.U + 1)], dtype=float)
        if np.sum(b) == 0:
            return [1.0 / len(self.k_anchors)] * len(self.k_anchors)

        P = np.zeros((len(self.k_anchors), self.U))
        for i, k in enumerate(self.k_anchors):
            total_hits = sum(self.empirical_counts[k].values())
            if total_hits == 0: continue
            for c in range(1, self.U + 1):
                P[i, c - 1] = self.empirical_counts[k].get(c, 0) / total_hits
                
        pesos, _ = nnls(P.T, b)
        
        suma_pesos = np.sum(pesos)
        if suma_pesos <= 1e-8:
            return [1.0 / len(self.k_anchors)] * len(self.k_anchors)
            
        return pesos / suma_pesos

    def _registrar_resultado(self, k, costo):
        if costo not in self.empirical_counts[k]:
            self.empirical_counts[k][costo] = 0
        self.empirical_counts[k][costo] += 1

    def _fibonacci_gen(self):
        a, b = 1, 2
        while True:
            yield a
            a, b = b, a + b

class FRGUniformEmpiricalGenerator(UniformEmpiricalGenerator):
    def __init__(self, H, S, seed):
        super().__init__(H, S, seed)

    def get_pool_initializer(self):
        # No requiere parámetros, retorna la función de inicialización de FRG
        return init_worker_frg, ()


class ModelUniformEmpiricalGenerator(UniformEmpiricalGenerator):
    def __init__(self, H, S, seed, model, input_adapter_config, batch_size=32):
        super().__init__(H, S, seed)
        
        self.model_cls = model.__class__
        self.model_params = getattr(model, 'hyperparams', {}) 
        self.weights = model.state_dict()
        
        self.input_adapter_config = input_adapter_config
        self.batch_size = batch_size

    def get_pool_initializer(self):
        args = (self.model_cls, self.model_params, self.weights, self.input_adapter_config, self.batch_size)
        return init_worker_model, args