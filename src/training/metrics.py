from abc import ABC, abstractmethod
import torch

class EpochMetrics():
    def __init__(self):
        self.metrics = {}

    def add_value(self, metric_cls, value):
        if metric_cls not in self.metrics:
            self.metrics[metric_cls] = []
            
        self.metrics[metric_cls].append(value)

    def get_last_value(self, metric_cls):
        return self.metrics[metric_cls][-1]

class Metric(ABC):
    def __init__(self, name, maximize=True):
        self.name = name
        self.maximize = maximize
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, logits, y):
        pass

    @abstractmethod
    def _compute(self):
        pass

    def compute(self):
        value = self._compute()
        self.reset()
        return value

    def format(self, value):
        return f"{value:.2f}"
    
class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy")

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def step(self, logits, y):
        batch_size = y.size(0)
        
        # 1. Índice de la acción predicha por el modelo (mayor logit)
        pred_indices = logits.argmax(dim=-1)
        
        # 2. Valor objetivo máximo real para cada muestra en el batch
        max_target_vals, _ = y.max(dim=-1)
        
        # 3. Valor objetivo correspondiente específicamente a la acción que el modelo predijo
        pred_target_vals = y[torch.arange(batch_size), pred_indices]
        
        # 4. Verificamos si el modelo predijo una acción que contiene el valor máximo
        correct = (pred_target_vals == max_target_vals)
        
        self.total_correct += correct.sum().item()
        self.total_samples += batch_size

    def _compute(self):
        return 100 * self.total_correct / self.total_samples
    
    def format(self, value):
        return f"{value:.2f}%"
    
class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__("CrossEntropy", False)

    def reset(self):
        self.total_samples = 0
        self.total_ce = 0
    
    def step(self, logits, y):
        y = y / y.sum(dim=1, keepdim=True)
        ce = torch.nn.functional.cross_entropy(logits, y)
        batch_size = y.size(0)
        self.total_ce += ce.item() * batch_size 
        self.total_samples += batch_size
        return ce

    def _compute(self):
        return self.total_ce / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class MSE(Metric):
    def __init__(self):
        super().__init__("MSE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse = 0
    
    def step(self, logits, y):
        mse = torch.nn.functional.mse_loss(logits, y.float())
        
        batch_size = y.size(0)
        self.total_mse += mse.item() * batch_size
        self.total_samples += batch_size
        
        return mse

    def _compute(self):
        if self.total_samples == 0: return 0.0
        return self.total_mse / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class ExpMSE(Metric):
    def __init__(self):
        super().__init__("ExpMSE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse_real = 0
    
    def step(self, logits, y_log):
        """
        logits: Salida del modelo (en escala logarítmica)
        y_log: Target original (en escala logarítmica)
        """
        # 1. Revertimos la transformación log para ambos
        preds_real = torch.exp(logits)
        targets_real = torch.exp(y_log)
        
        # 2. Calculamos el MSE en la escala original de pasos/desperdicio
        mse_real = torch.nn.functional.mse_loss(preds_real, targets_real.float())
        
        # 3. Acumulamos usando el batch size
        batch_size = y_log.size(0)
        self.total_mse_real += mse_real.item() * batch_size
        self.total_samples += batch_size
        
        return mse_real

    def _compute(self):
        if self.total_samples == 0: 
            return 0.0
        return self.total_mse_real / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class MAE(Metric):
    def __init__(self):
        super().__init__("MAE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse = 0
    
    def step(self, logits, y):
        mse = torch.nn.functional.l1_loss(logits, y.float())
        
        batch_size = y.size(0)
        self.total_mse += mse.item() * batch_size
        self.total_samples += batch_size
        
        return mse

    def _compute(self):
        if self.total_samples == 0: return 0.0
        return self.total_mse / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"
    
class ExpMAE(Metric):
    def __init__(self):
        super().__init__("ExpMAE", False)

    def reset(self):
        self.total_samples = 0
        self.total_mse_real = 0
    
    def step(self, logits, y_log):
        """
        logits: Salida del modelo (en escala logarítmica)
        y_log: Target original (en escala logarítmica)
        """
        # 1. Revertimos la transformación log para ambos
        preds_real = torch.exp(logits)
        targets_real = torch.exp(y_log)
        
        # 2. Calculamos el MSE en la escala original de pasos/desperdicio
        mse_real = torch.nn.functional.l1_loss(preds_real, targets_real.float())
        
        # 3. Acumulamos usando el batch size
        batch_size = y_log.size(0)
        self.total_mse_real += mse_real.item() * batch_size
        self.total_samples += batch_size
        
        return mse_real

    def _compute(self):
        if self.total_samples == 0: 
            return 0.0
        return self.total_mse_real / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"