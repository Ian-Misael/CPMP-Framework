import numpy as np
from scipy.stats import norm
from scipy.optimize import nnls
from instances.generators.instance_generator import InstanceGenerator
from instances.generators.random_moves import RandomMovesGenerator
from utils.utils import distribuir_suma_exacta

class UniformCostGenerator(InstanceGenerator):
    # Coeficientes actualizados para los nuevos modelos
    # mu_params: [k1, k2, k3, p, m, n]
    mu_params = [1.0905637662060876, -0.012739799939775441, 1.8070269315373046, 0.8383608929049343, 0.07384321919561357, 3.394380779842854]
    # sigma_params: [a0, a1, a2, b0, b1, b2, p]
    sigma_params = [0.7675597850356134, -0.3761636532711637, 1.2178744261021317, 2.228110888815153, 0.4865899877216909, 3.1339755774534117]


    def __init__(self, H, S, seed):
        super().__init__(H=H, S=S, N=S*(H-2), seed=seed)

    def generate_instances(self, amount):
        # Resolvemos cuántas instancias de cada R necesitamos
        n_opt, R = self.resolver_mezcla_distribuciones(self.H, self.S, amount)
        n_opt = distribuir_suma_exacta(n_opt, amount)

        for r, n in zip(R, n_opt):
            if n <= 0: continue

            # Generamos n instancias con r movimientos aleatorios
            gen = RandomMovesGenerator(self.H, self.S, self.N, r=int(round(r)), seed=self.seed)
            instances = gen.generate_instances(n)
            for instance in instances:
                self.add_instance(instance)
        return self.instances
    
    def resolver_mezcla_distribuciones(self, H, S, N):
        k1, k2, k3, _, _, _ = self.mu_params
        L_a = k1 * (S**k2) * (H**k3)
        K_max = int(np.floor(L_a - 1))
        pasos_k = np.arange(1, K_max + 1)

        # ── Muestrear uniformemente en espacio mu ──────────────────────────────
        mu_min = float(S)                  # dificultad mínima útil
        mu_max = L_a
        n_puntos = min(30, K_max)

        mu_targets = np.linspace(mu_min, mu_max, n_puntos)
        recursos_R = [self.calcR(H, S, mu_t) for mu_t in mu_targets]
        # ──────────────────────────────────────────────────────────────────────

        num_R = len(recursos_R)
        P_full = np.zeros((num_R, K_max))
        for i, R_i in enumerate(recursos_R):
            mu_i, sigma_i = self.dist(H, S, R_i)
            cdf_upper = norm.cdf(pasos_k + 0.5, loc=mu_i, scale=sigma_i)
            cdf_lower = norm.cdf(pasos_k - 0.5, loc=mu_i, scale=sigma_i)
            fila = cdf_upper - cdf_lower
            P_full[i, :] = fila / (np.sum(fila) + 1e-12)

        inicio_bin = int(S)
        P_reducida = P_full[:, inicio_bin - 1:]
        K_recortado = P_reducida.shape[1]

        # Objetivo uniforme simple — ahora P está bien condicionada
        b_reducido = np.full(K_recortado, N / K_recortado)

        lambda_reg = 0.05
        P_reg = np.vstack([P_reducida.T, lambda_reg * np.eye(num_R)])
        b_reg = np.concatenate([b_reducido, np.zeros(num_R)])

        n_opt, _ = nnls(P_reg, b_reg)

        '''
        distribucion_simulada = P_reducida.T @ n_opt
        bins = np.arange(inicio_bin, inicio_bin + K_recortado)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Izquierda: lo que NNLS "cree" que va a producir
        axes[0].bar(bins, distribucion_simulada)
        axes[0].axhline(N / K_recortado, color='red', linestyle='--', label='objetivo uniforme')
        axes[0].set_title("Distribución simulada por NNLS")
        axes[0].legend()

        # Derecha: n_opt por R
        mu_targets_plot = [self.mu(H, S, r) for r in recursos_R]
        axes[1].bar(mu_targets_plot, n_opt, width=1.2)
        axes[1].set_title("n_opt por valor de mu(R)")
        axes[1].set_xlabel("mu(R)")
        plt.tight_layout()
        plt.savefig("debug_nnls.png")
        print("Guardado debug_nnls.png")
        '''

        return n_opt, recursos_R

    def dist(self, H, S, R):
        return self.mu(H, S, R), self.sigma(H, S, R)

    def mu(self, H, S, R):
        return self._hill_model(H, S, R, self.mu_params, is_sigma=False)

    def sigma(self, H, S, R):
        # sigma usa el modelo de Hill para evitar valores explosivos
        return self._hill_model(H, S, R, self.sigma_params, is_sigma=True)

    def _hill_model(self, H, S, R, params, is_sigma=False):
        k1, k2, k3, p, m, n = params
        r_term = np.maximum(R - 1, 0)
        
        # Techo (Asíntota)
        L_a = k1 * (S**k2) * (H**k3)
        # Saturación
        C_a = np.exp(m * S + n)
        # Factor Hill
        factor = (r_term**p) / (C_a + r_term**p + 1e-12)
        
        if is_sigma:
            # Para sigma, el valor base cuando R=1 es ~0.5 para evitar colapso
            return 0.5 + L_a * factor
        else:
            # Para mu, el valor base cuando R=1 es 1.0
            return 1 + (L_a - 1) * factor

    def calcR(self, H, S, mu_target):
        k1, k2, k3, p, m, n = self.mu_params
        L_a = k1 * (S**k2) * (H**k3)
        C_a = np.exp(m * S + n)
        M = L_a - 1 
        
        Y = np.clip(mu_target - 1, 1e-6, M * 0.99)
        r_p = (Y * C_a) / (M - Y)
        return 1 + (r_p)**(1/p)