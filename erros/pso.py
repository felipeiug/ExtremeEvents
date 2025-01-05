import time
import numpy as np

w = np.random.uniform(0.5, 1.5)
c1 = np.random.uniform(0.5, 1.5)
c2 = np.random.uniform(0.5, 1.5)

# Função objetivo
def obj_func(X:np.ndarray):
    return np.pow(X[:, 0], 2) + np.pow(X[:, 1], 2)

n_var = 2
lim_inf = [-100, -100]
lim_sup = [100, 100]
n_particulas = 100000
n_iter = 10000
erro = 0.0001
iter_erro = 10

# Parâmetros do problema
n_var = 2
lim_inf = np.array([-100]*n_var)
lim_sup = np.array([100]*n_var)

# Parâmetros da otimização
n_particulas = 100000
n_iter = 10000
erro = 0.0001
iter_erro = 10

# Inicialização aleatória
X = np.random.uniform(-100, 100,  (n_particulas, n_var)) # Posição
V = np.random.uniform(-100, 100,  (n_particulas, n_var)) # Velocidade
Z = obj_func(X)                                          # Posição atual
X_best = X.copy()                                        # Parcela cognitiva, melhor posição já ocupada em cada partícula
Z_best = Z.copy()                                        # Melhor posição encontrada
X_best_G = np.zeros(n_var)                               # Melhor solução dentre todas as partículas
Z_best_G = Z_best.min()                                  # Melhor posição dentre todas as partículas

index = np.where(Z_best == Z_best_G)
X_best_G = X_best[index][-1]

resultados = []
for iter in range(n_iter):
    X = X + V
    Z = obj_func(X)
    
    # Atualizando os melhores valores
    Z_best = np.where(Z < Z_best, Z, Z_best)
    X_best = np.where((Z < Z_best)[:, np.newaxis], X, X_best)

    if Z_best.min() < Z_best_G:
        Z_best_G = Z_best.min()
        index = np.where(Z_best == Z_best_G)
        X_best_G = X_best[index][-1]
    
    V = (w * V + c1 * np.random.rand(n_particulas, 1) * (X_best - X) + c2 * np.random.rand(n_particulas, 1) * (X_best_G - X))

    if iter%100 == 0:
        print(f"Iter: {iter} | Z_best_min: {Z_best.min()} | Z_best_G: {Z_best_G}")
    
print(Z_best_G)

print("AAA")


