import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from pyls import behavioral_pls
from cca_zoo.linear import SCCA_PMD
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

# Centroids and distance matrix for G-PLS
img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:]
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))

# Load BrainSMASH nulls
nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         MODEL CLASS DEFINITIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class KernelPLS:
    def __init__(self, n_components=1, gamma=0.01):
        self.n_components = n_components
        self.gamma = gamma

    def _rbf_kernel(self, X1, X2):
        from scipy.spatial.distance import cdist
        dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-self.gamma * dists)

    def fit(self, X, Y):
        self.X_train_ = X
        K = self._rbf_kernel(X, X)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        self.K_c_train_ = K_c
        self.Y_mean_ = Y.mean(axis=0)
        Y_c = Y - self.Y_mean_

        self.t_weights_ = []
        self.y_weights_ = []

        Kc_curr = K_c.copy()
        Y_curr = Y_c.copy()

        for a in range(self.n_components):
            u = Y_curr[:, [0]].copy()
            for _ in range(100):
                t = Kc_curr @ u
                norm_t = np.linalg.norm(t)
                if norm_t > 0:
                    t /= norm_t
                c = Y_curr.T @ t
                u_new = Y_curr @ c
                norm_u = np.linalg.norm(u_new)
                if norm_u > 0:
                    u_new /= norm_u
                if np.linalg.norm(u - u_new) < 1e-6:
                    u = u_new
                    break
                u = u_new
            
            t = Kc_curr @ u
            norm_t = np.linalg.norm(t)
            if norm_t > 0:
                t /= norm_t
            c = Y_curr.T @ t

            P_t = np.eye(n) - t @ t.T
            Kc_curr = P_t @ Kc_curr @ P_t
            Y_curr = P_t @ Y_curr

            self.t_weights_.append(u)
            self.y_weights_.append(c)

        self.t_weights_ = np.hstack(self.t_weights_)
        self.y_weights_ = np.hstack(self.y_weights_)
        return self

    def transform(self, X_test, Y_test=None):
        K_test = self._rbf_kernel(X_test, self.X_train_)
        n_train = self.X_train_.shape[0]
        n_test = X_test.shape[0]
        
        ones_train = np.ones((n_train, n_train)) / n_train
        ones_test_train = np.ones((n_test, n_train)) / n_train
        K_test_c = K_test - ones_test_train @ self.K_c_train_ - K_test @ ones_train + ones_test_train @ self.K_c_train_ @ ones_train
        
        t_scores = K_test_c @ self.t_weights_
        if Y_test is not None:
            Y_test_c = Y_test - self.Y_mean_
            u_scores = Y_test_c @ self.y_weights_
            return t_scores, u_scores
        return t_scores


class GraphRegularizedPLS:
    def __init__(self, lam=1.0):
        self.lam = lam

    def _compute_laplacian(self, dist_matrix, sigma=15.0):
        W = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        deg = W.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = 1.0 / np.sqrt(deg)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        L_norm = np.eye(dist_matrix.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        return L_norm

    def fit(self, X, Y, dist_matrix):
        L = self._compute_laplacian(dist_matrix)
        Px = np.eye(X.shape[1]) + self.lam * (X.T @ L @ X)
        Py = np.eye(Y.shape[1]) + self.lam * (Y.T @ L @ Y)
        
        from scipy.linalg import pinv
        Px_inv = pinv(Px)
        Py_inv = pinv(Py)

        u = pinv(Py) @ Y.T @ X[:, [0]]
        u /= np.linalg.norm(u)
        
        for _ in range(100):
            v = Px_inv @ (X.T @ Y) @ u
            v_norm = np.sqrt(v.T @ Px @ v)
            if v_norm > 0:
                v /= v_norm
            
            u_new = Py_inv @ (Y.T @ X) @ v
            u_norm = np.sqrt(u_new.T @ Py @ u_new)
            if u_norm > 0:
                u_new /= u_norm
            
            if np.linalg.norm(u - u_new) < 1e-6:
                u = u_new
                break
            u = u_new

        v = Px_inv @ (X.T @ Y) @ u
        v_norm = np.sqrt(v.T @ Px @ v)
        if v_norm > 0:
            v /= v_norm

        self.x_weights_ = v
        self.y_weights_ = u
        return self

    def transform(self, X_test, Y_test=None):
        t_scores = X_test @ self.x_weights_
        if Y_test is not None:
            u_scores = Y_test @ self.y_weights_
            return t_scores, u_scores
        return t_scores

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         EMPIRICAL MODEL FITTING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Fitting empirical models...")

# 1. G-PLS Empirical Fit
gpls_emp = GraphRegularizedPLS(lam=1.0).fit(X, Y, distance)
t_gpls_emp, u_gpls_emp = gpls_emp.transform(X, Y)
r_emp_gpls = spearmanr(t_gpls_emp[:, 0], u_gpls_emp[:, 0])[0]

# 2. Ensemble (Stacked PLS) Empirical Fit
plsc_emp = behavioral_pls(X, Y, n_boot=0, n_perm=0, test_split=0)
t_plsc_emp = X @ plsc_emp["x_weights"][:, 0]
u_plsc_emp = Y @ plsc_emp["y_weights"][:, 0]

spls_emp = SCCA_PMD(latent_dimensions=1, tau=0.5, random_state=42, max_iter=200).fit([X, Y])
t_spls_emp, u_spls_emp = spls_emp.transform([X, Y])[0][:, 0], spls_emp.transform([X, Y])[1][:, 0]

kpls_emp = KernelPLS(gamma=0.01).fit(X, Y)
t_kpls_emp, u_kpls_emp = kpls_emp.transform(X, Y)
t_kpls_emp, u_kpls_emp = t_kpls_emp[:, 0], u_kpls_emp[:, 0]

# Align signs to PLSC
if pearsonr(t_spls_emp, t_plsc_emp)[0] < 0:
    t_spls_emp, u_spls_emp = -t_spls_emp, -u_spls_emp
if pearsonr(t_kpls_emp, t_plsc_emp)[0] < 0:
    t_kpls_emp, u_kpls_emp = -t_kpls_emp, -u_kpls_emp

T_X_emp = np.column_stack([t_plsc_emp, t_spls_emp, t_kpls_emp])
T_Y_emp = np.column_stack([u_plsc_emp, u_spls_emp, u_kpls_emp])

meta_pls_emp = behavioral_pls(T_X_emp, T_Y_emp, n_boot=0, n_perm=0, test_split=0)
t_stack_emp = T_X_emp @ meta_pls_emp["x_weights"][:, 0]
u_stack_emp = T_Y_emp @ meta_pls_emp["y_weights"][:, 0]
r_emp_stack = spearmanr(t_stack_emp, u_stack_emp)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         NULL MODELS PERMUTATIONS (500)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_perms = 500
gpls_null_corrs = []
stack_null_corrs = []

print(f"Running {n_perms} BrainSMASH permutations through G-PLS and Ensemble pipelines...")

for p in range(n_perms):
    # Construct z-scored null X
    X_null = np.zeros_like(X)
    for j in range(38):
        X_null[:, j] = nulls[j, :, p]
    X_null = zscore(X_null, ddof=1)

    # 1. G-PLS on Null
    g_null = GraphRegularizedPLS(lam=1.0).fit(X_null, Y, distance)
    tg_null, ug_null = g_null.transform(X_null, Y)
    gpls_null_corrs.append(spearmanr(tg_null[:, 0], ug_null[:, 0])[0])

    # 2. Ensemble on Null (Full Pipeline)
    # Base Model 1: PLSC
    plsc_null = behavioral_pls(X_null, Y, n_boot=0, n_perm=0, test_split=0)
    t_plsc_null = X_null @ plsc_null["x_weights"][:, 0]
    u_plsc_null = Y @ plsc_null["y_weights"][:, 0]

    # Base Model 2: sPLS
    try:
        spls_null = SCCA_PMD(latent_dimensions=1, tau=0.5, random_state=p, max_iter=100).fit([X_null, Y])
        t_spls_null = spls_null.transform([X_null, Y])[0][:, 0]
        u_spls_null = spls_null.transform([X_null, Y])[1][:, 0]
    except:
        t_spls_null, u_spls_null = t_plsc_null.copy(), u_plsc_null.copy()

    # Base Model 3: KPLS
    try:
        kpls_null = KernelPLS(gamma=0.01).fit(X_null, Y)
        t_kpls_null, u_kpls_null = kpls_null.transform(X_null, Y)
        t_kpls_null, u_kpls_null = t_kpls_null[:, 0], u_kpls_null[:, 0]
    except:
        t_kpls_null, u_kpls_null = t_plsc_null.copy(), u_plsc_null.copy()

    # Align signs to null PLSC
    if pearsonr(t_spls_null, t_plsc_null)[0] < 0:
        t_spls_null, u_spls_null = -t_spls_null, -u_spls_null
    if pearsonr(t_kpls_null, t_plsc_null)[0] < 0:
        t_kpls_null, u_kpls_null = -t_kpls_null, -u_kpls_null

    # Level 1 Stacked PLS Meta-learner
    T_X_null = np.column_stack([t_plsc_null, t_spls_null, t_kpls_null])
    T_Y_null = np.column_stack([u_plsc_null, u_spls_null, u_kpls_null])

    try:
        meta_pls_null = behavioral_pls(T_X_null, T_Y_null, n_boot=0, n_perm=0, test_split=0)
        t_stack_null = T_X_null @ meta_pls_null["x_weights"][:, 0]
        u_stack_null = T_Y_null @ meta_pls_null["y_weights"][:, 0]
        stack_null_corrs.append(spearmanr(t_stack_null, u_stack_null)[0])
    except:
        stack_null_corrs.append(spearmanr(t_plsc_null, u_plsc_null)[0])

gpls_null_corrs = np.array(gpls_null_corrs)
stack_null_corrs = np.array(stack_null_corrs)

# Compute statistics
p_gpls = (np.sum(np.abs(gpls_null_corrs) >= np.abs(r_emp_gpls)) + 1) / (n_perms + 1)
p_stack = (np.sum(np.abs(stack_null_corrs) >= np.abs(r_emp_stack)) + 1) / (n_perms + 1)

delta_gpls = r_emp_gpls - np.mean(gpls_null_corrs)
delta_stack = r_emp_stack - np.mean(stack_null_corrs)

print("\n--- RESULTS ---")
print(f"G-PLS Empirical: {r_emp_gpls:.4f} | Null Mean: {np.mean(gpls_null_corrs):.4f} | Delta r: {delta_gpls:.4f} | P_spin: {p_gpls:.4f}")
print(f"Ensemble Empirical: {r_emp_stack:.4f} | Null Mean: {np.mean(stack_null_corrs):.4f} | Delta r: {delta_stack:.4f} | P_spin: {p_stack:.4f}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOTTING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure(figsize=(10, 5), dpi=300)

# Subplot 1: G-PLS
plt.subplot(1, 2, 1)
sns.histplot(gpls_null_corrs, color='grey', kde=True, stat="density", label='BrainSMASH Null')
plt.axvline(r_emp_gpls, color='#C44E52', linestyle='--', linewidth=2.5, label=f'Empirical ({r_emp_gpls:.3f})')
plt.title(f"G-PLS Significance\n$\Delta r$ = {delta_gpls:.3f} | $p$ = {p_gpls:.4f}")
plt.xlabel("Spearman Correlation")
plt.ylabel("Density")
plt.legend(frameon=True, loc='upper left')
sns.despine()

# Subplot 2: Ensemble
plt.subplot(1, 2, 2)
sns.histplot(stack_null_corrs, color='grey', kde=True, stat="density", label='BrainSMASH Null')
plt.axvline(r_emp_stack, color='#4C72B0', linestyle='--', linewidth=2.5, label=f'Empirical ({r_emp_stack:.3f})')
plt.title(f"Ensemble Significance\n$\Delta r$ = {delta_stack:.3f} | $p$ = {p_stack:.4f}")
plt.xlabel("Spearman Correlation")
plt.ylabel("")
plt.legend(frameon=True, loc='upper left')
sns.despine()

plt.tight_layout()
plt.savefig('figs/ensemble_vs_gpls_brainsmash.pdf', bbox_inches='tight')
plt.close()
print("\nSignificance comparison complete. Plot saved to figs/ensemble_vs_gpls_brainsmash.pdf")
