import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
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
#                         CROSS-VALIDATION LOOP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7

results = {
    "PLSC (Baseline)": [],
    "G-PLS (Baseline)": [],
    "Equal Blend (with G-PLS)": [],
    "Weighted Blend (with G-PLS)": [],
    "Stacked PLS (G-PLS as Base)": [],
    "Stacked PLS (G-PLS as Meta)": []
}

print(f"Running {n_splits}-fold cross-validation incorporating G-PLS...")

for i in range(n_splits):
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    dist_train = distance[train_idx][:, train_idx]
    dist_test = distance[test_idx][:, test_idx]

    # --- 1. Train Base Learners ---
    
    # 1. PLSC
    try:
        plsc = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_plsc_train = Xtrain @ plsc["x_weights"][:, 0]
        u_plsc_train = Ytrain @ plsc["y_weights"][:, 0]
        t_plsc_test = Xtest @ plsc["x_weights"][:, 0]
        u_plsc_test = Ytest @ plsc["y_weights"][:, 0]
        r_plsc_train = spearmanr(t_plsc_train, u_plsc_train)[0]
    except Exception as e:
        continue

    # 2. sPLS
    try:
        spls = SCCA_PMD(latent_dimensions=1, tau=0.5, random_state=i, max_iter=100).fit([Xtrain, Ytrain])
        scores_train = spls.transform([Xtrain, Ytrain])
        scores_test = spls.transform([Xtest, Ytest])
        t_spls_train, u_spls_train = scores_train[0][:, 0], scores_train[1][:, 0]
        t_spls_test, u_spls_test = scores_test[0][:, 0], scores_test[1][:, 0]
        r_spls_train = spearmanr(t_spls_train, u_spls_train)[0]
    except Exception as e:
        t_spls_train, u_spls_train = t_plsc_train.copy(), u_plsc_train.copy()
        t_spls_test, u_spls_test = t_plsc_test.copy(), u_plsc_test.copy()
        r_spls_train = 0.0

    # 3. KPLS
    try:
        kpls = KernelPLS(gamma=0.01).fit(Xtrain, Ytrain)
        t_kpls_train, u_kpls_train = kpls.transform(Xtrain, Ytrain)
        t_kpls_test, u_kpls_test = kpls.transform(Xtest, Ytest)
        t_kpls_train, u_kpls_train = t_kpls_train[:, 0], u_kpls_train[:, 0]
        t_kpls_test, u_kpls_test = t_kpls_test[:, 0], u_kpls_test[:, 0]
        r_kpls_train = spearmanr(t_kpls_train, u_kpls_train)[0]
    except Exception as e:
        t_kpls_train, u_kpls_train = t_plsc_train.copy(), u_plsc_train.copy()
        t_kpls_test, u_kpls_test = t_plsc_test.copy(), u_plsc_test.copy()
        r_kpls_train = 0.0

    # 4. G-PLS
    try:
        gpls = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
        t_gpls_train, u_gpls_train = gpls.transform(Xtrain, Ytrain)
        t_gpls_test, u_gpls_test = gpls.transform(Xtest, Ytest)
        t_gpls_train, u_gpls_train = t_gpls_train[:, 0], u_gpls_train[:, 0]
        t_gpls_test, u_gpls_test = t_gpls_test[:, 0], u_gpls_test[:, 0]
        r_gpls_train = spearmanr(t_gpls_train, u_gpls_train)[0]
    except Exception as e:
        t_gpls_train, u_gpls_train = t_plsc_train.copy(), u_plsc_train.copy()
        t_gpls_test, u_gpls_test = t_plsc_test.copy(), u_plsc_test.copy()
        r_gpls_train = 0.0

    # --- 2. Align Signs to PLSC ---
    if pearsonr(t_spls_train, t_plsc_train)[0] < 0:
        t_spls_train, u_spls_train = -t_spls_train, -u_spls_train
        t_spls_test, u_spls_test = -t_spls_test, -u_spls_test
        
    if pearsonr(t_kpls_train, t_plsc_train)[0] < 0:
        t_kpls_train, u_kpls_train = -t_kpls_train, -u_kpls_train
        t_kpls_test, u_kpls_test = -t_kpls_test, -u_kpls_test

    if pearsonr(t_gpls_train, t_plsc_train)[0] < 0:
        t_gpls_train, u_gpls_train = -t_gpls_train, -u_gpls_train
        t_gpls_test, u_gpls_test = -t_gpls_test, -u_gpls_test

    # Record Baselines
    results["PLSC (Baseline)"].append(spearmanr(t_plsc_test, u_plsc_test)[0])
    results["G-PLS (Baseline)"].append(spearmanr(t_gpls_test, u_gpls_test)[0])

    # --- 3. Strategy A: Blending with G-PLS ---
    # Equal Blend (4 models)
    t_eq = (t_plsc_test + t_spls_test + t_kpls_test + t_gpls_test) / 4.0
    u_eq = (u_plsc_test + u_spls_test + u_kpls_test + u_gpls_test) / 4.0
    results["Equal Blend (with G-PLS)"].append(spearmanr(t_eq, u_eq)[0])

    # Performance Weighted Blend (4 models)
    train_corrs = np.array([max(r_plsc_train, 0), max(r_spls_train, 0), max(r_kpls_train, 0), max(r_gpls_train, 0)])
    weights = train_corrs / train_corrs.sum() if train_corrs.sum() > 0 else np.ones(4)/4.0
    
    t_w = (weights[0] * t_plsc_test + weights[1] * t_spls_test + weights[2] * t_kpls_test + weights[3] * t_gpls_test)
    u_w = (weights[0] * u_plsc_test + weights[1] * u_spls_test + weights[2] * u_kpls_test + weights[3] * u_gpls_test)
    results["Weighted Blend (with G-PLS)"].append(spearmanr(t_w, u_w)[0])

    # --- 4. Strategy B: Stacked PLS (G-PLS as Base, Level-1 PLSC) ---
    T_X_train_4 = np.column_stack([t_plsc_train, t_spls_train, t_kpls_train, t_gpls_train])
    T_Y_train_4 = np.column_stack([u_plsc_train, u_spls_train, u_kpls_train, u_gpls_train])
    T_X_test_4 = np.column_stack([t_plsc_test, t_spls_test, t_kpls_test, t_gpls_test])
    T_Y_test_4 = np.column_stack([u_plsc_test, u_spls_test, u_kpls_test, u_gpls_test])

    try:
        meta_pls = behavioral_pls(T_X_train_4, T_Y_train_4, n_boot=0, n_perm=0, test_split=0)
        t_stacked = T_X_test_4 @ meta_pls["x_weights"][:, 0]
        u_stacked = T_Y_test_4 @ meta_pls["y_weights"][:, 0]
        results["Stacked PLS (G-PLS as Base)"].append(spearmanr(t_stacked, u_stacked)[0])
    except:
        results["Stacked PLS (G-PLS as Base)"].append(spearmanr(t_gpls_test, u_gpls_test)[0])

    # --- 5. Strategy B: Stacked PLS (G-PLS as Meta-Learner, Level-1 G-PLS) ---
    # Here the base models are PLSC, sPLS, KPLS (non-spatial)
    T_X_train_3 = np.column_stack([t_plsc_train, t_spls_train, t_kpls_train])
    T_Y_train_3 = np.column_stack([u_plsc_train, u_spls_train, u_kpls_train])
    T_X_test_3 = np.column_stack([t_plsc_test, t_spls_test, t_kpls_test])
    T_Y_test_3 = np.column_stack([u_plsc_test, u_spls_test, u_kpls_test])

    try:
        # Fit G-PLS as meta-learner
        meta_gpls = GraphRegularizedPLS(lam=1.0).fit(T_X_train_3, T_Y_train_3, dist_train)
        t_stacked_meta = T_X_test_3 @ meta_gpls.x_weights_[:, 0]
        u_stacked_meta = T_Y_test_3 @ meta_gpls.y_weights_[:, 0]
        results["Stacked PLS (G-PLS as Meta)"].append(spearmanr(t_stacked_meta, u_stacked_meta)[0])
    except:
        results["Stacked PLS (G-PLS as Meta)"].append(spearmanr(t_gpls_test, u_gpls_test)[0])

# Print results
df_results = pd.DataFrame(results)
print("\nMean out-of-sample Spearman correlations:")
print(df_results.mean())
print("\nStandard deviation:")
print(df_results.std())

# Plot results
plt.figure(figsize=(10, 6), dpi=200)
palette_colors = sns.color_palette("Set2", 6)
sns.boxplot(data=df_results, palette=palette_colors, width=0.4, linewidth=1.0, showfliers=False)
sns.stripplot(data=df_results, palette=palette_colors, size=2.5, jitter=True, alpha=0.4, edgecolor="gray", linewidth=0.5)
plt.ylabel("Test score correlation (Spearman)")
plt.xticks(rotation=15, ha='right')
plt.title(f"Ensembling and Stacking with G-PLS ({n_splits} splits)")
sns.despine(trim=True)

plt.savefig('figs/gpls_ensemble_comparison_cv.pdf', bbox_inches='tight')
plt.close()
print("\nEvaluation complete. Plot saved to figs/gpls_ensemble_comparison_cv.pdf")
