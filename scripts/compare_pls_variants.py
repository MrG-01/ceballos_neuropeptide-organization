import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from pyls import behavioral_pls
from cca_zoo.linear import SCCA_PMD, SCCA_ADMM
import nibabel as nib
from utils import get_centroids

savefigs = True
n_splits = 100
test_size = 0.7
random_state_seed = 42

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load neurosynth terms (Y in switched PLS2)
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
# Load gene receptor data (X in switched PLS2)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

# Centroids and distance matrix for Graph-Regularized PLS
img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:] # discard background 0
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         CUSTOM ESTIMATORS
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
        self.t_scores_ = []

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
            self.t_scores_.append(t)

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
#                         CROSS-VALIDATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print(f"Running {n_splits}-fold cross-validation...")

results = {
    "PLSC": [],
    "sPLS": [],
    "KPLS": [],
    "sCCA": [],
    "G-PLS": []
}

for i in range(n_splits):
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]

    # 1. PLSC
    try:
        pls_cv = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_test = Xtest @ pls_cv["x_weights"][:, 0]
        u_test = Ytest @ pls_cv["y_weights"][:, 0]
        r_plsc, _ = spearmanr(t_test, u_test)
        results["PLSC"].append(r_plsc)
    except Exception as e:
        results["PLSC"].append(np.nan)

    # 2. sPLS (SCCA_PMD)
    try:
        spls_cv = SCCA_PMD(latent_dimensions=1, tau=0.5, random_state=i, max_iter=100).fit([Xtrain, Ytrain])
        scores_test = spls_cv.transform([Xtest, Ytest])
        r_spls, _ = spearmanr(scores_test[0][:, 0], scores_test[1][:, 0])
        results["sPLS"].append(r_spls)
    except Exception as e:
        results["sPLS"].append(np.nan)

    # 3. KPLS
    try:
        kpls_cv = KernelPLS(gamma=0.01).fit(Xtrain, Ytrain)
        t_test, u_test = kpls_cv.transform(Xtest, Ytest)
        r_kpls, _ = spearmanr(t_test[:, 0], u_test[:, 0])
        results["KPLS"].append(r_kpls)
    except Exception as e:
        results["KPLS"].append(np.nan)

    # 4. sCCA (SCCA_ADMM)
    try:
        scca_cv = SCCA_ADMM(latent_dimensions=1, tau=0.1, random_state=i, max_iter=100).fit([Xtrain, Ytrain])
        scores_test = scca_cv.transform([Xtest, Ytest])
        r_scca, _ = spearmanr(scores_test[0][:, 0], scores_test[1][:, 0])
        results["sCCA"].append(r_scca)
    except Exception as e:
        results["sCCA"].append(np.nan)

    # 5. G-PLS
    try:
        dist_train = distance[train_idx][:, train_idx]
        gpls_cv = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
        t_test, u_test = gpls_cv.transform(Xtest, Ytest)
        r_gpls, _ = spearmanr(t_test[:, 0], u_test[:, 0])
        results["G-PLS"].append(r_gpls)
    except Exception as e:
        results["G-PLS"].append(np.nan)

# Convert to DataFrame
df_results = pd.DataFrame(results)
print("Mean out-of-sample Spearman correlations:")
print(df_results.mean())

# Plot cross-validation results
plt.figure(figsize=(8, 6), dpi=200)
palette_colors = sns.color_palette("Set2", 5)
sns.boxplot(data=df_results, palette=palette_colors, width=0.5, linewidth=1.0)
plt.ylabel("Test score correlation (Spearman)")
plt.xlabel("Method")
plt.title(f"Cross-Validation Comparison ({n_splits} splits)")
sns.despine(trim=True)
if savefigs:
    plt.savefig('figs/pls_comparison_cv.pdf', bbox_inches='tight')
plt.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         FIT FULL MODEL AND EXTRACT LOADINGS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Fitting full models to extract receptor loadings...")

loadings_dict = {}

# 1. PLSC
# Run behavioral_pls with neurosynth (Y) as first argument and receptors (X) as second argument to get receptor loadings under y_loadings
pls_full = behavioral_pls(Y, X, n_boot=0, n_perm=0, test_split=0)
loadings_dict["PLSC"] = pls_full["y_loadings"][:, 0]

# 2. sPLS (SCCA_PMD)
spls_full = SCCA_PMD(latent_dimensions=1, tau=0.5, random_state=42, max_iter=200).fit([X, Y])
loadings_dict["sPLS"] = spls_full.weights_[0][:, 0]

# 3. KPLS
kpls_full = KernelPLS(gamma=0.01).fit(X, Y)
t_full = kpls_full.transform(X)
# For Kernel PLS, we define loadings as correlation of original X variables with latent scores
k_loadings = [spearmanr(X[:, j], t_full[:, 0])[0] for j in range(X.shape[1])]
loadings_dict["KPLS"] = np.array(k_loadings)

# 4. sCCA (SCCA_ADMM)
scca_full = SCCA_ADMM(latent_dimensions=1, tau=0.1, random_state=42, max_iter=200).fit([X, Y])
loadings_dict["sCCA"] = scca_full.weights_[0][:, 0]

# 5. G-PLS
gpls_full = GraphRegularizedPLS(lam=1.0).fit(X, Y, distance)
loadings_dict["G-PLS"] = gpls_full.x_weights_[:, 0]

# Create DataFrame of loadings
df_loadings = pd.DataFrame(loadings_dict, index=receptor_genes.columns)

# Standardize sign of weights for consistency (align to PLSC sign)
for col in df_loadings.columns:
    if col != "PLSC":
        # Check alignment sign
        corr = np.corrcoef(df_loadings["PLSC"], df_loadings[col])[0, 1]
        if corr < 0:
            df_loadings[col] = -df_loadings[col]

# Save loadings CSV
df_loadings.to_csv('results/pls_comparison_loadings.csv')

# Plot comparison heatmap
plt.figure(figsize=(10, 12), dpi=200)
sns.heatmap(df_loadings, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            cbar_kws={'label': 'Receptor Weight / Loading', 'shrink': 0.5},
            linewidths=0.5, linecolor='white')
plt.title("Receptor Loadings / Weights Across Cross-Decomposition Methods")
plt.ylabel("Neuropeptide Receptor")
plt.xlabel("Method")
plt.tight_layout()
if savefigs:
    plt.savefig('figs/pls_comparison_loadings.pdf', bbox_inches='tight')
plt.close()

print("Comparison scripts executed successfully. Plots saved to figs/.")
