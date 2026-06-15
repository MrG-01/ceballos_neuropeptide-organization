import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from pyls import behavioral_pls
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
#                         G-PLS CLASS DEFINITION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#                         HYBRID ENSEMBLING CV
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7
alpha = 0.7  # 70% weight on G-PLS topography, 30% weight on local/agnostic model

results = {
    "PLSC (Baseline)": [],
    "G-PLS (Baseline)": [],
    "G-PLS + Random Forest": [],
    "G-PLS + ElasticNet": []
}

print(f"Running {n_splits}-fold cross-validation for G-PLS Hybrid Ensembles...")

for i in range(n_splits):
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    dist_train = distance[train_idx][:, train_idx]

    # 1. Fit PLSC Baseline
    try:
        plsc = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_plsc_test = Xtest @ plsc["x_weights"][:, 0]
        u_plsc_test = Ytest @ plsc["y_weights"][:, 0]
        results["PLSC (Baseline)"].append(spearmanr(t_plsc_test, u_plsc_test)[0])
    except:
        continue

    # 2. Fit G-PLS Baseline
    gpls = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
    t_gpls_test, u_gpls_test = gpls.transform(Xtest, Ytest)
    t_gpls_test, u_gpls_test = t_gpls_test[:, 0], u_gpls_test[:, 0]
    results["G-PLS (Baseline)"].append(spearmanr(t_gpls_test, u_gpls_test)[0])

    # Calculate G-PLS training behavior scores to use as 1D regression target
    u_gpls_train = Ytrain @ gpls.y_weights_[:, 0]

    # 3. Fit 1D Random Forest Regressor to predict G-PLS behavior scores
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=i, n_jobs=-1)
    rf.fit(Xtrain, u_gpls_train)
    u_rf_test = rf.predict(Xtest)
    
    # Align RF score sign to G-PLS
    if pearsonr(u_rf_test, u_gpls_test)[0] < 0:
        u_rf_test = -u_rf_test
        
    # Blend scores
    u_blend_rf = alpha * t_gpls_test + (1 - alpha) * u_rf_test
    results["G-PLS + Random Forest"].append(spearmanr(t_gpls_test, u_blend_rf)[0])

    # 4. Fit 1D ElasticNet Regressor to predict G-PLS behavior scores
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=i, max_iter=2000)
    en.fit(Xtrain, u_gpls_train)
    u_en_test = en.predict(Xtest)
    
    # Align ElasticNet score sign to G-PLS
    if pearsonr(u_en_test, u_gpls_test)[0] < 0:
        u_en_test = -u_en_test
        
    # Blend scores
    u_blend_en = alpha * t_gpls_test + (1 - alpha) * u_en_test
    results["G-PLS + ElasticNet"].append(spearmanr(t_gpls_test, u_blend_en)[0])

# Convert to DataFrame
df_results = pd.DataFrame(results)
print("\nMean out-of-sample Spearman correlations:")
print(df_results.mean())
print("\nStandard deviation:")
print(df_results.std())

# Plot results
plt.figure(figsize=(8, 6), dpi=200)
palette_colors = sns.color_palette("Set2", 4)
sns.boxplot(data=df_results, palette=palette_colors, width=0.4, linewidth=1.0, showfliers=False)
sns.stripplot(data=df_results, palette=palette_colors, size=3.0, jitter=True, alpha=0.5, edgecolor="gray", linewidth=0.5)
plt.ylabel("Test score correlation (Spearman)")
plt.xlabel("Model")
plt.title(f"Hybrid G-PLS Ensemble Comparison ({n_splits} splits)")
sns.despine(trim=True)

plt.savefig('figs/gpls_hybrid_ensemble_cv.pdf', bbox_inches='tight')
plt.close()
print("\nHybrid evaluation complete. Plot saved to figs/gpls_hybrid_ensemble_cv.pdf")
