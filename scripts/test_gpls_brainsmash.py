import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load neurosynth terms (Y)
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
# Load gene receptor data (X)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

# Centroids and distance matrix for Graph-Regularized PLS
img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:] # discard background 0
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))

# Load BrainSMASH nulls
nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy')

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
#                         EVALUATE EMPIRICAL VS. NULL
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7

empirical_scores = []
null_scores = []

print(f"Running {n_splits} cross-validation splits...")
for i in range(n_splits):
    # 1. Empirical Split
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    dist_train = distance[train_idx][:, train_idx]
    
    gpls_emp = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
    t_test_emp, u_test_emp = gpls_emp.transform(Xtest, Ytest)
    r_emp, _ = spearmanr(t_test_emp[:, 0], u_test_emp[:, 0])
    empirical_scores.append(r_emp)

    # 2. BrainSMASH Null Split (using surrogate map `i`)
    # Construct surrogate X_null from nulls array for permutation `i`
    X_null = np.zeros_like(X)
    for j in range(38):
        X_null[:, j] = nulls[j, :, i]
    X_null = zscore(X_null, ddof=1)
    
    Xtrain_null = X_null[train_idx]
    Xtest_null = X_null[test_idx]
    
    gpls_null = GraphRegularizedPLS(lam=1.0).fit(Xtrain_null, Ytrain, dist_train)
    t_test_null, u_test_null = gpls_null.transform(Xtest_null, Ytest)
    r_null, _ = spearmanr(t_test_null[:, 0], u_test_null[:, 0])
    null_scores.append(r_null)

# Calculate p-value of G-PLS on the full model fit
print("Calculating full-model permutation significance...")
gpls_full_emp = GraphRegularizedPLS(lam=1.0).fit(X, Y, distance)
t_full_emp, u_full_emp = gpls_full_emp.transform(X, Y)
r_full_emp, _ = spearmanr(t_full_emp[:, 0], u_full_emp[:, 0])

null_full_corrs = []
# Evaluate 500 permutations for full-model permutation significance
n_perms_full = 500
for i in range(n_perms_full):
    X_null = np.zeros_like(X)
    for j in range(38):
        X_null[:, j] = nulls[j, :, i]
    X_null = zscore(X_null, ddof=1)
    
    gpls_full_null = GraphRegularizedPLS(lam=1.0).fit(X_null, Y, distance)
    t_full_null, u_full_null = gpls_full_null.transform(X_null, Y)
    r_full_null, _ = spearmanr(t_full_null[:, 0], u_full_null[:, 0])
    null_full_corrs.append(r_full_null)

null_full_corrs = np.array(null_full_corrs)
p_value = (np.sum(np.abs(null_full_corrs) >= np.abs(r_full_emp)) + 1) / (n_perms_full + 1)

print(f"Empirical Full-Model Spearman Correlation: {r_full_emp:.4f}")
print(f"BrainSMASH Permutation p-value: {p_value:.4f}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOTTING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df_plot = pd.DataFrame({
    "Test Correlation": empirical_scores + null_scores,
    "Group": ["Empirical G-PLS"] * n_splits + ["BrainSMASH Null G-PLS"] * n_splits
})

plt.figure(figsize=(6, 6), dpi=200)
palette_colors = {"Empirical G-PLS": "#4C72B0", "BrainSMASH Null G-PLS": "#C44E52"}

# Boxplot with individual data points overlaid
sns.boxplot(x="Group", y="Test Correlation", data=df_plot, palette=palette_colors, width=0.4, linewidth=1.0, showfliers=False)
sns.stripplot(x="Group", y="Test Correlation", data=df_plot, palette=palette_colors, size=4.0, jitter=True, alpha=0.6, linewidth=0.5, edgecolor="gray")

plt.ylabel("Test score correlation (Spearman)")
plt.xlabel("")
plt.title(f"G-PLS vs. BrainSMASH Spatial Null\nFull-Model Permutation p = {p_value:.4f}")
sns.despine(trim=True)

plt.savefig('figs/gpls_brainsmash_comparison.pdf', bbox_inches='tight')
plt.close()
print("Plots saved successfully as figs/gpls_brainsmash_comparison.pdf")
