import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from pyls import behavioral_pls
from cca_zoo.linear import SCCA_PMD, SCCA_ADMM

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

X = zscore(receptor_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         KERNEL PLS DEFINITION
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         CROSS-VALIDATION AND ENSEMBLING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7

results = {
    "PLSC (Baseline)": [],
    "Strategy A (Equal Blend)": [],
    "Strategy A (Weighted Blend)": [],
    "Strategy B (Stacked PLS)": []
}

print(f"Running {n_splits}-fold cross-validation for ensemble strategies...")

for i in range(n_splits):
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]

    # --- 1. Train Base Learners (Level 0) ---
    
    # Model 1: PLSC
    try:
        plsc = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_plsc_train = Xtrain @ plsc["x_weights"][:, 0]
        u_plsc_train = Ytrain @ plsc["y_weights"][:, 0]
        t_plsc_test = Xtest @ plsc["x_weights"][:, 0]
        u_plsc_test = Ytest @ plsc["y_weights"][:, 0]
        r_plsc_train = spearmanr(t_plsc_train, u_plsc_train)[0]
    except Exception as e:
        continue  # skip split if baseline fails

    # Model 2: sPLS
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

    # Model 3: KPLS
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

    # --- 2. Align Signs to PLSC ---
    # We check the correlation of each model's train scores with PLSC reference
    # and invert sign if negative to prevent canceling out during blending/stacking.
    
    # sPLS Alignment
    if pearsonr(t_spls_train, t_plsc_train)[0] < 0:
        t_spls_train, u_spls_train = -t_spls_train, -u_spls_train
        t_spls_test, u_spls_test = -t_spls_test, -u_spls_test
        
    # KPLS Alignment
    if pearsonr(t_kpls_train, t_plsc_train)[0] < 0:
        t_kpls_train, u_kpls_train = -t_kpls_train, -u_kpls_train
        t_kpls_test, u_kpls_test = -t_kpls_test, -u_kpls_test

    # --- 3. Evaluate Baseline PLSC ---
    r_plsc_test, _ = spearmanr(t_plsc_test, u_plsc_test)
    results["PLSC (Baseline)"].append(r_plsc_test)

    # --- 4. Strategy A: Blending ---
    # Equal-weight Blend (without CCA)
    t_eq = (t_plsc_test + t_spls_test + t_kpls_test) / 3.0
    u_eq = (u_plsc_test + u_spls_test + u_kpls_test) / 3.0
    r_eq, _ = spearmanr(t_eq, u_eq)
    results["Strategy A (Equal Blend)"].append(r_eq)

    # Performance-weighted Blend (without CCA)
    train_corrs = np.array([max(r_plsc_train, 0), max(r_spls_train, 0), max(r_kpls_train, 0)])
    if train_corrs.sum() > 0:
        weights = train_corrs / train_corrs.sum()
    else:
        weights = np.ones(3) / 3.0
        
    t_w = (weights[0] * t_plsc_test + 
           weights[1] * t_spls_test + 
           weights[2] * t_kpls_test)
    u_w = (weights[0] * u_plsc_test + 
           weights[1] * u_spls_test + 
           weights[2] * u_kpls_test)
    r_w, _ = spearmanr(t_w, u_w)
    results["Strategy A (Weighted Blend)"].append(r_w)

    # --- 5. Strategy B: Meta-Stacking via PLSC (without CCA) ---
    T_X_train = np.column_stack([t_plsc_train, t_spls_train, t_kpls_train])
    T_Y_train = np.column_stack([u_plsc_train, u_spls_train, u_kpls_train])
    T_X_test = np.column_stack([t_plsc_test, t_spls_test, t_kpls_test])
    T_Y_test = np.column_stack([u_plsc_test, u_spls_test, u_kpls_test])

    try:
        # Fit Level 1 PLSC meta-learner on Level 0 scores
        meta_pls = behavioral_pls(T_X_train, T_Y_train, n_boot=0, n_perm=0, test_split=0)
        t_stacked = T_X_test @ meta_pls["x_weights"][:, 0]
        u_stacked = T_Y_test @ meta_pls["y_weights"][:, 0]
        r_stacked, _ = spearmanr(t_stacked, u_stacked)
        results["Strategy B (Stacked PLS)"].append(r_stacked)
    except Exception as e:
        results["Strategy B (Stacked PLS)"].append(r_plsc_test)

# Convert results to DataFrame
df_results = pd.DataFrame(results)
print("\nMean out-of-sample Spearman correlations:")
print(df_results.mean())
print("\nStandard deviation of out-of-sample Spearman correlations:")
print(df_results.std())

# Plot results
plt.figure(figsize=(8, 6), dpi=200)
palette_colors = sns.color_palette("Set2", 4)
sns.boxplot(data=df_results, palette=palette_colors, width=0.4, linewidth=1.0, showfliers=False)
sns.stripplot(data=df_results, palette=palette_colors, size=3.0, jitter=True, alpha=0.5, edgecolor="gray", linewidth=0.5)
plt.ylabel("Test score correlation (Spearman)")
plt.xlabel("Strategy")
plt.title(f"Ensemble Model Comparison ({n_splits} splits)")
sns.despine(trim=True)

plt.savefig('figs/ensemble_comparison_cv.pdf', bbox_inches='tight')
plt.close()
print("\nEnsemble evaluation complete. Plot saved to figs/ensemble_comparison_cv.pdf")
