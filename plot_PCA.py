import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # --- Step 1: Load DataFrame ---
    features_formatted_fn = os.path.join('other_data', 'features_formatted.csv')
    outcomes_fn = os.path.join('other_data', 'outcomes.csv')
    outcomes = pd.read_csv(outcomes_fn)
    features_formatted = pd.read_csv(features_formatted_fn)
    df = features_formatted.merge(outcomes, on="example_id")

    # --- Step 2: Extract Feature Vectors and Labels ---
    feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
    X = df[feature_cols].to_numpy()
    y = df['has_tb'].to_numpy()  # 1 for TB, 0 for no TB

    # --- Step 3: Center the Data ---
    X_centered = X - X.mean(axis=0)

    # --- Step 4: PCA via Eigen Decomposition ---
    cov_matrix = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top 2 components
    top2 = eigenvectors[:, :2]
    X_projected = X_centered @ top2

    # --- Step 5: PCA Scatter Plot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(X_projected[y==0, 0], X_projected[y==0, 1], alpha=0.6, label='No TB')
    plt.scatter(X_projected[y==1, 0], X_projected[y==1, 1], alpha=0.6, label='TB')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection of Patient Vectors")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Step 6: Eigenvalue Spectrum Comparison ---
    X_TB = X[y == 1]
    X_no_TB = X[y == 0]

    # Center separately
    X_TB_centered = X_TB - X_TB.mean(axis=0)
    X_no_TB_centered = X_no_TB - X_no_TB.mean(axis=0)

    eigvals_TB = np.linalg.eigvals(np.cov(X_TB_centered.T))
    eigvals_no_TB = np.linalg.eigvals(np.cov(X_no_TB_centered.T))

    plt.figure(figsize=(8, 4))
    plt.plot(sorted(eigvals_TB, reverse=True), 'o-', label='TB Patients')
    plt.plot(sorted(eigvals_no_TB, reverse=True), 'o-', label='No TB Patients')
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue (Variance)")
    plt.title("Eigenvalue Spectrum Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()