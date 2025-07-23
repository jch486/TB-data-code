import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

frontier_val_fn = os.path.join('experiment_results', 'my_experiment', 'results', 'frontier_val.csv')
frontier_val = pd.read_csv(frontier_val_fn)

# TPR and FPR values
tpr = frontier_val['TPR'].astype(float).tolist()
fpr = frontier_val['FPR'].astype(float).tolist()

# Sort by FPR
fpr, tpr = zip(*sorted(zip(fpr, tpr)))

# Calculate AUC
auc = np.trapezoid(tpr, fpr)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
