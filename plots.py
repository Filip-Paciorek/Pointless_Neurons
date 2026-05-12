import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
from matplotlib.colors import LogNorm


# Setting background grid for all plots
sns.set_style("whitegrid")

# LOADING THE DATA 
# Reading the raw outputs generated earlier by our C program
loss_min_max = pd.read_csv('data_holder/loss_minmax.csv', header=None)
loss_z_score = pd.read_csv('data_holder/loss_zscore.csv', header=None)
y_pred_min_max = pd.read_csv('data_holder/y_pred_minmax.csv', header=None)
y_pred_z_score = pd.read_csv('data_holder/y_pred_zscore.csv', header=None)
# The actual ground truth: 1 means Pulsar, 0 means Noise
y_true = pd.read_csv('data_holder/y_true.csv', header=None) 


#LEARNING CURVE PLOT
#Creating a canvas for the plot
plt.figure(figsize=(10, 6))

#Plotting how the error (MSE) drops over time (epochs)
#Purple line for Min-Max scaling
plt.plot(loss_min_max[0], label="Min Max scaling", color="#6249a7", linewidth=1.5)
#Green line for Z-Score (slightly transparent 'alpha' so they don't block each other)
plt.plot(loss_z_score[0], label="Z_score scaling", color="#1b9744", alpha=0.7, linewidth=1.5)

#Adding titles, labels, and legend to make it readable
plt.title(f'Learning Curve Analysis: Min-Max vs Z-Score ({len(loss_min_max)} Epochs)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.legend(fontsize=11)

#Adjusting layout so nothing gets cut off, then saving and showing
plt.tight_layout()
#Uncomment the line below to save the plot for the README file
plt.savefig('Plots_holder/learning_curve.png', dpi=300) 
plt.show()


#CONFUSION MATRICES PLOTS
#Converting continuous probabilities (like 0.82) into strict yes/no decisions.
#Threshold = 0.7: If score > 0.7, we classify it as a Pulsar (1). Otherwise, Noise (0).
min_max_final = (y_pred_min_max > 0.7).astype(int)
zscore_final = (y_pred_z_score > 0.7).astype(int)

# Building the matrices. We force the [1, 0] order so Pulsar is in the top-left corner
cm_minmax = confusion_matrix(y_true, min_max_final, labels=[1, 0])
cm_zscore = confusion_matrix(y_true, zscore_final, labels=[1, 0])

# Setting up a side-by-side view (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot:Min-Max heatmap (Green theme)
sns.heatmap(cm_minmax, ax=ax1, annot=True, fmt='d', cmap="Greens", vmax=100, cbar=False, 
            xticklabels=["Pulsar (1)", "Noise (0)"], yticklabels=["Pulsar (1)", "Noise (0)"],
            linewidths=1, linecolor='black')
ax1.set_title("Min-Max Scaling")
ax1.set_ylabel('ACTUAL', fontsize=12, fontweight='bold')
ax1.set_xlabel('PREDICT', fontsize=12, fontweight='bold')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')


# Right plot: Z-Score heatmap (Blue theme)
sns.heatmap(cm_zscore, ax=ax2, annot=True, fmt='d', cmap="Blues", vmax=100, cbar=False, 
            xticklabels=["Pulsar (1)", "Noise (0)"], yticklabels=["Pulsar (1)", "Noise (0)"],
            linewidths=1, linecolor='black')

ax2.set_title("Z_Score Scaling")
ax2.set_ylabel('ACTUAL', fontsize=12, fontweight='bold')
ax2.set_xlabel('PREDICT', fontsize=12, fontweight='bold')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')

# Adding a main title above both matrices
plt.suptitle('Comparison of Confusion Matrices: Min-Max vs Z-Score', 
             fontsize=18, fontweight='bold', y=0.98) 

# Leaving some space at the top so the main title fits nicely
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

# Uncomment the line below to save the plot for the README file
plt.savefig('Plots_holder/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# ROC CURVE

# 'fpr'=False Positive Rate (mistakes), 'tpr'=True Positive Rate (correct catches)
#Calculating coordinates and AUC (Area Under Curve) for Min-Max
fpr_minmax, tpr_minmax, _ = roc_curve(y_true, y_pred_min_max)
auc_minmax = roc_auc_score(y_true, y_pred_min_max)

#Calculating coordinates and AUC for Z-Score
fpr_zscore, tpr_zscore, _ = roc_curve(y_true, y_pred_z_score)
auc_zscore = roc_auc_score(y_true, y_pred_z_score)

plt.figure(figsize=(8, 6))

#Drawing the model performance lines
plt.plot(fpr_minmax, tpr_minmax, color='green', label=f'Min-Max Scaling (AUC = {auc_minmax:.4f})')
plt.plot(fpr_zscore, tpr_zscore, color='blue', label=f'Z-Score Scaling (AUC = {auc_zscore:.4f})')

#the diagonal red dashed line represents a completely random "coin flip" model
plt.plot([0, 1], [0, 1], color='red', linestyle='--')

#Plot cosmetics
plt.title('ROC Curve Comparison', fontweight='bold', fontsize=14)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.legend(loc='lower right') # Best spot for ROC legend
plt.grid(alpha=0.3)

#Uncomment the line below to save the plot for the README file
plt.savefig('Plots_holder/roc_curve_comparison.png', dpi=300)
plt.show()


#PRECISION-RECALL CURVE

#Calculating Precision,Recall and Average Precision(AP)for Min-Max
precision_mm, recall_mm, _ = precision_recall_curve(y_true, y_pred_min_max)
ap_mm = average_precision_score(y_true, y_pred_min_max)

#Calculating Precision, Recall and Average Precision (AP) for Z-Score
precision_zs, recall_zs, _ = precision_recall_curve(y_true, y_pred_z_score)
ap_zs = average_precision_score(y_true, y_pred_z_score)

plt.figure(figsize=(8, 6))

#Drawing the PR curves
plt.plot(recall_mm, precision_mm, color='green', label=f'Min-Max (AP = {ap_mm:.4f})')
plt.plot(recall_zs, precision_zs, color='blue', label=f'Z-Score (AP = {ap_zs:.4f})')


baseline = y_true[0].sum() / len(y_true) # Using y_true[0] to prevent Pandas errors
plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')

#Plot cosmetics
plt.title('Precision-Recall Curve Comparison', fontweight='bold', fontsize=14)
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (Trustworthiness)', fontsize=12)
plt.legend(loc='lower left') 
plt.grid(alpha=0.3)

# Uncomment the line below to save the plot for the README file
plt.savefig('Plots_holder/precision_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.show()