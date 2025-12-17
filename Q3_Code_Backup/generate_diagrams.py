import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set a professional style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Model Comparison Bar Chart ---
def plot_model_comparison():
    df = pd.read_csv('model_comparison.csv')
    
    # Melt the DataFrame for easier plotting
    df_melt = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    # Filter for the key metrics
    df_melt = df_melt[df_melt['Metric'].isin(['AUC', 'Precision', 'Recall'])]
    
    plt.figure(figsize=(10, 6))
    
    # Define colors for each metric
    palette = {'AUC': '#1f77b4', 'Precision': '#ff7f0e', 'Recall': '#2ca02c'}
    
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melt, palette=palette)
    
    plt.title('Comparison of Predictive Model Performance (Q3)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('model_comparison_chart.png')
    plt.close()
    print("Saved model_comparison_chart.png")

# --- 2. Confusion Matrix Heatmaps ---
def plot_confusion_matrix(file_path, title):
    cm_df = pd.read_csv(file_path, index_col=0)
    
    # Convert to numpy array for plotting
    cm = cm_df.values
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=cm_df.columns, yticklabels=cm_df.index)
    
    plt.title(f'Confusion Matrix: {title}', fontsize=14)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'cm_{title.replace(" ", "_").lower()}.png')
    plt.close()
    print(f"Saved cm_{title.replace(' ', '_').lower()}.png")

if __name__ == '__main__':
    plot_model_comparison()
    
    # Plot for the two best-performing models (Random Forest and XGBoost)
    plot_confusion_matrix('cm_rf.csv', 'Random Forest')
    plot_confusion_matrix('cm_xgb.csv', 'XGBoost')
    
    # Also plot the baseline Logistic Regression for completeness
    plot_confusion_matrix('cm_lr.csv', 'Logistic Regression')
    
    print("All diagrams generated successfully.")
