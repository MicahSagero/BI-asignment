import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report

# 1. Load Data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

# Ensure column names are valid for XGBoost (replace < with _)
X_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns]
X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]

# 2. Model Training and Evaluation Function
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results = {
        'Model': model_name,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': cm.tolist(),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 30)
    
    return results, y_proba

# 3. Initialize Models
# Logistic Regression (Baseline, Interpretable)
lr_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

# Random Forest (Ensemble, Robust)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5)

# XGBoost (High Performance)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)

# 4. Train and Evaluate
all_results = []
lr_results, lr_proba = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Logistic Regression")
all_results.append(lr_results)

rf_results, rf_proba = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
all_results.append(rf_results)

xgb_results, xgb_proba = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
all_results.append(xgb_results)

# 5. Save Results
results_df = pd.DataFrame([
    {'Model': r['Model'], 'AUC': r['AUC'], 'Precision': r['Precision'], 'Recall': r['Recall']}
    for r in all_results
])
results_df.to_csv('model_comparison.csv', index=False)

# Save confusion matrices for diagram generation
cm_lr = pd.DataFrame(lr_results['Confusion Matrix'], index=['Actual <=50K', 'Actual >50K'], columns=['Predicted <=50K', 'Predicted >50K'])
cm_rf = pd.DataFrame(rf_results['Confusion Matrix'], index=['Actual <=50K', 'Actual >50K'], columns=['Predicted <=50K', 'Predicted >50K'])
cm_xgb = pd.DataFrame(xgb_results['Confusion Matrix'], index=['Actual <=50K', 'Actual >50K'], columns=['Predicted <=50K', 'Predicted >50K'])

cm_lr.to_csv('cm_lr.csv')
cm_rf.to_csv('cm_rf.csv')
cm_xgb.to_csv('cm_xgb.csv')

print("\nModel training and evaluation complete. Results saved to model_comparison.csv and confusion matrix files.")
