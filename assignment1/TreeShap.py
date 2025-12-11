# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import shap
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)

# %% Fit blackbox model
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)
p_stroke = y_prob[:, 1]
NEW_THRESHOLD = 0.1
y_pred_new = (p_stroke >= NEW_THRESHOLD).astype(int)

cm = confusion_matrix(y_test, y_pred_new)
recall_c1 = recall_score(y_test, y_pred_new, pos_label=1)
precision_c1 = precision_score(y_test, y_pred_new, pos_label=1)
f1_macro = f1_score(y_test, y_pred_new, average='macro')
accuracy = accuracy_score(y_test, y_pred_new)

print(f"\nThreshold Value: {NEW_THRESHOLD}")
print(f"Confusion Matrix:\n{cm}")
print(f"Recall (Finding Strokes): {recall_c1}")
print(f"Precision (Trustworthiness): {precision_c1}")
print(f"F1 Score: {f1_macro}")
print(f"Accuracy: {accuracy}\n")

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# %% Find TP TN FP FN sample indices
y_test_np = y_test.values
X_test_np = X_test.values
tp_indices = np.where((y_test_np == 1) & (y_pred_new == 1))[0]
tn_indices = np.where((y_test_np == 0) & (y_pred_new == 0))[0]
fp_indices = np.where((y_test_np == 0) & (y_pred_new == 1))[0]
fn_indices = np.where((y_test_np == 1) & (y_pred_new == 0))[0]
selected_indices = {}
category_map = {
    'TP': tp_indices,
    'TN': tn_indices,
    'FP': fp_indices,
    'FN': fn_indices
}

np.random.seed(2025)
for category, indices in category_map.items():
    if len(indices) < 3:
        selected_indices[category] = indices
        # print(f"Selected index for {category}: {selected_indices[category]}")
    else:
        selected_indices[category] = np.random.choice(indices, 3, replace=False)
        # print(f"Selected index for {category}: {selected_indices[category]}")

# %% Prepare the data for SHAP, using the selected indices
indices_to_explain = np.concatenate(list(selected_indices.values()))
X_explain = X_test.iloc[indices_to_explain]
print(f"Local Shap Explainer selected index: {indices_to_explain.tolist()}")
shap_values_selected = explainer.shap_values(X_explain)


for i, index in enumerate(indices_to_explain):
    for k, v in selected_indices.items():
        if index in v:
            category = k
            break
    single_X = X_explain[i:i + 1]
    prediction = rf.predict(single_X)[0]
    actual = y_test.iloc[index]

    print(f"\n--- Explaining Sample Index {index} ({category}) ---")
    print(f"Actual: {actual}, Predicted: {prediction}")

    shap.force_plot(explainer.expected_value[1],
                    shap_values_selected[1][i],
                    single_X,
                    matplotlib=True,
                    show=False)

    plt.savefig(f"tree_shap/tree_force_plot_index_{index}_category_{category}.png", bbox_inches='tight', dpi=300)
    plt.close()


# %% >> Visualize global features
# Feature summary
plt.figure()
shap_values_global = explainer.shap_values(X_test)
shap.summary_plot(shap_values_global[1], X_test, show=False)
plt.savefig('tree_summary_plot.png', bbox_inches='tight', dpi=300)

