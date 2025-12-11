# %% Imports
from utils import DataLoader
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import shap
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
feature_names = X_test.columns.tolist()

# %% Fit blackbox model
svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm.fit(X_train.values, y_train)
y_pred = svm.predict(X_test.values)

cm = confusion_matrix(y_test, y_pred)
recall_c1 = recall_score(y_test, y_pred, pos_label=1)
precision_c1 = precision_score(y_test, y_pred, pos_label=1)
f1_macro = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

print(f"\nConfusion Matrix:\n{cm}")
print(f"Recall (Finding Strokes): {recall_c1}")
print(f"Precision (Trustworthiness): {precision_c1}")
print(f"F1 Score: {f1_macro}")
print(f"Accuracy: {accuracy}\n")


def predict_wrapper(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return svm.predict_proba(x)


# %% Create Kernel SHAP explainer
background_data = shap.kmeans(X_train.values, 50)
explainer = shap.KernelExplainer(predict_wrapper, background_data)

# %% Prepare the data for SHAP, using the selected indices
indices_to_explain = [78, 433, 251, 19, 642, 701, 687, 76, 378, 584, 94, 382]
print(f"Local Shap Explainer selected index: {indices_to_explain}\n")
X_explain = X_test.iloc[indices_to_explain]
shap_values_single = explainer.shap_values(X_explain.values, nsamples=128)


def get_category(actual, predicted):
    if actual == 1 and predicted == 1:
        return 'TP'
    elif actual == 0 and predicted == 0:
        return 'TN'
    elif actual == 0 and predicted == 1:
        return 'FP'
    elif actual == 1 and predicted == 0:
        return 'FN'
    return 'UNKNOWN'


# %% >> Visualize local predictions
# Force plot
for i, index in enumerate(indices_to_explain):
    prediction = svm.predict(X_explain[i:i + 1].values)
    actual = y_test.iloc[index]
    category = get_category(actual, prediction)

    print(f"\n--- Explaining Sample Index {index} ({category}) ---")
    print(f"Actual: {actual}, Predicted: {prediction}")

    shap.force_plot(explainer.expected_value[1],
                    shap_values_single[1][i],
                    X_explain[i:i+1].values,
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False)  # for values

    plt.savefig(f"kernel_shap/kernel_force_plot_index_{index}_category_{category}.png", bbox_inches='tight', dpi=300)
    plt.close()


# %% >> Visualize global features
# Feature summary
plt.figure()
shap_values_global = explainer.shap_values(X_test.values, nsamples=128)
shap.summary_plot(shap_values_global[1], X_test.values, feature_names=feature_names, show=False)
plt.savefig('svm_summary_plot.png', bbox_inches='tight', dpi=300)

