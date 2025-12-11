# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
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
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 0
end_index = 1
shap_values_single = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]

# %% Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values_single[0].shape)
shap_values_single

# %% >> Visualize local predictions
shap.initjs()
# Force plot
prediction = rf.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")
force_plot = shap.force_plot(explainer.expected_value[1],
                shap_values_single[1],
                X_test[start_index:end_index]) # for values

shap.save_html("force_plot.html", force_plot)

shap.force_plot(explainer.expected_value[1],
                shap_values_single[1],
                X_test[start_index:end_index],
                matplotlib=True,
                show=False) # for values

plt.savefig('force_plot.png', bbox_inches='tight', dpi=300)

# %% >> Visualize global features
# Feature summary
plt.figure()
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig('summary_plot.png', bbox_inches='tight', dpi=300)