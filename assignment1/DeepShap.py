# %% Imports
from utils import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
import shap
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)

feature_names = X_test.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)


# %% Fit blackbox model
class StrokeModel(nn.Module):
    def __init__(self, input_size):
        super(StrokeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


model = StrokeModel(input_size=X_train_tensor.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)

    cm = confusion_matrix(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
    recall_c1 = recall_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), pos_label=1)
    precision_c1 = precision_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), pos_label=1)
    f1_macro = f1_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())

    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Recall (Finding Strokes): {recall_c1}")
    print(f"Precision (Trustworthiness): {precision_c1}")
    print(f"F1 Score: {f1_macro}")
    print(f"Accuracy: {accuracy}\n")


# %% Create SHAP explainer
background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(model, background)

# %% Prepare the data for SHAP, using the selected indices
indices_to_explain = [78, 433, 251, 19, 642, 701, 687, 76, 378, 584, 94, 382]
print(f"Local Shap Explainer selected index: {indices_to_explain}")
indices_tensor = torch.tensor(indices_to_explain, dtype=torch.long)
X_explain_tensor = X_test_tensor[indices_tensor]
shap_values_single = explainer.shap_values(X_explain_tensor)


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
    sample_tensor = X_explain_tensor[i:i + 1]
    with torch.no_grad():
        output = model(sample_tensor)
        prediction = torch.argmax(output, dim=1).item()

    actual = y_test.iloc[index]
    category = get_category(actual, prediction)

    print(f"\n--- Explaining Sample Index {index} ({category}) ---")
    print(f"Actual: {actual}, Predicted: {prediction}")

    shap.force_plot(explainer.expected_value[1],
                    shap_values_single[1][i],
                    X_test.iloc[index],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False)  # for values

    plt.savefig(f"deep_shap/deep_force_plot_index_{index}_category_{category}.png", bbox_inches='tight', dpi=300)
    plt.close()


# %% >> Visualize global features
# Feature summary
plt.figure()
shap_values_global = explainer.shap_values(X_test_tensor)
shap.summary_plot(shap_values_global[1], X_test, feature_names=feature_names, show=False)
plt.savefig('deep_summary_plot.png', bbox_inches='tight', dpi=300)

