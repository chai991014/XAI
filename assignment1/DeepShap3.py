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

print(X_train_tensor.shape)
print(X_test_tensor.shape)


# %% Fit blackbox model
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        return self.relu(out)


class StrokeModel(nn.Module):
    def __init__(self, input_size):
        super(StrokeModel, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output_layer(x)
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


# # %% Create SHAP explainer
# background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 100, replace=False)]
# explainer = shap.DeepExplainer(model, background)
#
# # Calculate shapley values for test data
# start_index = 0
# end_index = 10
# # ResNets + BatchNorm require check_additivity=False to ignore small precision errors
# shap_values_single = explainer.shap_values(X_test_tensor[start_index:end_index], check_additivity=False)
#
# # %% >> Visualize local predictions
# # Force plot
# for i in range(end_index):
#     sample_tensor = X_test_tensor[start_index + i: start_index + i + 1]
#     with torch.no_grad():
#         output = model(sample_tensor)
#         prediction = torch.argmax(output, dim=1).item()
#     print(f"Sample {i} ResNet predicted: {prediction}")
#     shap.force_plot(explainer.expected_value[1],
#                     shap_values_single[1][i],
#                     X_test[start_index + i: start_index + i + 1],
#                     feature_names=feature_names,
#                     matplotlib=True,
#                     show=False)  # for values
#     plt.savefig(f"deep3_force_plot_{i}.png", bbox_inches='tight', dpi=300)
#
# # %% >> Visualize global features
# # Feature summary
# plt.figure()
# shap_values_global = explainer.shap_values(X_test_tensor, check_additivity=False)
# shap.summary_plot(shap_values_global[1], X_test, feature_names=feature_names, show=False)
# plt.savefig('deep3_summary_plot.png', bbox_inches='tight', dpi=300)