import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import matplotlib.pyplot as plt
import os
import json
import joblib

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(0)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=380, latent_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))

class MLPModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, output_dim=32, dropout=0.2):
        super(MLPModel, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class WeightedMSELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weight', weight.view(1, -1))
    def forward(self, input, target):
        return ((self.weight * (input - target) ** 2).mean())

def evaluate_metrics_curvewise(true, pred):
    r2 = np.mean([r2_score(true[i], pred[i]) for i in range(len(true))])
    rmse = np.mean([np.sqrt(mean_squared_error(true[i], pred[i])) for i in range(len(true))])
    mae = np.mean([mean_absolute_error(true[i], pred[i]) for i in range(len(true))])
    re = np.mean([np.mean(np.abs((true[i] - pred[i]) / (true[i]))) for i in range(len(true))])
    mse = np.mean([mean_squared_error(true[i], pred[i]) for i in range(len(true))])
    return r2, rmse, mae, re, mse

def compute_modulus(strain, stress):
    # strain, stress: [batch_size, N]
    x = strain
    y = stress
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    numerator = ((x - x_mean) * (y - y_mean)).sum(dim=1)
    denominator = ((x - x_mean) ** 2).sum(dim=1) + 1e-8
    slope = numerator / denominator  # shape: [batch_size]
    return slope

img_path = r"E:\001-stimulation and procedure\10-data\MLP_AE5"
os.makedirs(img_path, exist_ok=True)
latent_dim = 32
early_stop_patience = 100

data = pd.read_csv(os.path.join(img_path, r"E:\001-stimulation and procedure\10-data\MLP\data/final data.csv"))
X = data.iloc[:, :6].values.astype(np.float32)
y = data.iloc[:, 6:6+380].values.astype(np.float32)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

joblib.dump(scaler_X, os.path.join(img_path, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(img_path, "scaler_y.pkl"))

weight_array = np.ones(380)
for i in range(1, 20, 2): weight_array[i] = 2.0
weight_tensor = torch.tensor(weight_array, dtype=torch.float32)

ae = AutoEncoder(latent_dim=latent_dim)
optimizer_ae = optim.Adam(ae.parameters(), lr=1e-3)
criterion_ae = WeightedMSELoss(weight_tensor)
best_loss = float('inf')
patience = 0
ae_train_losses = []
ae_test_losses = []

for epoch in range(200):
    ae.train()
    optimizer_ae.zero_grad()
    inputs_train = torch.tensor(y_train_val, dtype=torch.float32)
    output_train = ae(inputs_train)
    loss_train = criterion_ae(output_train, inputs_train)
    loss_train.backward()
    optimizer_ae.step()

    ae.eval()
    with torch.no_grad():
        inputs_test = torch.tensor(y_test, dtype=torch.float32)
        output_test = ae(inputs_test)
        loss_test = criterion_ae(output_test, inputs_test)

    ae_train_losses.append(loss_train.item())
    ae_test_losses.append(loss_test.item())

    if loss_train.item() < best_loss:
        best_loss = loss_train.item()
        best_ae_state = ae.state_dict()
        patience = 0
    else:
        patience += 1
    if patience >= early_stop_patience:
        break

ae.load_state_dict(best_ae_state)

with torch.no_grad():
    y_latent = ae.encode(torch.tensor(y_train_val, dtype=torch.float32)).numpy()
    y_test_latent = ae.encode(torch.tensor(y_test, dtype=torch.float32)).numpy()

search_space = {'hidden_dim': (128, 512), 'num_layers': (2, 5), 'dropout': (0.0, 0.3), 'lr': (1e-4, 1e-2)}
def objective(trial):
    model = MLPModel(
        hidden_dim=trial.suggest_int('hidden_dim', *search_space['hidden_dim']),
        num_layers=trial.suggest_int('num_layers', *search_space['num_layers']),
        dropout=trial.suggest_float('dropout', *search_space['dropout']),
        output_dim=latent_dim)
    lr = trial.suggest_float('lr', *search_space['lr'], log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X_train_val):
        for _ in range(50):
            model.train()
            optimizer.zero_grad()
            pred = model(torch.tensor(X_train_val[train_idx], dtype=torch.float32))
            loss = criterion(pred, torch.tensor(y_latent[train_idx], dtype=torch.float32))
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_train_val[val_idx], dtype=torch.float32)).numpy()
        scores.append(np.mean([r2_score(y_latent[val_idx][i], pred_val[i]) for i in range(len(val_idx))]))
    return -np.mean(scores)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
with open(os.path.join(img_path, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=4)

elastic_weight = 0.0001  

def compute_modulus(strain, stress):
    x = strain
    y = stress
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    numerator = ((x - x_mean) * (y - y_mean)).sum(dim=1)
    denominator = ((x - x_mean) ** 2).sum(dim=1) + 1e-8
    slope = numerator / denominator  
    return slope

final_mlp = MLPModel(
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    output_dim=latent_dim)
optimizer = optim.Adam(final_mlp.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()
train_losses, test_losses = [], []
best_val_loss = float('inf'); patience = 0

for epoch in range(200):
    final_mlp.train()
    optimizer.zero_grad()
    output = final_mlp(torch.tensor(X_train_val, dtype=torch.float32))
    loss_latent = criterion(output, torch.tensor(y_latent, dtype=torch.float32))

    decoded = ae.decode(output)
    strain_pred = decoded[:, ::2][:, :4]
    stress_pred = decoded[:, 1::2][:, :4]
    strain_true = torch.tensor(y_train_val, dtype=torch.float32)[:, ::2][:, :4]
    stress_true = torch.tensor(y_train_val, dtype=torch.float32)[:, 1::2][:, :4]
    E_pred = compute_modulus(strain_pred, stress_pred)
    E_true = compute_modulus(strain_true, stress_true)
    loss_elastic = F.mse_loss(E_pred, E_true)

    loss = loss_latent + elastic_weight * loss_elastic
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    final_mlp.eval()
    with torch.no_grad():
        test_output = final_mlp(torch.tensor(X_test, dtype=torch.float32))
        test_loss_latent = criterion(test_output, torch.tensor(y_test_latent, dtype=torch.float32))
        test_decoded = ae.decode(test_output)
        strain_pred = test_decoded[:, ::2][:, :3]
        stress_pred = test_decoded[:, 1::2][:, :3]
        strain_true = torch.tensor(y_test, dtype=torch.float32)[:, ::2][:, :3]
        stress_true = torch.tensor(y_test, dtype=torch.float32)[:, 1::2][:, :3]
        E_pred = compute_modulus(strain_pred, stress_pred)
        E_true = compute_modulus(strain_true, stress_true)
        loss_elastic_test = F.mse_loss(E_pred, E_true)
        total_test_loss = test_loss_latent + elastic_weight * loss_elastic_test
        test_losses.append(total_test_loss.item())

    if total_test_loss.item() < best_val_loss:
        best_val_loss = total_test_loss.item()
        best_model_state = final_mlp.state_dict()
        patience = 0
    else:
        patience += 1
    if patience >= early_stop_patience:
        break
final_mlp.load_state_dict(best_model_state)

with torch.no_grad():
    y_train_pred = ae.decode(final_mlp(torch.tensor(X_train_val, dtype=torch.float32))).numpy()
    y_test_pred = ae.decode(final_mlp(torch.tensor(X_test, dtype=torch.float32))).numpy()
y_train_true = scaler_y.inverse_transform(y_train_val)
y_test_true = scaler_y.inverse_transform(y_test)
y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_test_pred = scaler_y.inverse_transform(y_test_pred)

r2_train, rmse_train, mae_train, re_train, mse_train = evaluate_metrics_curvewise(y_train_true, y_train_pred)
r2_test, rmse_test, mae_test, re_test, mse_test = evaluate_metrics_curvewise(y_test_true, y_test_pred)

cv_metrics = {'R2': [], 'RMSE': [], 'MAE': [], 'RE': [], 'MSE': []}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    model = MLPModel(
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
        output_dim=latent_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(torch.tensor(X_train_val[train_idx], dtype=torch.float32))
        loss_latent = criterion(pred, torch.tensor(y_latent[train_idx], dtype=torch.float32))

        decoded = ae.decode(pred)
        strain_pred = decoded[:, ::2][:, :10]
        stress_pred = decoded[:, 1::2][:, :10]
        strain_true = torch.tensor(y_train_val[train_idx], dtype=torch.float32)[:, ::2][:, :10]
        stress_true = torch.tensor(y_train_val[train_idx], dtype=torch.float32)[:, 1::2][:, :10]
        E_pred = compute_modulus(strain_pred, stress_pred)
        E_true = compute_modulus(strain_true, stress_true)
        loss_elastic = F.mse_loss(E_pred, E_true)

        loss = loss_latent + elastic_weight * loss_elastic
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred_latent = model(torch.tensor(X_train_val[val_idx], dtype=torch.float32))
        val_pred_curve = ae.decode(val_pred_latent).numpy()
        val_true_curve = scaler_y.inverse_transform(y_train_val[val_idx])
        val_pred_curve = scaler_y.inverse_transform(val_pred_curve)

    r2, rmse, mae, re, mse = evaluate_metrics_curvewise(val_true_curve, val_pred_curve)
    for k, v in zip(cv_metrics.keys(), [r2, rmse, mae, re, mse]):
        cv_metrics[k].append(v)

summary_data = {
    'Metric': list(cv_metrics.keys()),
    'Fold 1': [cv_metrics[m][0] for m in cv_metrics],
    'Fold 2': [cv_metrics[m][1] for m in cv_metrics],
    'Fold 3': [cv_metrics[m][2] for m in cv_metrics],
    'Fold 4': [cv_metrics[m][3] for m in cv_metrics],
    'Fold 5': [cv_metrics[m][4] for m in cv_metrics],
    'Mean CV': [np.mean(cv_metrics[m]) for m in cv_metrics],
    'Train': [r2_train, rmse_train, mae_train, re_train, mse_train],
    'Test': [r2_test, rmse_test, mae_test, re_test, mse_test]
}
pd.DataFrame(summary_data).to_csv(os.path.join(img_path, "metrics_summary.csv"), index=False)

plt.figure(figsize=(6, 4))
plt.plot(ae_train_losses, label='AE Train Loss')
plt.plot(ae_test_losses, label='AE Test Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('AutoEncoder Weighted Loss')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(img_path, "ae_loss_curve.png"), dpi=300)
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('AE + MLP Loss')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(img_path, "ae_mlp_loss_curve.png"), dpi=300)

torch.save(ae.state_dict(), os.path.join(img_path, "autoencoder.pth"))
torch.save(final_mlp.state_dict(), os.path.join(img_path, "mlp_model.pth"))

ae_loss_df = pd.DataFrame({
    'Epoch': list(range(1, len(ae_train_losses) + 1)),
    'AE Train Loss': ae_train_losses,
    'AE Test Loss': ae_test_losses
})
ae_loss_df.to_csv(os.path.join(img_path, "ae_train_loss.csv"), index=False)
mlp_loss_df = pd.DataFrame({
    'Epoch': list(range(1, len(train_losses) + 1)),
    'MLP Train Loss': train_losses,
    'MLP Test Loss': test_losses
})
mlp_loss_df.to_csv(os.path.join(img_path, "mlp_train_test_loss.csv"), index=False)
