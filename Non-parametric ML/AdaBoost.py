import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import optuna
from optuna.samplers import TPESampler

df = pd.read_csv('E:/001-stimulation and procedure/10-data/final data.csv')
X = df[['down', 'right', 'up', 'left', 't', 'l']] 
y = df['PS']
X_train, X_final, y_train, y_final = train_test_split(X, y, test_size=0.2, random_state=42)
train_set = pd.concat([X_train, y_train], axis=1)
final_set = pd.concat([X_final, y_final], axis=1)
train_set.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/train_set.csv', index=False)
final_set.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/final_set.csv', index=False)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    model = XGBRegressor(**params, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []
    
    for train_index, val_index in cv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]       
        model.fit(X_train_fold, y_train_fold)      
        y_pred_val = model.predict(X_val_fold)       
        r2 = r2_score(y_val_fold, y_pred_val)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
    mean_r2 = np.mean(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    return mean_rmse

study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBoost Hyperparameter Optimization')
study.optimize(objective, n_trials=50)
print(f"Best hyperparameters: {study.best_params}")
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = []
rmse_scores = []

for fold, (train_index, val_index) in enumerate(cv.split(X_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    final_model.fit(X_train_fold, y_train_fold)
    y_pred_val = final_model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    fold_train_set = pd.concat([X_train_fold, y_train_fold], axis=1)
    fold_val_set = pd.concat([X_val_fold, y_val_fold], axis=1)
    fold_train_set.to_csv(f'E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/fold_{fold}_train.csv', index=False)
    fold_val_set.to_csv(f'E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/fold_{fold}_val.csv', index=False)

for i in range(5):
    print(f"Fold {i+1} - R²: {r2_scores[i]:.4f}, RMSE: {rmse_scores[i]:.4f}")
mean_r2 = np.mean(r2_scores)
mean_rmse = np.mean(rmse_scores)
print(f"Mean R² across all folds: {mean_r2:.4f}")
print(f"Mean RMSE across all folds: {mean_rmse:.4f}")
cv_results_df = pd.DataFrame({
    'Fold': range(1, 6),
    'R²': r2_scores,
    'RMSE': rmse_scores
})
cv_results_df.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/cv_results.csv', index=False)
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(range(1, 6), r2_scores, marker='o', color='b', label='R² per fold')
ax1.set_xlabel("Fold")
ax1.set_ylabel("R²", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.axhline(mean_r2, color='b', linestyle='--', linewidth=1)
ax2 = ax1.twinx()
ax2.plot(range(1, 6), rmse_scores, marker='s', color='g', label='RMSE per fold')
ax2.set_ylabel("RMSE", color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.axhline(mean_rmse, color='g', linestyle='--', linewidth=1)
fig.suptitle('Five-Fold Cross-Validation: R² and RMSE')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles + handles2, labels + labels2, loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

y_pred_final = final_model.predict(X_final)
plt.figure(figsize=(6, 6))
plt.scatter(y_final, y_pred_final, color='blue', edgecolors='k', alpha=0.7)
plt.plot([y_final.min(), y_final.max()], [y_final.min(), y_final.max()], color='red', linestyle='--')
plt.title("Final Regression Plot")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()

final_mse = mean_squared_error(y_final, y_pred_final)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(y_final, y_pred_final)
print(f"Final R² on Test Set: {final_r2}")
print(f"Final RMSE on Test Set: {final_rmse}")
final_results = pd.DataFrame({
    'Final R²': [final_r2],
    'Final RMSE': [final_rmse]
})
final_results.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/final_results.csv', index=False)

y_pred_train = final_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_pred_train)

train_results = pd.DataFrame({
    'Train R²': [train_r2],
    'Train RMSE': [train_rmse]
})
train_results.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/train_results.csv', index=False)

train_pred = pd.DataFrame({
    'True Value': y_train,
    'Predicted Value': y_pred_train
})
train_pred.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/train_predictions.csv', index=False)

final_pred = pd.DataFrame({
    'True Value': y_final,
    'Predicted Value': y_pred_final
})
final_pred.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/final_predictions.csv', index=False)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def calculate_re(y_true, y_pred):
    return np.abs(y_true - y_pred) / y_true
error_results = []
for fold, (train_index, val_index) in enumerate(cv.split(X_train), 1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    final_model.fit(X_train_fold, y_train_fold)
    y_pred_val = final_model.predict(X_val_fold)
    r2 = r2_score(y_val_fold, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
    mae = mean_absolute_error(y_val_fold, y_pred_val)
    mse = mean_squared_error(y_val_fold, y_pred_val)
    re = np.mean(calculate_re(y_val_fold, y_pred_val))  
    fold_error = pd.DataFrame({
        'Fold': [fold],
        'R²': [r2],
        'RMSE': [rmse],
        'MSE': [mse],
        'MAE': [mae],
        'RE': [re]
    })
    error_results.append(fold_error)

train_pred = final_model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
train_re = np.mean(calculate_re(y_train, train_pred))
train_mse = mean_squared_error(y_train, train_pred)
train_error = pd.DataFrame({
    'Fold': ['testset'],
    'R²': [train_r2],
    'RMSE': [train_rmse],
    'MSE': [train_mse],
    'MAE': [train_mae],
    'RE': [train_re]
})
error_results.append(train_error)
final_r2 = r2_score(y_final, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_final, y_pred_final))
final_mae = mean_absolute_error(y_final, y_pred_final)
final_re = np.mean(calculate_re(y_final, y_pred_final))
final_mse = mean_squared_error(y_final, y_pred_final)
final_error = pd.DataFrame({
    'Fold': ['testset'],
    'R²': [final_r2],
    'RMSE': [final_rmse],
    'MSE': [final_mse],
    'MAE': [final_mae],
    'RE': [final_re]
})
error_results.append(final_error)
error_df = pd.concat(error_results, ignore_index=True)
error_df.to_csv('E:/001-stimulation and procedure/10-data/01-model data/4-XGBoost/PS/error.csv', index=False)
print("save to 'error.csv' ")
