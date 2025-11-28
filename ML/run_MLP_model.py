from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from data_loader import data_loader
from MLP_model import MLP

# 1) Load data
train_df, valid_df, encoders = data_loader()

# 2) Feature columns (same as your LightGBM)
feature_cols = [
    "Division_enc", "HomeTeam_enc", "AwayTeam_enc",
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
    "HandiSize", "HandiHome", "HandiAway",
    "EloDiff", "EloSum",
    "Form3Diff", "Form5Diff",
    "ProbHome", "ProbDraw", "ProbAway", "BookerMargin",
    "OddHome_missing", "OddDraw_missing", "OddAway_missing",
    "HandiSize_missing", "HandiHome_missing", "HandiAway_missing",
]

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df["Target"].values.astype(np.int64)

X_valid = valid_df[feature_cols].values.astype(np.float32)
y_valid = valid_df["Target"].values.astype(np.int64)

# 3) Scale features (important for MLPs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# 🔹 Ensure no NaNs / inf in the scaled data
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

print("Train finite?", np.isfinite(X_train).all())
print("Valid finite?", np.isfinite(X_valid).all())

# 4) Convert to PyTorch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)

X_valid_t = torch.from_numpy(X_valid)
y_valid_t = torch.from_numpy(y_valid)

# Hyperparameters
input_dim = X_train.shape[1]
num_classes = 3
batch_size = 1024
epochs = 20
learning_rate = 1e-3
dropout_p = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP(input_dim, hidden1=128, hidden2=64, dropout_p=dropout_p, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoaders
train_ds = TensorDataset(X_train_t, y_train_t)
valid_ds = TensorDataset(X_valid_t, y_valid_t)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

for epoch in range(1, epochs + 1):
    # ---- Train ----
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_acc = accuracy_score(all_targets, all_preds)
    # Replace NaN and inf with safe values
    all_probs = np.nan_to_num(all_probs, nan=1.0/3.0, posinf=1.0, neginf=0.0)

    # Renormalize rows to sum to 1
    row_sums = all_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    all_probs = all_probs / row_sums

    # Clip for numerical stability
    eps = 1e-15
    all_probs = np.clip(all_probs, eps, 1 - eps)

    val_logloss = log_loss(all_targets, all_probs)

    print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | "
          f"Val acc: {val_acc:.4f} | Val log loss: {val_logloss:.4f}")
