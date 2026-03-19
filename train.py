import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from load_data import load_data
from model import SignCNN

# ── Load data ─────────────────────────
X_train, y_train = load_data(train=True)
X_test, y_test = load_data(train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset + DataLoader (IMPORTANT FIX) ──
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ── Model ─────────────────────────────
model = SignCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

# ── Training ───────────────────────────
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# ── Evaluation ─────────────────────────
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ── SAVE MODEL (CRITICAL FIX) ──────────
torch.save(model.state_dict(), "signcnn.pth")
print("Model saved to signcnn.pth")
