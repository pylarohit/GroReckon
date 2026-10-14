import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from georeckon_pipeline import KinoNetR2

model = KinoNetR2()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Create dummy IMU windows and speed targets (55 km/h = 15.27 m/s)
X = torch.randn(500, 6, 20)
y = torch.ones(500) * 15.27 + torch.randn(500) * 0.2
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

for epoch in range(15):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        mu, log_var = model(bx)
        loss = F.huber_loss(mu, by) + 0.05 * torch.mean(log_var**2)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    mu_test, _ = model(X[:5])
    print("True speed:", y[:5].numpy())
    print("Predicted speed:", mu_test.numpy())
