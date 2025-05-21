import torch
from deepseek_v2_lite import DeepSeekV2Lite

# Model hyperparameters
vocab_size = 32000
dim = 512
n_layers = 6
n_heads = 8
ff_dim = 2048
max_seq_len = 128
batch_size = 4
seq_len = 128

# Instantiate the model
model = DeepSeekV2Lite(
    vocab_size=vocab_size,
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    ff_dim=ff_dim,
    max_seq_len=max_seq_len
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dummy input (random tokens)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
target_ids = input_ids.clone()

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Forward pass
model.train()
logits = model(input_ids)  # shape: (B, T, vocab_size)

# Compute loss
loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

# Backward pass
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Training step - loss: {loss.item():.4f}")
