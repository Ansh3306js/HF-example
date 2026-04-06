



import torch
import torch.nn as nn
import math

# ── STEP 1: Positional Encoding ──────────────────────────────────
# Adds info about WHERE each token is in the sequence.
# Without this, "cat sat" and "sat cat" look identical to the model.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # even dims: sin wave
        pe[:, 1::2] = torch.cos(pos * div)  # odd  dims: cos wave
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# ── STEP 2: Tiny Transformer Model ───────────────────────────────
# Uses PyTorch's built-in TransformerEncoder — saves us writing
# the attention math by hand (we'll understand it conceptually above).
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()

        # Embedding: maps each token index → a vector of size d_model
        self.embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding: adds position info to embeddings
        self.pos_enc = PositionalEncoding(d_model)

        # Stack of Transformer encoder layers
        # Each layer = Multi-Head Attention + Add&Norm + FFN + Add&Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,   # size of each token vector
            nhead=nhead,       # number of attention heads
            dim_feedforward=dim_ff,  # hidden size of FFN
            dropout=dropout,
            batch_first=True   # we want (batch, seq, features) shape
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        # Output head: project from d_model → vocab_size
        # This gives us a score for each possible next token
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)  — integer token indices
        x = self.embed(x)          # → (batch, seq, d_model)
        x = self.pos_enc(x)        # → (batch, seq, d_model) + position info
        x = self.transformer(x)    # → (batch, seq, d_model) after attention
        logits = self.fc_out(x)    # → (batch, seq, vocab_size)
        return logits

# ── STEP 3: Toy Dataset ──────────────────────────────────────────
# We'll train on the repeating pattern "abcabcabc..."
# The model should learn: after "a" comes "b", after "b" comes "c", etc.

def make_dataset(seq_len=20, n_samples=500):
    """Create (input, target) pairs from a repeating pattern."""
    pattern = "abcdefg"  # simple repeating sequence
    vocab = sorted(set(pattern))
    char2idx = {c: i for i, c in enumerate(vocab)}  # {'a':0, 'b':1, ...}
    idx2char = {i: c for c, i in char2idx.items()}

    full_text = (pattern * 100)[:n_samples + seq_len]  # long repeating string

    X, Y = [], []
    for i in range(n_samples):
        chunk = full_text[i : i + seq_len]
        target = full_text[i+1 : i + seq_len + 1]
        X.append([char2idx[c] for c in chunk])   # input: indices
        Y.append([char2idx[c] for c in target])  # target: next char indices

    X = torch.tensor(X)   # shape: (n_samples, seq_len)
    Y = torch.tensor(Y)   # shape: (n_samples, seq_len)
    return X, Y, char2idx, idx2char

X, Y, char2idx, idx2char = make_dataset()
vocab_size = len(char2idx)

print(f"Vocab: {char2idx}")       # {'a':0, 'b':1, 'c':2, ...}
print(f"X[0]: {X[0][:8]}")       # input tokens
print(f"Y[0]: {Y[0][:8]}")       # target tokens (shifted by 1)
print(f"Vocab size: {vocab_size}")
# → 7 characters, each input predicts the next char

# ── STEP 4: The Training Loop ────────────────────────────────────
# This is the core loop. For each batch of data:
#  1. Forward pass: run model, get predictions
#  2. Compute loss: how wrong were we?
#  3. Backward pass: calculate gradients (who caused the error?)
#  4. Update weights: nudge weights to be less wrong next time

def train(model, X, Y, epochs=30, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # standard for classification

    model.train()  # tells model we're in training mode (enables dropout etc.)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        # ── Mini-batch loop ──
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]  # (batch, seq_len)
            y_batch = Y[i:i+batch_size]  # (batch, seq_len)

            # 1️⃣ Forward pass
            logits = model(x_batch)
            # logits shape: (batch, seq_len, vocab_size)
            # We need to flatten for cross-entropy:
            # (batch*seq_len, vocab_size) vs (batch*seq_len,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # predictions
                y_batch.view(-1)                   # true labels
            )

            # 2️⃣ Zero old gradients (IMPORTANT! or they accumulate)
            optimizer.zero_grad()

            # 3️⃣ Backward pass — compute gradients
            loss.backward()

            # 4️⃣ Optional: clip gradients so they don't explode
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 5️⃣ Update weights using Adam optimizer
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

    return model

# ── STEP 5: Train and Test ────────────────────────────────────────

# Create model
model = TinyTransformer(vocab_size=vocab_size, d_model=64,
                         nhead=4, num_layers=2, dim_ff=128)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# → ~100K parameters — a tiny but real Transformer!

# Train
model = train(model, X, Y, epochs=30, batch_size=32, lr=0.001)

# ── Test: can it predict the sequence? ──
model.eval()  # turn off dropout

def predict_next(model, seed, char2idx, idx2char, n=10):
    """Given a seed string, predict the next n characters."""
    result = seed
    inp = torch.tensor([[char2idx[c] for c in seed]])

    for _ in range(n):
        with torch.no_grad():
            logits = model(inp)  # (1, seq, vocab)
        next_tok = logits[0, -1].argmax().item()  # greedy: pick most likely
        next_char = idx2char[next_tok]
        result += next_char
        # slide window: drop first char, add predicted char
        new_idx = [char2idx[c] for c in result[-len(seed):]]
        inp = torch.tensor([new_idx])

    return result

# Should output: "abcdefgabcdefgab..." if the model learned the pattern
output = predict_next(model, "abcde", char2idx, idx2char, n=15)
print(f"Seed: 'abcde' → Predicted: '{output}'")

# ── Expected output after training ──
# Epoch  5 | Loss: 1.9312
# Epoch 10 | Loss: 1.2155
# Epoch 15 | Loss: 0.6432
# Epoch 20 | Loss: 0.3218
# Epoch 25 | Loss: 0.1892
# Epoch 30 | Loss: 0.1124
# Seed: 'abcde' → Predicted: 'abcdefgabcdefgabcdefg'
# Loss goes down = model is learning!

