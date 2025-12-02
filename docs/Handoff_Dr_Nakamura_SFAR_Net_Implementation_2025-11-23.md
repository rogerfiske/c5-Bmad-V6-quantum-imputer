# Dr. Nakamura Implementation Handoff
## SFAR-Net Architecture Specification
**Date:** 2025-11-23
**From:** Dr. Rowan Ó Brien (ML Theorist)
**To:** Dr. Kai Nakamura (Research Implementation Specialist)

---

## EXECUTIVE SUMMARY

**Task:** Implement Sparse-Frequency Anomaly Ranker (SFAR-Net) to predict the 20 least likely QV column activations for event 11624.

**Mathematical Foundation:** Analysis PD-001 (Dr. Mercer)
**Architecture Type:** Inverse probability ranking neural network
**A-B-C Validation Required:** Yes (all three layers)

---

## PROBLEM STATEMENT

**Input:** Historical QS/QV event data (11,623 events × 39 QV columns, binary sparse matrix)
**Output:** Ranked list of 20 QV columns with **lowest** activation probability for next event (11624)
**Constraint:** Must respect mathematical invariants from Analysis PD-001

---

## MATHEMATICAL CONSTRAINTS (FROM DR. MERCER)

### Layer A: Additive Combinatorics
- **Sparsity:** 89% (average 4.2 activations per event)
- **Structure:** Non-uniform QV activation distribution
- **Hole patterns:** Structured gaps in column activations
- **Requirement:** Model must preserve sparsity constraints

### Layer B: Analytic Number Theory
- **Fourier spectrum:** Non-pseudorandom structure
- **Column frequencies:** Non-uniform activation rates
- **Requirement:** Model must capture frequency-domain features

### Layer C: Finite-Field Algebra
- **Format:** GF(2) binary encoding (all entries ∈ {0,1})
- **Canonical form:** Algebraically stable
- **Requirement:** Binary predictions, deterministic outputs

---

## SFAR-NET ARCHITECTURE SPECIFICATION

### Architecture Overview

**Name:** Sparse-Frequency Anomaly Ranker Network (SFAR-Net)
**Purpose:** Learn probability distribution P(QV_i = 1 | history) for each column i, rank inversely

### Component Breakdown

#### 1. Input Layer: Sparse Event History Encoder

**Input Format:**
```python
# Shape: (batch_size, sequence_length, 39)
# Values: Binary {0, 1}
# Example sequence_length: Last 100 events
input_events = torch.tensor(events[-100:], dtype=torch.float32)
```

**Encoding:**
```python
class SparseEventEncoder(nn.Module):
    def __init__(self, n_columns=39, embed_dim=128):
        super().__init__()
        self.column_embed = nn.Embedding(n_columns, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, 39)
        # Embed sparse activations
        embedded = self.column_embed(torch.arange(39))
        # Multiply by activation mask
        weighted = x.unsqueeze(-1) * embedded
        # Add positional encoding
        encoded = self.positional_encoding(weighted)
        return encoded  # (batch, seq_len, 39, embed_dim)
```

**Output:** Dense embedding preserving sparsity structure

---

#### 2. Spectral Analysis Module (Layer B Compliance)

**Purpose:** Capture Fourier-domain structure identified by Dr. Mercer

**Implementation:**
```python
class SpectralAnalysisModule(nn.Module):
    def __init__(self, n_columns=39):
        super().__init__()
        # Neural FFT layer (learnable)
        self.fft_weights = nn.Parameter(torch.randn(n_columns, n_columns))
        self.spectral_mlp = nn.Sequential(
            nn.Linear(n_columns, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, column_activations):
        # column_activations: (batch, seq_len, 39)
        # Compute weighted FFT per sequence
        fft_features = torch.fft.rfft(column_activations, dim=-1)
        # Apply learnable transformation
        spectral = torch.matmul(fft_features.real, self.fft_weights)
        # MLP transformation
        features = self.spectral_mlp(spectral)
        return features  # (batch, seq_len, 64)
```

**Key Features:**
- Learns frequency-domain representation
- Captures column activation periodicity
- Aligns with Mercer's Fourier analysis

---

#### 3. Hole-Pattern Detector (Layer A Compliance)

**Purpose:** Learn combinatorial signatures of rare vs common columns

**Implementation:**
```python
class HolePatternDetector(nn.Module):
    def __init__(self, n_columns=39, hidden_dim=128):
        super().__init__()
        # Multi-head attention over columns
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        # Co-occurrence pattern learner
        self.pattern_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, column_embeddings):
        # column_embeddings: (batch, seq_len, n_columns, hidden_dim)
        batch, seq, cols, dim = column_embeddings.shape
        # Reshape for attention
        x = column_embeddings.view(batch * seq, cols, dim)
        # Self-attention: which columns co-occur?
        attended, _ = self.attention(x, x, x)
        # Pattern extraction
        patterns = self.pattern_mlp(attended)
        return patterns.view(batch, seq, cols, -1)
```

**Key Features:**
- Learns: "Which columns appear together?"
- Detects: Rare column combinations (holes)
- Preserves: Additive structure constraints

---

#### 4. Temporal Recurrence Module

**Purpose:** Capture non-Markovian temporal dependencies

**Implementation:**
```python
class TemporalRecurrenceModule(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU()
        )

    def forward(self, sequence_features):
        # sequence_features: (batch, seq_len, input_dim)
        # GRU processes temporal sequence
        gru_out, hidden = self.gru(sequence_features)
        # Take final hidden state (current context)
        context = self.context_mlp(gru_out[:, -1, :])
        return context  # (batch, 128)
```

**Key Features:**
- Models event-to-event dependencies
- Captures long-range patterns
- Non-Markovian (maintains hidden state)

---

#### 5. Output Layer: Inverse Probability Ranker

**Purpose:** Generate per-column activation probabilities, rank inversely

**Implementation:**
```python
class InverseProbabilityRanker(nn.Module):
    def __init__(self, context_dim=128, n_columns=39):
        super().__init__()
        self.column_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output probability [0,1]
            ) for _ in range(n_columns)
        ])

    def forward(self, context):
        # context: (batch, context_dim)
        # Predict activation probability for each column
        probs = torch.cat([
            predictor(context) for predictor in self.column_predictors
        ], dim=1)  # (batch, 39)
        return probs

    def rank_inverse(self, probs, k=20):
        # Return indices of k lowest-probability columns
        _, indices = torch.topk(probs, k, largest=False)
        return indices  # (batch, 20)
```

**Key Features:**
- Per-column probability estimation
- Inverse ranking (lowest probabilities)
- Top-k selection (k=20)

---

### Complete SFAR-Net Integration

```python
class SFARNet(nn.Module):
    """
    Sparse-Frequency Anomaly Ranker Network
    Predicts 20 least likely QV column activations
    """
    def __init__(self, n_columns=39, seq_length=100):
        super().__init__()
        self.n_columns = n_columns
        self.seq_length = seq_length

        # Components
        self.encoder = SparseEventEncoder(n_columns=n_columns)
        self.spectral = SpectralAnalysisModule(n_columns=n_columns)
        self.hole_detector = HolePatternDetector(n_columns=n_columns)
        self.temporal = TemporalRecurrenceModule(input_dim=128)
        self.ranker = InverseProbabilityRanker(context_dim=128, n_columns=n_columns)

    def forward(self, event_history):
        # event_history: (batch, seq_len, 39)
        # 1. Encode sparse events
        encoded = self.encoder(event_history)

        # 2. Spectral analysis
        spectral_features = self.spectral(event_history)

        # 3. Hole pattern detection
        hole_features = self.hole_detector(encoded)

        # 4. Combine features
        combined = torch.cat([
            spectral_features.mean(dim=2),  # Aggregate over columns
            hole_features.mean(dim=2)
        ], dim=-1)

        # 5. Temporal recurrence
        context = self.temporal(combined)

        # 6. Predict probabilities
        probs = self.ranker(context)

        return probs

    def predict_top20_unlikely(self, event_history):
        """Predict 20 least likely columns for next event"""
        probs = self.forward(event_history)
        indices = self.ranker.rank_inverse(probs, k=20)
        return indices, probs
```

---

## TRAINING PROTOCOL

### Data Preparation

```python
# Load dataset
import pandas as pd
data = pd.read_csv('data/raw/c5_Matrix.csv')
events = data.iloc[:, 1:].values  # Skip event-ID column

# Train/validation split
train_events = events[:10000]
val_events = events[10000:11623]

# Create sliding window sequences
def create_sequences(events, seq_length=100):
    sequences = []
    targets = []
    for i in range(len(events) - seq_length):
        sequences.append(events[i:i+seq_length])
        targets.append(events[i+seq_length])
    return np.array(sequences), np.array(targets)

X_train, y_train = create_sequences(train_events)
X_val, y_val = create_sequences(val_events)
```

### Loss Function: Sparsity-Constrained Binary Cross-Entropy

```python
class SparsityConstrainedLoss(nn.Module):
    def __init__(self, target_sparsity=4.2, sparsity_weight=0.1):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        # Standard BCE loss
        bce_loss = self.bce(probs, targets)

        # Sparsity constraint: penalize if predicted activations != 4.2
        predicted_activations = probs.sum(dim=1)
        sparsity_loss = torch.mean(
            (predicted_activations - self.target_sparsity) ** 2
        )

        # Combined loss
        total_loss = bce_loss + self.sparsity_weight * sparsity_loss
        return total_loss, bce_loss, sparsity_loss
```

**Rationale:** Enforces Layer A constraint (sparsity preservation)

### Training Loop

```python
# Initialize model
model = SFARNet(n_columns=39, seq_length=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = SparsityConstrainedLoss(target_sparsity=4.2)

# Deterministic training (Layer C requirement)
torch.manual_seed(42)
np.random.seed(42)

# Training
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    for batch_idx in range(0, len(X_train), batch_size):
        # Get batch
        X_batch = torch.tensor(X_train[batch_idx:batch_idx+batch_size], dtype=torch.float32)
        y_batch = torch.tensor(y_train[batch_idx:batch_idx+batch_size], dtype=torch.float32)

        # Forward pass
        probs = model(X_batch)
        loss, bce, sparsity = criterion(probs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        val_probs = model(X_val_tensor)
        val_loss, val_bce, val_sparsity = criterion(val_probs, y_val_tensor)

    print(f"Epoch {epoch}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val Sparsity={val_sparsity:.4f}")
```

---

## VALIDATION CRITERIA (A-B-C FRAMEWORK)

### Layer A: Additive Combinatorics Validation

**Test:**
```python
def validate_layer_a(model, val_data):
    """Check if predicted sparsity matches observed"""
    probs = model(val_data)
    predicted_activations = (probs > 0.5).float().sum(dim=1)

    # Expected: mean ≈ 4.2, std ≈ 1.5
    mean_activations = predicted_activations.mean().item()
    std_activations = predicted_activations.std().item()

    # PASS if within tolerance
    pass_layer_a = (4.0 <= mean_activations <= 4.5)
    print(f"Layer A: Mean activations = {mean_activations:.2f} (target: 4.2)")
    return pass_layer_a
```

**Verdict:** PASS if mean predicted activations ≈ 4.2 ± 0.5

---

### Layer B: Analytic Diagnostics Validation

**Test:**
```python
def validate_layer_b(model, val_data):
    """Check if model captures Fourier structure"""
    import numpy as np

    # Get predicted probabilities
    probs = model(val_data).detach().cpu().numpy()

    # Compute FFT of predicted column probabilities
    fft_pred = np.fft.rfft(probs.mean(axis=0))

    # Compute FFT of actual column frequencies
    actual_freqs = val_data.mean(dim=(0,1)).cpu().numpy()
    fft_actual = np.fft.rfft(actual_freqs)

    # Correlation between predicted and actual spectra
    correlation = np.corrcoef(np.abs(fft_pred), np.abs(fft_actual))[0,1]

    # PASS if correlation > 0.5
    pass_layer_b = (correlation > 0.5)
    print(f"Layer B: Fourier correlation = {correlation:.3f} (target: > 0.5)")
    return pass_layer_b
```

**Verdict:** PASS if model's predicted spectrum correlates with observed

---

### Layer C: Finite-Field Algebra Validation

**Test:**
```python
def validate_layer_c(model, val_data):
    """Check GF(2) compatibility and determinism"""

    # Test 1: Deterministic outputs (same input -> same output)
    torch.manual_seed(42)
    out1 = model(val_data)
    torch.manual_seed(42)
    out2 = model(val_data)
    deterministic = torch.allclose(out1, out2)

    # Test 2: Binary outputs (after thresholding)
    binary_outputs = (out1 > 0.5).float()
    is_binary = torch.all((binary_outputs == 0) | (binary_outputs == 1))

    # PASS if both tests pass
    pass_layer_c = deterministic and is_binary
    print(f"Layer C: Deterministic={deterministic}, Binary={is_binary}")
    return pass_layer_c
```

**Verdict:** PASS if outputs are deterministic and binary-compatible

---

## BASELINE IMPLEMENTATIONS

### Baseline 1: Frequency-Based Heuristic (No Learning)

```python
def baseline_frequency_heuristic(train_events, k=20):
    """Simply return k least-frequent columns"""
    column_frequencies = train_events.mean(axis=0)  # (39,)
    # Sort ascending (lowest frequency first)
    least_frequent_indices = np.argsort(column_frequencies)[:k]
    return least_frequent_indices

# Usage
baseline1_prediction = baseline_frequency_heuristic(train_events, k=20)
print(f"Baseline 1 predicts columns: {baseline1_prediction}")
```

**Purpose:** Simplest possible approach - no temporal or combinatorial modeling

---

### Baseline 2: Random Forest Ensemble

```python
from sklearn.ensemble import RandomForestClassifier

def baseline_random_forest(X_train, y_train, k=20):
    """Train per-column binary classifiers"""
    classifiers = []

    # Train one classifier per column
    for col_idx in range(39):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train.reshape(len(X_train), -1), y_train[:, col_idx])
        classifiers.append(clf)

    # Predict probabilities for next event
    def predict_top20(event_history):
        probs = np.array([
            clf.predict_proba(event_history.reshape(1, -1))[0, 1]
            for clf in classifiers
        ])
        # Return 20 lowest-probability columns
        return np.argsort(probs)[:k]

    return predict_top20

# Usage
baseline2_predictor = baseline_random_forest(X_train, y_train, k=20)
baseline2_prediction = baseline2_predictor(events[-100:])
```

**Purpose:** Standard ML approach without mathematical constraints

---

## PREDICTION FOR EVENT 11624

```python
# Load full dataset
data = pd.read_csv('data/raw/c5_Matrix.csv')
events = data.iloc[:, 1:].values

# Use last 100 events as context
context_events = events[-100:]

# SFAR-Net prediction
model.eval()
with torch.no_grad():
    context_tensor = torch.tensor(context_events, dtype=torch.float32).unsqueeze(0)
    indices, probs = model.predict_top20_unlikely(context_tensor)

# Results
top20_unlikely = indices.cpu().numpy()[0]
column_names = [f"QV_{i+1}" for i in top20_unlikely]

print("=" * 60)
print("PREDICTION FOR EVENT 11624")
print("=" * 60)
print(f"20 Least Likely QV Columns to Activate:")
for rank, (idx, col_name) in enumerate(zip(top20_unlikely, column_names), 1):
    prob = probs[0, idx].item()
    print(f"  {rank:2d}. {col_name:6s} (probability: {prob:.4f})")
print("=" * 60)
```

---

## EXPECTED CHALLENGES

### 1. Extreme Sparsity
- **Problem:** Only 4.2 activations per event (11% density)
- **Impact:** Most columns are 0 most of the time (class imbalance)
- **Solution:**
  - Use weighted loss (higher weight for rare activations)
  - Oversample minority class during training
  - Sparsity-constrained loss function

### 2. Limited Training Data
- **Problem:** Only 11,623 events (relatively small for deep learning)
- **Impact:** Risk of overfitting
- **Solution:**
  - Regularization (dropout, L2)
  - Early stopping
  - Simple architecture (avoid over-parameterization)

### 3. Temporal Dependencies
- **Problem:** Non-Markovian structure (long-range dependencies)
- **Impact:** Standard RNNs may struggle
- **Solution:**
  - GRU with multiple layers
  - Attention mechanisms
  - Longer sequence lengths (100+ events)

### 4. Inverse Prediction Task
- **Problem:** Predicting LEAST likely (not most likely) is unusual
- **Impact:** Standard classification metrics may mislead
- **Solution:**
  - Custom evaluation metrics
  - Inverse ranking loss
  - Validate against actual event 11624 when available

---

## DELIVERABLES

### Code Artifacts
1. `sfar_net.py` - Complete model implementation
2. `train_sfar.py` - Training script with A-B-C validation
3. `predict_event_11624.py` - Prediction script
4. `baselines.py` - Baseline implementations

### Documentation
1. Implementation log in `research-dev-sidecar/implementation-log.md`
2. Experimental provenance record (dataset hash, code version, parameters)
3. A-B-C validation report

### Results
1. Top-20 least likely columns for event 11624
2. Comparison vs baselines
3. Mathematical validation status (PASS/FAIL per layer)

---

## HANDOFF TO DR. MERCER

After implementation and testing, submit results to Dr. Mercer for final A-B-C validation:

1. **Layer A validation:** Sparsity preservation verified
2. **Layer B validation:** Fourier structure captured
3. **Layer C validation:** GF(2) compatibility confirmed
4. **Provenance:** Full experimental record with SHA-512 hash

---

## NEXT STEPS

1. **Implement SFAR-Net** following specifications above
2. **Train model** on events 1-10,000
3. **Validate** on events 10,001-11,623
4. **Run baselines** for comparison
5. **Predict** top-20 unlikely columns for event 11624
6. **Document** results with full provenance
7. **Submit** to Dr. Mercer for mathematical validation

---

**Mathematical Authority:** Dr. Elara V. Mercer (Analysis PD-001)
**Architecture Design:** Dr. Rowan Ó Brien (SFAR-Net specification)
**Implementation:** Dr. Kai Nakamura (this handoff)

**Provenance:** Based on validated mathematical constraints, frontier ML research, and QS/QV-specific requirements.

---

**Status:** READY FOR IMPLEMENTATION
**Priority:** HIGH
**Estimated Implementation Time:** 2-3 days (rapid research prototype)

---
