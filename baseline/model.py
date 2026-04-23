"""
CNN + LSTM Model for Counter-UAV Classification
================================================
Implements the exact architecture from the paper:
- CNN: 4 conv blocks with BN + GAP + FC
- LSTM: 2-layer LSTM for temporal tracking
- BFP feature concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN for micro-Doppler spectrogram feature extraction."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5)
        )
    
    def forward(self, x):
        # x: (B, 1, 128, 128)
        x = self.conv1(x)   # (B, 32, 64, 64)
        x = self.conv2(x)   # (B, 64, 32, 32)
        x = self.conv3(x)   # (B, 128, 16, 16)
        x = self.conv4(x)   # (B, 256, 8, 8)
        x = self.gap(x)     # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)      # (B, 128)
        return x


class CNNClassifier(nn.Module):
    """CNN-only classifier (no LSTM, no BFP)."""
    
    def __init__(self, n_classes=4):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.classifier = nn.Linear(128, n_classes)
    
    def forward(self, x):
        features = self.cnn(x)
        return self.classifier(features)


class CNNBPFClassifier(nn.Module):
    """CNN + BFP classifier (no LSTM)."""
    
    def __init__(self, n_classes=4, bfp_dim=3):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.classifier = nn.Linear(128 + bfp_dim, n_classes)
    
    def forward(self, x, bfp):
        features = self.cnn(x)
        combined = torch.cat([features, bfp], dim=1)
        return self.classifier(combined)


class LSTMTracker(nn.Module):
    """LSTM for temporal sequence classification."""
    
    def __init__(self, input_dim=131, n_classes=4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, n_classes)
    
    def forward(self, x):
        # x: (B, T, 131)
        out, _ = self.lstm1(x)    # (B, T, 64)
        out, _ = self.lstm2(out)  # (B, T, 32)
        out = out[:, -1, :]       # Last time step: (B, 32)
        return self.fc(out)


class CNNLSTMClassifier(nn.Module):
    """Full CNN + LSTM + BFP model."""
    
    def __init__(self, n_classes=4, bfp_dim=3, seq_len=10):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = LSTMTracker(input_dim=128 + bfp_dim, n_classes=n_classes)
        self.seq_len = seq_len
    
    def forward(self, x_seq, bfp_seq):
        """
        x_seq: (B, T, 1, 128, 128) - sequence of spectrograms
        bfp_seq: (B, T, 3) - sequence of BFP features
        """
        B, T = x_seq.shape[0], x_seq.shape[1]
        
        # Process ALL frames through CNN at once (batch B*T)
        x_flat = x_seq.reshape(B * T, *x_seq.shape[2:])  # (B*T, 1, 128, 128)
        cnn_flat = self.cnn(x_flat)                        # (B*T, 128)
        cnn_features = cnn_flat.reshape(B, T, -1)          # (B, T, 128)
        
        # Concatenate BFP features
        combined = torch.cat([cnn_features, bfp_seq], dim=2)  # (B, T, 131)
        
        # LSTM classification
        return self.lstm(combined)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_architectures():
    """Verify parameter counts match paper claims."""
    print("=" * 50)
    print("MODEL PARAMETER VERIFICATION")
    print("=" * 50)
    
    cnn = CNNFeatureExtractor()
    print(f"\nCNN Feature Extractor:")
    print(f"  Parameters: {count_parameters(cnn):,}")
    print(f"  Paper claims: ~420K")
    
    lstm = LSTMTracker(input_dim=131, n_classes=4)
    print(f"\nLSTM Tracker:")
    print(f"  Parameters: {count_parameters(lstm):,}")
    print(f"  Paper claims: ~62.7K")
    
    full_model = CNNLSTMClassifier()
    print(f"\nFull CNN+LSTM+BFP Model:")
    print(f"  Parameters: {count_parameters(full_model):,}")
    print(f"  Paper claims: ~483K")
    
    # Test forward pass
    print("\nForward pass test:")
    x_seq = torch.randn(2, 10, 1, 128, 128)
    bfp_seq = torch.randn(2, 10, 3)
    out = full_model(x_seq, bfp_seq)
    print(f"  Input: x_seq={x_seq.shape}, bfp_seq={bfp_seq.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Output probs: {F.softmax(out[0], dim=0).detach().numpy()}")
    
    return cnn, lstm, full_model


if __name__ == '__main__':
    verify_architectures()
