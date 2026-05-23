"""
neuromimetic_acoustic.py
========================
Neuromimetic Acoustic Processing via Differential Entropy (DE)
and Hjorth Parameters (Activity, Mobility, Complexity).

Motivation
----------
Depressed patients exhibit psychomotor retardation whose micro-tremor
signature in speech is analogous to the flattened neural oscillation
patterns seen in EEG studies of depression.  By borrowing BCI-grade
feature extraction we surface those sub-band dynamics before the
standard audio encoder ever sees the signal.

Pipeline
--------
  raw waveform  →  epoch (sliding window)
                →  band-pass filter into δ/θ/α/β/γ sub-bands
                →  per-epoch DE  (5 values per epoch)
                →  per-epoch Hjorth Activity / Mobility / Complexity
                →  concatenate → [B, n_epochs, 8]
                →  temporal Transformer encoder
                →  mean-pool → neuro_feat  [B, hidden_dim]

Usage
-----
    from neuromimetic_acoustic import NeuromimeticAcousticProcessor
    nap = NeuromimeticAcousticProcessor(sr=16000, hidden_dim=768).to(device)
    neuro_feat = nap(wav)          # wav: [B, T]  →  [B, hidden_dim]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Band-pass filter (FIR, causal, zero-phase via reflection padding)
# ---------------------------------------------------------------------------

def _make_bandpass_kernel(low_hz: float, high_hz: float, sr: int,
                           num_taps: int = 127) -> torch.Tensor:
    """Sinc-windowed band-pass FIR kernel, returned as a 1-D tensor."""
    assert num_taps % 2 == 1, "num_taps must be odd for a linear-phase FIR"
    t = torch.arange(num_taps) - num_taps // 2          # centre at 0
    t = t.float()

    def sinc(x):
        eps = 1e-8
        return torch.where(x.abs() < eps,
                           torch.ones_like(x),
                           torch.sin(math.pi * x) / (math.pi * x))

    low  = 2.0 * low_hz  / sr
    high = 2.0 * high_hz / sr
    kernel = high * sinc(high * t) - low * sinc(low * t)
    window = torch.blackman_window(num_taps, periodic=False)
    kernel = kernel * window
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel                                        # [num_taps]


class BandPassFilter(nn.Module):
    """
    Zero-phase FIR band-pass for a single frequency band.
    Works on batched 1-D signals: input [B, T].
    """
    # Classical EEG sub-bands (Hz)
    BANDS = {
        "delta": (0.5,  4.0),
        "theta": (4.0,  8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    def __init__(self, sr: int = 16000, num_taps: int = 127):
        super().__init__()
        self.sr       = sr
        self.num_taps = num_taps
        self.pad      = num_taps // 2

        # Register one kernel per band (non-trainable)
        for name, (lo, hi) in self.BANDS.items():
            k = _make_bandpass_kernel(lo, hi, sr, num_taps)
            self.register_buffer(f"kernel_{name}", k)

    def _apply_band(self, wav: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """wav: [B, T], kernel: [num_taps] → [B, T]"""
        B, T = wav.shape
        # reflection padding to approximate zero-phase
        padded = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        kernel = kernel.view(1, 1, -1)                   # [1, 1, num_taps]
        out = F.conv1d(padded, kernel, padding=0)        # [B, 1, T]
        return out.squeeze(1)                            # [B, T]

    def forward(self, wav: torch.Tensor):
        """Returns dict band_name → [B, T]"""
        results = {}
        for name in self.BANDS:
            k = getattr(self, f"kernel_{name}")
            results[name] = self._apply_band(wav, k)
        return results


# ---------------------------------------------------------------------------
# 2.  Differential Entropy per epoch
# ---------------------------------------------------------------------------

def differential_entropy(epoch: torch.Tensor) -> torch.Tensor:
    """
    DE of a Gaussian-approximated signal epoch.
    DE = 0.5 * log(2πe * σ²)
    epoch: [..., L]  →  scalar [...] per epoch
    """
    var = epoch.var(dim=-1).clamp(min=1e-10)
    de  = 0.5 * torch.log(2 * math.pi * math.e * var)
    return de                                            # [...]


# ---------------------------------------------------------------------------
# 3.  Hjorth Parameters per epoch
# ---------------------------------------------------------------------------

def hjorth_parameters(epoch: torch.Tensor):
    """
    Computes Activity, Mobility, Complexity for a signal epoch.
    epoch: [B, n_epochs, L]
    Returns three tensors each [B, n_epochs]
    """
    # Activity: variance of the signal
    activity   = epoch.var(dim=-1).clamp(min=1e-10)     # [B, n_epochs]

    # First derivative (finite difference)
    d1         = torch.diff(epoch, dim=-1)               # [B, n_epochs, L-1]
    var_d1     = d1.var(dim=-1).clamp(min=1e-10)

    # Mobility: sqrt(var(x') / var(x))
    mobility   = torch.sqrt(var_d1 / activity)           # [B, n_epochs]

    # Second derivative
    d2         = torch.diff(d1, dim=-1)                  # [B, n_epochs, L-2]
    var_d2     = d2.var(dim=-1).clamp(min=1e-10)
    mobility_d1 = torch.sqrt(var_d2 / var_d1.clamp(min=1e-10))

    # Complexity: mobility(x') / mobility(x)
    complexity  = mobility_d1 / mobility.clamp(min=1e-10)

    return activity, mobility, complexity


# ---------------------------------------------------------------------------
# 4.  Main module
# ---------------------------------------------------------------------------

class NeuromimeticAcousticProcessor(nn.Module):
    """
    Extracts DE + Hjorth features across 5 EEG-inspired frequency sub-bands
    from raw speech waveforms and projects them to `hidden_dim`.

    Args
    ----
    sr          : sample rate of the incoming waveform (default 16 000 Hz)
    epoch_sec   : epoch length in seconds (default 0.25 s = 250 ms)
    hop_sec     : hop size in seconds (default 0.125 s = 50% overlap)
    hidden_dim  : output feature dimensionality (matches model hidden_dim)
    n_heads     : attention heads in the temporal Transformer
    n_layers    : number of Transformer encoder layers
    num_taps    : FIR filter length (odd number, default 127)
    """

    N_BANDS     = 5   # δ θ α β γ
    FEATS_PER_BAND = 4  # DE, Activity, Mobility, Complexity
    RAW_DIM     = N_BANDS * FEATS_PER_BAND   # 20

    def __init__(
        self,
        sr:         int   = 16_000,
        epoch_sec:  float = 0.25,
        hop_sec:    float = 0.125,
        hidden_dim: int   = 768,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        num_taps:   int   = 127,
    ):
        super().__init__()
        self.sr          = sr
        self.epoch_len   = int(epoch_sec * sr)
        self.hop_len     = int(hop_sec   * sr)
        self.hidden_dim  = hidden_dim

        # Band-pass filter bank
        self.bpf = BandPassFilter(sr=sr, num_taps=num_taps)

        # Project raw neuromimetic features → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.RAW_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Lightweight temporal Transformer to model epoch-level dynamics
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Learnable positional encoding (up to 512 epochs)
        self.pos_emb = nn.Embedding(512, hidden_dim)

    # ------------------------------------------------------------------
    def _epoch_signal(self, band_wav: torch.Tensor) -> torch.Tensor:
        """
        Slice a filtered waveform into overlapping epochs.
        band_wav : [B, T]
        returns  : [B, n_epochs, epoch_len]
        """
        B, T = band_wav.shape
        epochs = band_wav.unfold(dimension=1,
                                  size=self.epoch_len,
                                  step=self.hop_len)  # [B, n_epochs, epoch_len]
        return epochs

    # ------------------------------------------------------------------
    def _extract_features(self, band_wav: torch.Tensor) -> torch.Tensor:
        """
        Compute [DE, Activity, Mobility, Complexity] per epoch.
        band_wav : [B, T]
        returns  : [B, n_epochs, 4]
        """
        epochs = self._epoch_signal(band_wav)          # [B, n_epochs, L]
        de     = differential_entropy(epochs)           # [B, n_epochs]
        act, mob, cplx = hjorth_parameters(epochs)     # each [B, n_epochs]
        feats  = torch.stack([de, act, mob, cplx], dim=-1)  # [B, n_epochs, 4]
        return feats

    # ------------------------------------------------------------------
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav : [B, T] (raw waveform, float32, range ~[-1, 1])
        returns neuro_feat : [B, hidden_dim]
        """
        # Safety: handle mono squeezed inputs
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Compute filtered bands
        band_dict = self.bpf(wav)                       # 5 × [B, T]

        # Per-band feature extraction
        band_feats = []
        for name in BandPassFilter.BANDS:               # preserves order
            f = self._extract_features(band_dict[name]) # [B, n_epochs, 4]
            band_feats.append(f)

        # Concatenate across bands → [B, n_epochs, 20]
        n_epochs = band_feats[0].shape[1]
        neuro_raw = torch.cat(band_feats, dim=-1)        # [B, n_epochs, 20]

        # Project to hidden_dim
        x = self.input_proj(neuro_raw)                  # [B, n_epochs, H]

        # Add positional encoding
        pos_ids = torch.arange(n_epochs, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)                  # [B, n_epochs, H]

        # Temporal Transformer over epochs
        x = self.transformer(x)                         # [B, n_epochs, H]

        # Mean-pool across epoch dimension
        neuro_feat = x.mean(dim=1)                      # [B, H]
        return neuro_feat