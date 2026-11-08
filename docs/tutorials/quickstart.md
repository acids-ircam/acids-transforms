# Quickstart

## Transforms are modules

Every transform is a {class}`torch.nn.Module` that derives from
{class}`~acids_transforms.transforms.base.AudioTransform`. A transform has:

- `forward(x)` (or `transform(x)`), which transforms its input;
- `invert(y)`, which gives back the input from the output, if the transform is invertible;
- the flags `invertible`, `scriptable` and `needs_scaling`, which tell you what it supports;
- `scale_data(x)`, which fits the data-dependent parameters of the transform (normalization statistics, for example).
  Call it once on your data before `forward` when `needs_scaling` is `True`.

All transforms can be imported from the top-level package:

```python
import torch
from acids_transforms import STFT, MuLaw, Normalize

x = torch.randn(2, 44100)                    # (channels, time) : 1 second of stereo noise at 44.1kHz

stft = STFT(n_fft=1024, hop_length=256)
y = stft(x)                                  # complex spectrum (2, 173, 513) : (channels, frames, bins)
print(stft.invertible, stft.scriptable)      # True True
x_rec = stft.invert(y)                       # (2, 44032) : exact, up to the last incomplete hop
```

## Shapes

Transforms work on the last dimensions and keep any leading batch dimensions. Audio is `(..., T)`, and spectral
transforms output `(..., n_frames, n_bins)`. With several output modes, they are stacked on a new axis before the
frames: `(..., n_modes, n_frames, n_bins)`.

```python
x = torch.randn(4, 2, 44100)                 # (batch, channels, time)
print(STFT(n_fft=1024, hop_length=256)(x).shape)                                 # (4, 2, 173, 513)
print(STFT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"])(x).shape)    # (4, 2, 2, 173, 513)
```

## Transforms with data-dependent parameters

Some transforms, like {class}`~acids_transforms.transforms.norm.Normalize`, need statistics of the data. Fit them
with `scale_data` first:

```python
x = 3 * torch.randn(16, 1024) + 2

norm = Normalize(mode="gaussian")            # also "unipolar" ([0, 1]) or "bipolar" ([-1, 1])
norm.scale_data(x)
y = norm(x)
print(y.mean().item(), y.std().item())       # ~0, ~1
assert torch.allclose(norm.invert(y), x, atol=1e-5)
```

## Inversion modes

Some transforms can be inverted in several ways. For example, a magnitude spectrum has no phase, so the phase must be
rebuilt. The available methods are given by `get_inversion_modes()`. You can set a default with the `inversion_mode`
argument of the constructor, and override it in each call to `invert`:

```python
print(STFT.get_inversion_modes())            # ['griffin_lim', 'keep_input', 'random', 'sinebank']

magnitude = STFT(n_fft=1024, hop_length=256, mode="magnitude", inversion_mode="griffin_lim")
x = torch.randn(1, 44100)
y = magnitude(x)
x_gl = magnitude.invert(y)                                   # Griffin-Lim
x_random = magnitude.invert(y, inversion_mode="random")      # random phase
```

## Lossy transforms

Not every inversion is exact. {class}`~acids_transforms.transforms.raw.MuLaw` quantizes audio into integer classes,
so its inversion only approximates the input:

```python
mulaw = MuLaw(channels=256)
x = torch.rand(1, 44100) * 2 - 1
codes = mulaw(x)                             # int64 codes in [0, 255]
x_rec = mulaw.invert(codes)                  # quantized approximation of x
```

Next, {doc}`composing` shows how to chain transforms.
