# Composing transforms

## Chaining with `+`

Adding transforms with `+` gives a
{class}`~acids_transforms.transforms.base.ComposeAudioTransform`. It applies the transforms from left to right in
`forward`, and inverts them from right to left in `invert`:

```python
import torch
from acids_transforms import Mono, STFT, DGT, Magnitude

x = torch.randn(2, 44100)                    # stereo noise

transform = Mono() + STFT(n_fft=1024, hop_length=256) + Magnitude(n_fft=1024, mel=False, mode="unipolar")
print(transform)
print(len(transform.transforms), transform[1])   # 3, STFT(...)
```

The flags of a composition combine those of its parts. A composition is invertible (or scriptable) only if all its
transforms are, and it needs scaling if any of them does:

```python
print(transform.invertible, transform.scriptable, transform.needs_scaling)   # True True True
```

## Fitting and inverting a chain

`scale_data` on a composition fits each transform on the output of the previous ones, so a single call on the raw
audio is enough:

```python
transform.scale_data(x)
y = transform(x)                             # normalized magnitude (173, 513), in [0, 1]
x_rec = transform.invert(y)                  # Griffin-Lim reconstruction of the mono mix
```

The `inversion_mode` argument of `invert` is passed to every transform of the chain. Transforms that do not use it
ignore it:

```python
x_random = transform.invert(y, inversion_mode="random")
```

## An invertible spectrogram

This chain is a common front-end for generative models. It gives a log-scaled mel spectrogram in `[0, 1]` that can be
turned back into audio:

```python
import torchaudio

x, sr = torchaudio.load("test/source_files/additive.wav")

transform = Mono() \
    + DGT(sr=sr, n_fft=1024, hop_length=256, inversion_mode="pghi") \
    + Magnitude(sr=sr, n_fft=1024, mel=True, mode="unipolar", contrast="log1p")

transform.scale_data(x)
spectrogram = transform(x)
audio = transform.invert(spectrogram)        # phase rebuilt with PGHI (see the DGT tutorial)
torchaudio.save("additive_pghi.wav", audio, sample_rate=sr)
```

{class}`~acids_transforms.transforms.spectral_repr.Magnitude` takes `contrast="log1p"` (`log(1 + x)`), `"log"`,
`"log10"` or `None`. Its `mode` sets the normalization: `"unipolar"`, `"bipolar"`, `"gaussian"` or `None`.

## Several representations at once

{class}`~acids_transforms.transforms.spectral_repr.Polar`,
{class}`~acids_transforms.transforms.spectral_repr.PolarIF` and
{class}`~acids_transforms.transforms.spectral_repr.Cartesian` turn a complex spectrum into two real channels, stacked
on the `stack` axis (`-2` by default). These are magnitude and phase, magnitude and instantaneous frequency, or real
and imaginary parts:

```python
from acids_transforms import Polar

transform = STFT(n_fft=1024, hop_length=256) + Polar(magnitude_args={"mode": "bipolar", "mel": False},
                                                     phase_args={"mode": "bipolar"})
x = torch.randn(1, 44100)
transform.scale_data(x)
y = transform(x)
print(y.shape)                               # (1, 173, 2, 513)
```
