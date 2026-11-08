# Phase reconstruction with the DGT

{class}`~acids_transforms.transforms.dgt.DGT` is a discrete Gabor transform: an STFT with a Gaussian window and its
canonical dual window for synthesis. A Gaussian window links the phase of a spectrogram to its magnitude, so the phase
can be estimated well from the magnitude alone. This is Phase Gradient Heuristic Integration (PGHI, Průša et al.,
2017), the default `inversion_mode` of the DGT.

```python
import torch, torchaudio
from acids_transforms import DGT

x, sr = torchaudio.load("test/source_files/additive.wav")
dgt = DGT(sr=sr, n_fft=1024, hop_length=256)
y = dgt(x)                                   # complex (channels, n_frames, 513)

x_exact = dgt.invert(y)                      # complex input : exact inversion
x_pghi = dgt.invert(y.abs())                 # magnitude only : phase estimated with PGHI
torchaudio.save("additive_pghi.wav", x_pghi, sample_rate=sr)
```

`DGT.get_inversion_modes()` lists the methods for magnitudes: `"pghi"`, `"griffin_lim"`, `"random"`,
`"keep_input"` and `"sinebank"`.

```python
x_gl = dgt.invert(y.abs(), inversion_mode="griffin_lim")
```

PGHI only reconstructs bins whose magnitude is above `tolerance` times the maximum (`1e-2` by default). The other bins
get random phases.

:::{note}
The window of a DGT is fixed by `n_fft`, so the DGT does not take the `window`, `win_length`, `center` or `mode`
arguments of {class}`~acids_transforms.transforms.stft.STFT`. Its output is always complex: use
{class}`~acids_transforms.transforms.spectral_repr.Magnitude` (or `.abs()`) after it to get magnitudes.
:::

:::{note}
PGHI and sine bank inversions depend on values, which `make_fx` cannot trace. See {ref}`tracing` for how they are
traced. They can always be scripted with {func}`torch.jit.script`.
:::
