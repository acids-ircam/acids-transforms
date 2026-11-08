# The STFT and its inversions

{class}`~acids_transforms.transforms.stft.STFT` turns audio `(..., T)` into spectra
`(..., n_frames, n_fft // 2 + 1)`. With its default parameters, it gives the same result as {func}`torch.stft`.

## Framing

Frames of `win_length` samples (`n_fft` by default) are windowed, then zero-padded at their end to `n_fft` samples.
A new frame starts every `hop_length` samples. The window can be `"hann"` (default), `"hamming"`, `"blackman"`,
`"bartlett"`, `"rectangular"` or `None`.

With `center=True` (default), the signal is padded with `win_length // 2` samples on both sides, so that frame `t` is
centered on sample `t * hop_length`. `pad_mode` sets how: `"reflect"` (default), `"constant"` (zeros), `"replicate"`
or `"circular"`.

```python
import torch
from acids_transforms import STFT

x = torch.randn(2, 44100)
stft = STFT(n_fft=1024, hop_length=256)
reference = torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True).transpose(-1, -2)
assert torch.allclose(stft(x), reference, atol=1e-4)
```

## Output modes

`mode` sets what the transform outputs. Give one mode, or a list of modes to stack them on a new axis
`(..., n_modes, n_frames, n_bins)`:

| mode | output |
| :--- | :--- |
| `"complex"` (default) | complex spectrum |
| `"magnitude"`, `"power"`, `"db"` | magnitude, power, and `10 log10` of the power |
| `"phase"` | phase, in $[-\pi, \pi]$ |
| `"real"`, `"imaginary"` | real and imaginary parts |
| `"power_density"` | power spectral density (V²/Hz), as `scipy.signal.spectrogram(scaling="density", mode="psd")` |
| `"instantaneous_phase"` | phase difference between consecutive frames, wrapped in $[-\pi, \pi]$ |
| `"instantaneous_frequency"` | frequency in Hz of each bin, from the phase difference between frames (phase vocoder) |

If `"complex"` is one of the modes, the output is complex and the other modes are cast to complex numbers. Otherwise
the output is real.

```python
stft = STFT(n_fft=2048, win_length=1024, hop_length=256, center=False, mode=["magnitude", "phase"])
y = stft(x)
print(y.shape, y.dtype)                      # (2, 2, 169, 1025) torch.float32
magnitude, phase = y[..., 0, :, :], y[..., 1, :, :]
```

The list of all modes is in `acids_transforms.SPECTRUM_MODES`.

## Inverting

`invert` turns spectra back into audio. Whether that is exact depends on the modes:

- **Exact**, from `"complex"`, from (`"real"`, `"imaginary"`), or from a magnitude (`"magnitude"`, `"power"`,
  `"db"`, `"power_density"`) together with `"phase"`.
- **Approximate**, from a magnitude together with `"instantaneous_phase"` or `"instantaneous_frequency"`. Phases are
  integrated over frames.
- **Estimated**, from a magnitude alone. The phase is rebuilt with `inversion_mode`:
  - `"griffin_lim"` (default): fast Griffin-Lim, with the framing of the transform;
  - `"random"`: random phases;
  - `"keep_input"`: the phases of the last call to `forward`;
  - `"sinebank"`: sinusoidal resynthesis. Spectral peaks are picked in each frame, their frequencies and amplitudes
    are refined by quadratic interpolation, and their phases are continued from frame to frame (see
    {class}`~acids_transforms.transforms.sinebank.SineBank`). It suits tonal sounds best.

```python
y = stft(x)
x_rec = stft.invert(y)                       # exact, except on the borders (center=False)

magnitude = STFT(n_fft=1024, hop_length=256, mode="magnitude")
y = magnitude(x)
x_gl = magnitude.invert(y)                                   # Griffin-Lim
x_sines = magnitude.invert(y, inversion_mode="sinebank")     # sine bank
x_kept = magnitude.invert(y, inversion_mode="keep_input")    # phases of x : exact
```

:::{tip}
`pytest test/test_stft_audio.py` writes audio examples of every inversion path on the sounds of `test/source_files`
into `test/reconstructions/stft/`, with a `report.md` that lists their SNR or spectral convergence. This is the quickest
way to hear how the modes compare.
:::

## Time stamps

`forward_with_time(x, time)` also returns the time (in seconds) of each frame, starting from `time`. This is useful to
align spectral frames with other annotations:

```python
stft = STFT(sr=44100, n_fft=1024, hop_length=256)
y, t = stft.forward_with_time(x, torch.zeros(2))
print(t.shape, t[0, :3])                     # (2, 173) : 0, 256 / 44100, 512 / 44100...
```
