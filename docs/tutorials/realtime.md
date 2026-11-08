# Real-time and streaming

Offline transforms see the whole signal at once. In real time, audio arrives in chunks (buffers of a few hundred
samples), and frames overlap from one chunk to the next. acids_transforms has three building blocks for this case:

- {class}`~acids_transforms.transforms.oadd.OverlapAdd` frames a stream of audio, keeping the samples needed by the
  next frames between calls. Its `invert` overlap-adds synthesis frames back into audio.
- {class}`~acids_transforms.transforms.stft.RealtimeSTFT` and {class}`~acids_transforms.transforms.dgt.RealtimeDGT`
  are frame-wise transforms. They turn frames `(..., n_frames, n_fft)` into spectra and back.
- {class}`~acids_transforms.transforms.stft.StreamingSTFT` combines `OverlapAdd` and `RealtimeSTFT` into a single
  transform, from chunks of audio to chunks of frames.

## StreamingSTFT

`forward` takes chunks of audio `(..., T)`, where `T` is a multiple of `hop_length`, and returns the `T // hop_length`
frames that these samples complete. `invert` gives back chunks of audio, delayed by
`latency = win_length - hop_length` samples. `STFT(...).streaming()` builds the `StreamingSTFT` of an STFT, with the
same parameters:

```python
import torch
from acids_transforms import STFT

stream = STFT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"]).streaming()
stream = torch.jit.script(stream)

x = torch.randn(1, 44032)
out = []
for chunk in x.split(512, -1):               # chunks of 2 * hop_length samples
    frames = stream(chunk)                   # (1, 2, 2, 513) : (channels, modes, frames, bins)
    out.append(stream.invert(frames))        # 512 samples, delayed by stream.latency
out = torch.cat(out, -1)

latency = stream.latency                     # 768 samples
assert torch.allclose(out[..., latency:], x[..., :-latency], atol=1e-4)
```

The frames are the frames of `STFT(center=False)` on the whole stream, preceded by
`win_length // hop_length - 1` frames that hold the initial zeros of the buffer.

Inversion is exact from complex outputs, or from magnitudes and phases. From magnitudes alone, `inversion_mode` can be
`"random"` (default), `"keep_input"` or `"sinebank"`. Griffin-Lim needs the whole signal, so it is not available when
streaming.

`StreamingSTFT` keeps state between calls. Call `reset()` before streaming a new signal:

```python
stream.reset()
```

## Frame-wise transforms with OverlapAdd

To stream a frame-wise transform such as {class}`~acids_transforms.transforms.dgt.RealtimeDGT`, put it between two
`OverlapAdd`: one frames the input, the other overlap-adds the output.

```python
from acids_transforms import OverlapAdd, RealtimeDGT

n_fft, hop_length = 1024, 256
analysis, synthesis = OverlapAdd(n_fft, hop_length), OverlapAdd(n_fft, hop_length)
dgt = RealtimeDGT(n_fft=n_fft, hop_length=hop_length, batch_size=1)

x = torch.randn(1, 44032)
out = []
for chunk in x.split(512, -1):
    frames = analysis(chunk)                 # (1, 2, 1024) : 2 new frames of n_fft samples
    spectra = dgt(frames)                    # (1, 2, 513), complex
    out.append(synthesis.invert(dgt.invert(spectra)))
out = torch.cat(out, -1)

latency = n_fft - hop_length
assert torch.allclose(out[..., latency:], x[..., :-latency], atol=1e-4)
```

`RealtimeDGT` keeps buffers of the previous frames for PGHI. Their shape is set by `batch_size`, which must match the
batch dimensions of your input.

```python
out = []
for chunk in x.split(512, -1):
    spectra = dgt(analysis(chunk))
    out.append(synthesis.invert(dgt.invert(spectra.abs(), inversion_mode="pghi")))
```

:::{warning}
Real-time PGHI with `RealtimeDGT` has known issues when streaming. A fix is coming.
:::

## From offline to real time

`realtime()` returns the frame-wise version of a transform:

```python
from acids_transforms import DGT

print(type(STFT().realtime()).__name__)      # RealtimeSTFT
print(type(DGT().realtime()).__name__)       # RealtimeDGT
```

On a composition, `realtime()` applies to each of its transforms.
