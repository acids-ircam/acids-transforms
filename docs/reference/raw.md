# Raw audio

```{eval-rst}
.. module:: acids_transforms.transforms.raw
```

Transforms on waveforms `(..., channels, T)`.

## Channels

{class}`Mono` turns stereo signals into mono: it averages both channels (`mode="mix"`, default), or keeps the
`"left"` or `"right"` one. With `squeeze=True` (default), it removes the channel axis. Its inversion gives a single
channel back, or duplicates it into two channels if the transform was built with `inversion_mode="stereo"`.

{class}`Stereo` duplicates mono signals into two channels. It raises an error on signals with more than two channels.

{class}`MidSide` turns left/right channels into mid (`(L + R) / 2`, divided by $\sqrt{2}$ if `pad_mid=True`) and
side (`(L - R) / 2`) channels, and back. Mono signals get a silent side channel.

```{eval-rst}
.. autoclass:: Mono
.. autoclass:: Stereo
.. autoclass:: MidSide
```

## Framing

{class}`Window` slices signals into frames of `window_size` samples every `hop_size` samples
(`(..., n_frames, window_size)`). Its inversion (`inversion_mode="crop"`) concatenates the first `hop_size`
samples of each frame, and the end of the last frame.

```{eval-rst}
.. autoclass:: Window
```

## Quantization

{class}`MuLaw` quantizes audio in `[-1, 1]` into `channels` integer classes with μ-law companding. With
`one_hot="channel"` or `"categorical"`, classes are one-hot encoded on the channel axis or on a new last axis. The
inversion is lossy.

```{eval-rst}
.. autoclass:: MuLaw
```
