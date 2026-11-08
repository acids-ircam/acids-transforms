# Miscellaneous

## Overlap-add

```{eval-rst}
.. module:: acids_transforms.transforms.oadd
```

{class}`OverlapAdd` frames a stream of audio chunks `(..., T)` into frames `(..., T // hop_length, n_fft)`, keeping
the last `n_fft - hop_length` samples between calls. Its `invert` overlap-adds synthesis frames back into chunks of
audio, compensating the gain of the overlap. Use it around frame-wise transforms such as
{class}`~acids_transforms.transforms.stft.RealtimeSTFT` or {class}`~acids_transforms.transforms.dgt.RealtimeDGT` (see
{doc}`/tutorials/realtime`).

```{eval-rst}
.. autoclass:: OverlapAdd
```

## Shape transforms

```{eval-rst}
.. module:: acids_transforms.transforms.misc
```

{class}`Unsqueeze` and {class}`Squeeze`
: add or remove the axis `dim`. `Squeeze` with `dim=None` (default) removes every axis of size 1: it then cannot be
  inverted, and raises {class}`~acids_transforms.transforms.base.NotInvertibleError`.

{class}`Transpose`
: swaps the two axes in `dims` (`(-2, -1)` by default). It is its own inverse.

{class}`OneHot`
: one-hot encodes integer classes. `scale_data` sets the number of classes from the data.

```{eval-rst}
.. autoclass:: Unsqueeze
.. autoclass:: Squeeze
.. autoclass:: Transpose
.. autoclass:: OneHot
```
