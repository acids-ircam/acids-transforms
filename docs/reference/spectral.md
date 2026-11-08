# Spectral transforms

Short-time Fourier transforms of audio `(..., T)` into spectra `(..., n_frames, n_fft // 2 + 1)`. See
{doc}`/tutorials/stft`, {doc}`/tutorials/dgt` and {doc}`/tutorials/realtime`.

## STFT

```{eval-rst}
.. module:: acids_transforms.transforms.stft

.. autoclass:: STFT
   :exclude-members: tracable, scriptable, invertible, needs_scaling

.. autoclass:: RealtimeSTFT

.. autoclass:: StreamingSTFT

.. autodata:: SPECTRUM_MODES
   :no-value:

   Output modes of :class:`STFT`: ``"complex"``, ``"magnitude"``, ``"power"``, ``"db"``, ``"phase"``,
   ``"real"``, ``"imaginary"``, ``"power_density"``, ``"instantaneous_phase"``, ``"instantaneous_frequency"``.

.. autodata:: WINDOWS
   :no-value:

   Analysis windows of :class:`STFT`: ``"hann"``, ``"hamming"``, ``"blackman"``, ``"bartlett"``, ``"rectangular"``.
```

## Discrete Gabor transform

{class}`~acids_transforms.transforms.dgt.DGT` is an STFT with a Gaussian window, whose magnitudes can be inverted
with Phase Gradient Heuristic Integration (`inversion_mode="pghi"`, default).
{class}`~acids_transforms.transforms.dgt.RealtimeDGT` is its frame-wise version, to use between two
{class}`~acids_transforms.transforms.oadd.OverlapAdd`.

```{eval-rst}
.. module:: acids_transforms.transforms.dgt

.. autoclass:: DGT

.. autoclass:: RealtimeDGT
```

## Sine bank

Sinusoidal resynthesis used by the `"sinebank"` inversion mode of {class}`~acids_transforms.transforms.stft.STFT`,
{class}`~acids_transforms.transforms.stft.RealtimeSTFT`, {class}`~acids_transforms.transforms.dgt.DGT` and
{class}`~acids_transforms.transforms.dgt.RealtimeDGT`. It turns magnitudes into complex spectra, that these transforms
then invert exactly.

```{eval-rst}
.. module:: acids_transforms.transforms.sinebank

.. autoclass:: SineBank
```

## MFCC

{class}`~acids_transforms.transforms.mel.MFCC` computes mel spectrograms with
{class}`torchaudio.transforms.MelSpectrogram` (`(..., n_mels, n_frames)`), optionally normalized with
{class}`~acids_transforms.transforms.norm.Normalize` (`norm_mode`). It is not invertible.

```{eval-rst}
.. module:: acids_transforms.transforms.mel

.. autoclass:: MFCC
```
