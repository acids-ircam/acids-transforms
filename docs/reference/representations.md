# Spectral representations

```{eval-rst}
.. module:: acids_transforms.transforms.spectral_repr
```

Transforms that turn complex spectra (from {class}`~acids_transforms.transforms.stft.STFT` or
{class}`~acids_transforms.transforms.dgt.DGT`) into real representations, for example to feed them to a neural network.
Their `mode` argument normalizes the output with {class}`~acids_transforms.transforms.norm.Normalize`
(`"unipolar"`, `"bipolar"`, `"gaussian"` or `None`): call `scale_data` first. With `keep_nyquist=False`, the first
bin is removed so that the number of bins is a power of two.

## Single representations

{class}`Real` and {class}`Imaginary`
: real and imaginary parts.

{class}`Magnitude`
: magnitude, optionally on a mel scale (`mel=True`, which needs `n_fft` and `sr`), and compressed with `contrast`:
  `"log1p"` (`log(1 + x)`, default), `"log"`, `"log10"` or `None`. Its inversion gives back magnitudes, so a spectral
  transform with a phase reconstruction (`inversion_mode`) must follow it in a chain.

{class}`Phase`
: phase, optionally unwrapped over frames (`unwrap=True`).

{class}`IF`
: instantaneous frequency: derivative of the unwrapped phase over frames. `method` is `"forward"`, `"backward"` or
  `"central"`.

```{eval-rst}
.. autoclass:: Real
.. autoclass:: Imaginary
.. autoclass:: Magnitude
.. autoclass:: Phase
.. autoclass:: IF
```

## Stacked representations

Two representations stacked on the `stack` axis (`-2` by default). Their arguments are passed with `*_args`
dictionaries.

{class}`Cartesian`
: {class}`Real` and {class}`Imaginary` (`real_args`, `imag_args`).

{class}`Polar`
: {class}`Magnitude` and {class}`Phase` (`magnitude_args`, `phase_args`).

{class}`PolarIF`
: {class}`Magnitude` and {class}`IF` (`magnitude_args`, `phase_args`).

```{eval-rst}
.. autoclass:: SpectralRepresentation
.. autoclass:: Cartesian
.. autoclass:: Polar
.. autoclass:: PolarIF
```
