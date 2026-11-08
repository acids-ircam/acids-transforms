# Base classes

```{eval-rst}
.. module:: acids_transforms.transforms.base
```

Every transform derives from {class}`AudioTransform`. Adding two transforms with `+` gives a
{class}`ComposeAudioTransform`.

Common interface:

`forward(x)`
: transforms `x`.

`invert(x, inversion_mode=None)`
: inverts the transform, if `invertible` is `True`. `inversion_mode` selects the inversion method of transforms that
  have several (see `get_inversion_modes()`).

`scale_data(x)`
: fits the data-dependent parameters of the transform (normalization statistics, for example). Call it before
  `forward` when `needs_scaling` is `True`.

`forward_with_time(x, time)`
: transforms `x`, and returns the time stamps of the outputs, starting from `time`.

`realtime()`
: returns the frame-wise, real-time version of the transform (itself by default).

`invertible`, `scriptable`, `tracable`, `needs_scaling`
: whether the transform can be inverted, scripted with {func}`torch.jit.script`, traced with `torch.fx`, and whether
  it needs `scale_data`.

`ratio`
: number of input samples for each output step (`hop_length` for spectral transforms, 1 otherwise).

```{eval-rst}
.. autoclass:: AudioTransform

.. autoclass:: ComposeAudioTransform

.. autoexception:: NotInvertibleError
```

## Tracing backends

See {ref}`tracing`.

```{eval-rst}
.. automodule:: acids_transforms.utils.tracing
   :members: get_backend, set_backend, BACKENDS
```
