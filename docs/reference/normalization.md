# Normalization

```{eval-rst}
.. module:: acids_transforms.transforms.norm
```

{class}`Normalize` scales data with statistics fitted by `scale_data`:

- `"unipolar"`: to `[0, 1]`, with the minimum and maximum;
- `"bipolar"`: to `[-1, 1]`, with the minimum and maximum;
- `"gaussian"`: to zero mean and unit variance.

Statistics are stored in buffers, so they are saved with the model and scripted modules.

```{eval-rst}
.. autoclass:: Normalize
```
