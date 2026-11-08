# acids_transforms

**acids_transforms** is a set of audio transforms for [PyTorch](https://pytorch.org), built on
[torchaudio](https://pytorch.org/audio). Most transforms can be inverted, and most can be compiled with TorchScript,
so they can run inside Python/C++ models in real time (for example in
[nn~](https://github.com/acids-ircam/nn_tilde) for Max/MSP and PureData).

```python
import torch, torchaudio
from acids_transforms import Mono, DGT, Magnitude

x, sr = torchaudio.load("sound.wav")
transform = Mono() + DGT(sr=sr, n_fft=1024, hop_length=256) + Magnitude(sr=sr, n_fft=1024)
transform = torch.jit.script(transform)
transform.scale_data(x)
spectrogram = transform(x)
x_reconstructed = transform.invert(spectrogram)   # phase is rebuilt with PGHI
```

## Available transforms

| transform | invertible | scriptable |
| :--- | :---: | :---: |
| **raw audio** | | |
| {class}`~acids_transforms.transforms.raw.Mono`, {class}`~acids_transforms.transforms.raw.Stereo`, {class}`~acids_transforms.transforms.raw.MidSide` | yes | yes |
| {class}`~acids_transforms.transforms.raw.Window` | yes | yes |
| {class}`~acids_transforms.transforms.raw.MuLaw` | yes | yes |
| **spectral** | | |
| {class}`~acids_transforms.transforms.stft.STFT`, {class}`~acids_transforms.transforms.stft.RealtimeSTFT`, {class}`~acids_transforms.transforms.stft.StreamingSTFT` | yes | yes |
| {class}`~acids_transforms.transforms.dgt.DGT`, {class}`~acids_transforms.transforms.dgt.RealtimeDGT` | yes | yes |
| {class}`~acids_transforms.transforms.mel.MFCC` | no | yes |
| **spectral representations** | | |
| {class}`~acids_transforms.transforms.spectral_repr.Real`, {class}`~acids_transforms.transforms.spectral_repr.Imaginary` | yes | yes |
| {class}`~acids_transforms.transforms.spectral_repr.Magnitude`, {class}`~acids_transforms.transforms.spectral_repr.Phase`, {class}`~acids_transforms.transforms.spectral_repr.IF` | yes | yes |
| {class}`~acids_transforms.transforms.spectral_repr.Cartesian`, {class}`~acids_transforms.transforms.spectral_repr.Polar`, {class}`~acids_transforms.transforms.spectral_repr.PolarIF` | yes | yes |
| **normalization** | | |
| {class}`~acids_transforms.transforms.norm.Normalize` | yes | yes |
| **miscellaneous** | | |
| {class}`~acids_transforms.transforms.oadd.OverlapAdd` | yes | yes |
| {class}`~acids_transforms.transforms.misc.Unsqueeze`, {class}`~acids_transforms.transforms.misc.Transpose`, {class}`~acids_transforms.transforms.misc.OneHot` | yes | yes |
| {class}`~acids_transforms.transforms.misc.Squeeze` | no | yes |

:::{warning}
This library is still experimental, and its API changes often.
:::

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

reference/index
```

```{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/domkirke/acids_transforms>
```
