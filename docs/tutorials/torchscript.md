# TorchScript and nn~

## Scripting transforms

Transforms whose `scriptable` flag is `True` can be compiled with {func}`torch.jit.script`. Their `forward`,
`invert`, `scale_data` and `forward_with_time` methods are all exported, so the scripted module can be saved and then
loaded from Python or C++ (libtorch) without acids_transforms:

```python
import torch
from acids_transforms import Mono, STFT, Magnitude

transform = Mono() + STFT(n_fft=1024, hop_length=256) + Magnitude(n_fft=1024, mel=False, mode="unipolar")
assert transform.scriptable

x = torch.randn(2, 44100)
transform.scale_data(x)                      # fit the normalization before scripting

scripted = torch.jit.script(transform)
torch.jit.save(scripted, "spectrogram.ts")

loaded = torch.jit.load("spectrogram.ts")
y = loaded(x)
x_rec = loaded.invert(y)
```

Data-dependent parameters, such as normalization statistics, are buffers. They are saved with the scripted module,
so call `scale_data` before saving.

(tracing)=
## Tracing with torch.fx

Transforms whose `tracable` flag is `True` can be traced with `make_fx` (`torch.fx.experimental.proxy_tensor`).
PGHI, the sine bank and `frame` loop and branch on values, which `make_fx` cannot trace. While tracing, they are
replaced according to the backend of {mod}`acids_transforms.utils.tracing`:

| backend | default when | traced as | traced graphs |
| :--- | :--- | :--- | :--- |
| `"custom_op"` | `torch.library.custom_op` exists (torch >= 2.4) | one opaque node running the exact algorithm | need `acids_transforms` imported to run |
| `"pure"` | older torch | plain aten operations | portable, but PGHI and the sine bank are approximations |

The pure PGHI integrates phases over frames with cumulative sums and locks them around spectral peaks, instead of
following a heap. The pure sine bank lets each bin follow the frequency of its nearest peak, and synthesizes the
sinusoids in the frequency domain. Outside tracing (eager mode) and with TorchScript, the exact implementations are
always used, so scripted models exported for nn~ do not depend on custom operators.

```python
from torch.fx.experimental.proxy_tensor import make_fx
from acids_transforms import DGT
from acids_transforms.utils import tracing

dgt = DGT(n_fft=1024, hop_length=256)
y = dgt(torch.randn(1, 44100)).abs()

tracing.set_backend("pure")                  # portable graph, approximate PGHI
graph = make_fx(lambda m: dgt.invert(m, inversion_mode="pghi"), tracing_mode="symbolic")(y)
```

Buffers of stateful transforms (phases of real-time PGHI and of the sine bank) are not updated while tracing: traced
graphs of real-time transforms start from the state they had when traced.

## Real-time models for Max/MSP and PureData

[nn~](https://github.com/acids-ircam/nn_tilde) runs TorchScript models on audio buffers in Max/MSP and PureData.
Streaming transforms ({doc}`realtime`) can be wrapped in an `nn_tilde.Module`. This minimal model transforms incoming
audio with a {class}`~acids_transforms.transforms.stft.StreamingSTFT`, then inverts it:

```python
import torch
import nn_tilde
from acids_transforms import StreamingSTFT


class SpectralLoop(nn_tilde.Module):

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.stft = StreamingSTFT(n_fft=n_fft, hop_length=hop_length)
        self.register_method("forward", in_channels=1, in_ratio=1, out_channels=1, out_ratio=1,
                             input_labels=["(signal) input"], output_labels=["(signal) output"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = self.stft(x)              # (batch, 1, n_frames, n_fft // 2 + 1) : process it here
        return self.stft.invert(spectrum).float()


SpectralLoop().export_to_ts("spectral_loop.ts")
```

Load it in Max/MSP or PureData with `[nn~ spectral_loop forward]`. The nn~ buffer size must be a multiple of
`hop_length`.

`test/test_nn_tilde.py` exports more complete examples into `test/nn_tilde/`: STFTs with several window sizes,
`RealtimeDGT` between two `OverlapAdd`, `MuLaw` and `MidSide`. In these models, an `inversion_mode` attribute selects
the inversion at run time. To export them without running the tests:

```bash
python test/test_nn_tilde.py [output_dir]
```
