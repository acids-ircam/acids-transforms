"""Device tests : every transform is moved to cpu / mps / cuda (skipped when unavailable) and checked to
    - keep its outputs (and inversions) on the device,
    - give the same outputs as on cpu (forward, and inversion when it is deterministic),
    - run every inversion mode on the device,
    - run on the device once scripted with TorchScript.

    pytest test/test_devices.py                 # all available devices
    pytest test/test_devices.py -k cuda         # only cuda
"""
import functools
import re
import pytest
import torch
from acids_transforms import transforms, AudioTransform
from acids_transforms.transforms.spectral_repr import _Representation, SpectralRepresentation
from acids_transforms.utils.misc import frame


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)  # torch >= 1.12
    return mps is not None and mps.is_available()


DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("mps", id="mps", marks=pytest.mark.skipif(not _mps_available(), reason="mps not available")),
    pytest.param("cuda", id="cuda",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")),
]

# tolerances w.r.t. cpu outputs (mps / cuda use different fft / reduction kernels)
RTOL, ATOL = 1e-3, 1e-4
# phases of (near-)silent bins are arbitrary, and may wrap differently from one device to another : forward outputs
# may differ from cpu on at most this fraction of elements
MAX_MISMATCH = 1e-2
SR = 44100


def skip_unsupported_ops(test):
    """skips (instead of failing) when torch has no kernel for an op on the tested device, e.g. older torch versions
    lack many mps kernels (aten::unfold before 1.13, fft before 2.1, angle before 2.7...)."""
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        try:
            return test(*args, **kwargs)
        except NotImplementedError as e:
            match = re.search(r"operator '([\w:.]+)' is not current(?:ly)? implemented for the (\w+) device", str(e))
            if match is None:
                raise
            pytest.skip(f"torch {torch.__version__} has no {match.group(2)} kernel for {match.group(1)}")
    return wrapper


def get_audio_transforms():
    names = [x for x in dir(transforms) if isinstance(getattr(transforms, x), type)
             and x not in ['AudioTransform', 'ComposeAudioTransform']]
    return [getattr(transforms, x) for x in names if issubclass(getattr(transforms, x), AudioTransform)]


def build(transform_type, x=None):
    """transform to test : spectral representations take STFT spectra, OneHot takes mu-law codes. Transforms that
    need scaling are scaled on x (on cpu, before being moved to the device)."""
    if issubclass(transform_type, (_Representation, SpectralRepresentation)):
        transform = transforms.STFT() + transform_type()
    elif transform_type is transforms.OneHot:
        transform = transforms.MuLaw(channels=256) + transforms.OneHot(n_classes=256)
    else:
        transform = transform_type()
    if x is not None and transform.needs_scaling:
        transform.scale_data(x)
    return transform


def make_input(transform_type, batch=(2, 2), n_samples=16384):
    """test audio (batch, n_samples) ; realtime transforms take a single frame."""
    torch.manual_seed(0)
    t = torch.arange(n_samples) / SR
    x = (0.5 * torch.sin(2 * torch.pi * 440 * t) + 0.1 * torch.randn(n_samples)).expand(*batch, n_samples).clone()
    if transform_type is transforms.RealtimeSTFT:
        t = transform_type()
        x = frame(x, t.win_length, t.hop_length, -1)[..., :1, :].contiguous()  # (..., 1, win_length)
    elif transform_type is transforms.RealtimeDGT:
        t = transform_type()
        x = frame(x, t.n_fft, t.hop_length, -1)[..., 0, :].contiguous()  # (..., n_fft)
    return x


def get_inversion_modes(transform):
    try:
        modes = transform.get_inversion_modes()
    except TypeError:
        return []
    return list(modes) if modes is not None else []


def invert_seeded(transform, y, seed: int, *args):
    torch.manual_seed(seed)
    return transform.invert(y, *args)


def is_deterministic(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return all(is_deterministic(a_i, b_i) for a_i, b_i in zip(a, b))
    return torch.equal(a, b)


def assert_on_device(y, device: str):
    if isinstance(y, (tuple, list)):
        for y_i in y:
            assert_on_device(y_i, device)
    elif isinstance(y, torch.Tensor):
        assert y.device.type == device, f"output on {y.device}, expected {device}"


def assert_close(y, y_ref, what: str, max_mismatch: float = 0.):
    if isinstance(y, (tuple, list)):
        for y_i, y_ref_i in zip(y, y_ref):
            assert_close(y_i, y_ref_i, what, max_mismatch)
        return
    y = y.cpu()
    assert y.shape == y_ref.shape, f"{what}: shape {tuple(y.shape)} != cpu shape {tuple(y_ref.shape)}"
    assert y.dtype == y_ref.dtype, f"{what}: dtype {y.dtype} != cpu dtype {y_ref.dtype}"
    if y.dtype.is_floating_point or y.is_complex():
        mismatch = ~torch.isclose(y, y_ref, rtol=RTOL, atol=ATOL)
        ratio = mismatch.float().mean().item()
        err = (y - y_ref).abs().max().item()
        assert ratio <= max_mismatch, \
            f"{what}: {100 * ratio:.2f}% of elements differ from cpu (max abs error {err:.3e})"
    else:
        assert torch.equal(y, y_ref), f"{what}: differs from cpu"


def assert_finite(y, what: str):
    if isinstance(y, (tuple, list)):
        for y_i in y:
            assert_finite(y_i, what)
    elif isinstance(y, torch.Tensor) and (y.dtype.is_floating_point or y.is_complex()):
        assert torch.isfinite(y).all(), f"{what}: non-finite values"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform_type", get_audio_transforms())
@skip_unsupported_ops
def test_device_forward(transform_type, device):
    x = make_input(transform_type)
    transform_cpu = build(transform_type, x)
    y_ref = transform_cpu(x)
    transform = build(transform_type, x).to(device)
    y = transform(x.to(device))
    assert_on_device(y, device)
    assert_close(y, y_ref, "forward", MAX_MISMATCH)
    # time-stamped forward (time : (batch,) onsets in seconds)
    y_t = build(transform_type, x).to(device).forward_with_time(x.to(device), torch.zeros(x.shape[:-1], device=device))
    assert_on_device(y_t, device)
    if transform.invertible:
        x_inv = invert_seeded(transform, y, 0)
        assert_on_device(x_inv, device)
        assert_finite(x_inv, "invert")
        # compare with cpu only if the inversion is deterministic (no random phase...), from the cpu forward
        x_inv_ref = invert_seeded(transform_cpu, y_ref, 0)
        if is_deterministic(x_inv_ref, invert_seeded(transform_cpu, y_ref, 1)):
            x_inv = invert_seeded(transform, y_ref.to(device) if torch.is_tensor(y_ref) else y_ref, 0)
            assert_close(x_inv, x_inv_ref, "invert")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform_type", get_audio_transforms())
@skip_unsupported_ops
def test_device_inversion_modes(transform_type, device):
    """every inversion mode runs on the device (no cpu comparison : some modes are random, e.g. random phase).
    Uses a short input, as phase reconstruction (e.g. PGHI) loops over bins and is very slow on gpus."""
    x = make_input(transform_type, batch=(1,), n_samples=4096)
    transform = build(transform_type, x).to(device)
    if not transform.invertible:
        pytest.skip("not invertible")
    modes = [m for m in get_inversion_modes(transform) if m != "direct"]
    if len(modes) == 0:
        pytest.skip("no inversion modes")
    y = transform(x.to(device))
    for mode in modes:
        x_inv = transform.invert(y.abs() if torch.is_complex(y) else y, mode)
        assert_on_device(x_inv, device)
        assert_finite(x_inv, f"inversion mode {mode}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("transform_type", [t for t in get_audio_transforms() if build(t).scriptable])
@skip_unsupported_ops
def test_device_scripted(transform_type, device):
    x = make_input(transform_type)
    transform = build(transform_type, x)
    y_ref = transform(x)
    scripted = torch.jit.script(build(transform_type, x)).to(device)
    y = scripted(x.to(device))
    assert_on_device(y, device)
    assert_close(y, y_ref, "scripted forward", MAX_MISMATCH)
    if transform.invertible:
        x_inv = scripted.invert(y)
        assert_on_device(x_inv, device)
        assert_finite(x_inv, "scripted invert")
