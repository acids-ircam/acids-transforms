import torch
from pathlib import Path
import torchaudio
import os
import pytest
from acids_transforms import transforms, AudioTransform
from acids_transforms.utils import tracing


def get_audio_transforms():
    audio_transforms = list(filter(lambda x: (type(getattr(transforms, x)) == type) and (
        not x in ['AudioTransform', 'ComposeAudioTransform']), dir(transforms)))
    audio_transforms = [getattr(transforms, x) for x in audio_transforms]
    audio_transforms = list(
        filter(lambda x: issubclass(x, AudioTransform), audio_transforms))
    return audio_transforms


def get_scriptable_transforms():
    audio_transforms = get_audio_transforms()
    audio_transforms = list(filter(lambda x: x().scriptable, audio_transforms))
    return audio_transforms


def get_invertible_transforms():
    audio_transforms = get_audio_transforms()
    audio_transforms = list(filter(lambda x: x().invertible, audio_transforms))
    return audio_transforms

def get_tracable_transforms(): 
    audio_transforms = get_audio_transforms()
    audio_transforms = list(filter(lambda x: x().tracable, audio_transforms))
    return audio_transforms


@pytest.mark.parametrize("transform", get_audio_transforms())
def test_forward(test_files, transform: AudioTransform):
    transform = transform()
    raw, name = test_files
    time = torch.zeros(raw.shape[:-1])
    y = transform.test_forward(raw)
    y, time = transform.test_forward(raw, time)

@pytest.mark.parametrize("transform", get_audio_transforms())
def test_realtime(test_files, transform: AudioTransform):
    transform = transform()
    raw, name = test_files
    time = torch.zeros(raw.shape[:-1])
    realtime_transform = transform.realtime()
    realtime_transform.test_forward(raw, time)
    
@pytest.mark.parametrize("transform", get_invertible_transforms())
def test_inversion(test_files, transform: AudioTransform):
    if not os.path.isdir("test/reconstructions"):
        os.makedirs("test/reconstructions")
    raw, names = test_files
    transform_name = transform.__name__.split(".")[-1]
    transform = transform()
    x_inv = transform.test_inversion(raw)
    for k, v in x_inv.items():
        for i in range(raw.shape[0]):
            x_tmp = v[i]
            if x_tmp.ndim == 1:
                x_tmp = x_tmp.unsqueeze(0)
            torchaudio.save(
                f"test/reconstructions/{names[i]}_{transform_name}_{k}.wav", x_tmp, sample_rate=44100)
    return True



@pytest.mark.parametrize("transform_type", get_scriptable_transforms())
def test_scriptable(transform_type):
    transform = transform_type()
    scripted_transform = torch.jit.script(transform)
    transform_type.test_scripted_transform(
        scripted_transform, invert=transform.invertible)
    return True

AUDIO_OUT_DIR = Path("test/tracing")


@pytest.fixture(params=tracing.BACKENDS)
def tracing_backend(request):
    """traces data-dependent algorithms (PGHI, sine bank, frame) with each backend (see utils.tracing)."""
    if request.param == "custom_op" and not tracing.HAS_CUSTOM_OPS:
        pytest.skip("custom operators need torch >= 2.4")
    previous = tracing.get_backend()
    tracing.set_backend(request.param)
    yield request.param
    tracing.set_backend(previous)


@pytest.mark.parametrize("transform_type", get_tracable_transforms())
def test_tracing(test_files, transform_type, tracing_backend):
    out_dir = AUDIO_OUT_DIR / tracing_backend
    os.makedirs(str(out_dir), exist_ok=True)
    transform = transform_type()
    fns, outs = transform_type.test_tracable_transform(transform, test_files)
    for name, fn in fns.items():
        opaque = [n for n in fn.graph.nodes if "acids_transforms" in str(n.target)]
        if tracing_backend == "pure":
            assert not opaque, "pure backend : %s has opaque nodes %s" % (name, opaque)
    for k, v in outs.items():
        for i in range(test_files[0].shape[0]):
            torchaudio.save(str(out_dir / f"{k}_{test_files[1][i]}.wav"), v[i], sample_rate = transform.sr)
    return True



combinations = {
    "stft+magnitude": transforms.STFT() + transforms.Magnitude(),
    # "scaled+dgt": 10.0 * transforms.DGT(),
    "stereo+mulaw+onehot": transforms.Stereo() + transforms.MuLaw(channels=256) + transforms.OneHot(n_classes=256),
    "stft+polar": transforms.STFT() + transforms.Polar(),
    "overlap+stft": transforms.OverlapAdd() + transforms.RealtimeSTFT()
}


@pytest.mark.parametrize("transform", combinations)
def test_combinations(test_files, transform):
    if not os.path.isdir("test/reconstructions"):
        os.makedirs("test/reconstructions")
    raw, names = test_files
    transform_name = transform
    transform = combinations[transform]
    transform.realtime()
    if transform.needs_scaling:
        transform.scale_data(raw)
    # if transform.scriptable:
    #     transform = torch.jit.script(transform)
    time = torch.zeros(*raw.shape[:-1])
    x_t, time = transform.forward_with_time(raw, time)
    if transform.invertible:
        x_inv = transform.invert(x_t)
    for i in range(x_inv.shape[0]):
        x_tmp = x_inv[i]
        if x_tmp.ndim == 1:
            x_tmp = x_tmp.unsqueeze(0)
        torchaudio.save(
            f"test/reconstructions/{names[i]}_{transform_name}.wav", x_tmp, sample_rate=44100)
    return True

