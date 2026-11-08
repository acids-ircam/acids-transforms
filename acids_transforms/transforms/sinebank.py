import math
import torch
import torch.nn as nn
from typing import Tuple
from ..utils.tracing import opaque_op, is_fx_tracing, select_implementation

__all__ = ["SineBank", "sinebank"]


def _wrap(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


def _peaks(mag: torch.Tensor, window_sum: torch.Tensor, n_fft: int, sr: int, threshold_db: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """magnitudes (B, n_frames, n_bins) -> peak mask, frequencies (Hz) and amplitudes (B, n_frames, n_bins)."""
    log_mag = torch.log(mag.clamp(min=1e-10))
    a, b, c = log_mag[..., :-2], log_mag[..., 1:-1], log_mag[..., 2:]
    threshold = mag.amax(-1, keepdim=True) * (10 ** (threshold_db / 20))
    is_peak = (b > a) & (b >= c) & (mag[..., 1:-1] > threshold) & (mag[..., 1:-1] > 1e-8)
    # vertex of the parabola through (a, b, c)
    curvature = a - 2 * b + c
    offset = torch.where(curvature < 0, 0.5 * (a - c) / curvature.clamp(max=-1e-10), torch.zeros_like(b))
    offset = offset.clamp(-0.5, 0.5)
    bins = torch.arange(1, mag.size(-1) - 1, dtype=mag.dtype, device=mag.device)
    freqs = (bins + offset) * sr / n_fft
    # the rfft of a windowed sinusoid of amplitude A peaks at A * sum(window) / 2
    amps = torch.exp(b - 0.25 * (a - c) * offset) * 2 / window_sum
    pad = [1, 1]
    is_peak = torch.nn.functional.pad(is_peak.to(torch.uint8), pad).to(torch.bool)
    return is_peak, torch.nn.functional.pad(freqs, pad), torch.nn.functional.pad(amps, pad) * is_peak


def _phases(is_peak: torch.Tensor, freqs: torch.Tensor, hop_length: int, sr: int, max_jump: int,
            last_peaks: torch.Tensor, last_freqs: torch.Tensor, last_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """phases (B, n_frames, n_bins) of peaks at the center of frames, continued from frame to frame (and from the
    last frame of the previous call, given by last_*), and peaks, frequencies and phases of the last frame."""
    batch, n_frames, n_bins = is_peak.size(0), is_peak.size(1), is_peak.size(2)
    if list(last_peaks.shape) != [batch, n_bins] or last_peaks.device != is_peak.device:
        last_peaks = torch.zeros(batch, n_bins, dtype=torch.bool, device=is_peak.device)
        last_freqs = torch.zeros(batch, n_bins, dtype=freqs.dtype, device=freqs.device)
        last_phases = torch.zeros(batch, n_bins, dtype=freqs.dtype, device=freqs.device)
    last_freqs, last_phases = last_freqs.to(freqs.dtype), last_phases.to(freqs.dtype)
    index = torch.arange(n_bins, device=is_peak.device).expand(batch, n_bins)
    none = torch.full_like(index, 4 * n_bins)
    phases = []
    for t in range(n_frames):
        peak, freq = is_peak[:, t], freqs[:, t]
        # nearest peak of the previous frame, on each side
        left = torch.where(last_peaks, index, -none).cummax(-1).values
        right = torch.where(last_peaks, index, none).flip(-1).cummin(-1).values.flip(-1)
        dist_left, dist_right = index - left, right - index
        nearest = torch.where(dist_left <= dist_right, left, right).clamp(0, n_bins - 1)
        continued = peak & (torch.minimum(dist_left, dist_right) <= max_jump)
        advanced = last_phases.gather(-1, nearest) \
            + math.pi * (last_freqs.gather(-1, nearest) + freq) * hop_length / sr
        phase = torch.where(continued, advanced, 2 * math.pi * torch.rand_like(freq))
        phase = torch.where(peak, _wrap(phase), torch.zeros_like(phase))
        last_peaks, last_freqs, last_phases = peak, freq, phase
        phases.append(phase)
    if n_frames == 0:
        return torch.zeros_like(freqs), last_peaks.clone(), last_freqs.clone(), last_phases.clone()
    return torch.stack(phases, 1), last_peaks.clone(), last_freqs.clone(), last_phases.clone()


def _synthesize(amps: torch.Tensor, freqs: torch.Tensor, phases: torch.Tensor, win_length: int, sr: int) -> torch.Tensor:
    """sums of sinusoids on frames (B, n_frames, win_length), with phases given at the center of frames."""
    batch, n_frames = amps.size(0), amps.size(1)
    n_sines = int((amps > 0).sum(-1).max()) if amps.numel() > 0 else 0
    if n_sines == 0:
        return torch.zeros(batch, n_frames, win_length, dtype=amps.dtype, device=amps.device)
    # keep the n_sines largest partials of each frame (the others have zero amplitude)
    selected = amps.topk(n_sines, -1).indices
    amps, freqs, phases = amps.gather(-1, selected), freqs.gather(-1, selected), phases.gather(-1, selected)
    time = (torch.arange(win_length, dtype=amps.dtype, device=amps.device) - win_length // 2) / sr
    # synthesize by blocks of frames to bound memory
    block = max(1, (1 << 22) // max(1, batch * n_sines * win_length))
    frames = []
    for start in range(0, n_frames, block):
        end = min(start + block, n_frames)
        arg = 2 * math.pi * freqs[:, start:end, :, None] * time + phases[:, start:end, :, None]
        frames.append((amps[:, start:end, :, None] * torch.cos(arg)).sum(-2))
    return torch.cat(frames, 1)


def _sinebank(mag: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, sr: int, threshold_db: float,
              last_peaks: torch.Tensor, last_freqs: torch.Tensor, last_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """magnitudes (B, n_frames, n_fft // 2 + 1) -> (complex spectra of the sinusoidal resynthesis, peaks, frequencies
    and phases of the last frame)."""
    win_length = window.size(0)
    window = window.to(mag.dtype)
    is_peak, freqs, amps = _peaks(mag, window.sum(), n_fft, sr, threshold_db)
    # partials move by less than the main lobe of the window between frames
    max_jump = max(1, int(round(2. * n_fft / win_length)))
    phases, last_peaks, last_freqs, last_phases = _phases(is_peak, freqs, hop_length, sr, max_jump,
                                                          last_peaks, last_freqs, last_phases)
    frames = _synthesize(amps, freqs, phases, win_length, sr)
    return torch.fft.rfft(frames * window, n=n_fft), last_peaks, last_freqs, last_phases


def _sinebank_fake(mag, window, n_fft, hop_length, sr, threshold_db, last_peaks, last_freqs, last_phases):
    batch, n_frames, n_bins = mag.shape
    complex_dtype = torch.promote_types(mag.dtype, torch.complex64)
    return (mag.new_empty((batch, n_frames, n_fft // 2 + 1), dtype=complex_dtype),
            mag.new_empty((batch, n_bins), dtype=torch.bool),
            mag.new_empty((batch, n_bins)), mag.new_empty((batch, n_bins)))


# -- pure implementation : plain aten operations, without data-dependent control flow (traceable by make_fx)

def _nearest_peak(is_peak: torch.Tensor) -> torch.Tensor:
    """index of the nearest peak of each bin, in its frame (..., n_bins) (0 if the frame has no peak)."""
    n_bins = is_peak.size(-1)
    index = torch.arange(n_bins, device=is_peak.device).expand(is_peak.shape)
    none = torch.full_like(index, 4 * n_bins)
    left = torch.where(is_peak, index, -none).cummax(-1).values
    right = torch.where(is_peak, index, none).flip(-1).cummin(-1).values.flip(-1)
    return torch.where(index - left <= right - index, left, right).clamp(0, n_bins - 1)


def _window_spectrum(window: torch.Tensor, n_fft: int, half_width: int, oversampling: int) -> torch.Tensor:
    """DTFT of window, centered on its sample len(window) // 2, at offsets of -(half_width + 1) to half_width + 1 bins
    (oversampling points per bin)."""
    center = window.size(0) // 2
    offsets = torch.arange(-(half_width + 1) * oversampling, (half_width + 1) * oversampling + 1, dtype=window.dtype,
                           device=window.device) / oversampling
    time = torch.arange(window.size(0), dtype=window.dtype, device=window.device) - center
    angle = -2 * math.pi * offsets.unsqueeze(-1) * time / n_fft
    return torch.complex((window * torch.cos(angle)).sum(-1), (window * torch.sin(angle)).sum(-1))


def _interpolate(table: torch.Tensor, offset: torch.Tensor, half_width: int, oversampling: int) -> torch.Tensor:
    """linear interpolation of _window_spectrum at offsets (bins) ; 0 beyond half_width + 1 bins."""
    position = (offset + half_width + 1) * oversampling
    inside = (position >= 0) & (position <= table.size(0) - 1)
    position = position.clamp(0, table.size(0) - 1.001)
    low = position.floor()
    frac = (position - low).to(table.dtype)
    low = low.long()
    value = table[low] * (1 - frac) + table[low + 1] * frac
    return torch.where(inside, value, torch.zeros_like(value))


def _synthesize_spectrum(amps: torch.Tensor, freqs: torch.Tensor, phases: torch.Tensor, window: torch.Tensor,
                         n_fft: int, sr: int) -> torch.Tensor:
    """rfft (B, n_frames, n_fft // 2 + 1) of frames of sinusoids windowed by window (as _synthesize, then
    rfft(frames * window, n_fft)), computed in the frequency domain : each sinusoid adds the spectrum of the window,
    shifted to its frequency, on the bins of its main lobe."""
    n_bins = amps.size(-1)
    half_width = int(math.ceil(4. * n_fft / window.size(0)))
    oversampling = 32
    table = _window_spectrum(window, n_fft, half_width, oversampling)
    nu = freqs * n_fft / sr
    shift = nu - torch.arange(n_bins, dtype=nu.dtype, device=nu.device)       # offset of each peak from its bin
    positive = torch.polar(amps / 2, phases)
    negative = torch.polar(amps / 2, -phases)
    spectrum = torch.zeros_like(positive)
    for d in range(-half_width, half_width + 1):
        # positive frequencies : peak j adds to bin j + d
        contribution = positive * _interpolate(table, d - shift, half_width, oversampling)
        if d > 0:
            contribution = torch.cat([torch.zeros_like(contribution[..., :d]), contribution[..., :-d]], -1)
        elif d < 0:
            contribution = torch.cat([contribution[..., -d:], torch.zeros_like(contribution[..., :-d])], -1)
        spectrum = spectrum + contribution
    for d in range(0, min(half_width, n_bins - 1) + 1):
        # negative frequencies of low partials : peak j adds to bin d - j
        contribution = (negative * _interpolate(table, d + shift, half_width, oversampling))[..., :d + 1].flip(-1)
        spectrum = spectrum + torch.cat([contribution, torch.zeros_like(spectrum[..., d + 1:])], -1)
    # phases are given at the center of frames, and rfft starts at their first sample
    center = window.size(0) // 2
    bins = torch.arange(n_bins, dtype=amps.dtype, device=amps.device)
    return spectrum * torch.polar(torch.ones_like(bins), -2 * math.pi * bins * center / n_fft)


def _sinebank_pure(mag: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, sr: int, threshold_db: float,
                   last_peaks: torch.Tensor, last_freqs: torch.Tensor, last_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """traceable approximation of _sinebank : each bin follows the frequency of the nearest peak of its frame, and
    its phase is the integral of this frequency (cumulative sum over frames). Partials moving to neighboring bins of
    the same region keep a continuous phase, as in phase-locked vocoders ; there is no matching of peaks between
    frames. Sinusoids are synthesized in the frequency domain (see _synthesize_spectrum). The state is the frequency
    and the phase of each bin in the last frame."""
    batch, n_bins = mag.size(0), mag.size(2)
    window = window.to(mag.dtype)
    is_peak, freqs, amps = _peaks(mag, window.sum(), n_fft, sr, threshold_db)
    has_peak = is_peak.any(-1, keepdim=True)
    region_freqs = torch.where(has_peak, freqs.gather(-1, _nearest_peak(is_peak)), torch.zeros_like(freqs))
    if list(last_freqs.shape) != [batch, n_bins]:
        last_freqs = torch.zeros_like(mag[:, 0])
        last_phases = torch.zeros_like(mag[:, 0])
    previous_freqs = torch.cat([last_freqs.to(mag).unsqueeze(1), region_freqs[:, :-1]], 1)
    # increments are wrapped, and summed in double precision, to keep phases precise over long signals
    increments = _wrap(math.pi * (previous_freqs + region_freqs) * hop_length / sr)
    phases = _wrap(last_phases.to(mag).unsqueeze(1).double() + increments.double().cumsum(1)).to(mag.dtype)
    spectrum = _synthesize_spectrum(amps, freqs, phases, window, n_fft, sr)
    return spectrum, is_peak[:, -1].clone(), region_freqs[:, -1].clone(), phases[:, -1].clone()


# -- opaque operator for make_fx, wrapping the exact implementation (see utils.tracing)

def _sinebank_fake(mag, window, n_fft, hop_length, sr, threshold_db, last_peaks, last_freqs, last_phases):
    batch, n_frames, n_bins = mag.shape
    complex_dtype = torch.promote_types(mag.dtype, torch.complex64)
    return (mag.new_empty((batch, n_frames, n_fft // 2 + 1), dtype=complex_dtype),
            mag.new_empty((batch, n_bins), dtype=torch.bool),
            mag.new_empty((batch, n_bins)), mag.new_empty((batch, n_bins)))


_sinebank_op = opaque_op("acids_transforms::sinebank", _sinebank_fake)(_sinebank)


@torch.jit.unused
def _sinebank_eager(mag: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, sr: int, threshold_db: float,
                    last_peaks: torch.Tensor, last_freqs: torch.Tensor, last_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    implementation = select_implementation(_sinebank, _sinebank_pure, _sinebank_op)
    return implementation(mag, window, n_fft, hop_length, sr, threshold_db, last_peaks, last_freqs, last_phases)


def sinebank(mag: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, sr: int, threshold_db: float,
             last_peaks: torch.Tensor, last_freqs: torch.Tensor, last_phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """functional sine bank (see SineBank) : magnitudes (B, n_frames, n_fft // 2 + 1) and the state of the last call
    (empty tensors to start) -> complex spectra and new state. Exact in eager mode and with TorchScript ; while
    tracing, an opaque operator or a pure approximation, depending on the backend (see utils.tracing)."""
    if torch.jit.is_scripting():
        return _sinebank(mag, window, n_fft, hop_length, sr, threshold_db, last_peaks, last_freqs, last_phases)
    return _sinebank_eager(mag, window, n_fft, hop_length, sr, threshold_db, last_peaks, last_freqs, last_phases)


class SineBank(nn.Module):
    """Sinusoidal resynthesis of magnitude spectra (inversion_mode "sinebank" of spectral transforms).

    Each frame is analyzed as a sum of sinusoids (McAulay & Quatieri, 1986) :

    - peaks : bins that are local maxima, above threshold_db relative to the maximum of their frame ;
    - frequencies and amplitudes : quadratic interpolation of the log-magnitude around each peak (exact for gaussian
      windows, as in the DGT, and close for the others) ;
    - phases : each peak continues the phase of the nearest peak of the previous frame (if it is close enough),
      advanced by their mean frequency over a hop ; new peaks start with a random phase.

    Sinusoids are synthesized on frames of len(window) samples, which are windowed and transformed back to spectra :
    spectral transforms then invert them with their exact inversion. Contrary to sines at the center frequencies of
    bins (that make a periodic signal of period n_fft samples, heard as a drone at sr / n_fft Hz), sinusoids follow
    the frequencies of partials.

    With keep_state, phases of the last frame are kept between calls for streaming ; call reset() between streams.
    The algorithm depends on values : while tracing with make_fx, it is an opaque operator or a pure approximation,
    depending on the backend (see utils.tracing), and the state is not updated.
    """

    def __init__(self, threshold_db: float = -80.):
        super().__init__()
        self.threshold_db = float(threshold_db)
        # state of the last synthesized frame (B, n_bins), not saved in state dicts
        self.register_buffer("last_peaks", torch.zeros(0, dtype=torch.bool), persistent=False)
        self.register_buffer("last_freqs", torch.zeros(0), persistent=False)
        self.register_buffer("last_phases", torch.zeros(0), persistent=False)

    @torch.jit.export
    def reset(self) -> None:
        self.last_peaks = torch.zeros(0, dtype=torch.bool, device=self.last_peaks.device)
        self.last_freqs = torch.zeros(0, device=self.last_freqs.device)
        self.last_phases = torch.zeros(0, device=self.last_phases.device)

    def forward(self, mag: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, sr: int,
                keep_state: bool = True) -> torch.Tensor:
        """magnitudes (..., n_frames, n_fft // 2 + 1) -> complex spectra of the sinusoidal resynthesis, with frames of
        len(window) samples windowed by window and zero-padded to n_fft."""
        batch_shape = mag.shape[:-2]
        mag = mag.reshape(-1, mag.size(-2), mag.size(-1))
        if keep_state:
            last_peaks, last_freqs, last_phases = self.last_peaks, self.last_freqs, self.last_phases
        else:
            last_peaks = torch.zeros(0, dtype=torch.bool, device=mag.device)
            last_freqs, last_phases = torch.zeros(0, device=mag.device), torch.zeros(0, device=mag.device)
        spectrum, last_peaks, last_freqs, last_phases = sinebank(mag, window, n_fft, hop_length, sr, self.threshold_db,
                                                                 last_peaks, last_freqs, last_phases)
        if keep_state:
            if torch.jit.is_scripting() or not is_fx_tracing():
                self.last_peaks, self.last_freqs, self.last_phases = last_peaks, last_freqs, last_phases
        return spectrum.reshape(batch_shape + spectrum.shape[-2:])
