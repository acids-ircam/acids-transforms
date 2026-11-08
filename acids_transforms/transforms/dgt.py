import torch
import math
from enum import Enum
from .stft import realtime_frames, overlap_add_frames, STFT, RealtimeSTFT
from .base import AudioTransform, InversionEnumType
from .sinebank import SineBank
from ..utils.tracing import opaque_op, is_fx_tracing, select_implementation
from ..utils.heapq import heappop, heappush
from ..utils.misc import frame, reshape_batches
from typing import Tuple, Dict, Union, List


__all__ = ['DGT', 'RealtimeDGT']

MAX_NFFT = 16384
eps = torch.finfo(torch.get_default_dtype()).eps


# PGHI (Prusa et al., 2017) : phases integrated from their time / frequency derivatives, estimated from the magnitude
# with a gaussian window. Integration follows a heap of magnitudes, which depends on values : while tracing, pghi and
# pghi_realtime are opaque operators, or pure approximations, depending on the backend (see utils.tracing).

def _phasegrad_offline(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:
    fmul = gamma/(hop_length * n_fft)
    Y = torch.empty((mag.shape[0]+2, mag.shape[1]+2)).to(mag)
    Y[1:-1, 1:-1] = torch.log(mag)
    Y[0, :] = Y[1, :]
    Y[-1, :] = Y[-2, :]
    Y[:, 0] = Y[:, 1]
    Y[:, -1] = Y[:, -2]
    dxdw = (Y[1:-1, 2:]-Y[1:-1, :-2])/2
    dxdt = (Y[2:, 1:-1]-Y[:-2, 1:-1])/2
    # dxdt = (3 * Y[..., 2:, 1:-1] - 4 * Y[..., 1:-1, 1:-1] + Y[..., 0:-2, 1:-1])/2
    fgradw = dxdw/fmul + (2*torch.pi*hop_length/n_fft) * \
        torch.arange(int(n_fft/2)+1).unsqueeze(0).to(mag)
    tgradw = -fmul*dxdt + torch.pi
    return (tgradw, fgradw)


def _hgi_offline(X: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor, abstol: float, tol: float) -> torch.Tensor:
    spectrogram = X
    phase = torch.zeros_like(spectrogram)
    M2 = spectrogram.shape[0]
    N = spectrogram.shape[1]
    max_val = spectrogram.max()  # Find maximum value to start integration
    max_pos_f = torch.nonzero(spectrogram == max_val)[0]
    magnitude_heap = [(-max_val, (max_pos_f[0], max_pos_f[1]))]
    spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
    spectrogram = torch.where(
        spectrogram < max_val * tol, torch.full_like(spectrogram, abstol), spectrogram)
    while max_val > abstol:
        while len(magnitude_heap) > 0:  # Integrate around maximum value until reaching silence
            max_val, max_pos = heappop(magnitude_heap)
            col = max_pos[0]
            row = max_pos[1]
            N_pos = col+1, row
            S_pos = col-1, row
            E_pos = col, row+1
            W_pos = col, row-1
            if max_pos[0] < M2-1 and spectrogram[N_pos[0], N_pos[1]] > abstol:
                phase[N_pos[0], N_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                    (fgradw[max_pos[0], max_pos[1]] +
                     fgradw[N_pos[0], N_pos[1]])/2
                heappush(magnitude_heap,
                         (-spectrogram[N_pos[0], N_pos[1]], N_pos))
                spectrogram[N_pos[0], N_pos[1]] = abstol
            if max_pos[0] > 0 and spectrogram[S_pos[0], S_pos[1]] > abstol:
                phase[S_pos[0], S_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                    (fgradw[max_pos[0], max_pos[1]] +
                     fgradw[S_pos[0], S_pos[1]])/2
                heappush(magnitude_heap,
                         (-spectrogram[S_pos[0], S_pos[1]], S_pos))
                spectrogram[S_pos[0], S_pos[1]] = abstol
            if max_pos[1] < N-1 and spectrogram[E_pos[0], E_pos[1]] > abstol:
                phase[E_pos[0], E_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                    (tgradw[max_pos[0], max_pos[1]] +
                     tgradw[E_pos[0], E_pos[1]])/2
                heappush(magnitude_heap,
                         (-spectrogram[E_pos[0], E_pos[1]], E_pos))
                spectrogram[E_pos[0], E_pos[1]] = abstol
            if max_pos[1] > 0 and spectrogram[W_pos[0], W_pos[1]] > abstol:
                phase[W_pos[0], W_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                    (tgradw[max_pos[0], max_pos[1]] +
                     tgradw[W_pos[0], W_pos[1]])/2
                heappush(magnitude_heap,
                         (-spectrogram[W_pos[0], W_pos[1]], W_pos))
                spectrogram[W_pos[0], W_pos[1]] = abstol
        max_val = spectrogram.max()
        max_pos_f = torch.nonzero(spectrogram == max_val)[0]
        heappush(magnitude_heap, (-max_val, (max_pos_f[0], max_pos_f[1])))
        spectrogram[max_pos_f[0], max_pos_f[1]] = abstol
    return phase


def _phasegrad_realtime(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    taken from
    https://github.com/andimarafioti/tifresi/blob/676db371d5c472a5f3199506bf3863367a2ecde4/tifresi/phase/modGabPhaseGrad.py#L77
    a : length of time shift
    g : window function
    M : number of frequency channels
    """
    #if gamma is None: gamma = 2*torch.pi*((-n_fft**2/(8*torch.log(0.01)))**.5)**2
    fmul = gamma/(hop_length * n_fft)
    Y = torch.empty((mag.shape[0], mag.shape[1]+2,
                     mag.shape[2]+2), dtype=mag.dtype, device=mag.device)
    Y[:, 1:-1, 1:-1] = torch.log(mag)
    Y[:, :, 0] = Y[:, :, 1]
    Y[:, :, -1] = Y[:, :, -2]
    # time edges (read by dxdt) : left uninitialized, they gave random nans in pghi
    Y[:, 0, :] = Y[:, 1, :]
    Y[:, -1, :] = Y[:, -2, :]
    dxdw = (Y[:, 1:-1, 2:] - Y[:, 1:-1, :-2])/2
    dxdt = (3 * Y[..., 2:, 1:-1] - 4 * Y[..., 1:-1, 1:-1] + Y[..., 0:-2, 1:-1])/2
    fgradw = dxdw/fmul + (2*torch.pi*hop_length/n_fft) * (torch.arange(math.floor(n_fft/2)+1, device=mag.device))
    tgradw = -fmul * dxdt + torch.pi
    return (tgradw, fgradw)


def _hgi_realtime(spectrogram: torch.Tensor, previous_phase: torch.Tensor, tgradw: torch.Tensor, fgradw: torch.Tensor, tol: float, eps: float) -> torch.Tensor:
    abstol = torch.clamp(tol * spectrogram.max(), eps, None)
    # Random init phase when X < abstol
    phase = torch.zeros_like(spectrogram)
    phase[1] = previous_phase
    phase[2:] = torch.where(spectrogram[2:] > abstol, torch.zeros_like(
        spectrogram[2:]), torch.randn_like(spectrogram[2:]))

    # pad gradients for convenient indices
    pad = torch.zeros(2, tgradw.shape[1], device=tgradw.device, dtype=tgradw.dtype)
    tgradw = torch.cat([pad, tgradw], 0)
    fgradw = torch.cat([pad, fgradw], 0)
    spectrogram_history = spectrogram.clone()

    for f in range(2, spectrogram.shape[0]):
        # Create I set
        max_val = spectrogram[f].max()
        if max_val <= abstol:
            continue

        # Init heap
        # zero_tensor = torch.zeros(1).long()
        # one_tensor = torch.ones(1).long()
        zero_tensor = torch.full((1,), f-1).long()
        one_tensor = torch.full((1,), f).long()
        max_val = spectrogram[f].max()

        max_pos_f = torch.nonzero(spectrogram[f] == max_val)
        magnitude_heap = [(-max_val, (one_tensor, max_pos_f[0].unsqueeze(0)))]
        for item in torch.nonzero(spectrogram_history[f-1] > abstol):
            heappush(
                magnitude_heap, (-spectrogram_history[f-1, item[0]], (zero_tensor, item[0].unsqueeze(0))))
        if len(magnitude_heap) == 0:
            continue
        while max_val > abstol:
            while len(magnitude_heap) > 0:
                max_val, max_pos = heappop(magnitude_heap)
                if max_pos[0] == f-1:
                    E_pos = (max_pos[0]+1, max_pos[1])
                    if spectrogram[E_pos[0], E_pos[1]] > abstol:
                        phase[E_pos[0], E_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                            0.5 * (tgradw[max_pos[0], max_pos[1]] + tgradw[E_pos[0], E_pos[1]])
                        heappush(
                            magnitude_heap, (-spectrogram[E_pos[0], E_pos[1]], (E_pos[0], E_pos[1])))
                        spectrogram[E_pos[0], E_pos[1]] = abstol
                if max_pos[0] == f:
                    if max_pos[1] + 1 < spectrogram.shape[1]:
                        N_pos = (max_pos[0], max_pos[1]+1)
                        if spectrogram[N_pos[0], N_pos[1]] > abstol:
                            phase[N_pos[0], N_pos[1]] = phase[max_pos[0], max_pos[1]] + \
                                0.5 * (fgradw[max_pos[0], max_pos[1]]+fgradw[N_pos[0], N_pos[1]])
                            heappush(
                                magnitude_heap, (-spectrogram[N_pos[0], N_pos[1]], (N_pos[0], N_pos[1])))
                            spectrogram[N_pos[0], N_pos[1]] = abstol
                    if max_pos[1] - 1 > 0:
                        S_pos = (max_pos[0], max_pos[1]-1)
                        if spectrogram[S_pos[0], S_pos[1]] > abstol:
                            phase[S_pos[0], S_pos[1]] = phase[max_pos[0], max_pos[1]] - \
                                0.5 * (fgradw[max_pos[0], max_pos[1]]+fgradw[S_pos[0], S_pos[1]])
                            heappush(
                                magnitude_heap, (-spectrogram[S_pos[0], S_pos[1]], (S_pos[0], S_pos[1])))
                            spectrogram[S_pos[0], S_pos[1]] = abstol
            max_val = spectrogram[f].max()
            max_pos_f = torch.nonzero(spectrogram[f] == max_val)
            heappush(magnitude_heap, (-max_val,
                                    (one_tensor, max_pos_f[0].unsqueeze(0))))
            spectrogram[f, max_pos_f[0]] = abstol
    return phase[2:]


def _pghi(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """phases (B, n_frames, n_bins) of magnitudes (B, n_frames, n_bins), integrated over the whole spectrogram."""
    phases = []
    for n in range(mag.size(0)):
        m = torch.clamp(mag[n].clone(), eps, None)
        tgradw, fgradw = _phasegrad_offline(m, gamma, hop_length, n_fft)
        phases.append(_hgi_offline(m, tgradw, fgradw, eps, tolerance))
    return torch.stack(phases)


def _pghi_realtime(mag: torch.Tensor, mag_buffer: torch.Tensor, phase_buffer: torch.Tensor, gamma: float,
                   hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """phases (B, n_frames, n_bins) of magnitudes (B, n_frames, n_bins), integrated from the two previous frames
    (magnitudes mag_buffer (B, 2, n_bins)) and the phase of the last one (phase_buffer (B, n_bins))."""
    mag = torch.clamp(torch.cat([mag_buffer.to(mag), mag], -2), eps, None)
    tgradw, fgradw = _phasegrad_realtime(mag, gamma, hop_length, n_fft)
    phases = []
    for i in range(mag.size(0)):
        phases.append(_hgi_realtime(mag[i], phase_buffer[i].to(mag), tgradw[i], fgradw[i], tolerance, eps))
    return torch.stack(phases)


# -- pure implementation : plain aten operations, without data-dependent control flow (traceable by make_fx)

def _phasegrad_pure(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """phase derivatives (B, n_frames, n_bins) of log-magnitudes (B, n_frames, n_bins), as _phasegrad_offline : phase
    increments from a frame to the next one, and from a bin to the next one."""
    log_mag = torch.log(mag)
    log_f = torch.cat([log_mag[..., :1], log_mag, log_mag[..., -1:]], -1)
    log_t = torch.cat([log_mag[:, :1], log_mag, log_mag[:, -1:]], 1)
    d_freq = (log_f[..., 2:] - log_f[..., :-2]) / 2
    d_time = (log_t[:, 2:] - log_t[:, :-2]) / 2
    fmul = gamma / (hop_length * n_fft)
    bins = torch.arange(mag.size(-1), dtype=mag.dtype, device=mag.device)
    along_frames = d_freq / fmul + (2 * math.pi * hop_length / n_fft) * bins
    along_bins = -fmul * d_time + math.pi
    return along_frames, along_bins


def _integrate_pure(mag: torch.Tensor, along_frames: torch.Tensor, along_bins: torch.Tensor,
                    first_phase: torch.Tensor, tolerance: float) -> torch.Tensor:
    """phases (B, n_frames - 1, n_bins) of the frames following the first one (of phase first_phase (B, n_bins)) :
    integrated over frames for each bin, then locked in frequency around the peaks of each frame (bins take the phase
    of their nearest peak, integrated over bins). Bins below tolerance * max(mag) get random phases."""
    increments = 0.5 * (along_frames[:, 1:] + along_frames[:, :-1])
    # wrapped increments, summed in double precision, keep phases precise over long signals
    increments = torch.remainder(increments + math.pi, 2 * math.pi) - math.pi
    time_phase = first_phase.unsqueeze(1).double() + increments.double().cumsum(1)
    mag, along_bins = mag[:, 1:], along_bins[:, 1:]
    # phase locking around peaks
    left, center, right = mag[..., :-2], mag[..., 1:-1], mag[..., 2:]
    threshold = mag.amax((-2, -1), keepdim=True) * tolerance
    is_peak = torch.nn.functional.pad(((center > left) & (center >= right) & (center > threshold)).to(torch.uint8),
                                      [1, 1]).to(torch.bool)
    n_bins = mag.size(-1)
    index = torch.arange(n_bins, device=mag.device).expand(is_peak.shape)
    none = torch.full_like(index, 4 * n_bins)
    nearest_left = torch.where(is_peak, index, -none).cummax(-1).values
    nearest_right = torch.where(is_peak, index, none).flip(-1).cummin(-1).values.flip(-1)
    nearest = torch.where(index - nearest_left <= nearest_right - index, nearest_left, nearest_right).clamp(0, n_bins - 1)
    bin_phase = torch.cat([torch.zeros_like(along_bins[..., :1]),
                           (0.5 * (along_bins[..., 1:] + along_bins[..., :-1])).double().cumsum(-1).to(along_bins)], -1)
    locked = time_phase.gather(-1, nearest) + (bin_phase - bin_phase.gather(-1, nearest)).double()
    phase = torch.where(is_peak.any(-1, keepdim=True), locked, time_phase)
    phase = (torch.remainder(phase + math.pi, 2 * math.pi) - math.pi).to(mag.dtype)
    return torch.where(mag > threshold, phase, 2 * math.pi * torch.rand_like(phase))


def _pghi_pure(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """traceable approximation of _pghi : phases integrated over frames for each bin (cumulative sums), and locked in
    frequency around spectral peaks, instead of the heap integration of PGHI."""
    mag = mag.clamp(min=eps)
    along_frames, along_bins = _phasegrad_pure(mag, gamma, hop_length, n_fft)
    # the first frame is integrated from a zero phase, as a frame before the signal
    zero = torch.zeros_like(mag[:, :1])
    return _integrate_pure(torch.cat([zero + eps, mag], 1), torch.cat([along_frames[:, :1], along_frames], 1),
                           torch.cat([along_bins[:, :1], along_bins], 1), zero[:, 0], tolerance)


def _pghi_realtime_pure(mag: torch.Tensor, mag_buffer: torch.Tensor, phase_buffer: torch.Tensor, gamma: float,
                        hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """traceable approximation of _pghi_realtime (see _pghi_pure), continued from the last frame (magnitudes
    mag_buffer (B, 2, n_bins), phase of the last one phase_buffer (B, n_bins))."""
    full = torch.cat([mag_buffer.to(mag), mag], -2).clamp(min=eps)
    # gamma of the gaussian window of the DGT (as DGT._get_gamma) : the gamma of RealtimeDGT (used by _pghi_realtime)
    # gives wrong phase derivatives
    gamma = 2 * math.pi * (-n_fft ** 2 / (8 * math.log(0.01)))
    along_frames, along_bins = _phasegrad_pure(full, gamma, hop_length, n_fft)
    return _integrate_pure(full[:, 1:], along_frames[:, 1:], along_bins[:, 1:], phase_buffer.to(mag), tolerance)


# -- opaque operators for make_fx, wrapping the exact implementations (see utils.tracing)

def _pghi_fake(mag, gamma, hop_length, n_fft, eps, tolerance):
    return torch.empty_like(mag)


def _pghi_realtime_fake(mag, mag_buffer, phase_buffer, gamma, hop_length, n_fft, eps, tolerance):
    return torch.empty_like(mag)


_pghi_op = opaque_op("acids_transforms::pghi", _pghi_fake)(_pghi)
_pghi_realtime_op = opaque_op("acids_transforms::pghi_realtime", _pghi_realtime_fake)(_pghi_realtime)


@torch.jit.unused
def _pghi_eager(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    return select_implementation(_pghi, _pghi_pure, _pghi_op)(mag, gamma, hop_length, n_fft, eps, tolerance)


@torch.jit.unused
def _pghi_realtime_eager(mag: torch.Tensor, mag_buffer: torch.Tensor, phase_buffer: torch.Tensor, gamma: float,
                         hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    implementation = select_implementation(_pghi_realtime, _pghi_realtime_pure, _pghi_realtime_op)
    return implementation(mag, mag_buffer, phase_buffer, gamma, hop_length, n_fft, eps, tolerance)


def pghi(mag: torch.Tensor, gamma: float, hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """PGHI phases of magnitudes (B, n_frames, n_bins). Exact in eager mode and with TorchScript ; while tracing, an
    opaque operator or a pure approximation, depending on the backend (see utils.tracing)."""
    if torch.jit.is_scripting():
        return _pghi(mag, gamma, hop_length, n_fft, eps, tolerance)
    return _pghi_eager(mag, gamma, hop_length, n_fft, eps, tolerance)


def pghi_realtime(mag: torch.Tensor, mag_buffer: torch.Tensor, phase_buffer: torch.Tensor, gamma: float,
                  hop_length: int, n_fft: int, eps: float, tolerance: float) -> torch.Tensor:
    """frame-wise PGHI phases of magnitudes (B, n_frames, n_bins), continued from the previous frames. Exact in eager
    mode and with TorchScript ; while tracing, an opaque operator or a pure approximation (see utils.tracing)."""
    if torch.jit.is_scripting():
        return _pghi_realtime(mag, mag_buffer, phase_buffer, gamma, hop_length, n_fft, eps, tolerance)
    return _pghi_realtime_eager(mag, mag_buffer, phase_buffer, gamma, hop_length, n_fft, eps, tolerance)


class DGT_INVERSION_MODES(Enum):
    KEEP_INPUT = 0
    GRIFFIN_LIM = 1
    PGHI = 2
    RANDOM = 3


class DGT(STFT):

    def __repr__(self):
        repr_str = f"DGT(n_fft={self.n_fft}, hop_length={self.hop_length}, " \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    def __init__(self, sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, dtype: torch.dtype = None, 
                 inversion_mode: InversionEnumType = "pghi", tolerance: float = 1.e-2):
        AudioTransform.__init__(self, sr)
        dtype = dtype or torch.get_default_dtype()
        self.register_buffer("window", torch.zeros(MAX_NFFT))
        self.register_buffer("inv_window", torch.zeros(MAX_NFFT))
        self.register_buffer("eps",  torch.tensor(
            torch.finfo(dtype).eps, dtype=dtype))
        self.register_buffer("phase_buffer", torch.zeros(0))
        self.register_buffer("tolerance", torch.tensor(tolerance))
        self.sinebank = SineBank()
        if (n_fft is not None):
            assert hop_length is not None, "n_fft and hop_length must be given together"
        if (hop_length is not None):
            assert n_fft is not None, "n_fft and hop_length must be given together"
        if (n_fft is not None) and (hop_length is not None):
            self.set_params(n_fft, hop_length)
        if inversion_mode in type(self).get_inversion_modes():
            self.inversion_mode = inversion_mode
        else:
            raise ValueError("Inversion mode %s not known" % inversion_mode)

    def realtime(self):
        inversion_mode = self.inversion_mode if self.inversion_mode in RealtimeDGT.get_inversion_modes() else "pghi"
        return RealtimeDGT(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, inversion_mode = inversion_mode)

    @property
    def ratio(self):
        return self.hop_length

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -1)
        window = self.window[:self.n_fft]
        x_dgt = torch.stft(x, n_fft=int(self.n_fft), hop_length=int(self.hop_length), window=window, return_complex=True, onesided=True).transpose(-2, -1)
        self._replace_phase_buffer(x_dgt.angle())
        return x_dgt.reshape(batch_shape + x_dgt.shape[-2:])

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor): 
        transform = self.forward(x)
        n_chunks = transform.size(-2)
        shifts = torch.arange(n_chunks, device=transform.device) * self.hop_length / self.sr
        new_shape = [t for t in transform.shape[:-2]]
        new_shape.append(n_chunks)
        new_strides = [0] * (x.ndim-1)
        new_strides.append(1)
        shifts = shifts.as_strided(new_shape, new_strides)
        new_time = shifts + time.unsqueeze(-1)
        return transform, new_time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        x, batch_shape = reshape_batches(x, -2)
        if not torch.is_complex(x):
            x_inv = self.invert_without_phase(x, inversion_mode)
        else:
            # torch.istft divides by the overlap of squared windows : with the analysis window, it gives the dual window
            window = self.window[:self.n_fft]
            x_inv = torch.istft(x.transpose(-2, -1), n_fft=self.n_fft, hop_length=self.hop_length, window=window, onesided=True)
        return x_inv.reshape(batch_shape + x_inv.shape[-1:])            

    @torch.jit.export
    def set_params(self, n_fft: int, hop_length: int) -> None:
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.window.zero_()
        self.inv_window.zero_()
        self.window[:self.n_fft] = self._get_window()
        self.inv_window[:self.n_fft] = self._get_dual_window()
        self.gamma = self._get_gamma()

    def _get_gamma(self) -> float:
        return 2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2

    def _get_window(self) -> torch.FloatTensor:
        lambda_ = (-self.n_fft**2/(8*math.log(0.01)))**.5
        n = torch.arange(0, 2*self.n_fft+1) - (2*self.n_fft) / 2
        w = torch.exp(-n**2 / (2 * (lambda_ * 2)**2))
        return w[1:2*self.n_fft+1:2]

    def _get_dual_window(self) -> torch.Tensor:
        return self._canonical_dual_window()

    def _canonical_dual_window(self) -> torch.Tensor:
        gsynth = torch.zeros(self.n_fft)
        for l in range(self.n_fft):
            denom = 0
            for n in range(int(-self.n_fft//self.hop_length), int(self.n_fft//self.hop_length + 1)):
                dl = l - n*self.hop_length
                if dl >= 0 and dl < self.n_fft:
                    denom += self.window[dl]**2
            gsynth[l] = self.window[l]/denom
        return gsynth

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        # analysis window : torch.istft computes the dual window (see invert)
        window = self.window[:self.n_fft]
        phase = torch.tensor(0)
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer(x)
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "griffin_lim"):
            return self.griffin_lim(x)
        elif (inversion_mode == "pghi"):
            phase = self.pghi(x, float(self.tolerance))
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "sinebank"):
            return torch.istft(self.get_sinebank_inversion(x, False).transpose(-2, -1), n_fft=int(self.n_fft),
                               hop_length=int(self.hop_length), window=window, onesided=True)
        else:
            raise ValueError("inversion mode %s not valid." %
                             self.inversion_mode)
        x = x * torch.exp(phase * torch.full(phase.size(),
                                             1j, device=phase.device))
        return torch.istft(x.transpose(-2, -1), n_fft=int(self.n_fft), hop_length=int(self.hop_length), window=window, onesided=True)

    def get_sinebank_inversion(self, magnitude: torch.Tensor, keep_state: bool = True) -> torch.Tensor:
        """complex spectrum of the sinusoidal resynthesis of magnitude (see SineBank)."""
        return self.sinebank(magnitude, self.window[:self.n_fft], self.n_fft, self.hop_length, self.sr, keep_state)

    def pghi(self, mag: torch.Tensor, tolerance: float = 1.e-4) -> torch.Tensor:
        """PGHI phases of magnitudes (..., n_frames, n_bins) (traced by make_fx as a single node)."""
        mag, batch_shape = reshape_batches(mag, -2)
        phase = pghi(mag, self.gamma, self.hop_length, self.n_fft, float(self.eps), tolerance)
        return phase.reshape(batch_shape + phase.shape[-2:])

    @staticmethod
    def get_inversion_modes():
        return ["pghi", "griffin_lim", "random", "keep_input", "sinebank"]

    # keep_input depends on the last forward call (PGHI and sine bank are traced as opaque operators)
    untracable_inversion_modes = ["keep_input"]
    
class RealtimeDGT(DGT):
    def __init__(self, sr: int = 44100, n_fft=1024, hop_length=256, dtype=None, batch_size: Union[int, List[int]] = 2, inversion_mode: InversionEnumType = "pghi"):
        super().__init__(sr=sr, n_fft=n_fft, hop_length=hop_length, dtype=dtype, inversion_mode=inversion_mode)
        if isinstance(batch_size, int):
            self.batch_size = [batch_size]
        else:
            self.batch_size = batch_size

        self.register_buffer(
            'hgi_mag_buffer', torch.zeros(*self.batch_size, 2, n_fft//2+1))
        self.register_buffer("hgi_phase_buffer",
                             torch.zeros(*self.batch_size, n_fft//2+1))

    def __repr__(self):
        repr_str = f"RealtimeDGT(n_fft={self.n_fft}, hop_length={self.hop_length}" \
                   f"inversion_mode = {self.inversion_mode})"
        return repr_str

    @staticmethod
    def get_inversion_modes():
        return ['random', 'pghi', 'keep_input', 'sinebank']

    @classmethod
    def _tracing_input(cls, transform: AudioTransform, x: torch.Tensor) -> torch.Tensor:
        """frame-wise transforms take frames (..., n_frames, n_fft), as given by OverlapAdd."""
        return realtime_frames(x, transform.n_fft, transform.hop_length)

    @classmethod
    def _tracing_output(cls, transform: AudioTransform, y: torch.Tensor) -> torch.Tensor:
        return overlap_add_frames(y, transform.hop_length)

    @classmethod
    def _before_tracing(cls, transform: AudioTransform, x: torch.Tensor) -> None:
        # PGHI buffers are reset when the batch shape changes : resetting them inside a trace would store symbolic
        # shapes in the module, that leak into later traces
        transform.reset(list(x.shape[:-2]))

    def _get_dual_window(self) -> torch.Tensor:
        # synthesis frames are overlap-added by OverlapAdd, that compensates a gain of n_fft / hop_length
        return self._canonical_dual_window() * (float(self.n_fft) / self.hop_length)

    @torch.jit.export
    def get_batch_size(self) -> List[int]:
        return [int(b) for b in self.batch_size]

    @torch.jit.export
    def set_batch_size(self, batch_size: Union[int, List[int]]):
        self.reset(batch_size)

    @torch.jit.export
    def batch_size(self) -> List[int]:
        return self.hgi_mag_buffer.shape[:-2]

    @torch.jit.export
    def reset(self, batch_size: Union[int, List[int]]) -> None:
        if isinstance(batch_size, int):
            self.batch_size = [batch_size]
        else:
            self.batch_size = torch.Size(batch_size)
        device = self.hgi_mag_buffer.device
        self.hgi_mag_buffer = torch.zeros(torch.Size(self.batch_size) + torch.Size([2, int(self.n_fft)//2+1]), device=device)
        self.hgi_phase_buffer = torch.zeros(torch.Size(self.batch_size) + torch.Size([int(self.n_fft)//2+1]), device=device)
        self.sinebank.reset()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window[:self.n_fft]
        x_dgt = torch.fft.rfft(x * window)
        self._replace_phase_buffer(x_dgt.angle())
        return x_dgt

    @torch.jit.export
    def forward_with_time(self, x: torch.Tensor, time: torch.Tensor): 
        return self(x), time

    @torch.jit.export
    def invert(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        if not torch.is_complex(x):
            x_rec = self.invert_without_phase(x, inversion_mode)
            return x_rec
        else:
            inv_window = self.inv_window[:self.n_fft]
            return torch.fft.irfft(x) * inv_window

    def invert_without_phase(self, x: torch.Tensor, inversion_mode: InversionEnumType = None) -> torch.Tensor:
        batch_size = x.shape[:-2]
        if batch_size != self.batch_size:
            self.reset(batch_size)
        window = self.inv_window[:self.n_fft]
        phase = torch.tensor(0)
        if inversion_mode is None:
            inversion_mode = self.inversion_mode
        if (inversion_mode == "keep_input"):
            phase = self._get_phase_buffer(x)
            if phase.shape[0] == 0:
                phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "pghi"):
            phase = self.pghi(x, float(self.tolerance))
        elif (inversion_mode == "random"):
            phase = torch.pi * 2 * torch.rand_like(x)
        elif (inversion_mode == "sinebank"):
            # phases of partials are continued from the last call
            x = self.get_sinebank_inversion(x)
            self.update_buffers(x)
            return torch.fft.irfft(x) * window
        else:
            raise ValueError("inversion mode %s not valid." %
                             self.inversion_mode)
        x = x * torch.exp(phase * torch.full(phase.shape,
                                             1j, device=phase.device))
        self.update_buffers(x)
        return torch.fft.irfft(x) * window

    def update_buffers(self, x):
        # traced tensors must not be stored in the module (they would leak in later calls)
        if not torch.jit.is_scripting():
            if is_fx_tracing():
                return
        if x.shape[-2] > 1:
            self.hgi_mag_buffer = x[..., -2:, :].abs()
        else:
            self.hgi_mag_buffer = torch.stack(
                [self.hgi_mag_buffer[..., 1, :], x[..., -1, :].abs()], -2)
        self.hgi_phase_buffer = x[..., -1, :].angle()

    def pghi(self, mag: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
        """PGHI phases of magnitudes (..., n_frames, n_bins), continued from the previous frames (traced by make_fx as
        a single node)."""
        mag, batch_shape = reshape_batches(mag, -2)
        hgi_mag_buffer, _ = reshape_batches(self.hgi_mag_buffer, -2)
        hgi_phase_buffer, _ = reshape_batches(self.hgi_phase_buffer, -1)
        phases = pghi_realtime(mag, hgi_mag_buffer, hgi_phase_buffer, self.gamma, self.hop_length, self.n_fft,
                               float(self.eps), tolerance)
        return phases.reshape(batch_shape + phases.shape[1:])

    def _get_gamma(self) -> float:
        return (-self.n_fft**2/(8*math.log(0.01)))**.5
        # return 2*torch.pi*((-self.n_fft**2/(8*math.log(0.01)))**.5)**2
        # return - (torch.pi / 4) * (self.n_fft**2 / math.log(self.window[0])

    def test_forward(self, x: torch.Tensor, time: torch.Tensor = None):
        x = frame(x, self.n_fft, self.hop_length, -1)
        transform = []
        for i in range(x.shape[-2]):
            transform.append(self(x[..., i, :]))
        transform = torch.stack(transform, -2)
        if time is None:
            return transform
        else:
            return transform, None

    # Tests
    def test_inversion(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        from .oadd import OverlapAdd
        self.reset(x.shape[:-1])
        n_fft = self.n_fft
        chunk_size = n_fft * 4
        hop_length = self.hop_length
        oadd = OverlapAdd(n_fft, hop_length)
        outs = {}
        x_framed = x.split(chunk_size, -1)
        # test direct inversion
        outs["direct"] = []
        for n in range(len(x_framed)):
            x_in = oadd(x_framed[n])
            x_t = self(x_in)
            x_i = oadd.invert(self.invert(x_t))
            outs['direct'].append(x_i)
        outs['direct'] = torch.cat(outs['direct'], -1)
        # test spectrogram inversion modes
        for inv_mode in self.get_inversion_modes():
            self.inversion_mode = inv_mode
            oadd = OverlapAdd(n_fft, hop_length)
            outs[inv_mode] = []
            for n in range(len(x_framed)):
                x_in = oadd(x_framed[n])
                x_t = self(x_in)
                x_i = oadd.invert(self.invert(x_t.abs(), inversion_mode=inv_mode))
                outs[inv_mode].append(x_i)
            outs[inv_mode] = torch.cat(outs[inv_mode], -1)
        return outs


    @classmethod
    def test_scripted_transform(cls, transform: AudioTransform, invert: bool = True):
        x = torch.zeros(2, 1, transform.n_fft)
        transform.reset(list(x.shape[:-1]))
        x_t = transform(x)
        if cls.invertible:
            x_inv = transform.invert(x_t)
            for inv_type in cls.get_inversion_modes():
                x_inv = transform.invert(x_t.abs(), inversion_mode=inv_type)
