from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
from scipy.signal import stft


@dataclass(frozen=True)
class AudioAssessment:
    passed: bool
    duration_seconds: float
    sample_rate: int
    channels: int
    rms: float
    clipping_fraction: float
    silent_frame_fraction: float
    speech_activity_proxy: float
    features: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_audio(audio_bytes: bytes) -> AudioAssessment:
    if not audio_bytes:
        raise ValueError("The recording is empty")
    try:
        samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Unsupported or unreadable audio format; record a WAV file") from error
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("The recording contains no usable audio samples")

    mono = samples.mean(axis=1)
    duration = len(mono) / sample_rate
    rms = float(np.sqrt(np.mean(np.square(mono))))
    clipping_fraction = float(np.mean(np.abs(samples) >= 0.999))
    frame_length = max(1, int(sample_rate * 0.02))
    frame_count = len(mono) // frame_length
    if frame_count:
        frames = mono[:frame_count * frame_length].reshape(frame_count, frame_length)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        silent_fraction = float(np.mean(frame_rms < 10 ** (-45 / 20)))
        activity_proxy = float(np.mean(frame_rms >= 10 ** (-40 / 20)))
        zero_crossings = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        zcr = float(np.mean(zero_crossings))
        frame_energy = frame_rms
    else:
        silent_fraction = 1.0
        activity_proxy = 0.0
        zcr = 0.0
        frame_energy = np.array([0.0])

    frequencies, _, spectrum = stft(mono, fs=sample_rate, nperseg=min(1024, len(mono)), noverlap=None, boundary=None)
    magnitudes = np.abs(spectrum)
    spectral_power = magnitudes ** 2
    power_sum = np.sum(spectral_power, axis=0)
    valid = power_sum > np.finfo(float).eps
    if valid.any():
        normalized = spectral_power[:, valid] / power_sum[valid]
        centroid = float(np.mean(np.sum(frequencies[:, None] * normalized, axis=0)))
        geometric = np.exp(np.mean(np.log(spectral_power[:, valid] + 1e-12), axis=0))
        flatness = float(np.mean(geometric / np.mean(spectral_power[:, valid], axis=0)))
    else:
        centroid = 0.0
        flatness = 0.0

    reasons: list[str] = []
    if duration < 3.0 or duration > 15.0:
        reasons.append("Record between 3 and 15 seconds.")
    if sample_rate not in {16000, 22050, 24000, 44100, 48000}:
        reasons.append("Use a supported sample rate (16, 22.05, 24, 44.1, or 48 kHz).")
    if rms < 0.005:
        reasons.append("Recording level is too low; move closer to the microphone.")
    if rms > 0.45:
        reasons.append("Recording level is unusually high; move farther from the microphone.")
    if clipping_fraction > 0.005:
        reasons.append("Clipping detected; lower the microphone level and record again.")
    if silent_fraction > 0.60 or activity_proxy < 0.15:
        reasons.append("Too much silence or too little signal; record again in a quiet room.")

    features = {
        "rms_mean": rms,
        "rms_std": float(np.std(frame_energy)),
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "speech_activity_proxy": activity_proxy,
    }
    return AudioAssessment(
        passed=not reasons,
        duration_seconds=float(duration),
        sample_rate=int(sample_rate),
        channels=int(samples.shape[1]),
        rms=rms,
        clipping_fraction=clipping_fraction,
        silent_frame_fraction=silent_fraction,
        speech_activity_proxy=activity_proxy,
        features=features,
        reasons=tuple(reasons),
    )