"""Milestone 4A: envelope, stereo, rhythm, and decay tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuroacoustic.analysis.envelope import compute_envelope_features
from neuroacoustic.analysis.reverb import compute_reverb_features
from neuroacoustic.analysis.rhythm import compute_rhythm_features
from neuroacoustic.analysis.stats import to_mono
from neuroacoustic.analysis.stereo import compute_stereo_features
from neuroacoustic.audio.loader import load_audio
from neuroacoustic.config import AppConfig
from neuroacoustic.fingerprint.models import AcousticFingerprint
from neuroacoustic.pipeline import run_pipeline


def _mono(path: Path, cfg: AppConfig):
    loaded = load_audio(path, cfg)
    return to_mono(loaded.samples), loaded.analysis_sample_rate, loaded


def test_envelope_percussive_vs_slow_attack(fixtures_dir: Path, app_config: AppConfig) -> None:
    perc_y, sr, _ = _mono(fixtures_dir / "impulse_decay.wav", app_config)
    slow_y, _, _ = _mono(fixtures_dir / "slow_attack_sine.wav", app_config)
    perc = compute_envelope_features(perc_y, sr, app_config.analysis)
    slow = compute_envelope_features(slow_y, sr, app_config.analysis)

    assert perc.attack_time_s is not None
    assert slow.attack_time_s is not None
    assert perc.attack_time_s < slow.attack_time_s
    assert perc.attack_time_s < 0.08
    assert slow.attack_time_s > 0.2
    assert perc.sustain_level is not None and perc.sustain_level < 0.35
    assert "not_synthesizer_adsr" in perc.warnings


def test_envelope_sustained_high_sustain(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "sustained_tone.wav", app_config)
    env = compute_envelope_features(y, sr, app_config.analysis)
    assert env.sustain_level is not None and env.sustain_level > 0.7
    assert env.decay_time_s == 0.0 or env.decay_time_s is not None
    assert "sustained_like_envelope" in env.warnings


def test_envelope_silence_nulls(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "silence.wav", app_config)
    env = compute_envelope_features(y, sr, app_config.analysis)
    assert env.onset_time_s is None
    assert env.attack_time_s is None
    assert env.sustain_level is None
    assert env.confidence == 0.0


def test_stereo_mono_no_fabricated_width(fixtures_dir: Path, app_config: AppConfig) -> None:
    loaded = load_audio(fixtures_dir / "mono_sine.wav", app_config)
    st = compute_stereo_features(loaded.samples, loaded.analysis_sample_rate, app_config.analysis)
    assert st.is_mono is True
    assert st.stereo_width_estimate is None
    assert st.correlation is None
    assert st.right_rms is None
    assert st.left_rms is not None and st.left_rms > 0


def test_stereo_identical_inverted_independent(
    fixtures_dir: Path, app_config: AppConfig
) -> None:
    identical = load_audio(fixtures_dir / "stereo_identical.wav", app_config)
    inverted = load_audio(fixtures_dir / "stereo_inverted.wav", app_config)
    independent = load_audio(fixtures_dir / "stereo_independent.wav", app_config)

    si = compute_stereo_features(identical.samples, identical.analysis_sample_rate, app_config.analysis)
    inv = compute_stereo_features(inverted.samples, inverted.analysis_sample_rate, app_config.analysis)
    ind = compute_stereo_features(
        independent.samples, independent.analysis_sample_rate, app_config.analysis
    )

    assert si.correlation is not None and si.correlation > 0.99
    assert si.stereo_width_estimate is not None and si.stereo_width_estimate < 0.02
    assert si.side_to_mid_ratio is not None and si.side_to_mid_ratio < 0.05

    assert inv.correlation is not None and inv.correlation < -0.99
    assert inv.stereo_width_estimate is not None and inv.stereo_width_estimate > 0.98
    # Mid cancels for inverted identical content
    assert inv.mid_energy is not None and inv.side_energy is not None
    assert inv.side_energy > inv.mid_energy
    assert inv.side_to_mid_ratio is None
    assert "phase_opposition" in inv.warnings
    assert "mono_compatibility_risk" in inv.warnings

    assert ind.correlation is not None and abs(ind.correlation) < 0.2
    assert ind.stereo_width_estimate is not None
    assert 0.35 < ind.stereo_width_estimate < 0.65


def test_stereo_partial_correlation(fixtures_dir: Path, app_config: AppConfig) -> None:
    loaded = load_audio(fixtures_dir / "stereo_correlated.wav", app_config)
    st = compute_stereo_features(loaded.samples, loaded.analysis_sample_rate, app_config.analysis)
    assert st.correlation is not None
    assert 0.5 < st.correlation < 0.95
    assert st.stereo_width_estimate is not None
    assert 0.0 < st.stereo_width_estimate < 0.5


def test_stereo_silent_channels_safe(app_config: AppConfig) -> None:
    n = 44100
    silent = np.zeros((n, 2), dtype=np.float32)
    st = compute_stereo_features(silent, 44100, app_config.analysis)
    assert st.correlation is None
    assert st.stereo_width_estimate is None
    assert st.side_to_mid_ratio is None


def test_rhythm_click_track_tempo(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "click_track_120bpm.wav", app_config)
    rhythm = compute_rhythm_features(y, sr, app_config.analysis)
    assert rhythm.onset_event_count >= 4
    assert rhythm.tempo_bpm is not None
    # Allow half-/double-time ambiguity around 120 BPM
    candidates = [rhythm.tempo_bpm, rhythm.tempo_bpm * 2, rhythm.tempo_bpm / 2]
    assert any(abs(c - 120.0) < 8.0 for c in candidates)
    assert rhythm.tempo_periodicity is not None
    assert rhythm.tempo_periodicity >= app_config.analysis.min_tempo_periodicity
    assert rhythm.beat_count >= app_config.analysis.min_beats_for_tempo
    assert rhythm.beat_times_s
    assert rhythm.confidence is not None and rhythm.confidence >= 0.4
    assert "half_double_time_ambiguity" in rhythm.warnings


def test_rhythm_noise_no_reliable_tempo(fixtures_dir: Path, app_config: AppConfig) -> None:
    """Broadband / uncorrelated noise must not invent a reliable tempo.

    librosa may propose a BPM and even a regular beat grid; the periodicity
    gate (onset-envelope R(τ)) must reject it.
    """
    for name in ("white_noise.wav", "stereo_independent.wav"):
        y, sr, _ = _mono(fixtures_dir / name, app_config)
        rhythm = compute_rhythm_features(y, sr, app_config.analysis)
        assert rhythm.tempo_bpm is None, f"{name} leaked tempo={rhythm.tempo_bpm}"
        assert rhythm.beat_times_s == []
        assert rhythm.confidence is not None and rhythm.confidence < 0.4
        assert any(
            w.startswith("tempo_suppressed_") or w in (
                "too_few_onsets_for_tempo",
                "too_short_for_tempo",
                "low_onset_contrast_drone_or_noise",
            )
            for w in rhythm.warnings
        ), rhythm.warnings
        # Periodicity evidence, when defined, stays below the acceptance floor
        if rhythm.tempo_periodicity is not None:
            assert rhythm.tempo_periodicity < app_config.analysis.min_tempo_periodicity


def test_rhythm_drone_null_tempo(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "drone.wav", app_config)
    rhythm = compute_rhythm_features(y, sr, app_config.analysis)
    assert rhythm.tempo_bpm is None
    assert rhythm.beat_times_s == []
    assert rhythm.confidence is not None and rhythm.confidence < 0.4
    assert any(
        w in rhythm.warnings
        for w in (
            "low_onset_contrast_drone_or_noise",
            "too_few_onsets_for_tempo",
            "tempo_suppressed_insufficient_periodicity",
            "tempo_suppressed_too_few_beats",
        )
    )


def test_rhythm_silence_and_short(fixtures_dir: Path, app_config: AppConfig) -> None:
    silence_y, sr, _ = _mono(fixtures_dir / "silence.wav", app_config)
    short_y, _, _ = _mono(fixtures_dir / "short_sine_440hz.wav", app_config)
    r_sil = compute_rhythm_features(silence_y, sr, app_config.analysis)
    r_short = compute_rhythm_features(short_y, sr, app_config.analysis)
    assert r_sil.tempo_bpm is None
    assert r_sil.beat_times_s == []
    assert r_short.tempo_bpm is None
    assert r_short.beat_times_s == []


def test_stereo_inverted_phase_warnings(fixtures_dir: Path, app_config: AppConfig) -> None:
    inverted = load_audio(fixtures_dir / "stereo_inverted.wav", app_config)
    inv = compute_stereo_features(
        inverted.samples, inverted.analysis_sample_rate, app_config.analysis
    )
    assert inv.side_to_mid_ratio is None
    assert "mid_energy_near_zero" in inv.warnings
    assert "phase_opposition" in inv.warnings
    assert "mono_compatibility_risk" in inv.warnings
    assert inv.stereo_width_estimate is not None and inv.stereo_width_estimate > 0.98

def test_reverb_exp_decay_estimate(fixtures_dir: Path, app_config: AppConfig) -> None:
    """Amplitude envelope A*exp(-t/τ) → energy ~ exp(-2t/τ).

    Energy-dB slope ≈ −(20 / (τ·ln(10))) dB/s for amplitude τ,
    so T60_energy = 60 / |slope| ≈ τ · ln(10) * 3 ≈ 6.908 · τ.
    For τ=0.12 → T60 ≈ 0.83 s (order-of-magnitude check).
    """
    y, sr, _ = _mono(fixtures_dir / "exp_decay_noise.wav", app_config)
    rev = compute_reverb_features(y, sr, app_config.analysis)
    assert rev.tail_decay_t60_estimate_seconds is not None
    assert rev.fit_r_squared is not None and rev.fit_r_squared >= 0.85
    assert rev.decay_slope_db_per_s is not None and rev.decay_slope_db_per_s < 0
    # Physical ballpark for τ=0.12 amplitude decay
    assert 0.3 < rev.tail_decay_t60_estimate_seconds < 2.5
    assert "file_tail_decay_not_room_rt60" in rev.warnings
    assert "not_exact_rt60_for_mixed_music" in rev.warnings


def test_reverb_impulse_decay(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "impulse_decay.wav", app_config)
    rev = compute_reverb_features(y, sr, app_config.analysis)
    # Short τ=0.05 → faster T60; may still fit
    if rev.tail_decay_t60_estimate_seconds is not None:
        assert rev.tail_decay_t60_estimate_seconds < 1.5
        assert rev.fit_r_squared is not None and rev.fit_r_squared > 0.7
    else:
        # Accept null if too short / insufficient range — must not invent RT60
        assert rev.confidence is not None and rev.confidence < 0.5


def test_reverb_sustained_null(fixtures_dir: Path, app_config: AppConfig) -> None:
    y, sr, _ = _mono(fixtures_dir / "sustained_tone.wav", app_config)
    rev = compute_reverb_features(y, sr, app_config.analysis)
    assert rev.tail_decay_t60_estimate_seconds is None


def test_fingerprint_m4a_sections_and_vector(
    fixtures_dir: Path, app_config: AppConfig, tmp_path: Path
) -> None:
    result = run_pipeline(
        fixtures_dir / "impulse_decay.wav",
        app_config,
        output_dir=tmp_path,
        plots=False,
    )
    fp = result.fingerprint
    assert fp.schema_version == "0.4.0"
    assert fp.analysis_version == "0.4.0"
    assert fp.vector_meta is not None
    assert fp.vector_meta.version == "0.4.0-preliminary"
    assert len(fp.vector) == 36
    assert fp.vector_meta.labels[0] == "peak_freq_norm"
    assert fp.vector_meta.labels[12] == "voiced_ratio"
    assert fp.vector_meta.labels[21] == "harmonics_confidence"
    assert fp.vector_meta.labels[22] == "attack_time_norm"
    assert fp.vector_meta.labels[35] == "reverb_confidence"
    assert fp.envelope.attack_time_s is not None or fp.envelope.confidence == 0.0
    assert fp.stereo.is_mono is True
    assert isinstance(fp.rhythm.onset_event_count, int)
    assert "file_tail_decay_not_room_rt60" in fp.reverberation.warnings

    # Round-trip Pydantic + finite scan
    payload = fp.to_json_dict()
    AcousticFingerprint.model_validate(payload)
    _assert_finite(payload)


def _assert_finite(obj: object) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_finite(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_finite(v)
    elif isinstance(obj, float):
        assert obj == obj and obj not in (float("inf"), float("-inf"))
