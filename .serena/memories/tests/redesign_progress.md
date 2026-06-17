# Test Redesign Progress

Status as of 2026-06-17; verify worktree state before relying on these notes.

- Direction: migrate pytest tests toward `mem:instructions/testing_conventions`: pytest-native parametrization, shared fixtures in `tests/conftest.py`, deterministic samples under `tests/dataSamples/`, expected contracts under `tests/dataSamples/expected/`.
- Current active target: `readAudioFile`.
- New shared path source: `tests/_theSSOT.py` defines `pathDataSamples` and `pathDataSamplesExpected`; `tests/__init__.py` re-exports them.
- `tests/conftest.py` is in transition. New pieces include `approx_rel`, `approx_abs`, `assert_approx`, `assert_array_equal`, `messageTestFailure_ndarray`, fixture-param `pathFilename` over seven audio samples, `dtype_str` inferred as `'int16'` when the sample filename stem contains `s16`, and `expected` loading `.npy` arrays from `tests/dataSamples/expected/`.
- Expected-array naming pattern: `readAudioFile__{pathFilename.stem}__sampleRateDesired{int(sampleRateDesired)}Hz__dtype{dtype_str or 'default'}.npy`.
- Test data progress: `tests/dataSamples/expected/` contains 14 expected arrays for seven samples times desired rates `44100` and `48000`. New sample name uses `Tone1000Hz_ch2_44100Hz_29s_LUFS-23_s16.wav`; the old unsuffixed tone WAV is deleted in the worktree.
- `tests/test_io.py` currently contains `test_readAudioFile`, parametrized over `sampleRateDesired` values `44100` and `48000`; it uses the `pathFilename`, `dtype_str`, and `expected` fixtures and asserts exact array equality with `assert_array_equal`.
- `tests/test_resample.py` exists but has no code symbols/content yet.
- Implementation dependency: current in-progress `src/hunterHearsPy/_io.py` signature is `readAudioFile(pathFilename, sampleRateDesired=None, dtype_str=None)`, and `test_readAudioFile` passes `dtype_str` through. Root `hunterHearsPy.__init__` currently re-exports `个` for conftest typing.
- Old test support still remains in `tests/conftest.py` under old-system banners: `standardizedEqualTo`, prototype NumPy helpers, `WaveformAndMetadata`, `ingestSampleData`, `sampleData`, and labeled-sample paths. Decide keep/adapt/remove as each target migrates.
- Validation not run during this memory update. Before treating the `readAudioFile` redesign as stable, run focused `uv run pytest tests/test_io.py -q`.