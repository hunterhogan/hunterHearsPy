# Tech Stack

- Python package, `requires-python >=3.10`; package name `hunterHearsPy`.
- Build backend: `uv_build`; `[tool.uv].build-backend` sets `module-name = "hunterHearsPy"`, `module-root = "src"`.
- Runtime dependencies: `hunterMakesPy`, `numpy`, `resampy`, `scipy`, `soundfile`, `tqdm`.
- Optional dependency group: project optional extra `torch = ["torch"]`; dependency group `pytorch` pins `torch==2.10.0`, `torchaudio==2.10.0` from PyTorch CUDA 12.8 index.
- Dev/test dependency groups: `dev = ["astToolkit", "pytest-cov", "scipy-stubs", "types-resampy"]`; `testing = ["pytest", "pytest-mock", "pytest-xdist", "torch"]`.
- Pytest config belongs in `[tool.pytest.ini_options]`; current config uses `testpaths = ["tests"]`, `pythonpath = ["src"]`, xdist `-n 4`, and colored output.
- Coverage config writes data under `tests/coverage/.coverage`, runs branch coverage with multiprocessing concurrency, omits `tests/*`, and currently uses `source = ["."]`.
- Source uses NumPy/SciPy audio/signal operations; `ioAudio.py` depends on `hunterMakesPy.filesystemToolkit.makeDirsSafely`.