# hunterHearsPy

A comprehensive collection of Python utilities for audio processing.

## Audio Processing Made Simple

### Load and Save Audio Files

Read audio files with automatic stereo conversion and sample rate control:

```python
from hunterHearsPy import readAudioFile, writeWAV

# Load audio with sample rate conversion
waveform = readAudioFile('input.wav', sampleRate=44100)

# Save in WAV format (always 32-bit float)
writeWAV('output.wav', waveform)
```

### Process Multiple Audio Files at Once

Load and process batches of audio files:

```python
from hunterHearsPy import loadWaveforms

# Load multiple files with consistent formatting
array_waveforms = loadWaveforms(['file1.wav', 'file2.wav', 'file3.wav'])

# The result is a unified array with shape (channels, samples, file_count)
```

### Work with Spectrograms

Convert between waveforms and spectrograms:

```python
from hunterHearsPy import stft, halfsine

# Create a spectrogram with a half-sine window
spectrogram = stft(waveform, windowingFunction=halfsine(1024))

# Convert back to a waveform
reconstructed = stft(spectrogram, inverse=True, lengthWaveform=original_length)
```

### Process Audio in the Frequency Domain

Create functions that operate on spectrograms:

```python
from hunterHearsPy import waveformSpectrogramWaveform

def boost_low_frequencies(spectrogram):
    # Boost frequencies below 500 Hz
    spectrogram[:, :10, :] *= 2.0
    return spectrogram

# Create a processor that handles the STFT/ISTFT automatically
processor = waveformSpectrogramWaveform(boost_low_frequencies)

# Apply the processor to a waveform
processed_waveform = processor(original_waveform)
```

## Development

I want:

- consistent waveforms,
- consistent spectrograms, and
- efficient, reliable transformations.

Therefore, I want one package, this package, to manage those objectives with a single source of truth for configurable universal settings.

- I need good default universal settings.
- I need an easy way to change a universal setting.
- I want _ad hoc_ overrides for some or all universal settings.

I don't know a good way to implement user-configurable universal settings. Therefore, as I normally do, I will use my HARDCODED system as a placeholder.

### Semiotics

- channel, rarely channels.
- time is more generic than samples and often preferred.
- array is generic.
- when talking about a NumPy `ndarray`, write `ndarray` not array.
- Use `sampleRate` unless I have a compelling reason not to.

### Preferred packages

- NumPy
- scipy
- hunterMakesPy
- tqdm (for status messages)
- cytoolz via the [Z0Z_tools](https://github.com/hunterhogan/Z0Z_tools) package (until it finds a forever home)
- more_itertools
- `astToolKit`
- `pytest`

### Probably won't need

- `analyzeAudio`
- `gmpy2`
- `numba`
- `platformdirs`
- `sympy`
- `torch-einops-kit`

### Not using `PyTorch` for "business" logic

- But, I must have `torch` compatibility.
- At a minimum, a transformation to and from `torch` that the user must call.
- `astToolkit` easily creates real `torch` APIs and identifiers, however, for the `windowingFunctions` module in `windowingFunctionsTensor`.

### Not using librosa

- Dependency bloated.
- Slow.
- Less precise than I want.
- I generally dislike the API and identifiers.

### Disfavored packages

- `pandas`

### More vectorization

- Few or no `for` loops.
- Few or no `for` object comprehensions.

## Installation

```bash
pip install hunterHearsPy
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/hunterHearsPy/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
