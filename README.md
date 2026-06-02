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

## Installation

```bash
pip install hunterHearsPy
```

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/hunterHearsPy/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
