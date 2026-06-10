# ruff: noqa: D100
from __future__ import annotations

from hunterHearsPy import ParametersUniversal, tukey
from hunterMakesPy import PackageSettings
from numpy import complex64, float32
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterHearsPy import ParametersShortTimeFFT, ParametersSTFT

settingsPackage = PackageSettings('hunterHearsPy')

# TODO I want reliably consistent waveforms, spectrograms, and transformations.
# Therefore, I want one package, this package, to manage those objects.
# Within the package, I need a single source of truth for the settings of those objects.
# I need good default universal settings.
# But if I want a different universal setting, I need an easy way to change the universal setting.
# Until I learn how to make an easy way to change the universal settings, I will use the HARDCODED
# system as a placeholder.
# Finally, I want to be able to override the universal settings as needed on a per-call basis.

#================== Hardcoded =====================================================================

universalDtypeWaveform = float32
universalDtypeSpectrogram = complex64
parametersShortTimeFFTUniversal: ParametersShortTimeFFT = {'fft_mode': 'onesided'}
parametersSTFTUniversal: ParametersSTFT = {'padding': 'even', 'axis': -1}

lengthWindowingFunctionDEFAULT = 1024
windowingFunctionCallableDEFAULT = tukey
parametersDEFAULT = ParametersUniversal(
	lengthFFT=2048
	, lengthHop=512
	, lengthWindowingFunction=lengthWindowingFunctionDEFAULT
	, sampleRate=44100
	, windowingFunction=windowingFunctionCallableDEFAULT(lengthWindowingFunctionDEFAULT),
)

setParametersUniversal = None

windowingFunctionCallableUniversal = windowingFunctionCallableDEFAULT
if not setParametersUniversal:
	parametersUniversal: ParametersUniversal = parametersDEFAULT
