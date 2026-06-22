from __future__ import annotations

from pathlib import Path

pathDataSamples = Path('tests/dataSamples')
pathDataSamplesExpected: Path = pathDataSamples / 'expected'

dtypeTokens: frozenset[str] = frozenset(('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64', 'complex64', 'complex128'))
