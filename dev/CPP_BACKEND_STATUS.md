# C++ Backend Status

## Goal

Move the hot board-analysis path from Python into a C++ extension and wire it back into the existing Python env/bot code without changing training semantics.

## Completed

- Added C++ source at [cpp/RewardEvaluator.cpp](/d:/code/gomoku_ai/cpp/RewardEvaluator.cpp)
- Added Python wrapper at [gomoku_ai/cpp_backend.py](/d:/code/gomoku_ai/gomoku_ai/cpp_backend.py)
- Added local build entrypoint at [setup.py](/d:/code/gomoku_ai/setup.py)
- Integrated the backend into [gomoku_ai/env.py](/d:/code/gomoku_ai/gomoku_ai/env.py) with Python fallback
- Updated build dependencies in [pyproject.toml](/d:/code/gomoku_ai/pyproject.toml) and [requirements.txt](/d:/code/gomoku_ai/requirements.txt)
- Configured the build to copy `libwinpthread-1.dll` next to the extension on Windows MinGW builds

## C++ Functions Implemented

- `classify_move_counts`
- `immediate_winning_actions`
- `threat_summary`
- `affected_actions`

These cover the dominant hot path used by:

- environment reward calculation
- threat recomputation
- rule-based bot move scoring

## Integration Notes

- `env.py` now checks whether the compiled backend is available.
- If the extension imports successfully, hot functions dispatch to C++.
- If the extension is missing, the previous Python implementation is still used.

## Local Build Command

```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace --force --compiler=mingw32
```

## Verification Results

### Tests

```text
17 passed in 0.06s
```

### Benchmark

Before C++ backend:

```text
steps_per_sec ~= 78.24
```

After C++ backend:

```text
steps_per_sec ~= 2013.78
```

### cProfile

Before C++ backend:

```text
steps_per_sec ~= 26.04
```

After C++ backend:

```text
steps_per_sec ~= 26.50 under cProfile, but Python-level hot calls collapse into C-extension calls
```

The non-profiled benchmark is the meaningful performance signal here.

## Remaining Work

- If desired, move `neighboring_actions` or full environment step logic into C++ as a second phase.
- Add a dedicated build script or CI job if this backend should be rebuilt automatically across machines.
