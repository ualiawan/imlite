# Logging & Progress Bars

## Logging

imlite uses Python's standard `logging` module with the named logger `"imlite"`.
The library **never adds handlers by itself** - you are in full control.

### Setting verbosity

```python
import imlite

imlite.set_verbosity("DEBUG")    # very verbose: all internal steps
imlite.set_verbosity("INFO")     # progress messages (default when enabled)
imlite.set_verbosity("WARNING")  # only warnings and errors (library default)
imlite.set_verbosity("ERROR")    # errors only
imlite.set_verbosity("SILENT")   # completely quiet
```

`set_verbosity` also accepts integer levels (`logging.DEBUG`, `logging.INFO`, ...).

When you call `set_verbosity`, imlite automatically attaches a `StreamHandler` to `stderr`
if no handler has been configured yet - so you always see output without extra setup.

### Integrating with your own logging config

If your application already configures the root logger, imlite will respect it:

```python
import logging
logging.basicConfig(level=logging.INFO)

import imlite
imlite.set_verbosity("DEBUG")
```

## Progress Bars

Long-running operations (frame extraction, frame saving, video encoding) display a
`tqdm` progress bar automatically.

### Disabling progress bars

```python
imlite.set_progress(False)    # suppress all progress bars
imlite.set_progress(True)     # re-enable (default)
```

This is a global flag. Disabling is recommended in automated pipelines and CI:

```python
# In pytest conftest.py - keeps test output clean
import imlite
imlite.set_progress(False)
```

### Per-call override

The low-level functions (`extract_frames`, `merge_frames`, `save_frames`) accept a
`show_progress` keyword argument that overrides the global setting for that one call:

```python
imlite.extract_frames("clip.mp4", "frames/", show_progress=False)
```
