# imlite examples

Runnable scripts. Each one is self-contained and generates whatever media it
needs, so nothing here depends on files that are not in the repository.

```bash
pip install py-imlite
python examples/01_image_pipeline.py
```

| Script | What it shows |
|---|---|
| `01_image_pipeline.py` | Loading, chaining transforms, the functional API, colour spaces |
| `02_video_pipeline.py` | Video to frames to video, all of it lazy |
| `03_batch_thumbnails.py` | Processing a directory of images |
| `04_numpy_interop.py` | Dropping imlite into an existing numpy pipeline |

Most of what these scripts do is also available from the shell - see
`imlite --help` and the [CLI guide](../docs/guides/cli.md).
