# pitchvlm

Soccer broadcast clip in. Annotated video, 2D pitch radar and a measured score out.

One vision language model (Gemini) finds the people, the pitch landmarks and the shirt numbers.
Plain code does the geometry, the tracking, the export and the drawing. Each model step is
measured against a gate before the pipeline relies on it.

Status: first build. Nothing has been measured on real footage yet.

- Move to the run machine: [docs/HANDOVER.md](docs/HANDOVER.md)
- Plan: [docs/PLAN.md](docs/PLAN.md)
- Module contracts: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Commands: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Decisions: [docs/DECISIONS.md](docs/DECISIONS.md)

## Quick start (offline, no API key)

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
source .venv/bin/activate
python -m pytest
python -m pitchvlm.cli demo --out outputs/demo
```

The demo renders a synthetic clip with a fake model. It needs no network and no footage.
