# pitchvlm — project rules

Soccer broadcast clip in; annotated video, 2D pitch radar and a measured score out. One vision
language model (Gemini) does the perception. Plain code does geometry, tracking, export, drawing.

Read first, in this order: `docs/HANDOVER.md` (the move to the run machine, step by step),
`docs/RUNBOOK.md` (commands), `docs/PLAN.md` (what and why), `docs/DECISIONS.md` (decision log).
For code changes also: `docs/ARCHITECTURE.md` (module contracts), `configs/gates.yaml` (gates).

## Two machines

- **Build machine**: has no model API key, makes no API call, holds no footage, no dataset and no
  label files. It writes code, runs the offline tests and reads returned text results.
- **Run machine**: has the Gemini key in the environment variable `GEMINI_API_KEY`, and the
  dataset and footage under `~/pitchvlm-data` (outside this repository).
- How to tell: if `GEMINI_API_KEY` is not set (in the environment or in `.env`) and `~/pitchvlm-data` does not exist, you are on the
  build machine. Then never call the API, never download data, and install nothing that is not
  already installed. If a package is missing, stop and say so.
- A new run machine has neither of the two at first. So the owner makes the folder
  `~/pitchvlm-data` before the first session there, and says that it is the run machine
  (`docs/HANDOVER.md`, section 4).

## Hard rules

1. No secret in a tracked file: no API key, no token. On the run machine the keys may live in a
   `.env` file in the top folder. It is git-ignored, has mode 600, and is never committed, read
   out or printed. A command loads it with
   `set -a; . ./.env; set +a` in the same shell call as the command.
2. No pixels and no dataset files in the repository: no images, no video, no crops, no label files.
   Result files that go on the `results` branch are text only: metrics, cached model answers, the
   cost ledger. They hold no box, no pitch position and no role, team or shirt number from a
   label file.
3. No machine-specific text in tracked files: no absolute home paths, no user names, no private
   host names, no index addresses, and nothing about the owner or the policies of a machine:
   the two machines are "build machine" and "run machine" in every file. No lock file is
   committed (`uv.lock`, `pylock.toml`); the pinned list of the run machine is a requirements
   file under `locks/`.
4. Only `src/pitchvlm/vlm/client.py` imports the model SDK. Everything else must run without it.
   Only `tools/run_tool.py` imports an external tool (RF-DETR, PnLCalib). It runs in the
   environment of that tool, and the package never imports it.
5. Model calls: paid tier only, `store=False`, no sampling parameters. Players are identified by
   shirt number plus roster, never by face.
6. Money: every paid run prints its estimate first and the measured cost after. A run stops at 1.3x
   its estimate. The contract checks of `doctor` are the one exception: each doctor run stops at
   the spend limit of gate G0 (2 USD). Hard stop at 135 USD in total. Do not start a paid run
   without the owner's go-ahead.
7. Measurement: gates are fixed in `configs/gates.yaml` before a run. The 10 test clips are run
   once, after the freeze tag. Do not tune on them and do not look at their results before.
8. A specialist model may replace a Gemini step only after that step's gate failed and the
   specialist passed the same gate on the same frames. The report names the engine of every step.
9. The guard (`tools/guard.py`) is a git hook. It runs before `git commit`
   and before a push that sends commits. Do not bypass them: `--no-verify`, the variable `SKIP`
   of the hook runner and another hook folder (`core.hooksPath`) are not allowed. The guard
   knows private words only from the deny-list of the clone, the file `.git/guard-deny.txt`.
   The owner writes it on each machine (`docs/RUNBOOK.md`, section 9). It is never committed
   and never copied into a tracked file. No hook runs for a push that sends only names (a tag,
   or a new branch on a commit that the remote has): run `python3 tools/guard.py --push` by
   hand before such a push.
10. Commit only when the owner asks. Never push without being asked. A commit message has no
    co-author line. The remote is public: a pushed commit cannot be taken back.

## How to work

- Small, tested changes. `pytest` must pass offline, with no SDK installed.
- Keep "measured" apart from "estimated" in every note and report. Give sample sizes.
- Run one paid command at a time, and use one cache folder for all paid runs: the cost ledger
  in it is the memory of the budget guard, and two commands do not see the spend of each other.
- Write each decision with its numbers into `docs/DECISIONS.md` on the day it is made.
- Conventions (coordinates, records, error handling) are in `docs/ARCHITECTURE.md` section 2.

## Commands

```bash
python -m pytest                      # offline tests
python -m pitchvlm.cli demo --out outputs/demo   # synthetic clip, fake model, no network
python3 tools/guard.py --tracked      # leak check of all tracked files
python3 tools/guard.py --push         # leak check of what a push can send: commits, tags, new names
```

On the build machine the package is not installed in the environment (there is no editable
install). Run its commands there from the top folder of the repository with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m pitchvlm.cli demo --out outputs/demo
```

`python -m pytest` needs no prefix: it takes its Python path from `pyproject.toml`.

`python -m pitchvlm.cli doctor --no-api` checks the machine and calls no model. It fails until
git runs both hooks (`pre-commit install --hook-type pre-commit --hook-type pre-push`) and until
the deny-list `.git/guard-deny.txt` is there. The owner does both on each machine.

The run-machine commands (`doctor`, `data`, `oracle`, `probe`, `tool-job`, `tool-score`, `run`,
`eval`, `sync`) are in `docs/RUNBOOK.md`.
