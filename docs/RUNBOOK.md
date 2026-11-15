# Runbook for the run machine

This file is for the run machine and for an assistant that works there.
Read `CLAUDE.md` first, then `docs/HANDOVER.md`: it gives the order of the move to this machine
and says which steps the owner does alone. The plan is in `docs/PLAN.md`, the gates are in
`configs/gates.yaml`.

Where things are:

| What | Where |
|---|---|
| Code | the clone of this repository; run every command in its top folder |
| Dataset, footage, videos, scorer | `~/pitchvlm-data` (outside the repository) |
| Text results | `../pitchvlm-results/results` (work tree of the branch `results`, next to the repository) |
| Stored model answers and cost ledger | `../pitchvlm-results/results/cache` (the default of `--cache`); the ledger is `ledger.jsonl`, then `ledger-0002.jsonl` and so on |
| Metrics files | `../pitchvlm-results/results/metrics` |

Every command has `--help`. Exit code 0 means done, 1 means that a check or a gate failed, 2 means
that the command could not run. Every error gives the exit code 2 and ends with a line
`error: ...`, also when a Python traceback is printed before that line. So the exit code 1 never
means an error.

The output of a command is written line by line, also into a pipe or a file. So the cost
estimate and the budget line can be read before the first model call, and `error:` is the last
line also when both streams go to one file. A settings file, a roster file, a label file or an
experiment file with wrong content gives one line that names the file and the fault, without a
traceback.

## 1 Setup once

```bash
# Tools.
brew install uv gitleaks
uv tool install pre-commit

# Main environment: Python 3.13. Then run the offline tests.
uv venv --python 3.13
uv pip install -e ".[dev,gemini]"
source .venv/bin/activate
python -m pytest

# Pinned package list of this machine. uv.lock is never committed.
mkdir -p locks
uv pip compile pyproject.toml --extra dev --extra gemini -o locks/run-machine.txt

# Hooks: the guard and the secret scan run before a commit and before a push (section 9).
pre-commit install --hook-type pre-commit --hook-type pre-push

# Personal git identity, for this repository only. The guard refuses a commit and a push until
# both values are set here, and it needs the same two values on both machines.
git config --local user.name "<your name>"
git config --local user.email "<your personal e-mail>"

# Deny-list of this clone: the private words that the guard refuses. Section 9 says what to put in it.
printf '%s\n' "# One word per line. This file is never committed." >> .git/guard-deny.txt

# Data folder, outside the repository. The owner makes it before the first session of the
# assistant (docs/HANDOVER.md, section 4.1).
mkdir -p ~/pitchvlm-data

# Work tree for the text results, on the branch "results".
git worktree add -b results ../pitchvlm-results
# When the branch exists already (a later clone): git worktree add ../pitchvlm-results results

# Scorer: its own environment (Python 3.12, numpy 1.26) and a clone of its repository at one tag.
# The package brings the dependencies. The code that scores is the code of the clone.
uv venv --python 3.12 ~/pitchvlm-data/envs/scorer
uv pip install --python ~/pitchvlm-data/envs/scorer/bin/python "numpy>=1.26,<1.27" "sn-trackeval==0.4.0"
git clone --branch v0.4.1 --depth 1 https://github.com/SoccerNet/sn-trackeval ~/pitchvlm-data/sn-trackeval
git -C ~/pitchvlm-data/sn-trackeval rev-parse HEAD
```

- The scorer script `scripts/run_soccernet_gs.py` puts the folder of its clone first on the Python
  path. So the `trackeval` code of the clone runs, and not the installed package 0.4.0. The clone
  is fixed to the tag `v0.4.1` for this reason. Checked on 14 Nov 2026: this tag is the tag
  `v0.4.0` plus one line, and that line repairs an error message and changes no score.
- Write the commit that the last command prints into `docs/DECISIONS.md` at the setup, before
  the commit of the setup changes (`docs/HANDOVER.md`, section 4, steps 12 and 14). The files
  `oracle.json` and `scores.json` record the commit of the scorer clone under
  `provenance.scorer_commit`.
- The model SDK must be a version 3.x (`google-genai>=3.0,<4` in `pyproject.toml`). An older SDK
  repeats a failed call by itself, and `doctor` then fails before any model call.

The key lives in the environment variable `GEMINI_API_KEY`. Set it in each new terminal:

```bash
# The key is typed at the prompt. It is not shown and not stored in the shell history.
printf "Gemini API key: " && read -rs GEMINI_API_KEY && export GEMINI_API_KEY && echo
source .venv/bin/activate
```

- Never write the key into a file, a command line or a chat message. Only the owner types it,
  at this prompt. An assistant gets the key only when the owner starts it from the terminal
  that has the key; it does not run the prompt command and does not print the environment.
- A key that Google AI Studio makes now starts with `AQ.`; an older key starts with `AIza`. The
  guard and the secret scan know both forms. The length of the new form is not public: the guard
  expects 20 or more characters after the prefix. On day 1, count the characters of the real key
  without writing it anywhere (`printf %s "$GEMINI_API_KEY" | wc -c`). Tell the owner when the
  count is below 23: the pattern in `tools/guard.py` and in `.gitleaks.toml` must then change.
  Section 8 (fact 11) has a command that tests the whole pattern and prints no part of the key.
- The work tree `../pitchvlm-results` also holds a copy of the code. Do not run code from there.
- The new file `locks/run-machine.txt` belongs to the code branch. Commit it when the owner asks,
  after the hooks, the git identity and the deny-list are set. A paid command refuses to start
  while the file is not committed, because the git work tree then has changes.

## 2 Day 1

Ask the owner before each paid command. A paid command prints its cost estimate first and the
measured cost after. It refuses to start when the git work tree has changes: commit first.
Run one paid command at a time (section 4, money).

```bash
# 1. The machine. No model call.
python -m pitchvlm.cli doctor --no-api

# 2. The contract of the model API, gate G0. Paid: about 0.13 USD for these two models (an
#    estimate that is not measured yet). The checks stop above the 2 USD of gate G0.
python -m pitchvlm.cli doctor --models gemini-4-argon,gemini-3.8-flash \
    --out ../pitchvlm-results/results/metrics/doctor.json

# 3. Data: download the validation split (11.2 GB), extract it and write the split list.
python -m pitchvlm.cli data --root ~/pitchvlm-data --split valid \
    --manifest configs/splits.json --seed 20261113
```

The split list `configs/splits.json` is fixed now. Commit it (when the owner asks) before the
first probe. The test split is downloaded only at the freeze, with `--split test` and the same
`--manifest` and `--seed`.

```bash
# 4. Free label stage, gate G1. First one clip, then all clips of the split.
python -m pitchvlm.cli oracle --root ~/pitchvlm-data --split valid --clips <clip> \
    --scorer-python ~/pitchvlm-data/envs/scorer/bin/python \
    --scorer-script ~/pitchvlm-data/sn-trackeval/scripts/run_soccernet_gs.py \
    --out ~/pitchvlm-data/oracle/one-clip
python -m pitchvlm.cli oracle --root ~/pitchvlm-data --split valid \
    --scorer-python ~/pitchvlm-data/envs/scorer/bin/python \
    --scorer-script ~/pitchvlm-data/sn-trackeval/scripts/run_soccernet_gs.py \
    --out ~/pitchvlm-data/oracle/valid
cp ~/pitchvlm-data/oracle/valid/oracle.json ../pitchvlm-results/results/metrics/oracle_valid.json

# 5. Cost pilot: the first 2 clips of the probe group. Paid: about 0.7 USD for the three probes.
python -m pitchvlm.cli probe people --experiment configs/experiments/people_probe.yaml \
    --root ~/pitchvlm-data --max-clips 2 --out ../pitchvlm-results/results/metrics/pilot_people.json
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \
    --root ~/pitchvlm-data --max-clips 2 --out ../pitchvlm-results/results/metrics/pilot_pitch.json
python -m pitchvlm.cli probe number --experiment configs/experiments/number_probe.yaml \
    --root ~/pitchvlm-data --max-clips 2 --out ../pitchvlm-results/results/metrics/pilot_number.json
```

- `doctor --no-api` fails until git runs the hooks of the pre-commit framework and until the
  deny-list `.git/guard-deny.txt` is there (section 9). A hook file of another tool, a hook file
  without the executable bit, and a `core.hooksPath` setting that points to another folder all
  fail the check `guard hooks`.
- The doctor prints two kinds of lines for the boxes and the landmarks. A failed check `boxes`
  or `landmarks` (and a failed check `store flag`, `token sum`, `served tier` or `flex tier`) is
  a fault of the client: repair it before any footage is sent. The lines `box accuracy` and
  `landmark accuracy` are facts without a verdict: a value that does not reach the value of the
  gate file, next to a passed contract check, means that the model is less exact on the small
  synthetic boxes or confuses a landmark. It is not a client fault. Write both values into
  `docs/DECISIONS.md`.
- A check `flex tier` is a client fault only when its line starts with `[fail]`. A line
  `[info] flex tier: not checked: the flex tier had no capacity ...` means that the provider
  answered 503 or 429 for 900 seconds. This is the documented answer of a busy flex tier: gate
  G0 passes, the flex tier is not verified, and the doctor must run again before the first run
  through the flex tier.
- The contract checks of the doctor stop above the spend limit of gate G0
  (`g0_contract.spend_usd_max` in `configs/gates.yaml`, 2 USD), and not above 1.3 times their
  estimate like every other paid run. Their estimate is the least certain one: the first real
  answers of a model come from these checks. The limit holds for one doctor run with all its
  models together, and the budget line of the doctor names it. The checks do not start when
  their estimate is above the limit, also with `--start-over-budget`: name fewer models then.
- When the doctor stops at this limit (exit code 2), the screen shows `[fail] spend` and the
  error line names the spent amount and the limit. The report file is still written: it holds
  the checks of the models that were finished and the cost of every paid answer, and no check
  of the model that was in progress. Tell the owner: the checks cost many times their estimate.
  Put the mean tokens of the "measured cost" lines into `DEFAULT_TOKEN_ESTIMATES` (with the
  owner's go-ahead) and commit before the doctor runs again.
- The answers of the doctor are not stored, so every doctor run pays again, up to its limit at
  each start. After the token correction the estimate of the same command is above the limit,
  so the doctor does not start with the same models. Run it with one model at a time, each with
  its own `--out` file. When one model alone is estimated above the limit, stop and ask the
  owner: that model can run only with a higher `spend_usd_max`, which is a gate change with an
  entry in `docs/DECISIONS.md`.
- A request through the flex tier can wait up to 15 minutes for its answer (client timeout
  900 s). A standard request waits 5 minutes. The line `flex tier` of the doctor names the time
  that the flex answer took. While the flex tier is busy (status 503 or 429), the request is
  tried again for up to 900 seconds after its first attempt, with waits of up to 60 seconds.
  Every wait is a log line, and these attempts are not billed. When the tier stays busy, a
  `run` or a `probe` stops with exit code 2 and the error "the flex tier had no capacity":
  start the command again later, the stored answers are used again.
- Without `--cache` a command stores and reads its answers in
  `../pitchvlm-results/results/cache`. It refuses to start when `../pitchvlm-results` is not a
  git work tree on the branch `results`: create the work tree first (section 1).
- G0 and G1 must pass before any footage is sent to the model.
- The report of the label stage (`oracle.json`, part `team_side_rule`) holds the share of clips
  where the team-side rule is right, the share where the players alone are right, the clips
  that the goalkeepers decided and, for every clip, the gap between the mean positions of the
  two teams. The rule has two limits that are estimates (3 m for this gap, the depth of the
  penalty area for a goalkeeper): compare them with these values (section 8).
- The lines after "measured cost" give the mean tokens of one answer for each step. Write them
  into `docs/DECISIONS.md`. They replace the estimates in `src/pitchvlm/vlm/cost.py`
  (`DEFAULT_TOKEN_ESTIMATES`) and the same numbers in `docs/ARCHITECTURE.md` (section
  `vlm/cost.py`). The output count of an estimate is the sum of the "out" and the "thinking"
  tokens.
- After this edit, run `python -m pytest`. No test pins the values of this table. The token
  counts of the fake model (`src/pitchvlm/vlm/fake.py`) stay as they are: the tests of the
  command line make their estimates from them.
- The answers of the pilot stay in the cache. The full probes use them again and do not pay twice.
- Planned budget for doctor and cost pilot together: 2 USD. One doctor run can use this amount
  alone, because it stops only above 2 USD. When the doctor runs cost more than about 1.3 USD
  together, tell the owner before the cost pilot (about 0.7 USD).

## 3 Probes

Each probe reads one experiment file: the group of the split list, the frames or tracks per clip,
the seed, the repeat index and the model settings.

| Probe | Gate | Planned budget, all settings together |
|---|---|---|
| people | G2: recall, precision, small boxes, foot-point error | 14 USD |
| pitch | G3: position error near the goals and in midfield, share of frames with a mapping | 15 USD |
| number | G5a: exact-match rate on crops of labelled boxes (upper bound, no pass line) | 11 USD |

```bash
python -m pitchvlm.cli probe people --experiment configs/experiments/people_probe.yaml \
    --root ~/pitchvlm-data --out ../pitchvlm-results/results/metrics/people_probe.json
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \
    --root ~/pitchvlm-data --out ../pitchvlm-results/results/metrics/pitch_probe.json
python -m pitchvlm.cli probe number --experiment configs/experiments/number_probe.yaml \
    --root ~/pitchvlm-data --out ../pitchvlm-results/results/metrics/number_probe.json
```

- Pitch probe: run the first 4 clips (20 frames) with `--max-clips 4` first. Compare `valid_share`
  and `error_m.median` of the metrics file with the kill switch of gate G3 in `configs/gates.yaml`.
- Another setting (another model, another resolution, the flex tier): copy the experiment file,
  change `name` and one value under `step`, commit, and run it with its own `--out` file.
- A repeat run: set `repeat: 1` in a copy of the file. The cache does not answer it, so it pays
  for every call again. The command prints a line that says so.
- The model `gemini-robotics-er-2-preview` has no flex tier. A settings file or an experiment
  file that asks for `service_tier: flex` with this model is refused before the estimate is
  printed. A thinking level that a model does not have is refused when the file is read
  (`THINKING_LEVELS` in `src/pitchvlm/config.py`).
- A probe never chooses a setting. Compare the metrics files with the gates, and write the decision
  with its numbers into `docs/DECISIONS.md`.
- `--replay` runs a probe again from the stored answers, for free.

### The two specialist tools on the same frames (free)

The owner decided on 2026-11-14 that the RF-DETR detector and the PnLCalib calibration are
measured on the frames of the people probe and of the pitch probe, with their pretrained weights
and without training. So the decision of 24 Nov has measured values for both engines of each
step. No model is called and nothing is paid. Rule 8 of section 7 does not change: a tool
replaces a Gemini step only after that step failed its gate and the tool passed the same gate on
the same frames.

| Tool | Experiment file | Frames and gate |
|---|---|---|
| RF-DETR Small | `configs/experiments/people_probe_rfdetr_small.yaml` | people probe, G2 |
| RF-DETR Small, antialiased resize | `configs/experiments/people_probe_rfdetr_small_antialias.yaml` | people probe, G2 |
| RF-DETR Large | `configs/experiments/people_probe_rfdetr_large.yaml` | people probe, G2 |
| RF-DETR Large, antialiased resize | `configs/experiments/people_probe_rfdetr_large_antialias.yaml` | people probe, G2 |
| PnLCalib | `configs/experiments/pitch_probe_pnlcalib.yaml` | pitch probe, G3 |

Each tool gets its own environment, because its packages do not fit the environment of the
package. Ask the owner before the install: it downloads packages and about 1 GB of weights.
These commands were written from the pages of the two tools. None of them ran before.

```bash
# RF-DETR. The script tools/run_tool.py was written for the versions 1.12.0 and 1.12.1 of the package.
uv venv --python 3.12 ~/pitchvlm-data/envs/rfdetr
uv pip install --python ~/pitchvlm-data/envs/rfdetr/bin/python "rfdetr==1.12.0"

# PnLCalib: a clone at the commit that the script was written for, its own environment, two weight files.
git clone https://github.com/mguti97/PnLCalib ~/pitchvlm-data/PnLCalib
git -C ~/pitchvlm-data/PnLCalib checkout 42088864170f75ea02e76d20e9a9a1db59edb3e3
uv venv --python 3.12 ~/pitchvlm-data/envs/pnlcalib
uv pip install --python ~/pitchvlm-data/envs/pnlcalib/bin/python \
    "torch==2.3.1" "torchvision==0.18.1" "numpy==2.0.2" "scipy==1.13.1" "opencv-python==4.10.0.84" \
    "shapely==2.0.7" "matplotlib==3.9.4" "pillow==11.1.0" "PyYAML==6.0.2"
mkdir -p ~/pitchvlm-data/weights/pnlcalib
curl -L --fail -o ~/pitchvlm-data/weights/pnlcalib/SV_kp \
    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L --fail -o ~/pitchvlm-data/weights/pnlcalib/SV_lines \
    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
ls -l ~/pitchvlm-data/weights/pnlcalib    # SV_kp: 264964645 bytes, SV_lines: 264857893 bytes
```

- The install of RF-DETR takes the packages that `rfdetr` needs without a fixed version. Give
  it the package age limit of the setup (section 8, fact 19).
- RF-DETR downloads the weights of a model class when the class is used for the first time
  (Small: 386 MB, Large: 136 MB) from the host of its maker into `~/.roboflow/models`.
- The requirements file of PnLCalib is for Linux with an NVIDIA GPU and cannot be installed
  here. The install above names the packages that its files `model/` and `utils/` import, each
  in the version of that requirements file. The package index has a file of each one for an
  Apple Silicon machine with Python 3.12 (read on 2026-11-14). Do not take a newer
  `opencv-python`: its version 5 moved the camera functions that the tool calls, and the tool
  was not tested with it.
- Write the versions that the two environments got into `docs/DECISIONS.md`
  (`uv pip freeze --python <the interpreter of the environment>`).

Then three commands for each experiment file: write the job, run the tool, measure. The examples
are for RF-DETR Small and for PnLCalib. The device `mps` is the GPU of an Apple Silicon machine.

```bash
mkdir -p ~/pitchvlm-data/jobs

python -m pitchvlm.cli tool-job --experiment configs/experiments/people_probe_rfdetr_small.yaml \
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/people_probe_rfdetr_small.json
~/pitchvlm-data/envs/rfdetr/bin/python tools/run_tool.py rfdetr --device mps \
    --job ~/pitchvlm-data/jobs/people_probe_rfdetr_small.json --root ~/pitchvlm-data \
    --out ../pitchvlm-results/results/tools/people_probe_rfdetr_small.json
python -m pitchvlm.cli tool-score --experiment configs/experiments/people_probe_rfdetr_small.yaml \
    --root ~/pitchvlm-data --run ../pitchvlm-results/results/tools/people_probe_rfdetr_small.json \
    --out ../pitchvlm-results/results/metrics/people_probe_rfdetr_small.json

python -m pitchvlm.cli tool-job --experiment configs/experiments/pitch_probe_pnlcalib.yaml \
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/pitch_probe_pnlcalib.json
~/pitchvlm-data/envs/pnlcalib/bin/python tools/run_tool.py pnlcalib --device mps \
    --job ~/pitchvlm-data/jobs/pitch_probe_pnlcalib.json --root ~/pitchvlm-data \
    --repo ~/pitchvlm-data/PnLCalib \
    --weights-kp ~/pitchvlm-data/weights/pnlcalib/SV_kp \
    --weights-lines ~/pitchvlm-data/weights/pnlcalib/SV_lines \
    --out ../pitchvlm-results/results/tools/pitch_probe_pnlcalib.json
python -m pitchvlm.cli tool-score --experiment configs/experiments/pitch_probe_pnlcalib.yaml \
    --root ~/pitchvlm-data --run ../pitchvlm-results/results/tools/pitch_probe_pnlcalib.json \
    --out ../pitchvlm-results/results/metrics/pitch_probe_pnlcalib.json
```

- First run: add `--max-clips 2` to `tool-job` and to `tool-score`, and do the first check of
  section 8 (row "Tool script") before the full run. The script never ran with a real tool.
- The frames are those of the model probe: the tool file and the probe file have the same group,
  frames per clip and seed. Give `tool-job`, `tool-score` and the model probe the same
  `--max-clips` when you compare them.
- The metrics file has the keys of the model probe without the values that only a model step has
  (`prompts`, unparsed answers, the values of the landmark fit). Compare it with the same gate.
- `seconds_per_frame` is the time of the tool for one frame on this machine, without the reading
  of the image file. For RF-DETR it is the time of the plain `predict` of the tool. The tool also
  has a compiled predictor, which its pages give as about twice as fast on an M4 Max (11.1 ms
  against 20.0 ms for the Small model): that value is published, not measured here. The paper of
  PnLCalib gives 439 ms per frame on an NVIDIA GPU for the settings of this experiment file.
- `tool-score` refuses a run file that was made for another experiment name or with other tool
  settings, and a run file that has no answer for a frame of the probe. After a change of the
  experiment file, write the job again and run the tool again.
- `tool-score` refuses a work tree with changes, like `probe`.
- Pitch: read `axis_check` first. It holds the median position error for the matrices as they
  are (`as_given`) and with the sign of x, of y or of both changed. `as_given` must be the
  smallest of the four. When it is not, the script gives the matrices in another pitch
  convention: that is a fault of `tools/run_tool.py`, and the other values say nothing about
  the tool.
- Pitch: PnLCalib gives the best camera that it finds and never refuses one. So its
  `valid_share` is high by the way the tool works, and a weak camera shows in the error values
  and in `plausible_share`. Read these next to `valid_share`. The landmark fit of the model step
  refuses a weak fit, so its `valid_share` is lower for the same reason.
- Two engines on the same frames: `per_clip` of each metrics file holds the value of every clip.
  Take the difference clip by clip, as the plan says for every comparison of two settings.
- People: a detector that was trained on everyday pictures can also find people in the stands.
  They count as wrong boxes, as they do for the model step. The run file keeps every detection
  above the score limit with its class name, so another count can be made later without a new
  run of the tool.
- RF-DETR makes a frame of 1920 pixels about 4 times narrower for its input. Since its version
  1.9 it does that without antialiasing, and its weights are older than that version. No page
  of the tool says which resize fits the weights. So each model class has two experiment files:
  one with the resize that the tool uses by itself (`antialias: false`) and one with
  antialiasing (`antialias: true`). Run both. The setting that is taken later is confirmed on
  fresh frames, as the plan says for every chosen setting.
- Another setting: copy the experiment file, change `name` and one value under `tool`, commit,
  and run the three commands again. For RF-DETR the values are the model class (`RFDETRNano`,
  `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`), the score limit `min_score` (0.5 is the
  threshold that the tool uses by itself) and `antialias`. For PnLCalib the two thresholds of
  the file are those of the tool's script `inference.py`; its benchmark script for main-camera
  pictures uses 0.0966 for the keypoints and 0.3441 for the lines.
- No source says that the GPU of an Apple Silicon machine gives the same values as the CPU for
  these tools: the pages of RF-DETR give weak evidence that it does, and PnLCalib was written
  for NVIDIA GPUs. Run the first two clips also with `--device cpu` and another `--out` file, and
  compare the two metrics files before the full run.
- The run files are text results: boxes that the detector found and camera matrices, no pixels
  and no label data. They go to the results branch with `sync` (section 6). The job files stay
  in the data folder.
- Licences: the code of RF-DETR and the weights of its four model classes above are under
  Apache-2.0. The code of PnLCalib is under GPL-2.0-or-later and its weights under CC BY 4.0
  (name the paper of the tool when the weights are used). `tools/run_tool.py` imports code of
  the PnLCalib clone when it runs. Check what that means before this repository is published
  (`docs/PLAN.md`, risks).

## 4 First video

```bash
# Free dry run of a dev clip: it prints the frames, the frame rate, the keyframes and the cost
# estimate. Then it stops, because no answer is stored yet.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --out ~/pitchvlm-data/runs/dev --replay

# Paid run: the same command without --replay. About 2.3 USD for a clip of 30 seconds.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --out ~/pitchvlm-data/runs/dev

# The demo clip (a video file) with a roster: first the dry run, then the paid run.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/demo/<clip>.mp4 \
    --roster ~/pitchvlm-data/demo/roster.yaml --out ~/pitchvlm-data/runs/demo --replay
python -m pitchvlm.cli run --clip ~/pitchvlm-data/demo/<clip>.mp4 \
    --roster ~/pitchvlm-data/demo/roster.yaml --out ~/pitchvlm-data/runs/demo
```

- Take `<clip>` from the group `dev` of `configs/splits.json`. Do not use a test clip.
- Output of a clip: `<out>/<clip>/<clip>.mp4` (video and radar) and `<out>/<clip>/result.json`.
  With `--labels`: `<out>/predictions/SoccerNetGS-<split>/<tracker>/<clip>.json` for the scorer.
- With `--labels` the command prints the number of scored frames without a prediction and
  writes it into `result.json` (`prediction_file`). Such a frame is not a format problem: the
  scorer counts its people as missed. The exit code is 1 only when no scored frame has a
  prediction (without `--labels`: when no frame holds a person). The line `failed: ...` then
  names the usual cause, a clip without a pitch mapping. The files are written also then, so
  the scorer counts the clip as missed. A clip without any pitch mapping sends no shirt-number
  request.
- The cost estimate of a clip assumes 50 number requests. This is a guess from the synthetic
  clip (48 to 50 requests in 30 seconds); the real number is in `result.json` under
  `provenance.cost.requests`. With the default token estimates a clip of 30 seconds is estimated
  at about 2.3 USD.
- A run that stops because of the network, or that is interrupted (Ctrl-C, a plain kill, a
  closed terminal), keeps the answers that it received in the cache. Start it again: it goes
  on. The model calls in flight finish first, so a command can take up to the client timeout
  (300 s, flex 900 s) to end after Ctrl-C, a plain kill or a closed terminal; a second signal
  does not shorten this. A request that waits for a new attempt is not sent again. One dropped
  connection does not stop a run, because the call is tried again. When a call ends without an
  answer that can be read (a timeout, a connection that stays down, or an answer that the SDK
  cannot read), the error line and the log say that the call may still be billed. The ledger
  has no line for such a call: compare the ledger with the bill of the provider.
- After a budget stop, do not just start again: the stored answers count against the limit of
  the next run, so it pays for no further answer. Read the mean tokens in the lines after
  "measured cost", correct `DEFAULT_TOKEN_ESTIMATES` with the owner's go-ahead, commit, and then
  start again.
- The stored answers of a clip belong to its pixels. When a stored answer was recorded with
  other image bytes than the clip shows now, the command stops with an error before any model
  call. For new footage, give the clip another file or folder name. For the same footage (an
  update of the image library, a kit legend that was asked from other frames, a video decoder
  that does not give the same pixels each time), start the command again with
  `--allow-other-images`.
- A process that is killed hard (`kill -9`) can leave a temporary file in the folder of a
  stored answer, `<cache>/<clip>/<step>`. Its name is
  `.<32 hex digits>.json.<process id>.<thread id>.tmp`. `sync` names such a file and refuses
  the folder: delete the file.
- Other settings: copy `configs/default.yaml`, change the copy and pass it with `--config`. A key
  that is left out keeps its default. Three tracking settings to know: `tracking.stitch_speed_mps`
  (4.0) is the speed that limits the distance between two fragments that are joined;
  `tracking.min_observations` drops short tracks before fragments are joined;
  `tracking.mismatch_penalty` is added when role or team key differ from the vote of the track,
  once more for every keyframe that the track missed.
- A video file is decoded while it is read, so a long clip does not need more memory. The decoder
  runs forward only: a read of an earlier frame starts again at the first frame. A folder of
  frame images with `--fps` does not have this cost.
- Roster file (`left` is the team that defends the left goal as the main camera sees it):

```yaml
left:
  name: Home FC
  players:
    1: First Keeper
    10: First Playmaker
right:
  name: Away United
  players:
    9: Second Striker
```

- A shirt number, a side or a team name that is written twice in the roster file is an error
  that names the key and, for YAML, the line. The file must be UTF-8 text.

Money:

- Run one paid command at a time. The limit of 8 USD per ten minutes is kept inside one
  command; two commands at the same time, or a new start right after a stop, do not see the
  spend of each other. Two commands that work on the same clip at the same time also pay for
  the same answers twice; the log then has a warning, and the ledger has two lines with the
  same `request`.
- Use one cache folder for all paid runs. The hard stop counts what the ledger of the cache
  folder of the command holds: a command with another `--cache` folder starts at 0 USD. The
  budget line of every paid command prints the path of its ledger and what it holds.
- A paid command refuses to start when the cost ledger plus its estimate is above the hard
  stop. The option `--start-over-budget` starts it anyway, for example when most answers are
  stored already; the run still stops at the hard stop. The budget line names the planned
  budget of 120 USD, and a warning line is printed when the ledger is above it.
- A budget stop comes after the answer that crosses a limit, and the requests in flight at that
  time are paid too. So a run can pass 1.3 times its estimate, and the ledger the hard stop, by
  the cost of `max_parallel - 1` answers. `max_parallel` is 4 by default and at most 8, so this
  is at most 7 answers.
- One run has another limit than 1.3 times its estimate: the contract checks of the doctor stop
  above the 2 USD of gate G0 (section 2).
- The run limit also counts the number requests above the 50 of the estimate. Computed with the
  default token estimates, not measured: a clip with 64 keyframes reaches its limit at the
  129th number request. When `provenance.cost.requests` of a result shows many more than 50
  number requests, the number of requests is the thing to correct
  (`EXPECTED_NUMBER_REQUESTS` in `src/pitchvlm/cli.py`, with the owner's go-ahead), not the
  token estimates.
- The cost ledger must be a complete series of files (`ledger.jsonl`, `ledger-0002.jsonl`, ...
  without a gap). When a file of the series is missing, or a line has a negative cost, every
  paid command stops before its first model call. The error names the files that it found, or
  the file and the line. Bring the missing file back from the branch `results` before the next
  paid command.
- No request limits its output tokens, but one call is one request. An answer that stops at
  the token cap of one request (32,768 tokens of thinking and output on `gemini-4-argon`) is
  not continued: it is recorded as a billed answer that cannot be used (about 0.33 USD at the
  November price, 0.66 USD from 1 December 2026), and the run stops. So this is the largest
  cost of one call on that model. Section 8 says how the cost pilot settles a limit.

## 5 Scoring

```bash
python -m pitchvlm.cli eval --root ~/pitchvlm-data --pred ~/pitchvlm-data/runs/dev/predictions \
    --split valid --tracker pitchvlm --clips <clip> <clip> \
    --scorer-python ~/pitchvlm-data/envs/scorer/bin/python \
    --scorer-script ~/pitchvlm-data/sn-trackeval/scripts/run_soccernet_gs.py \
    --out ~/pitchvlm-data/eval/dev-pass-1
cp ~/pitchvlm-data/eval/dev-pass-1/scores.json ../pitchvlm-results/results/metrics/scores_dev_pass_1.json
```

- The command prints GS-HOTA, DetA and AssA for each clip and for all clips together. The file
  `scores.json` records the commit of the scorer clone (`provenance.scorer_commit`). The commit
  is found also when the path of the scorer script is typed in another letter case.
- The evaluator is the code of another project. The commands `oracle` and `eval` start it
  without the variable `GEMINI_API_KEY` in its environment.
- Use a new, empty `--out` folder for each scoring run.
- The 10 test clips are run and scored one time only, after the freeze tag. Before that paid run,
  download the test split (`data --split test` with the same `--manifest` and `--seed`) and repeat
  the free label stage on it (`oracle --split test`).

## 6 Sending text results back

Only text goes back: metrics files, stored model answers and the cost ledger. They live in
`../pitchvlm-results/results`. Videos, images, label files, `result.json` and prediction files
stay in `~/pitchvlm-data`.

```bash
python -m pitchvlm.cli sync --results ../pitchvlm-results/results
```

The command lists the files, checks them with `tools/guard.py` and with the secret scan of the
git hooks, and prints three git commands (`add`, `commit`, `push`). It runs none of them. The
owner runs them and decides about the push. The command refuses the folder when it holds
anything but `.json` and `.jsonl` files, and when the guard or the secret scan finds a problem.
On the build machine, fetch the branch `results` and read the files there.

- The secret scan of `sync` is `gitleaks` with the file `.gitleaks.toml` of the work tree
  `../pitchvlm-results`, as the commit hook of that work tree runs it. When that file is
  missing, the command stops and names the git command that brings it. Without `gitleaks` the
  command says that the scan did not run.
- The guard refuses a label file and the result file of a clip run by name
  (`Labels-GameState.json`, `result.json`). It also reads every `.json` and `.jsonl` file, also a
  small one: label data, prediction data, the result of a clip run and records of the label
  format (a box with a `track_id`) are refused at any depth of the file, and when each line of
  the file is one record. Its report line has the rule `dataset`. The rule reads no other file
  type; `sync` accepts no other file type.
- The cost ledger is `ledger.jsonl` and, when that file holds 1,000,000 bytes, `ledger-0002.jsonl`,
  `ledger-0003.jsonl` and so on, all in the cache folder. So every ledger file stays below the
  size limit of the guard (2 MB for every file). A line has about 370 bytes: it holds one new
  answer and names its repeat index, the cache key of its request (`request`) and the number of
  attempts. Two lines with the same `request` are two payments for the same question.
- The hooks in `../pitchvlm-results` run the guard and the settings files of the branch `results`,
  not those of the code branch. After a change of `tools/`, `.gitleaks.toml`, `.gitignore` or
  `.pre-commit-config.yaml` on the code branch, merge the code branch into `results` before the
  next commit of results: `git -C ../pitchvlm-results merge <code branch>`.
- When the secret scan stops a commit, it prints the file and the line of each finding. The
  secret itself is not printed.

## 7 Rules

1. No secret in the repository: no API key, no token, no `.env` file.
2. No pixels and no dataset files in the repository; the branch `results` holds text only, with no box, no pitch position and no role, team or shirt number from a label file.
3. No machine-specific text in tracked files, and nothing about the owner or the policies of a machine; no lock file is committed, the pinned list is in `locks/`.
4. Only `src/pitchvlm/vlm/client.py` imports the model SDK.
5. Model calls: paid tier only, `store=False`, no sampling parameters; players are identified by shirt number and roster, never by face.
6. Money: estimate first, measured cost after; a run stops at 1.3 times its estimate (the contract checks of `doctor`: at the 2 USD of gate G0); hard stop at 135 USD; no paid run without the owner's go-ahead.
7. Gates are fixed in `configs/gates.yaml` before a run; the 10 test clips are run once, after the freeze tag.
8. A specialist model replaces a Gemini step only after that step failed its gate and the specialist passed the same gate on the same frames.
9. The guard and the secret scan run before `git commit` and before a push that sends commits; `--no-verify`, the variable `SKIP` of the hook runner and another hook folder (`core.hooksPath`) are not allowed. Before a push that sends only names (a tag or a new branch), run `python3 tools/guard.py --push` by hand. The deny-list `.git/guard-deny.txt` is never committed (section 9).
10. Commit only when the owner asks; never push without being asked; a commit message has no co-author line.

## 8 What was never tested before the run machine

The build machine has no key, no data and no scorer. So these parts run for the first time here.
Check each one with its first check before you rely on it.

| Part | File | First check | When it fails |
|---|---|---|---|
| Offline tests on this machine | `tests/` | `python -m pytest -rs` after the setup (section 1): every test passes, and the skipped tests are listed with their reason. The tests ran on one machine only, with Python 3.13. Some tests need a disk that tells upper and lower case apart, and some a disk that does not: one of the two groups never ran | A skipped test names a missing tool (`git`, `gitleaks`, `pre-commit`) or the kind of disk: install the tool. A failed test names the code that this machine runs in another way: repair the code, not the test. |
| Setup commands | section 1 | Each command of section 1 ends without an error, and `doctor --no-api` shows the SDK as a version 3.x. Most commands of section 1 never ran: the install with the extra `gemini`, the pinned list, the work tree and the scorer | Read the error of the command. Fact 19 below is about the package age limit of `uv pip`. |
| Gemini client | `src/pitchvlm/vlm/client.py` | `doctor` with one model (step 2 of day 1). The cost estimate and the budget line must be on the screen before the first check line of a model | Read the failed check and the report file. A failed contract check (`boxes`, `landmarks`, `store flag`, `token sum`, `served tier`, `flex tier`) is a client fault: repair the client before any footage is sent. A value of `box accuracy` or `landmark accuracy` that does not reach the value of the gate file is not a client fault. A line `[info] flex tier: not checked` is no client fault either: the flex tier was busy (section 2). When the check `boxes` fails, read the answers under `contract.<model>.answers` in the report file first: a model that names no person for the coloured rectangles of the synthetic frames fails this check without a client fault. |
| Download | `src/pitchvlm/datasets.py` | `data --split valid`; then `ls ~/pitchvlm-data/SoccerNetGS-valid` shows about 58 clip folders | Run the command again: it continues the download. An archive whose `.part` file in `~/pitchvlm-data/archives` already holds every byte gets its final name at the next start: the server answers the range with the status 416 and the size of the archive, and the command compares that size with the file. When the sizes differ, or the answer names no size, the command stops with an error that names the `.part` file: compare its size with the size on the dataset page, or delete it to start again. When the host refuses the request or its certificate (the download itself was never run), download `valid.zip` by hand into `~/pitchvlm-data/archives` and run the command again: it then only extracts. |
| Label reader on real files | `src/pitchvlm/labels.py` | `oracle` with one clip: it reads the label file first | The error names the file and the place in it. Repair the reader, not the label file (check 4 below). |
| Scorer wrapper and its output layout | `src/pitchvlm/evalx/scorer.py` | `oracle` with one clip: the ground truth must score at least 99.5. The wrapper gives the evaluator `<root>/SoccerNetGS-<split>` as the label folder and `<predictions>/SoccerNetGS-<split>` as the prediction folder, with `--SKIP_SPLIT_FOL True`. The source of the evaluator reads no file `seqinfo.ini` for this data; this run confirms that `--SEQ_INFO` with clip names only is enough | The error shows the last lines of the scorer. A line `GT file not found` or `Tracker file not found` names the path that the evaluator looked for: compare it with the place of the file. When the lines name a missing `seqinfo.ini`, the evaluator needs the clip lengths: repair the wrapper, do not write such files into the dataset. The evaluator writes its files into the folder of the prediction files of the tracker: three `*_detailed.csv` files, and a summary file and two plot files for each of them. For `oracle` this folder is `<out>/predictions/SoccerNetGS-<split>/<tracker>/`, and the command moves the `*_detailed.csv` files from there to `<out>/scorer/<tracker>/`. For `eval` it is `<pred>/SoccerNetGS-<split>/<tracker>/`, and the files move to `<out>/`. The scores come from `person_detailed.csv`. When the command finds no such file, look where the evaluator wrote its files. The wrapper leaves the plots of the evaluator on, and the evaluator draws a plot before it writes the detailed file of a class: an error of the plot library in the last lines also leaves no detailed file. The evaluator was never imported under Python 3.12 with the packages that the setup resolves. |
| Video encoding | `src/pitchvlm/render.py` | `python -m pitchvlm.cli demo --out outputs/demo`, then open `outputs/demo/demo.mp4` | Check the OpenCV build (codec `mp4v`). |
| Video decoding | `src/pitchvlm/frames.py` | The free dry run of the demo clip (section 4) prints the frames, the frame rate and the keyframes: frames divided by the rate must be the length of the clip in seconds, and the clip must have about 2.1 keyframes per second at any frame rate. Only synthetic clips of 25 frames per second ran before | A video with a variable frame rate, rotation data or interlaced frames was never read here. Convert it to a constant frame rate, or extract its frames into a folder and use `--fps`. |
| Roster file | `src/pitchvlm/roster.py` | The same dry run with `--roster` reads the roster file before any model call. After the paid run, the video shows a name next to every displayed number of the roster | The error names the key and, for YAML, the line (section 4). |
| Tool script | `tools/run_tool.py`, `src/pitchvlm/fallbacks/` | After the install of the two tools (section 3): the three commands of each tool with `--max-clips 2`. The script was written from the source of the tools (`rfdetr` 1.12.0 and 1.12.1, PnLCalib at one commit) and never ran with them: the tests replace each tool by a stand-in. Its image reader needs Pillow and ran nowhere. RF-DETR: the recall is far above 0, and the largest box values of the run file are near the width and the height of the frames. PnLCalib: `axis_check.as_given` is the smallest of its four values, `valid_share` is above 0, and the payloads of the run file show a camera on the near side above the grass (`camera_position_m`: y above 34, z below 0). Then compare the device `mps` with `cpu` on the same two clips | An `error:` line about a version or a commit: install the version that the line names. A traceback inside a tool: the line before it names the frame on which the tool stopped. When the first frame fails, the tool does not have the interface that the script expects: read the source of the installed version, repair `tools/run_tool.py` with a test, and do not change the tool. When one later frame fails, the tool has an error for that picture: tell the owner; the script counts no frame as a frame without an answer because of an error. An error of `torch.load` about the weights of PnLCalib: the file is not plain weights. Do not switch that check off without the owner: a file that is not plain weights can run code when it is loaded. A recall near 0 with many boxes: the boxes are not in pixels of the frame, or the class names are not those of the tool (read `label` in the run file). `axis_check` with another smallest value: the matrix is in another pitch convention; repair `camera_homography` of the script. Values that differ between `mps` and `cpu`: tell the owner, and use `cpu` for the measurement. A `ValueError` of an adapter names the key or the value of a payload that does not follow its format. |
| Git hooks | `.pre-commit-config.yaml` | `doctor --no-api` shows `[ok  ] guard hooks`. The first commit that the owner asks for must list the guard and the secret scan as passed, and the first push the two push hooks. `git -C ../pitchvlm-results rev-parse --git-path hooks` must name the hook folder of this clone: the results work tree then runs the same hooks | `pre-commit install --hook-type pre-commit --hook-type pre-push`. The hooks ran on the build machine in scratch clones only, never in this repository, never against a real remote and never with signed commits. A results work tree that names another hook folder has its own `core.hooksPath`: remove that setting. |

The cost pilot is also the first use of the stored answers and of the ledger with real probe
requests. After it, check that the ledger files in `../pitchvlm-results/results/cache` have one
line for each new answer, and that a second run with `--replay` gives the same metrics for free.
Then run `sync` (section 6): it is the first time that the guard and the secret scan read real
model answers. A report line names the rule and the file. A false alarm on the text of an
answer needs a change of the rule with a test, not a change of the answer.

Facts that the review could not settle. No public source answers them, or the build machine
could not run the check. Each row names the command or the observation that settles it here.

| # | Open fact | First check | What the result decides |
|---|---|---|---|
| 1 | Does the field `service_tier` of an answer body name the tier that served the call? The provider documents only the response header `x-gemini-service-tier`. The client and the check `served tier` of the doctor read the body field | Paid, three small calls, with the owner's go-ahead: send one text prompt with `store=False` to a model that has a flex tier through `client.interactions.with_raw_response.create`, once with `service_tier` flex, once with standard and once without a tier. Print the response header `x-gemini-service-tier` and the field `service_tier` of the parsed answer of each call | The body field reports the serving tier only when it is filled without a requested tier and equals the header in all three calls. Else the check `served tier` proves nothing: make `doctor.RecordingClient` read the header, or say in the check that it compares the body field with the request. |
| 2 | How often is the team-side rule right on real clips? The paper measures the mean-x rule at GS-HOTA 92.00 on the test split with ground truth for the other modules, so the gate of 0.97 can fail although the code is right. The published baseline scores 18.05 on the validation split and 22.26 on the test split | Free: `oracle --split valid` on all clips (step 4 of day 1). Read `team_side_rule` in `oracle.json`: `share_right`, `share_right_by_players_alone`, `clips_wrong`, `clips_decided_by_goalkeepers`, and `player_mean_gap_m` and the goalkeeper counts under `facts` | A share below 0.97 while the ground truth scores at least 99.5: the rule misses the gate, and the code is right. Write the measured shares into `docs/DECISIONS.md`. Then the owner chooses a lower threshold or other limits of the rule (`CLEAR_TEAM_GAP_M` and `GOAL_ZONE_MIN_X_M` in `src/pitchvlm/tracking.py`). Compare the dev score of gate G6 with the validation baseline 18.05, not with 22.26. |
| 3 | Do keyframes without a pitch position exist in real clips? The resampled oracle variant fills a track only between two keyframes that both hold it. One keyframe without labelled positions therefore removes up to 22 frames (measured on a synthetic clip), and the variant is then below the ceiling of the keyframe design | Free: the script "Keyframes without a labelled person" below, on every clip of the split | Zero on all clips: the resampled value is the ceiling, and nothing changes. Else the resampled GS-HOTA is too low by about the share of scored frames that the script prints: move a keyframe that is not scored to the nearest scored frame in `cli.oracle_variants` before you judge the gate of 92. |
| 4 | Do real label files hold values that the reader refuses and the scorer ignores or accepts (a box without area, `lines` null, two pitch annotations of one image, a line point without y, a shirt number that is not digits on a goalkeeper or a referee, a player number above 99, the team text `nan`)? Do they start with a byte order mark? Do they hold a shirt number with a leading zero, which the reader turns into another text? | Free: `oracle --split valid` on all clips. The reader stops at the first such value and names the file and the place. A shirt number with a leading zero shows as a GS-HOTA of the ground truth below 99.5 | No stop and a ground truth of at least 99.5: the reader is right as it is. A stop: change only the rule that the real value breaks, with a test. Keep the shirt number of goalkeepers: `gsr_frame_people` gives it for every role. |
| 5 | Which classes does the SDK raise for a lost connection and for a client timeout? The client knows them by the names `APIConnectionError` and `APITimeoutError` | Free: `grep -rn -e "class APIConnectionError" -e "class APITimeoutError" "$(python -c 'import google.genai, os; print(os.path.dirname(google.genai.__file__))')"` prints one line for each class | No line, or a network failure that ends a call with "status None" and without the words "may still be billed": correct `SDK_CONNECTION_ERROR` and `SDK_TIMEOUT_ERROR` in `src/pitchvlm/vlm/client.py`. |
| 6 | What does the API answer to a thinking level or a service tier that a model does not have? The tables `THINKING_LEVELS` (`src/pitchvlm/config.py`) and `FLEX_TIER_MODELS` (`src/pitchvlm/vlm/cost.py`) come from the documentation pages, not from a call | Paid, about 0.1 USD for each model: `doctor --models` with every model that a probe or a run will use. The doctor sends the level `low`; another level is checked by the first request of the probe that uses it | A call that the API refuses with an error about the thinking level, the resolution or the service tier costs nothing: correct the table, with a test. |
| 7 | How many output tokens does a real answer need, and does a limit on them count the thinking tokens? No request limits its output tokens. An answer that reaches 32,768 tokens of thinking and output on `gemini-4-argon` is not continued: it is billed, it cannot be used, and it stops the run | After the cost pilot, free: the script "Largest answers" below prints the largest output plus thinking tokens of one answer for each step. A stop with the words `requires_continuation` in the error line is such an answer | The owner decides the limit. The review proposed a field `max_output_tokens` of a request (not a part of the cache key) at 8 times the output estimate of its step, and a doctor check that sends one request with a small limit and reports the status and the token counts of the cut answer. |
| 8 | Does the installed `gitleaks` read the rules file? `.gitleaks.toml` uses an allowlist form that was checked with gitleaks 8.30.1 only | Free: `gitleaks version`, then `python -m pytest tests -k "secret_scan" -rs`: every test passes and none is skipped | A skipped or failed test: install a gitleaks version from 8.30.1 on. Without it the commit hook refuses every commit of stored answers. |
| 9 | In which time zone do the prices change on 1 December 2026 and on 1 January 2027? The code prices an answer by its UTC day | Compare the `usd` of the first ledger lines after 00:00 UTC of 1 December with the bill of the provider | Another time zone: the ledger counts a few answers at the wrong price. The plan ends the standard-price runs on 30 November for this reason. |
| 10 | Does a broadcast graphic (a scoreboard) disturb the measured camera motion? Measured on a synthetic clip of 1920 x 1080 pixels with a static overlay: the 90th percentile of the motion error over 12 frames rises from 0.5 to 1.3 pixels with a rich grass texture and from 3 to 14 pixels with a poor one | After the first dev clip with labels, free: the script "Position error at and between the keyframes" below | A much larger error between the keyframes than at them: the camera motion is off. Mask the graphic in `motion.compute_motion_field` (it masks only the people of the keyframes), with a test on synthetic frames. |
| 11 | Does the real key have the form that the guard and the secret scan expect: the prefix `AQ.` and then 20 or more letters, digits, `_` or `-`? No public source gives the form | Free: the command "Form of the key" below. It prints no part of the key | "the pattern does not fit": tell the owner. The pattern in `tools/guard.py` (`NEW_GOOGLE_KEY`) and in `.gitleaks.toml` must change before any text result is committed. |
| 12 | How many landmark fits does the share rule refuse on real answers? A fit needs 40% of its landmarks within 3 times the RANSAC threshold. Measured on synthetic cameras only | After the pitch probe, free: compare `valid_share` with `valid_share_without_share_rule` in its metrics file, for all frames and under `by_view` | A difference of some percent of the frames: the rule refuses fits on real answers. Give the two shares to the owner before gate G3 is judged; the rule is `MIN_NEAR_SHARE` and `NEAR_LIMIT_FACTOR` in `src/pitchvlm/geometry.py`. |
| 13 | Does a real model box one person two times with a small shift? A second box with the same role and team key is dropped above an IoU of 0.6; the 0.6 is an estimate | After the people probe, free: the script "Boxes that nearly repeat each other" below reads the stored people answers and the labels | Pairs on one labelled person below an IoU of 0.6, or pairs on two labelled people above it: the owner changes `REPEATED_BOX_SAME_LABELS_MIN_IOU` in `src/pitchvlm/pipeline.py` before the first dev clip. No pair at all: the limit stays. |
| 14 | Do two crops of one person that are less than one second apart agree on a wrong number? A track that is seen for a short time gets two such crops, and two crops that agree show the number in the video and link fragments. Does a real model read a number of two digits in parts? The fake model reads every crop on its own and never in parts | After the number probe, free: the script "Short tracks and numbers in parts" below reads the stored number answers and the labels | Many short tracks whose two crops agree on a wrong number: the owner decides whether a short track may display a number (`pipeline.crop_choice`). Readings in parts are counted for the export number (`tracking.decide_number`); no such reading means that this rule is not used. |
| 15 | Is the mapping error small against the 6 m that tell two people with one shirt number apart (`TWO_PEOPLE_MARGIN_M` in `src/pitchvlm/tracking.py`)? The 6 m come from synthetic clips with 10 pixels of landmark noise | After the pitch probe, free: read `error_m.p90` in its metrics file | A 90th percentile above 3 m (an estimate: two errors of 3 m in opposite directions are 6 m): two positions of one person can be more than 6 m apart, and the link by the shirt number is refused too often. The owner chooses a larger margin. |
| 16 | Does the provider bill a call that the client gave up (an interruption, a client timeout, a lost connection)? The ledger has no line for such a call | After the cost pilot: compare the total of the ledger with the bill of the provider for the same days. Every such call has a log line with the words "may still be billed" | A bill above the ledger: write the difference into `docs/DECISIONS.md`. The hard stop counts the ledger, so plan the difference into the budget. |
| 17 | How often is the flex tier busy? A flex request waits up to 900 seconds for capacity, and then the command stops | The line `flex tier` of the doctor, and the log of the first run through the flex tier: every wait is a line "The flex tier has no capacity" | Many waits: a flex run takes much longer than a standard run. A stop with "the flex tier had no capacity": start the command again later. |
| 18 | Which rate limits does the project have for each model (requests and tokens per minute)? The public page sends the reader to the console of the project; a clip sends about 180 requests, 4 at the same time | The owner reads the limits on the console page of the project. In the first paid run, a warning with "status 429" for a standard request is a hit of a limit | Hits of a limit: lower `max_parallel` in a copy of `configs/default.yaml` (it is at most 8). |
| 19 | Does `uv pip` read the package age limit of `pyproject.toml` (`[tool.uv]`, `exclude-newer`)? The page of the tool did not say | Free, at the setup: run the `uv pip compile` command of section 1 a second time with the option `--exclude-newer` and the date of 7 days ago, into another file, and compare the two lists | The lists differ: `uv pip` does not read the setting. Give the option on the command line of `uv pip install` and `uv pip compile`, and write that into section 1. |
| 20 | Does the package run under Python 3.12? `pyproject.toml` allows it, and only Python 3.13 ran the tests | Free and not needed for the plan: `uv venv --python 3.12 ~/pitchvlm-data/envs/py312`, `uv pip install --python ~/pitchvlm-data/envs/py312/bin/python -e ".[dev]"`, then `~/pitchvlm-data/envs/py312/bin/python -m pytest` | A failed test: the owner sets `requires-python` to 3.13 or repairs the code. |
| 21 | How do a merged track, a missing state and a state at a wrong place trade off in GS-HOTA? The bench of the reviews measured the share of a true track under one id, the id switches and the tracks per person, never GS-HOTA | After the first scored dev clip, free: compare DetA and AssA of `eval` with the position errors and the identity values that the script "Position error at and between the keyframes" prints | It decides which rule to change first when gate G4 or G6 is missed: a low AssA with a good DetA points at the tracks, the opposite at the people step and the pitch mapping. |

Form of the key (fact 11):

```bash
printf %s "$GEMINI_API_KEY" | LC_ALL=C grep -Eqx 'AQ\.[0-9A-Za-z_-]{20,}' \
    && echo "the pattern fits" || echo "the pattern does not fit"
```

Keyframes without a labelled person (fact 3):

```python
from pathlib import Path

from pitchvlm.frames import keyframe_indices
from pitchvlm.labels import gsr_frame_states, load_gsr_labels

for label_file in sorted(Path("~/pitchvlm-data/SoccerNetGS-valid").expanduser().glob("*/Labels-GameState.json")):
    labels = load_gsr_labels(label_file)
    truth = gsr_frame_states(labels)
    keys = keyframe_indices(len(labels.frames), labels.fps, 0.48)
    empty = [position for position, frame in enumerate(keys) if not truth[frame].players]
    lost: set[int] = set()
    for position in empty:
        first, last = keys[max(position - 1, 0)], keys[min(position + 1, len(keys) - 1)]
        lost.update(frame for frame in range(first + 1, last) if labels.frames[frame].scored)
    scored = sum(frame.scored for frame in labels.frames)
    not_scored = sum(not labels.frames[keys[position]].scored for position in empty)
    print(f"{labels.clip_id}: {len(empty)} keyframes without a person ({not_scored} of them not scored), "
          f"{len(lost)} of {scored} scored frames lose their prediction")
```

Largest answers (fact 7):

```python
import json
from pathlib import Path

largest: dict[str, int] = {}
for ledger in sorted(Path("../pitchvlm-results/results/cache").glob("ledger*.jsonl")):
    for line in ledger.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        tokens = entry["output_tokens"] + entry["thought_tokens"]
        largest[entry["step"]] = max(largest.get(entry["step"], 0), tokens)
print(largest)
```

Position error at and between the keyframes (fact 10), and the identity values of the clip
(fact 21). Put the paths of the run into the first two lines:

```python
import json
from pathlib import Path

from pitchvlm.evalx.metrics import identity_scores, position_errors_m, summarize
from pitchvlm.frames import keyframe_indices
from pitchvlm.labels import gsr_frame_states, load_gsr_labels
from pitchvlm.schemas import ClipResult

result_file = Path("~/pitchvlm-data/runs/dev/<clip>/result.json").expanduser()
label_file = Path("~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json").expanduser()
document = json.loads(result_file.read_text(encoding="utf-8"))
result = ClipResult.model_validate(document["result"])
truth = gsr_frame_states(load_gsr_labels(label_file))
interval_s = document["provenance"]["config"]["keyframe_interval_s"]
keys = set(keyframe_indices(result.info.n_frames, result.info.fps, interval_s))
for name, wanted in (("at the keyframes", True), ("between the keyframes", False)):
    predicted = [state for state in result.frames if (state.frame in keys) is wanted]
    labelled = [state for state in truth if (state.frame in keys) is wanted]
    print(name, summarize(position_errors_m(predicted, labelled)))
# The share of a true track under one id, the id switches and the predicted tracks per true track.
print(identity_scores(result.frames, truth))
```

Boxes that nearly repeat each other (fact 13). It reads the people answers of the cache and
the label files:

```python
import json
from itertools import combinations
from pathlib import Path

from pitchvlm.labels import gsr_frame_people, load_gsr_labels
from pitchvlm.vlm.coords import box_from_model

data = Path("~/pitchvlm-data/SoccerNetGS-valid").expanduser()
cache = Path("../pitchvlm-results/results/cache")
n_answers = n_pairs = n_one_person = 0
for cassette in sorted(cache.glob("*/people/*.json")):
    stored = json.loads(cassette.read_text(encoding="utf-8"))
    clip, frame = stored["meta"]["clip_id"], stored["meta"]["frame"]
    labels = load_gsr_labels(data / clip / "Labels-GameState.json")
    labelled = [person.box for person in gsr_frame_people(labels, frame).people]
    people = json.loads(stored["response"]["text"])["people"]
    boxes = [box_from_model(person["box_2d"], labels.width, labels.height) for person in people]
    n_answers += 1
    for first, second in combinations(range(len(people)), 2):
        same_labels = all(people[first][key] == people[second][key] for key in ("role", "team"))
        if not same_labels or boxes[first] is None or boxes[second] is None:
            continue
        overlap = boxes[first].iou(boxes[second])
        if 0.5 <= overlap <= 0.9:
            # One labelled person under both boxes: the model boxed that person two times. Two: a duel.
            shared = sum(1 for box in labelled if min(box.iou(boxes[first]), box.iou(boxes[second])) >= 0.3)
            n_pairs += 1
            n_one_person += shared == 1
            print(f"{clip} frame {frame}: IoU {overlap:.2f}, labelled people under both boxes: {shared}")
print(f"{n_pairs} pairs with the same labels and an IoU from 0.5 to 0.9 in {n_answers} people answers; "
      f"{n_one_person} of them lie on one labelled person")
```

Short tracks and numbers in parts (fact 14). It reads the number answers of the number probe
and the label files:

```python
import json
from collections import Counter
from pathlib import Path

from pitchvlm.labels import load_gsr_labels

data = Path("~/pitchvlm-data/SoccerNetGS-valid").expanduser()
cache = Path("../pitchvlm-results/results/cache")
n_short = n_agree = n_agree_wrong = n_two_digits = n_in_parts = 0
for cassette in sorted(cache.glob("*/number/*.json")):
    stored = json.loads(cassette.read_text(encoding="utf-8"))
    item = stored["meta"]["item"]
    if not item.startswith("track-"):
        continue  # A request of the pipeline names no track of the label file.
    track_id = int(item.split("-")[1])
    labels = load_gsr_labels(data / stored["meta"]["clip_id"] / "Labels-GameState.json")
    numbers = Counter(
        athlete.jersey
        for frame in labels.frames
        for athlete in frame.athletes
        if athlete.track_id == track_id and athlete.jersey is not None
    )
    label_number = numbers.most_common(1)[0][0] if numbers else None
    answer = json.loads(stored["response"]["text"])
    crops, frames = answer["per_crop"], stored["context"]["frames"]
    if len(frames) == 2 and frames[1] - frames[0] < labels.fps:
        n_short += 1
        if crops[0] is not None and crops[0] == crops[1]:
            n_agree += 1
            n_agree_wrong += crops[0] != label_number
    if answer["number"] is not None and answer["number"] >= 10:
        n_two_digits += 1
        digits = set(divmod(answer["number"], 10))
        n_in_parts += any(crop in digits for crop in crops if crop != answer["number"])
print(f"tracks with two crops less than one second apart: {n_short}; both crops show one number: {n_agree}; "
      f"that number is not the labelled one: {n_agree_wrong}")
print(f"readings of a number with two digits: {n_two_digits}; with a crop that shows one of its digits alone: {n_in_parts}")
```

## 9 The guard, the secret scan and the deny-list

Two checks run before a commit and before a push: the guard (`tools/guard.py`) and the secret
scan (`gitleaks` with `.gitleaks.toml`). The file `.pre-commit-config.yaml` starts them. The
commit hooks run for `git commit`. A commit that another git command makes, for example the
merge commit of `git merge`, starts no commit hook: the push hooks check it before it leaves the
machine. The hook runner skips a hook that the environment variable `SKIP` names. Do not set
it for these hooks: it is a bypass, like `--no-verify`. Another hook folder
(`git -c core.hooksPath=...`) is a bypass too: git then starts no hook of this repository.

| Mode of the guard | When | What it reads |
|---|---|---|
| `--staged` | before a commit (hook) | the files of the git index, and the author and the committer of the next commit |
| `--push` | before a push (hook), or by hand | every commit that the remote does not have yet and every local annotated tag: files, messages, author, committer, tagger. Every local tag: what it points at. The names of the local branches that the remote does not have, and of every local tag |
| `--tracked` | by hand | every tracked file of the working tree |
| `--paths <file> ...` | the command `sync` | the given files |

A report line is `<path>:<line>: <rule>: <message>`. A line for a commit ends with its name, for
example `[commit 1a2b3c4d5e]`. The rules:

| Rule | Refuses |
|---|---|
| `ext`, `binary`, `size` | a file type that is not on the list, a file that is not text, a file above 2 MB |
| `base64`, `pixels` | encoded binary data: a run of 200 base64 characters, a wrapped run of 2000, a `data:` address, a table of 20000 byte values |
| `dataset` | a label file and the result file of a clip run, by name; label data, prediction data, the result of a clip run and records of the label format in a `.json` or `.jsonl` file |
| `secret` | API keys, tokens, private key headers, a password in an address, a literal value after a secret name |
| `lock`, `ignored` | a lock file; a file that `.gitignore` names (added with `git add -f`), and a path that git cannot test against the ignore rules |
| `host`, `home` | a host that is not in `tools/allowed_hosts.txt`, an address of a private network; a path into a home folder or into another folder of one user |
| `deny` | a word of the deny-list, in a file, a path, a message, an identity or the name of a branch or of a tag |
| `identity` | a clone without a local `user.name` or `user.email`; an author, a committer or a tagger whose name or e-mail is not the local one |
| `tag` | a tag that does not point at a commit, for example a tag at the content of a file |

The rules `secret`, `host`, `home` and `deny` also read names: the path of every checked file
and, in the push mode, the names of branches and tags. The secret scan reads the content of
files only.

The deny-list:

- It is the file `.git/guard-deny.txt` of the clone. It is never committed, and its words are
  never written into a tracked file. All work trees of the clone share it.
- One entry per line. A line that starts with `#` is a comment. An entry matches as a part of a
  word, in any letter case. A report names the line of the entry, never its text.
- An entry also matches the name and the e-mail of a commit. So do not list a word that is a
  part of the personal git name or e-mail.
- Put into it on each machine: the user name of the account, the name of the machine, and every
  name that must not leave the machine. Examples: the name of an organisation, the domain of an
  e-mail address that is not the personal one, the suffix of a private host.
- The guard cannot know such words by itself. Without the file the rule `deny` is off, the
  guard prints one notice line, and the check `deny-list` of `doctor` fails. A file with comments
  only stops the notice and passes the check.

When a push is refused:

- The report names the commit. Change that commit (`git commit --amend`, or `git rebase -i` for
  an older one). A new commit that removes the text does not help: the old commit goes to the
  remote with the push.
- The guard checks every local branch and every tag, not only the branch of the push. The hook
  runner tells a hook about one branch only, and a push can send several.
- A report line can name a branch or a tag (`the name of branch ...`, `the name of tag ...`, or
  the rule `tag`). Give the branch another name (`git branch -m <old name> <new name>`), or
  delete the tag (`git tag -d <tag>`).
- Git does not record which tags the remote has. So the guard checks every local tag at every
  push: its name and, for an annotated tag, its message and its tagger. A tag that came from
  the other machine is checked against the local `user.name`, the local `user.email` and the
  local deny-list. Both clones need the same name and the same e-mail for this reason. A local
  tag that was deleted (`git tag -d <tag>`) comes back with the next fetch;
  `git fetch --no-tags` leaves the tags out.
- A push that sends only names starts no hook: a tag, or a new branch, on a commit that the
  remote has already. Before such a push (for example the freeze tag), run
  `python3 tools/guard.py --push` by hand. It checks the names, the tag messages, the taggers
  and what every tag points at.
- A report line with the rule `ignored` can name a symbolic link: a folder of the working tree
  is a link, and a commit of another branch holds a file below that folder. Remove the link,
  then push again.

Text that the guard refuses although it is no leak. Write it in another way:

- A route such as `/home/<word>` with a real word: it has the form of a home folder.
- A bare list of check sums or commit names, one per line, from 2000 characters on (32 values
  of SHA-256, 50 commit names): it has the form of wrapped hexadecimal data. Write the file
  name after each check sum, as `sha256sum` prints it.

What the guard does not see:

- A private host name that stands alone, and a private name in the path of an allowed host. The
  deny-list covers both.
- A private IPv6 address that stands alone.
- Pixels in other forms than base64 text and a long table of byte values: a table below 20000
  values, a list of short text pieces, a byte string in Python code.
- Label data and prediction data in a file that is not a `.json` or `.jsonl` file, and in a JSON
  file of several lines that is not valid JSON, for example after a comment line.
- A secret that is cut into pieces or written in another code. The secret scan has the same limit.
- A bare value without a digit after a secret name (`password: <words>`): the guard takes it
  for a name of the code. A value of 20 or more characters in quotes is refused with or without
  a digit.
- A secret of another service in a commit message, in a tag message or in a name: the guard
  knows only the key forms of its own list there, and the secret scan reads the content of
  files only.
- A name that exists only in the push command (`git push origin <branch>:<other name>`), when
  the push sends no new commit or when it is not the first branch of the push: the hook runner
  gives the guard one such name, for the first branch with new commits.
