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
| Run files of the two specialist tools | `../pitchvlm-results/results/tools` (section 3) |
| Job files of the two specialist tools | `~/pitchvlm-data/jobs`; they name image paths and stay there |

Every command has `--help`. Exit code 0 means done, 1 means that a check or a gate failed, 2 means
that the command could not run. Every error gives the exit code 2 and ends with a line
`error: ...`, also when a Python traceback is printed before that line. So the exit code 1 never
means an error.

The output of a command is written line by line, also into a pipe or a file. So the cost
estimate and the budget line can be read before the first model call, and `error:` is the last
line also when both streams go to one file. A settings file, a roster file, a label file, an
experiment file, a split list or a run file of a tool with wrong content gives one line that
names the file and the fault, without a traceback. So does a clip folder that does not fit its
label file.

## 1 Setup once

```bash
# Tools.
brew install uv gitleaks
uv tool install pre-commit

# Main environment: Python 3.13. Then run the offline tests: the full run once here.
uv venv --python 3.13
uv pip install -e ".[dev,gemini]"
source .venv/bin/activate
python -m pytest -m "slow or not slow"

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
env -u GEMINI_API_KEY uv pip install --python ~/pitchvlm-data/envs/scorer/bin/python "numpy>=1.26,<1.27" "sn-trackeval==0.4.0"
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

The offline tests have two runs (`docs/DECISIONS.md`, entry "A faster default run of the offline
tests" of 2026-11-22):

```bash
python -m pytest                          # default run, after each change: all but the 9 slow tests
python -m pytest -m "slow or not slow"    # full run, before a commit: every test
python -m pytest -m slow                  # the 9 slow tests alone
```

- The default run leaves out only the tests marked `slow` (`pyproject.toml`, section
  `[tool.pytest.ini_options]`). They need a long synthetic clip and check no hard rule. Every
  check of the guard, the secret scan, the money stops, the cache keys and rule 4 is in the
  default run.
- `tests/conftest.py` keeps pure results that many tests compute again (frames of a synthetic
  scene, frames read from image files, the measured camera motion, the paint alignment, JPEG
  bytes, the git state of this checkout) once per test process. `PITCHVLM_TEST_CACHE=0 python -m
  pytest` runs without these caches.
- On a machine that has the plugin `pytest-xdist`, `python -m pytest -n auto` runs the tests in
  several processes. The plugin is not a dependency of the project and is not installed on the
  build machine; such a run was never tried there.

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
| people | G2: recall and the lower end of its interval over the clips (`main_metric.low`), precision, small boxes, foot-point error median and 90th percentile (`foot_error_m.p90`) | 14 USD |
| pitch | G3: median position error near the goals and in midfield, 90th percentile of the position error (`error_p90_m_max`), share of frames with a mapping | 15 USD |
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
  The kill switch also names five overlay images that are checked by eye. No command writes
  such images yet: tell the owner before the kill switch is judged (`docs/DECISIONS.md`, last
  list).
- Another setting (another model, another resolution, the flex tier): copy the experiment file,
  change `name` and one value under `step`, commit when the owner asks, and run it with its own
  `--out` file.
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
and without training. So the decision of 24 Nov has measured values of both tools next to the
values of the two model steps: boxes for RF-DETR, pitch mappings for PnLCalib. No model is
called and nothing is paid. Rule 8 of section 7 does not change: a tool replaces a Gemini step
only after that step failed its gate and the tool passed the same gate on the same frames.

| Tool | Experiment file | Frames and gate |
|---|---|---|
| RF-DETR Small | `configs/experiments/people_probe_rfdetr_small.yaml` | people probe, G2 |
| RF-DETR Small, antialiased resize | `configs/experiments/people_probe_rfdetr_small_antialias.yaml` | people probe, G2 |
| RF-DETR Large | `configs/experiments/people_probe_rfdetr_large.yaml` | people probe, G2 |
| RF-DETR Large, antialiased resize | `configs/experiments/people_probe_rfdetr_large_antialias.yaml` | people probe, G2 |
| PnLCalib | `configs/experiments/pitch_probe_pnlcalib.yaml` | pitch probe, G3 |
| RF-DETR Medium, the default detector of the detector engine (section 11) | `configs/experiments/people_probe_rfdetr_medium.yaml` | people probe, G2 |

Each tool gets its own environment, because its packages do not fit the environment of the
package. Ask the owner before the install: it downloads packages and about 1 GB of weights.
Ask at the same time for the licence decision of the last list of `docs/DECISIONS.md`.
These commands were written from the pages of the two tools. None of them ran before.

```bash
# RF-DETR. The script tools/run_tool.py was written for the versions 1.12.0 and 1.12.1 of the package.
uv venv --python 3.12 ~/pitchvlm-data/envs/rfdetr
env -u GEMINI_API_KEY uv pip install --python ~/pitchvlm-data/envs/rfdetr/bin/python "rfdetr==1.12.0"

# PnLCalib: a clone at the commit that the script was written for, its own environment, two weight files.
git clone https://github.com/mguti97/PnLCalib ~/pitchvlm-data/PnLCalib
git -C ~/pitchvlm-data/PnLCalib checkout 42088864170f75ea02e76d20e9a9a1db59edb3e3
uv venv --python 3.12 ~/pitchvlm-data/envs/pnlcalib
env -u GEMINI_API_KEY uv pip install --python ~/pitchvlm-data/envs/pnlcalib/bin/python \
    "torch==2.3.1" "torchvision==0.18.1" "numpy==2.0.2" "scipy==1.13.1" "opencv-python==4.10.0.84" \
    "shapely==2.0.7" "matplotlib==3.9.4" "pillow==11.1.0" "PyYAML==6.0.2"
mkdir -p ~/pitchvlm-data/weights/pnlcalib
curl -L --fail -o ~/pitchvlm-data/weights/pnlcalib/SV_kp \
    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp
curl -L --fail -o ~/pitchvlm-data/weights/pnlcalib/SV_lines \
    https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines
ls -l ~/pitchvlm-data/weights/pnlcalib    # SV_kp: 264964645 bytes, SV_lines: 264857893 bytes
```

- Both installs take packages without a fixed version: `rfdetr` brings what it needs, and the
  nine packages of PnLCalib bring theirs (for example `sympy` and `fonttools`). Give both the
  package age limit of the setup (section 8, fact 19).
- RF-DETR downloads the weights of a model class when the class is used for the first time
  (Small: 386 MB, Large: 136 MB) from the host of its maker into `~/.roboflow/models`.
- The requirements file of PnLCalib is for Linux with an NVIDIA GPU and cannot be installed
  here. The install above names the packages that the script needs, each in the version of
  that requirements file. The package index has a file of each one for an Apple Silicon machine
  with Python 3.12 (read on 2026-11-14). Do not take a newer `opencv-python`: the tool was not
  tested with OpenCV 5.
- Write the versions that the two environments got into `docs/DECISIONS.md`
  (`env -u GEMINI_API_KEY uv pip freeze --python <the interpreter of the environment>`).
- Every command that starts the interpreter of a tool environment begins with
  `env -u GEMINI_API_KEY`: the start-up code of a package of that environment runs before any
  line of a script.

Then three commands for each experiment file: write the job, run the tool, measure. The examples
are for RF-DETR Small and for PnLCalib. The device `mps` is the GPU of an Apple Silicon machine.

```bash
mkdir -p ~/pitchvlm-data/jobs

python -m pitchvlm.cli tool-job --experiment configs/experiments/people_probe_rfdetr_small.yaml \
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/people_probe_rfdetr_small.json
env -u GEMINI_API_KEY ~/pitchvlm-data/envs/rfdetr/bin/python tools/run_tool.py rfdetr --device mps \
    --job ~/pitchvlm-data/jobs/people_probe_rfdetr_small.json --root ~/pitchvlm-data \
    --out ../pitchvlm-results/results/tools/people_probe_rfdetr_small.json
python -m pitchvlm.cli tool-score --experiment configs/experiments/people_probe_rfdetr_small.yaml \
    --root ~/pitchvlm-data --run ../pitchvlm-results/results/tools/people_probe_rfdetr_small.json \
    --out ../pitchvlm-results/results/metrics/people_probe_rfdetr_small.json

python -m pitchvlm.cli tool-job --experiment configs/experiments/pitch_probe_pnlcalib.yaml \
    --root ~/pitchvlm-data --out ~/pitchvlm-data/jobs/pitch_probe_pnlcalib.json
env -u GEMINI_API_KEY ~/pitchvlm-data/envs/pnlcalib/bin/python tools/run_tool.py pnlcalib --device mps \
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
- These commands measure a tool on the probe frames. They do not switch it on: the settings
  of `run` name the engine of each step, and section 11 runs the tools on every frame of a clip
  for the detector engine.
- The two tools are the code of other projects. The two commands above start the script without
  the variable `GEMINI_API_KEY` (`env -u`), and the script also takes the variable out of its
  environment before it loads a tool. Code of the tool environment runs before that line of the
  script (the start-up code of a package, and NumPy). So always start the script with
  `env -u GEMINI_API_KEY`, also in a command that you type by hand.
- The frames are those of the model probe: the tool file and the probe file have the same group,
  frames per clip and seed. Give `tool-job`, `tool-score` and the model probe the same
  `--max-clips` when you compare them.
- The metrics file has the keys of the model probe without the values that only a model step has
  (`prompts`, unparsed answers, the values of the landmark fit). Compare it with the same gate.
  Its provenance names the run file with its SHA-256 (`run_file_sha256`) and says whether the
  script that made the run file is the `tools/run_tool.py` of this work tree
  (`script_is_current`). When it is not, `tool-score` prints a line: the commit of the metrics
  file then does not name the script that ran the tool.
- `seconds_per_frame` is the time of the tool for one frame on this machine, without the reading
  of the image file. For RF-DETR it is the time of the plain `predict` of the tool. The tool
  also has a compiled predictor, which its pages give as about twice as fast on an M4 Max (11.1
  ms against 20.0 ms for the Small model): that value is published, not measured here. The paper
  of PnLCalib gives 439 ms per frame for the same voting with line refinement (the two networks
  on an NVIDIA RTX 2080 Ti, the calibration on a CPU). That time was measured with the reading
  of the image, at 960 x 540 pixels, with other weights and other thresholds: it is published,
  not measured here, and not for the settings of this experiment file.
- `tool-score` refuses a run file that was made for another experiment name or with other tool
  settings, and a run file that has no answer for a frame of the probe. After a change of the
  experiment file, write the job again and run the tool again. Both commands refuse an `--out`
  file that is the experiment file or the split list, and `tool-score` also one that is the
  run file.
- `tool-score` refuses a work tree with changes, like `probe`.
- Pitch: read `axis_check` first; `tool-score` prints it. It holds the median position error for
  the matrices as they are (`as_given`) and with the sign of x, of y or of both changed. The
  pitch convention is right when `as_given` is the smallest of the four and small (some metres
  at most; the other three are then many times larger). When another value is smaller,
  `tool-score` prints `axis check failed`, writes the metrics file and ends with the exit
  code 1. Then look at the size of the values. `as_given` large and one other value small:
  either the script gives the matrices in another pitch convention, which is a fault of
  `tools/run_tool.py`, or the tool mirrors the pitch on more than half of the frames.
  `plausible_share` tells the two apart. A wrong convention changes every frame, so
  `plausible_share` is then 0 (measured on the synthetic clip of the tests, for each of the
  three sign changes). When it is above 0, a part of the frames is right as given: do not change
  `camera_homography`, and tell the owner. In both cases the other values say nothing about the
  tool. All four values large (about 10 m and more), also when `as_given` is the smallest of
  them: the check decides nothing, because the output of the tool is wrong for another reason
  (device, weights, frame size). Compare with `--device cpu` first, and do not change
  `camera_homography`.
- Pitch: PnLCalib gives the best camera that it finds and never refuses one. The landmark fit
  of the model step refuses a weak fit, so its `valid_share` is lower. So `error_m`, the errors
  under `by_view`, `plausible_error_m`, `main_metric` and `per_clip.error_median_m` are counted
  on the frames where that engine gave a mapping: the two engines are not measured on the same
  frames there. A weak camera of the tool shows in `error_m.p90` and `error_m.mean`. It does
  not show in the medians until about half of the cameras are weak, and not in
  `plausible_share` when the camera is only some metres off. So the tool must also meet
  `error_p90_m_max` of `configs/gates.yaml`, like the model step.
- Pitch: `plausible_share` of a tool is counted with a stricter rule than that of a landmark
  fit. A record of a tool has no count of agreeing landmarks, so its plausibility check always
  wants 9 of 10 people on the pitch; a landmark fit with 7 or more agreeing landmarks needs 2
  of 3. The gate values (`error_m`, `valid_share`) do not use this check.
- Pitch: the main metric leaves out the clips without a valid frame, and its printed line names
  how many clips it used ("over N of M clips"). Compare it between two results only when both
  used all clips. Between a tool and the model step the frames inside a clip still differ (see
  above): judge each engine against the gate lines. The values under `by_view`, which decide G3,
  are counted over the frames of all clips together: they have no value per clip and no
  interval. So a difference near a gate line of G3 cannot be tested from the metrics files.
- Pitch: the benchmark script of PnLCalib for main-camera pictures drops a camera whose
  reprojection error is above 100 pixels at 960 x 540 pixels (`--max_reproj_err`). The
  measurement here drops none. The run file keeps `rep_err` of every frame, in pixels of the
  frame (100 pixels at 960 x 540 are about 200 pixels of a frame of 1920 x 1080), so such a
  limit can be counted later without a new run of the tool. It is a setting that is not built:
  it needs a key in the experiment file.
- People: `per_clip` of each metrics file holds the recall, the precision and the small-box
  recall of every clip, on the same frames and the same labels for both engines. Take the
  difference clip by clip, as the plan says for every comparison of two settings.
- People: a detector that was trained on everyday pictures can also find people in the stands.
  They count as wrong boxes. The people prompt tells the model to leave the spectators out, and
  the pipeline drops every person whose foot point lies more than 3 m outside the pitch. So
  read the precision of a detector next to `counts.fp_off_pitch`: the wrong boxes whose foot
  point the label mapping of the frame puts more than 3 m outside the pitch. A frame without a
  label mapping adds nothing to this count. The metrics file of the model step has the same
  count. The gate line uses `precision` as it is. When only the boxes off the pitch miss the
  line, tell the owner: another count is a change of the gate (rule 7). The run file keeps
  every detection above the score limit with its class name, so another count can be made
  later without a new run of the tool.
- People: `foot_error_m` is counted on the matched boxes only. Read it next to
  `small_boxes.recall`: the engine that finds fewer small boxes has the easier boxes.
- RF-DETR makes a frame of 1920 pixels about 4 times narrower for the input of the Small model
  and about 3 times for the Large model. Since its version 1.9 it does that without
  antialiasing, and its weights are older than that version. No page of the tool says which
  resize fits the weights. So Small and Large each have two experiment files: one with the
  resize that the tool uses by itself (`antialias: false`) and one with antialiasing
  (`antialias: true`). Run both. The setting that is taken later is confirmed on clips of the
  group `reserve`, as the plan says for every chosen setting: a copy of the experiment file
  with its own `name` and `group: reserve`, then the three commands. Do not use the group
  `fresh`: gate G6 runs it once.
- Another setting: copy the experiment file, change `name` and one value under `tool`, commit
  when the owner asks, and run the three commands again. Put each copy of a tool experiment
  file, also the one for the group `reserve`, into a sub-folder, for example
  `configs/experiments/more/`: one offline test names the five tool files of
  `configs/experiments` and fails for a sixth file there. For RF-DETR the values are the model
  class (`RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`), the score limit
  `min_score` (0.5 is the threshold that the tool uses by itself) and `antialias`. A low score
  limit makes the run file large. The guard refuses a file above 2,000,000 bytes (about 14,000
  detections, for example 120 per frame on 120 frames), and `sync` then prints no git command
  for the whole folder: keep such a run file in `~/pitchvlm-data`. For PnLCalib the two
  thresholds of the file are those of the tool's script `inference.py`; its benchmark script
  for main-camera pictures uses 0.0966 for the keypoints and 0.3441 for the lines.
- No source says that the GPU of an Apple Silicon machine gives the same values as the CPU for
  these tools: the pages of RF-DETR give weak evidence that it does, and PnLCalib was written
  for NVIDIA GPUs. Run the first two clips also with `--device cpu` and another `--out` file, and
  compare the two metrics files before the full run: `counts`, `valid_share`, `axis_check` and
  the error values. `seconds_per_frame`, `provenance` and `written_at` always differ. Tell the
  owner about every other difference.
- The run files are text results: boxes that the detector found and camera matrices, no pixels
  and no label data. The commands above write them into the results work tree, and `sync`
  (section 6) lists them as tool runs. Whether they go to the results branch is an open owner
  decision (`docs/DECISIONS.md`, last list). The `add` command that `sync` prints takes the
  whole folder `results`, also `results/tools`: ask the owner for this decision before the
  first commit of results after a tool run. Without a yes, move the folder `results/tools` to
  `~/pitchvlm-data` first. The job files stay in the data folder.
- Licences: the code of RF-DETR and the weights of its four model classes above are under
  Apache-2.0. The code of PnLCalib is under GPL-2.0-or-later and its weights under CC BY 4.0
  (name the paper of the tool when the weights are used). `tools/run_tool.py` imports code of
  the PnLCalib clone when it runs. The repository is public since 2026-11-14, and the check of
  what that means is not done: it is an open owner decision (`docs/DECISIONS.md`, last list).

### Paint alignment of the pitch mappings (free)

Every metrics file of the pitch probe and of `tool-score` for PnLCalib has the key
`paint_alignment` (`docs/DECISIONS.md`, entry "Paint alignment of pitch mappings, and hard
cuts" of 2026-11-22): a score from 0 to 1 per valid mapping, how well the projected pitch
lines lie on the white paint of the frame, and how well the score separates the mappings with
a median position error above 3 m from those at or below 2 m. It is a diagnostic: no rule uses
it yet. The measurement needs no model call. It runs on the frames
of the group `probe` of the split list, which are valid clips only; never run it on test clips.

When the pitch probe ran before this code was there, replay it from the stored answers (free):

```bash
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \
    --root ~/pitchvlm-data --replay --out ../pitchvlm-results/results/metrics/pitch_probe_paint.json
python -m pitchvlm.cli tool-score --experiment configs/experiments/pitch_probe_pnlcalib.yaml \
    --root ~/pitchvlm-data --run ../pitchvlm-results/results/tools/pitch_probe_pnlcalib.json \
    --out ../pitchvlm-results/results/metrics/pitch_probe_pnlcalib_paint.json
```

- Read `paint_alignment.separation` of each file: `auroc`, `auroc_standard_error`, `n_good`,
  `n_bad`, `score_good` and `score_bad`. Also read `n_cannot_judge` and `cannot_judge_reasons`:
  a frame with too little paint, or with too much white, gets no score.
- Write the numbers of both engines into `docs/DECISIONS.md` with the date. The rule that would
  follow from them is in that file (entry "Paint alignment of pitch mappings, and hard cuts",
  item 4); it needs an entry of its own.
- `paint_alignment.frames` holds one entry per mapping: clip, frame, score and median position
  error. They are text and go back with the metrics file.
- The constants of the score were chosen on synthetic frames only. Before the numbers are read,
  check three things on the first clip: `n_cannot_judge` is small (a large count means that the
  paint mask misses the lines of real frames), the median score of the good mappings is high,
  and the score of a frame does not depend on the run (it is deterministic).

`demo` and `run` print the hard cuts of a clip and the median paint score of its keyframe
mappings (the landmark fits of the pitch engine `model`, or the cameras of the pitch engine
`calibration`), and the result file of `run` holds both (`cuts`, `paint`). The clips of SoccerNet GSR have no
cut, so no gate measures the cut test: it guards the demo clip and other broadcast footage. A
cut that `run` names on a dataset clip is a false cut: tell the owner.

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

### The track repairs against gate G4 (arms)

Two track repairs are arms of gate G4, off by default: the split of team swaps and the team slots
(`docs/ARCHITECTURE.md`, `tracking.py`; `docs/DECISIONS.md`, 2026-11-22). Each arm is a settings
file in `configs/arms`. Measure them on the 6 clips of the group `dev` after the runs of the
current behaviour above (`--out ~/pitchvlm-data/runs/dev`), with the same cache folder. For each
clip of the group:

```bash
# Team slots: they act after the shirt numbers, so every answer is stored. Free.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --config configs/arms/g4_team_slots.yaml --out ~/pitchvlm-data/runs/g4-team-slots --replay

# Split of team swaps: a track that is cut gets new crops, so its number request is new. First the
# dry run: it stops at the first answer that is not stored. Then the paid run.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --config configs/arms/g4_split_team_swaps.yaml --out ~/pitchvlm-data/runs/g4-split-swaps --replay
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --config configs/arms/g4_split_team_swaps.yaml --out ~/pitchvlm-data/runs/g4-split-swaps

# Both repairs: the tracks before the votes are those of the split arm, so every answer is stored. Free.
python -m pitchvlm.cli run --clip ~/pitchvlm-data/SoccerNetGS-valid/<clip> \
    --labels ~/pitchvlm-data/SoccerNetGS-valid/<clip>/Labels-GameState.json \
    --config configs/arms/g4_both_repairs.yaml --out ~/pitchvlm-data/runs/g4-both --replay
```

- Money: the estimate line of the paid run counts every request of the clip (about 2.3 USD), but
  the stored answers cost nothing. Only the number requests of the tracks that the split changed
  are new: about 0.009 USD each with the default token estimates (an estimate, not measured).
  The run limit of 1.3 times the estimate does not stop such a run. Run one clip at a time.
- Each run prints the tracking values without labels (births inside the image, steps faster than
  9 m/s, team-key changes, tracks per team) and writes them into `result.json` under
  `diagnostics`. No gate uses them.

The G4 values of each arm, per clip and over the group. Put the clip names of the group `dev`
into the list (`configs/splits.json`, split `valid`, group `dev`):

```python
import json
from pathlib import Path

import yaml

from pitchvlm.evalx.metrics import identity_scores
from pitchvlm.labels import gsr_frame_states, load_gsr_labels
from pitchvlm.schemas import ClipResult

data = Path("~/pitchvlm-data").expanduser()
clips = json.loads(Path("configs/splits.json").read_text(encoding="utf-8"))["valid"]["dev"]
gate = yaml.safe_load(Path("configs/gates.yaml").read_text(encoding="utf-8"))["g4_identity"]
arms = {"current": "dev", "team slots": "g4-team-slots", "split swaps": "g4-split-swaps", "both": "g4-both"}
for arm, folder in arms.items():
    scores = {}
    for clip in clips:
        document = json.loads((data / "runs" / folder / clip / "result.json").read_text(encoding="utf-8"))
        result = ClipResult.model_validate(document["result"])
        truth = gsr_frame_states(load_gsr_labels(data / "SoccerNetGS-valid" / clip / "Labels-GameState.json"))
        scores[clip] = identity_scores(result.frames, truth)
        print(arm, clip, scores[clip])
    coverage = sum(one.coverage for one in scores.values()) / len(scores)
    worst = min(one.coverage for one in scores.values())
    switches = max(one.switches for one in scores.values())
    ratio = max(one.track_ratio for one in scores.values())
    passes = (
        coverage >= gate["coverage_min"]
        and worst >= gate["coverage_worst_clip_min"]
        and switches <= gate["switches_per_clip_max"]
        and ratio <= gate["track_ratio_max"]
    )
    print(f"{arm}: coverage {coverage:.3f} (worst clip {worst:.3f}), most switches {switches}, "
          f"largest track ratio {ratio:.2f}, G4 {'passes' if passes else 'fails'} on {len(scores)} clips")
```

- The script reads the track ratio limit per clip (the largest ratio of the group). The gate file
  does not say whether the limit is per clip or over the group: the owner decides before the run.
- The G4 values do not see a wrong join: a track that follows two people one after the other.
  `purity` (printed with the values) does, and so does AssA of `eval`. Score each arm with `eval`
  (section 5, `--pred ~/pitchvlm-data/runs/<folder>/predictions`, a new `--out` per arm) and
  compare AssA with the arm `current` on the same clips before a repair becomes the default.
- A repair becomes the default only with a decision entry in `docs/DECISIONS.md` that names its
  G4 values and AssA against the current behaviour on the same 6 clips.

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

The remote `origin` is public since 2026-11-14. A pushed result file is public at once. Do not
push the branch `results` to that remote before the owner has decided where the branch goes
(`docs/DECISIONS.md`, last list).

```bash
python -m pitchvlm.cli sync --results ../pitchvlm-results/results
```

The command lists the files, checks them with `tools/guard.py` and with the secret scan of the
git hooks, and prints three git commands (`add`, `commit`, `push`). It runs none of them. The
owner runs `add` and `commit`. After a tool run, see section 3 first: the folder
`results/tools` waits for its own owner decision. The push waits for the decision above, and
the line under the three commands says so. The command refuses the
folder when it holds anything but `.json` and `.jsonl` files, and when the guard or the secret
scan finds a problem. After that decision, fetch the branch `results` on the build machine and
read the files there.

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
4. Only `src/pitchvlm/vlm/client.py` imports the model SDK. Only `tools/run_tool.py` imports an external tool.
5. Model calls: paid tier only, `store=False`, no sampling parameters; players are identified by shirt number and roster, never by face.
6. Money: estimate first, measured cost after; a run stops at 1.3 times its estimate (the contract checks of `doctor`: at the 2 USD of gate G0); hard stop at 135 USD; no paid run without the owner's go-ahead.
7. Gates are fixed in `configs/gates.yaml` before a run; the 10 test clips are run once, after the freeze tag.
8. A specialist model replaces a Gemini step only after that step failed its gate and the specialist passed the same gate on the same frames.
9. The guard and the secret scan run before `git commit` and before a push that sends commits; `--no-verify`, the variable `SKIP` of the hook runner and another hook folder (`core.hooksPath`) are not allowed. Before a push that sends only names (a tag or a new branch), run `python3 tools/guard.py --push` by hand. The deny-list `.git/guard-deny.txt` is never committed (section 9).
10. Commit only when the owner asks; never push without being asked; a commit message has no co-author line; the remote is public: a pushed commit cannot be taken back.

## 8 What was never tested before the run machine

The build machine has no key, no data and no scorer. So these parts run for the first time here.
Check each one with its first check before you rely on it.

| Part | File | First check | When it fails |
|---|---|---|---|
| Offline tests on this machine | `tests/` | `python -m pytest -m "slow or not slow" -rs` (the full run) after the setup (section 1): every test passes, and the skipped tests are listed with their reason. The tests ran on one machine only, with Python 3.13. Some tests need a disk that tells upper and lower case apart, and some a disk that does not: one of the two groups never ran | A skipped test names a missing tool (`git`, `gitleaks`, `pre-commit`) or the kind of disk: install the tool. One test of the tool script is skipped without Pillow. Pillow belongs to the environments of the two specialist tools (row "Tool script"): leave this test skipped. A failed test names the code that this machine runs in another way: repair the code, not the test. |
| Setup commands | section 1 | Each command of section 1 ends without an error, and `doctor --no-api` shows the SDK as a version 3.x. Most commands of section 1 never ran: the install with the extra `gemini`, the pinned list, the work tree and the scorer | Read the error of the command. Fact 19 below is about the package age limit of `uv pip`. |
| Gemini client | `src/pitchvlm/vlm/client.py` | `doctor` with one model (step 2 of day 1). The cost estimate and the budget line must be on the screen before the first check line of a model | Read the failed check and the report file. A failed contract check (`boxes`, `landmarks`, `store flag`, `token sum`, `served tier`, `flex tier`) is a client fault: repair the client before any footage is sent. A value of `box accuracy` or `landmark accuracy` that does not reach the value of the gate file is not a client fault. A line `[info] flex tier: not checked` is no client fault either: the flex tier was busy (section 2). When the check `boxes` fails, read the answers under `contract.<model>.answers` in the report file first: a model that names no person for the coloured rectangles of the synthetic frames fails this check without a client fault. |
| Download | `src/pitchvlm/datasets.py` | `data --split valid`; then `ls ~/pitchvlm-data/SoccerNetGS-valid` shows about 58 clip folders | Run the command again: it continues the download. An archive whose `.part` file in `~/pitchvlm-data/archives` already holds every byte gets its final name at the next start: the server answers the range with the status 416 and the size of the archive, and the command compares that size with the file. When the sizes differ, or the answer names no size, the command stops with an error that names the `.part` file: compare its size with the size on the dataset page, or delete it to start again. When the host refuses the request or its certificate (the download itself was never run), download `valid.zip` by hand into `~/pitchvlm-data/archives` and run the command again: it then only extracts. |
| Label reader on real files | `src/pitchvlm/labels.py` | `oracle` with one clip: it reads the label file first | The error names the file and the place in it. Repair the reader, not the label file (check 4 below). |
| Scorer wrapper and its output layout | `src/pitchvlm/evalx/scorer.py` | `oracle` with one clip: the ground truth must score at least 99.5. The wrapper gives the evaluator `<root>/SoccerNetGS-<split>` as the label folder and `<predictions>/SoccerNetGS-<split>` as the prediction folder, with `--SKIP_SPLIT_FOL True`. The source of the evaluator reads no file `seqinfo.ini` for this data; this run confirms that `--SEQ_INFO` with clip names only is enough | The error shows the last lines of the scorer. A line `GT file not found` or `Tracker file not found` names the path that the evaluator looked for: compare it with the place of the file. When the lines name a missing `seqinfo.ini`, the evaluator needs the clip lengths: repair the wrapper, do not write such files into the dataset. The evaluator writes its files into the folder of the prediction files of the tracker: three `*_detailed.csv` files, and a summary file and two plot files for each of them. For `oracle` this folder is `<out>/predictions/SoccerNetGS-<split>/<tracker>/`, and the command moves the `*_detailed.csv` files from there to `<out>/scorer/<tracker>/`. For `eval` it is `<pred>/SoccerNetGS-<split>/<tracker>/`, and the files move to `<out>/`. The scores come from `person_detailed.csv`. When the command finds no such file, look where the evaluator wrote its files. The wrapper leaves the plots of the evaluator on, and the evaluator draws a plot before it writes the detailed file of a class: an error of the plot library in the last lines also leaves no detailed file. The evaluator was never imported under Python 3.12 with the packages that the setup resolves. |
| Video encoding | `src/pitchvlm/render.py` | `python -m pitchvlm.cli demo --out outputs/demo`, then open `outputs/demo/demo.mp4` | Check the OpenCV build (codec `mp4v`). |
| Video decoding | `src/pitchvlm/frames.py` | The free dry run of the demo clip (section 4) prints the frames, the frame rate and the keyframes: frames divided by the rate must be the length of the clip in seconds, and the clip must have about 2.1 keyframes per second at any frame rate. Only synthetic clips of 25 frames per second ran before | A video with a variable frame rate, rotation data or interlaced frames was never read here. Convert it to a constant frame rate, or extract its frames into a folder and use `--fps`. |
| Roster file | `src/pitchvlm/roster.py` | The same dry run with `--roster` reads the roster file before any model call. After the paid run, the video shows a name next to every displayed number of the roster | The error names the key and, for YAML, the line (section 4). |
| Tool script | `tools/run_tool.py`, `src/pitchvlm/fallbacks/` | After the install of the two tools (section 3): the three commands of each tool, with `--max-clips 2` for `tool-job` and `tool-score`. The script was written from the source of the tools (`rfdetr` 1.12.0 and 1.12.1, PnLCalib at one commit) and never ran with them: the tests replace each tool by a stand-in. Its image reader needs Pillow and ran nowhere. RF-DETR: the recall is far above 0, and the largest box values of the run file are near the width and the height of the frames. PnLCalib: `axis_check.as_given` is the smallest of its four values and small (some metres at most), `tool-score` ends with the exit code 0, `valid_share` is above 0, and the payloads of the run file show a camera on the near side above the grass (`camera_position_m`: y above 34, z below 0). Then compare the device `mps` with `cpu` on the same two clips | An `error:` line about a version or a commit: install the version that the line names. A traceback inside a tool: the line before it names the frame on which the tool stopped. A traceback without such a line comes from before the first frame: the import of the tool or the building of its model. When it ends with `ModuleNotFoundError` or `ImportError`, a package of the tool environment is missing or has another version: repair the environment, not the script. When the first frame fails, the tool does not have the interface that the script expects: read the source of the installed version, repair `tools/run_tool.py` with a test, and do not change the tool. When one later frame fails, the tool has an error for that picture: tell the owner; the script counts no frame as a frame without an answer because of an error. An error of the operating system inside a tool (for example a full disk) gives the frame line, its traceback and one `error:` line. An error of `torch.load` about the weights of PnLCalib: the file is not plain weights. Do not switch that check off without the owner: a file that is not plain weights can run code when it is loaded. A recall near 0 with many boxes: the boxes are not in pixels of the frame, or the class names are not those of the tool (read `label` in the run file). `axis check failed` with a large `as_given`, one small other value and `plausible_share` 0: the matrix is in another pitch convention; repair `camera_homography` of the script, with a test. The same with `plausible_share` above 0: the tool mirrors the pitch on a part of the frames; do not change the script, tell the owner. All four values of `axis_check` large (about 10 m and more): the check decides nothing, and the output of the tool is wrong for another reason (device, weights, frame size); compare with `--device cpu` first and do not change `camera_homography`. No frame with a camera (`valid_share` 0), no box at all, a camera on the wrong side, or four large values also on `cpu`: read `fault` and the payloads of the run file, change no threshold and no line of the script without a cause, and tell the owner. Values of `counts`, `valid_share`, `axis_check` or the errors that differ between `mps` and `cpu`: tell the owner, and use `cpu` for the measurement. A `ValueError` of an adapter names the key or the value of a payload that does not follow its format. |
| Git hooks | `.pre-commit-config.yaml` | `doctor --no-api` shows `[ok  ] guard hooks`. The first commit that the owner asks for must list the guard and the secret scan as passed, and the first push the two push hooks. `git -C ../pitchvlm-results rev-parse --git-path hooks` must name the hook folder of this clone: the results work tree then runs the same hooks | `pre-commit install --hook-type pre-commit --hook-type pre-push`. Before 14 Nov 2026 the hooks ran in scratch clones only. Since then they ran on the build machine in this repository, for the first commits and for the pushes to `origin`. They never ran on the run machine and never with signed commits. A results work tree that names another hook folder has its own `core.hooksPath`: remove that setting. |

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

One run of the guard starts a few git processes, however many files it reads: `--paths` 2,
`--tracked` 3, `--staged` at most 8 (5 when no file is staged), `--push` at most 10 (6 when
nothing is new), and one more for the commit that the hook runner names. One process gives
the top folder and the git folder, and one process gives the type and the size of every object
that the run needs. Nothing is kept from one run to the next. When git fails, or a path of the
repository holds a line break, the guard asks for the two folders one at a time, so that an
error message names the git command that failed.

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
- An `identity` line can name a merge commit that the git host made (a pull request that was
  merged on its page). Such a commit has another author e-mail than the commits of the two
  machines, and the git host as its committer. A push to `origin` does not judge it, because
  `origin` has it. A push to another remote judges it and is refused, unless that remote has
  the commit too and `git fetch <remote>` has run: the guard knows the commits of a remote
  from the last fetch. The commit cannot be changed here: tell the owner
  (`docs/DECISIONS.md`, last list, branch `results`).
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

## 10 Image models through OpenRouter

Every model step can use an image model of OpenRouter in the place of Gemini
(`docs/DECISIONS.md`, entry "Image models through OpenRouter" of 2026-11-22). A step names
its provider, its model and the coordinate convention of that model (`provider`, `model`,
`coords` in `configs/default.yaml` and in the `step` section of an experiment file). Nothing here
ran against the real API before this machine: the first check of each model is its doctor run.

| Setting | Gemini | OpenRouter |
|---|---|---|
| `provider` | `gemini` | `openrouter` |
| `model` | a Gemini name, `gemini-4-argon` | the slug of the model page, `author/name`; a slug that ends with `:free` is refused (rule 5) |
| `thinking_level` | `minimal` to `xhigh`, the levels of the model (`config.THINKING_LEVELS`) | sent as the reasoning effort; `model_default` sends none, for a model without one |
| `resolution` | `low` to `ultra_high` | `low` or `high` (the `detail` of an image) |
| `service_tier` | `standard` or `flex` | `standard` only |
| `coords` | `yx_1000` only | `yx_1000`, `xy_1000`, `xy_unit` or `xy_pixel`: the convention that the model was trained on |

Rules that the client keeps for every OpenRouter request:

- The inputs stay private: the request is routed only to endpoints with zero data retention
  (`zdr`), never to a provider that collects data (`data_collection: deny`), only to providers
  that support every parameter of the request (`require_parameters`), and never falls back to
  another provider (`allow_fallbacks: false`). When no provider of a model meets these settings,
  the request fails with the status 503 and its error line says so. Choose another model then;
  never loosen the settings.
- No sampling parameter, no limit on the output tokens, `store` does not exist there.
- The cost of a call is the cost that the answer reports (`usage.cost`), and the ledger takes it
  as it is. The estimate before a run comes from `configs/openrouter_prices.yaml`: a model that
  is not in that file has no estimate, and its run does not start. A call served with a provider
  key of the account (BYOK) is refused as a call whose cost is unknown: keep no provider key in
  the OpenRouter account.
- Gemini and OpenRouter share one cache folder, one ledger, the 1.3 times stop and the hard stop
  at 135 USD. Each provider has its own spend window of `budget.max_usd_per_10_min`.

The key lives in the environment variable `OPENROUTER_API_KEY`, only on this machine. Set it in
each terminal that runs an OpenRouter step, the same way as the Gemini key:

```bash
# The key is typed at the prompt. It is not shown and not stored in the shell history.
printf "OpenRouter API key: " && read -rs OPENROUTER_API_KEY && export OPENROUTER_API_KEY && echo
```

- The same rules hold as for the Gemini key (section 1). An OpenRouter key starts with
  `sk-or-v1-` and has 64 hexadecimal digits after it; the guard and the secret scan know this form.
- A program of another project gets neither key: start the interpreter of a tool environment
  with `env -u GEMINI_API_KEY -u OPENROUTER_API_KEY` (the scorer gets neither from the package).
- Before the first paid call, check in the privacy settings of the OpenRouter account that the
  training of providers on the inputs is not allowed. The request settings deny it too.

First check of one OpenRouter model, gate G0. Ask the owner first: it is paid. The doctor needs
neither the Gemini key nor the SDK for an OpenRouter model.

```bash
# The machine and both keys. No model call.
python -m pitchvlm.cli doctor --no-api

# Contract checks of one OpenRouter model, in the convention that its model card names. Paid:
# 0.015 USD at the published prices of the price table (6 calls; an estimate from the Gemini
# token counts, not measured). The checks stop above the 2 USD of gate G0.
python -m pitchvlm.cli doctor --models openrouter:qwen/qwen3-vl-235b-a22b-instruct \
    --coords xy_1000 --thinking-level model_default \
    --out ../pitchvlm-results/results/metrics/doctor_qwen3_vl_xy_1000.json
```

- `--coords` holds for every OpenRouter model of the run; a Gemini model of the same run keeps
  `yx_1000`. Check one OpenRouter model per run.
- The checks `boxes` and `landmarks` are the proof of the convention. When both fail, read the
  answers under `contract.<model>.answers` of the report: boxes that are the true boxes with x and
  y swapped, or with values in pixels or in shares of the side, name the convention of the model.
  Run the doctor again with that `--coords` and a new `--out` file (it pays again). Only a
  convention that passed both checks goes into a settings file or an experiment file.
- The new lines of an OpenRouter model: `data policy` (every request and every sent call carries
  the private routing settings; a failure is a client fault), `served by` (a fact: the provider
  that served each call), `reported cost` (fails when an answer reports no cost; its detail
  compares the reported cost with the price table: above 1 means that the table price is too
  low). The line `store flag` and `served tier` are for Gemini only.
- A request that fails makes its check fail, and the doctor goes on with the next check. The
  status 503 after all 4 attempts can mean that no provider of the model meets the privacy
  settings, or only that the service was busy: run the doctor again later before you take it for
  the first. The status 400 means that a provider refused a part of the request, for example a
  keyword of the answer schema in its strict mode (`minItems`, `maxItems`). Tell the owner; do
  not change the routing settings.
- Write into `docs/DECISIONS.md`: the convention that passed, the two accuracy values, the
  served providers, the reported cost against the estimate, and the mean tokens of the measured
  cost lines. Image tokens of other models differ from those of Gemini, so the token estimates
  of `DEFAULT_TOKEN_ESTIMATES` can be far off for them.

People probe with an OpenRouter model, on the frames of the people probe of Gemini (same group,
same seed). Run it only after the doctor of the same model and convention passed. Ask the owner
first: it is paid. The experiment file is `configs/experiments/people_probe_openrouter.yaml`;
change its `coords` to the convention that passed and commit before the run.

```bash
# Cost pilot first: 2 clips. The estimate is printed before the first call.
python -m pitchvlm.cli probe people --experiment configs/experiments/people_probe_openrouter.yaml \
    --root ~/pitchvlm-data --max-clips 2 --out ../pitchvlm-results/results/metrics/pilot_people_openrouter.json
# Then the whole probe group. The answers of the pilot are used again.
python -m pitchvlm.cli probe people --experiment configs/experiments/people_probe_openrouter.yaml \
    --root ~/pitchvlm-data --out ../pitchvlm-results/results/metrics/people_probe_openrouter.json
```

- The probe asks the kit legend with the same model. Compare the metrics file with gate G2 and
  with the people probe of Gemini on the same frames. Which engine a step uses is the owner's
  decision, written into `docs/DECISIONS.md` with its numbers.
- Another OpenRouter model: add its price to `configs/openrouter_prices.yaml` from its model page
  (the highest price of its providers, with the day as `valid_from`), commit, and run its doctor
  first.
- Video input is not built (`docs/DECISIONS.md`, entry "Image models through OpenRouter" of
  2026-11-22, item 7).

What was never tested before this machine, for OpenRouter:

| Part | First check | When it fails |
|---|---|---|
| OpenRouter client (`OpenRouterBackend`, `UrllibTransport`) | the doctor of one model: the estimate and the budget line before the first check line; `data policy`, `token sum` and `reported cost` pass | A failed `data policy` or `token sum` is a client fault: repair the client. The answers were written by hand from the documentation, so the first real answer can have another shape: an error line that names a missing field (`usage`, `cost`, `choices`) means that the reader of the answer must change. |
| Cost in credits | `reported cost` of the first doctor: the sum of the reported costs, and the spend of the account page for the same calls | The documentation calls the cost "credits" in one place and USD in another. When the account page shows another amount than the ledger, stop paid OpenRouter runs and tell the owner. |
| Strict answer schema | the doctor answers parse, `boxes` and `landmarks` pass | A status 400 that names the schema: a provider does not take a keyword. Tell the owner. |
| Coordinate convention | `boxes` and `landmarks` pass with the `--coords` of the model card | Read the answers and run again with the convention that they show. |
| Reasoning effort | a doctor with `--thinking-level low` for a model whose page lists `reasoning` | A status 503 or 404: no provider of the model takes the reasoning setting. Use `model_default`. |

## 11 Detector engine (owner decision of 2026-11-22)

The pipeline has a second engine for the people step and for the pitch mapping
(`docs/DECISIONS.md`, entry "A real-time detector engine next to the model engine" of
2026-11-22). The people engine `detector` takes the people and the ball of every frame from a
detector run, gives each person a team key and a role from the kit colours, and follows the
ball. The pitch engine `calibration` takes the camera of every frame from a PnLCalib run. With
the settings file `configs/engines/detector_calibration.yaml` no step calls a model, so the run
is free. `configs/engines/detector_calibration_numbers.yaml` adds the paid shirt-number step (one
kit legend request and one number request per track): it prints its estimate like every paid run
and needs the owner's go-ahead.

The track repairs of section 4 ("The track repairs against gate G4") work with this engine too.
`run` reads one settings file, so an arm of the detector engine needs a file with the `engines`
section of `configs/engines/detector_calibration.yaml` and the `tracking` section of the arm.
Commit that file before the run (rule 7). The tracking values without labels of a detector run
count every observed frame, not keyframes: do not compare them with those of the model engine.

The tools run as in section 3, in their own environments, through `tools/run_tool.py`. The
package never imports a tool. Three commands per clip: write the clip job, run the tool on every
frame, run the pipeline. None of them ran before with a real tool, a real clip or real weights.

```bash
CLIP=~/pitchvlm-data/SoccerNetGS-valid/<clip>
SETTINGS=configs/engines/detector_calibration.yaml
mkdir -p ~/pitchvlm-data/jobs/<clip> ../pitchvlm-results/results/tools/<clip>

# 1. Clip jobs: every frame of the clip, with the tool settings of the settings file.
python -m pitchvlm.cli clip-job --tool rfdetr --config $SETTINGS --clip $CLIP --root ~/pitchvlm-data \
    --out ~/pitchvlm-data/jobs/<clip>/rfdetr.json
python -m pitchvlm.cli clip-job --tool pnlcalib --config $SETTINGS --clip $CLIP --root ~/pitchvlm-data \
    --out ~/pitchvlm-data/jobs/<clip>/pnlcalib.json

# 2. The tools on every frame. The run is written in parts of at most 1,000,000 bytes:
#    detections.json, detections-0002.json, ... A clip of 30 s with 25 people is about 0.75 MB, one part.
env -u GEMINI_API_KEY -u OPENROUTER_API_KEY ~/pitchvlm-data/envs/rfdetr/bin/python tools/run_tool.py rfdetr \
    --device mps --job ~/pitchvlm-data/jobs/<clip>/rfdetr.json --root ~/pitchvlm-data \
    --out ../pitchvlm-results/results/tools/<clip>/detections.json
env -u GEMINI_API_KEY -u OPENROUTER_API_KEY ~/pitchvlm-data/envs/pnlcalib/bin/python tools/run_tool.py pnlcalib \
    --device mps --job ~/pitchvlm-data/jobs/<clip>/pnlcalib.json --root ~/pitchvlm-data \
    --repo ~/pitchvlm-data/PnLCalib \
    --weights-kp ~/pitchvlm-data/weights/pnlcalib/SV_kp \
    --weights-lines ~/pitchvlm-data/weights/pnlcalib/SV_lines \
    --out ../pitchvlm-results/results/tools/<clip>/calibration.json

# 3. The pipeline with both tool engines. No model call: it needs no cache folder and costs nothing.
python -m pitchvlm.cli run --clip $CLIP --labels $CLIP/Labels-GameState.json --config $SETTINGS \
    --detections ../pitchvlm-results/results/tools/<clip>/detections.json \
    --calibration ../pitchvlm-results/results/tools/<clip>/calibration.json \
    --out ~/pitchvlm-data/runs/dev-detector --tracker pitchvlm-detector
```

- `run` refuses a clip run whose tool settings differ from the settings file, a run of another
  clip or of another number of frames, a missing part, a part of another run, and a payload that
  cannot be used (the line names the file and the frame). It refuses `--detections` when the
  settings do not name the people engine `detector`, and the same for `--calibration`.
- The script refuses to start when files with the names of later parts lie next to `--out`
  (`detections-0002.json` of an older run): remove them first. It deletes nothing itself.
- A clip job reads a folder of frame images. A video file is refused: its frames must first be
  written as numbered images below the data folder (no command does this yet).
- `result.json` holds `timing`: seconds, milliseconds per frame and frames per second of each
  stage, and `total`. The stages `detector` and `calibration` are the times that the script
  measured per frame on its device (without reading the image file); the other stages are the
  times of the package on this machine, with the reading of the frames. `provenance.tool_runs`
  names each clip run, its parts, its device and versions (`runner`) and whether the script is
  the one of this work tree. Nothing is "real time" until these values are measured here and
  the total is above the frame rate of the clip. Write the measured values into
  `docs/DECISIONS.md` with the device.
- Without the number step no player has a number: `run` prints a line that says so, and the
  engine table of `result.json` has `"number": "none"`. The scorer then counts the numbers of
  numbered players as wrong.
- The ball goes to the video and to the radar only. The prediction file has no ball, and the
  pitch position of a ball in the air is wrong (it is mapped as if it were on the grass).
- The clip runs are tool runs: they stay out of the results branch until the owner decides on
  the tool run files (`docs/DECISIONS.md`, last list). Each part passes the size rule of the
  guard (measured: 37.9 bytes per detection; a part of 26 detections on each of 750 frames is
  738,558 bytes, and the guard passed on it).

Measurement, fixed before any run (rule 7):

1. Gate G2 on the same frames as the people step: the experiment file
   `configs/experiments/people_probe_rfdetr_medium.yaml` (the default detector) next to the four
   files of section 3, with `tool-job`, `tools/run_tool.py` and `tool-score` as in section 3. The
   detector engine adds no box: its boxes are those of the tool, kept by the same score limit,
   so G2 measures the boxes of the engine. Compare only with the people probe of the same frames.
2. End to end on the dev clips with the same scorer: run every clip of the group `dev` with
   the commands above, each engine with its own `--tracker` name, then `eval` for each tracker
   (section 5) on the same clips. Compare GS-HOTA, DetA and AssA clip by clip, and the role and
   team-side accuracy of gate G5b. The settings of both runs are fixed before the first clip.

## 12 Plan: a detector trained on soccer (not done; no download, no training yet)

The weights of the everyday classes (COCO) know `person` and `sports ball`: they do not tell a
player from a referee, and they miss small and blurred balls. A detector trained on soccer
frames gives the classes `player`, `goalkeeper`, `referee` and `ball`. Nothing of this plan was
run; it needs the owner's go-ahead, a GPU machine and its own budget line.

Data:

- SoccerNet-GSR train split (the same source as the valid split of this project): clips of
  30 s with boxes and roles (`player`, `goalkeeper`, `referee`, `other`); the ball has
  annotations with the role `ball` that `labels.py` skips (to confirm on a label file). The
  labels are on Hugging Face (`SoccerNet/SN-GSR-2025`, tagged `gpl-3.0`, not gated); the footage
  is broadcast video for research. Treat the data and every weight file trained on it as
  research-only until the owner has read the terms (open owner decision).
- Not used: the Roboflow football datasets (licence page not readable, unverified) and the
  SoccerNet tracking video (behind an NDA).
- The valid split (probe, dev, reserve groups) and the test split are never training data.

Steps (none of them is built: the command `data` takes only `--split valid` and `--split test`
and refuses `train`, the converter does not exist, and the training has no job in
`tools/run_tool.py`):

```bash
# 1. To build: a download of the train split into the data folder (its size is not known here;
#    check the free disk first). The command `data` refuses `--split train` now.

# 2. To build: a converter that writes the COCO layout that RF-DETR trains on, with links to the
#    frame images (no copy): ~/pitchvlm-data/finetune/coco/{train,valid}/_annotations.coco.json.
#    Classes: 0 player, 1 goalkeeper, 2 referee, 3 ball; "other" is left out. Every 5th frame of
#    each train clip; 10% of the train clips (by clip, never by frame) as its valid folder.

# 3. To build: a training job of `tools/run_tool.py` (rule 4 of `CLAUDE.md`: only that script
#    imports a tool), run on the GPU machine in the RF-DETR environment without the model keys
#    (`env -u GEMINI_API_KEY -u OPENROUTER_API_KEY`). The job calls, with the API of rfdetr
#    1.12.1: RFDETRMedium().train(dataset_dir=<the coco folder of step 2>, epochs=50,
#    batch_size=8, grad_accum_steps=2, lr=1e-4, output_dir=~/pitchvlm-data/weights/rfdetr-soccer).
```

- Weights live in `~/pitchvlm-data/weights/rfdetr-soccer` on the machine that trains, and are
  copied to the run machine by hand. Never into the repository, never into the results branch.
- Before the engine can use them, three things are built: a key in the detector settings that
  names the weight file (its SHA-256 goes into the facts of the run), the load of that file in
  `tools/run_tool.py` (the source of rfdetr 1.12.1 says how a model class takes its own
  weights; to confirm), and the class names of the trained weights in `fallbacks/rfdetr.py`
  (`player`, `goalkeeper`, `referee` give a role; `ball` is a ball candidate).
- Measure the trained detector like the others: gate G2 on the probe frames, then end to end on
  the dev clips (section 11). Ultralytics weights are not an option for this repository: their
  code and every model trained with it are AGPL-3.0 (`docs/DECISIONS.md`, entry "A real-time
  detector engine next to the model engine" of 2026-11-22).
