# Handover: from the build machine to the run machine

This file says what to do, in order, when the project moves from the build machine to the run
machine. It is the first file to read on the run machine after `CLAUDE.md`. The commands and
their details are in `docs/RUNBOOK.md`; this file names the section each time.

Words:

- **Build machine**: it writes code and runs the offline tests. **Run machine**: it has the model
  API key, the dataset and the footage.
- **Owner**: the person who decides about money, commits and pushes. **Assistant**: the coding
  assistant that works for the owner on a machine.
- **Gate** (G0 to G6): a pass line that is fixed before a run, in `configs/gates.yaml`.
- **Doctor**: the command that checks the machine and, with a paid call, the contract of the
  model API (gate G0). A **contract check** tells whether the client and the API agree.
- **Oracle**: the free label stage. It scores the labels themselves to test the scoring chain
  (gate G1).
- **Probe**: a measurement of one model step on labelled frames (people, pitch, number). The
  **cost pilot** is the three probes on 2 clips.
- **Flex**: the half-price service tier of the provider. It is slower.
- **GS-HOTA**: the score of the benchmark, from 0 to 100. **Freeze tag**: the git tag after which
  no setting changes. The test clips run after it.

## 1 State at the handover (written 15 Nov 2026, updated 22 and 23 Nov 2026)

| Topic | State |
|---|---|
| Milestone | M1 (first build) is done. M2 was planned to start on 17 Nov; on 23 Nov it has not started (`docs/PLAN.md`, "Milestones"; section 12 below) |
| Code | Branch `main` of the remote (public since 14 Nov 2026). Package `pitchvlm` under `src/`, tests under `tests/` |
| Added on 22 Nov | A second model provider (OpenRouter, RUNBOOK section 10), the paint alignment of pitch mappings and the cut test (section 3), two track repairs as arms of G4 (`configs/arms`, section 4), a detector engine with `clip-job`, `run --detections --calibration` and `video-frames` (section 11), the check images of G3 (`overlay`, section 3), two test tiers (section 1), a faster guard |
| Added on 23 Nov | Video input for the model steps: `doctor --video` and three experiment files `people_probe_video*.yaml` (RUNBOOK section 13); a second detector family, D-FINE and RT-DETRv2 through transformers, with five experiment files and an install proposal (section 11); the streaming path `stream` (section 14); new tests of the money guards |
| Offline tests | `3419 passed, 1 skipped` on the build machine (Python 3.13, no model SDK installed), full run of `main` 0c2ac24 on 23 Nov; this commit adds the tests of the documents. The skipped test reads an image with Pillow, which the environment of the package does not have |
| Demo | `demo` gives a video, a radar and a valid prediction file from a synthetic clip and a fake model; `demo --engine detector` does the same with stand-ins of the two tools |
| Reviews | Two reviews tested the first build, a third one the measurement of the two specialist tools, a fourth one the fixes of the third, and a fifth one the fixes of the fourth; two more read the code of 22 Nov. Each finding was repaired, kept with its reason, or recorded as not built or as an open owner decision (`docs/DECISIONS.md`) |
| Rehearsal | On 22 Nov this file and the RUNBOOK were walked from the first line to the last on the build machine, with a synthetic dataset, the fake model, a stand-in scorer and stand-ins of the tools (`docs/DECISIONS.md`, entry "Rehearsal of the handover and the runbook"). On 23 Nov the commands of 23 Nov were walked the same way, and the estimates of every paid step were printed at both prices (entry "Rehearsal of the commands of 23 Nov, the budget at both prices and the owner decisions") |
| Real API, real scorer, real footage | Never used. Every result so far is synthetic |
| Money spent | 0 USD of the planned 120 USD |

What "never used" means in detail is in `docs/RUNBOOK.md`, section 8. Read it before day 1: the
Gemini client, the scorer wrapper, the download and the label reader run for the first time on
the run machine.

## 2 Before the code leaves the build machine

Do this section, and steps 1 to 4 of section 4.1, by 16 Nov 2026 (`docs/PLAN.md`, "Your
to-dos"). The owner does each step, or asks the assistant to do it. The assistant never commits
or pushes without being asked.

State on 22 Nov 2026: all four steps are done; nothing of this section is left to do. The
steps stay here as the record of the move. On 14 Nov 2026 the owner chose a public repository
(`docs/DECISIONS.md`, section "The code repository is public"). Its branch `main` holds the
first build and the fixes of a review: the pull request of the branch `m1-first-build` was
merged on the page of the git host.

1. Set the personal git identity for this repository only. The guard refuses a commit without
   both values.

   ```bash
   git config --local user.name "<your name>"
   git config --local user.email "<your personal e-mail>"
   ```

2. Make the first commit. A commit message has no co-author line (rule 10). The output of
   `git commit` must list the guard and the secret scan as passed.
3. Make an empty repository on a git host. Both machines need read and write access to it.
   Put no token into the address of the remote. Add it as the remote and push the branch. The
   two push hooks must pass.

   ```bash
   git remote add origin <address of the repository>
   git push -u origin m1-first-build
   ```

4. Keep the build machine as it is. It reads the text results later (section 9).

When a commit is refused, repair the file or the identity and commit again. When a push is
refused, change the refused commit itself (`git commit --amend`): a new commit on top does not
help, because the old commit goes to the remote too (`docs/RUNBOOK.md`, section 9). The hooks
never ran against a real remote before this push.

## 3 What git does not carry

Git does not carry these things. Make each one on the run machine.

| Thing | Why it does not travel | Made in |
|---|---|---|
| Python environment (`.venv`) | Ignored by git | Section 4, step 6 |
| Pinned package list | `uv.lock` is ignored. The list of the run machine is `locks/run-machine.txt` | Section 4, step 7 |
| Git hooks (`.git/hooks`) | A clone does not copy the hook files | Section 4, step 8 |
| Git identity of the repository | Local setting of one clone | Section 4, step 9 |
| Deny-list (`.git/guard-deny.txt`) | Never tracked. It holds the private words of one machine | Section 4, step 10 |
| Data folder, results work tree, scorer | They exist only on the run machine | Section 4, steps 3, 11 and 12 |
| Model API key | It lives in an environment variable of the run machine, never in a file | Section 4, steps 4 and 5 |
| Notes outside the repository | Only tracked files travel. They hold everything that the run machine needs | - |

## 4 Setup on the run machine (once)

The commands are in `docs/RUNBOOK.md`, section 1. Follow the order of the two tables below.
Most commands of this section never ran: the install with the extra `gemini`, the pinned list,
the work tree and the scorer. So read each error.

### 4.1 The owner alone, before the first session of the assistant

| # | Step | Done when |
|---|---|---|
| 1 | Install the tools: `git`, `uv`, `gitleaks` (version 8.30.1 or later), `pre-commit`, and the assistant. Keep about 45 GB of disk free | Each tool prints its version |
| 2 | Clone the repository. The clone starts on the branch `main` | `git status` shows a clean tree on that branch |
| 3 | Make the data folder: `mkdir -p ~/pitchvlm-data` | The folder exists, outside the repository |
| 4 | Google AI Studio: a project with billing (paid tier only), a budget alert, an API key | The owner has the key. It is in no file |
| 5 | Open a terminal in the top folder of the clone. Type the key with the first line of the prompt block of RUNBOOK section 1. Do not run its second line yet: the environment does not exist before step 6. Start the assistant from this terminal | The assistant runs `test -n "$GEMINI_API_KEY" && echo set`, and it prints `set` |

Why steps 3 and 5 come first:

- `CLAUDE.md` tells an assistant how to know its machine: without the key and without the
  folder `~/pitchvlm-data` it is on the build machine, and there it installs and downloads
  nothing. A new run machine has neither. So the owner makes the folder first, and tells the
  assistant: "this is the run machine".
- Only the owner types the key, at the prompt, never in a file, a command line or a chat
  message. The commands of the assistant get the key only from the terminal that started it.
  The assistant does not run the prompt command, does not ask for the key and does not print
  the environment (`env`, `printenv`). Type the key again in each new terminal.

### 4.2 With the assistant

The owner does steps 8 and 10 (`CLAUDE.md`) and step 9. The assistant does the other steps
with the owner. Do not commit before step 10 is done. Before step 8 a commit runs no guard and
can take another git identity. Before step 10 the guard does not know the private words of this
machine.

| # | Step | Done when |
|---|---|---|
| 6 | Make the environment with Python 3.13, install the package with the extras `dev` and `gemini`, run `python -m pytest -m "slow or not slow" -rs` (the full run). In each new terminal: `source .venv/bin/activate` | Every test passes. A skipped test names its reason |
| 7 | Write the pinned list `locks/run-machine.txt`. Then do the check of fact 19 (RUNBOOK, section 8): it shows whether `uv pip` reads the package age limit of `pyproject.toml`. Write the second list of that check outside the repository, for example into `~/pitchvlm-data` | The file exists. The two lists of the check are equal, or the owner knows that they differ |
| 8 | Owner: install the hooks: `pre-commit install --hook-type pre-commit --hook-type pre-push` | `doctor --no-api` shows `[ok  ] guard hooks` |
| 9 | Owner: set the git identity of this clone, with the name and e-mail of the commits of the build machine (`git log -1 --no-merges --format='%an <%ae>'` prints them; a merge commit that the git host made has another e-mail, so leave it out) | `git config --local --get user.name` and `git config --local --get user.email` print the same two values |
| 10 | Owner: write the deny-list `.git/guard-deny.txt`: one name per line that must not leave this machine (RUNBOOK, section 9). Do not list a word that is a part of the git name or e-mail: the guard then refuses every commit | `doctor --no-api` shows `[ok  ] deny-list` |
| 11 | Make the work tree of the branch `results` next to the repository: `../pitchvlm-results` | `git worktree list` shows it. `git -C ../pitchvlm-results rev-parse --git-path hooks` names the hook folder of this clone |
| 12 | Make the scorer environment (Python 3.12, numpy 1.26) and clone the repository of the scorer (`sn-trackeval`) at the tag `v0.4.1` | The last command prints the commit. Write it into `docs/DECISIONS.md` |
| 13 | Run `python -m pitchvlm.cli doctor --no-api`. Then run the command "Form of the key" (RUNBOOK, section 8, fact 11): it prints no part of the key | Exit code 0. The lines `guard hooks`, `deny-list` and `model SDK` start with `[ok  ]`, and the SDK is a version 3.x. The line `API key` is `[info] API key: the variable OPENROUTER_API_KEY is not set; the variable GEMINI_API_KEY is set` while only the Gemini key is typed, and `[ok  ]` when both keys are set (RUNBOOK, section 10). The key command prints "the pattern fits" |
| 14 | Commit the setup changes when the owner asks: `locks/run-machine.txt` and the scorer commit in `docs/DECISIONS.md` | `git status --short` prints nothing. The output of `git commit` lists the guard and the secret scan as passed |

Four things to know in this section:

- The tests never ran with the model SDK installed, and never on another machine. A failed test
  on the run machine is a finding: repair the code, not the test.
- With `--no-api`, a missing SDK and a missing key are `[info]` lines, and the exit code is
  still 0. So read the lines of step 13, not only the exit code.
- A paid command refuses to start when the git work tree has changes. This is why step 14
  comes before day 1.
- The hooks call `python3` and `gitleaks` from the path. The work tree `../pitchvlm-results`
  also holds a copy of the code: do not run code from there.

## 5 First session of the assistant on the run machine

Owner: the assistant there starts without the history of the build machine. Start it as step 5
of section 4 says. Tell it that this is the run machine. Tell it to read, in this order:
`CLAUDE.md` (rules), this file, `docs/RUNBOOK.md` (sections 1, 2 and 8 first, then 10, 11, 13
and 14), `docs/PLAN.md` (gates and budget), `docs/DECISIONS.md` (why things are as they are). The
open decisions of the owner are in one list: section 11 of this file.

Then it does section 4.2 with the owner, and then section 6. It asks before each paid command.

## 6 First session on the run machine, step by step

Do section 4.2 first. Then follow these steps in this order, one at a time. Each step names its
command, what it must print to pass, what to do when it does not, and its cost. The commands
and their details are in `docs/RUNBOOK.md` (the section is named in each step). The words to
look for were printed on 22 and 23 Nov by the same commands on the build machine, with a
synthetic dataset, the fake model, a stand-in scorer and stand-ins of the tools; the estimates
are those that the code printed on 23 Nov with the token estimates of that day, at the prices
until 30 Nov and from 1 Dec (section 8, "Budget of the first sessions").

The smallest path to a first real result is steps 1 to 11, then 14 and 15: gate G0, gate G1 and
the cost pilot on 2 clips with its measured tokens. It pays 0.785 USD until 30 Nov and 1.536 USD
from 1 Dec (estimated, not measured). Step 12 gives the first verdict of a gate (the kill switch
of G3) and is worth the same session. The other steps are optional.

| Step | What | Paid | Estimate, to 30 Nov / from 1 Dec (USD) | First result |
|---|---|---|---|---|
| 1 | Guard tests on the git of this machine | free | - | needed |
| 2 | Full run of the offline tests | free | - | needed |
| 3 | `doctor --no-api` | free | - | needed |
| 4 | Form of the key | free | - | needed |
| 5 | Doctor, gate G0 | paid | 0.133 / 0.230 | needed |
| 6 | Data (11.2 GB download) | free | - | needed |
| 7 | Free label stage, gate G1 | free | - | needed |
| 8 | Commit before the first probe | free | - | needed |
| 9 | Cost pilot: three probes on 2 clips | paid | 0.652 / 1.306 | needed |
| 10 | Replay of the pilot, paint alignment | free | - | needed |
| 11 | Measured tokens into the estimates | free | - | needed |
| 12 | Kill switch of G3 and its check images | paid | 0.248 / 0.496 (about half of it is paid: 10 of its 20 answers are stored by step 9) | recommended |
| 13 | Doctor of an OpenRouter model | paid | 0.015 / 0.015 | optional |
| 14 | Send the text results back | free | - | needed |
| 15 | Commit and push the code branch | free | - | needed |
| 16 to 19 | Install and first check of RF-DETR and PnLCalib, pixels of a video frame | free (downloads; owner's yes) | - | optional |
| 20 | Doctor with the video check | paid | 0.144 / 0.289 | optional |
| 21 | Cost pilot of the video input | paid | 0.175 / 0.350 | optional |
| 22 | First check of the second detector family | free (downloads; owner's yes) | - | optional |
| 23 | Streaming path on a dev clip | free | - | optional |

Money: a paid step starts only after the owner's go-ahead for that step. Run one paid command at
a time, and never pass `--cache`. Before each paid step, read its estimate line first: the
command prints it before the first model call, with the prices of the day of the run (UTC). The
estimates above count an empty cache; step 11 changes the estimates of every later step. A paid
command refuses a work tree with changes: commit first, when the owner asks.

**Step 1. The guard tests on the git of this machine.** Free. Section 4.2, step 6 ran the full
run once; run the tests of the guard alone too, because the guard is the commit hook and its
tests start git many times:

```bash
git --version
python -m pytest tests/test_guard.py -rs
```

- Passes: the last line starts with `152 passed` (the count of `main` 0c2ac24 on 23 Nov), with no
  failure. A skipped test names its reason.
- When not: a failed test names a git behaviour that this machine has and the build machine did
  not. Commit nothing until it is repaired: the guard is the hook of every commit. Repair
  `tools/guard.py` with a test, never the test alone, and never bypass the hook (rule 9). Write
  the git version and the finding into `docs/DECISIONS.md`.

**Step 2. The full run of the offline tests.** Free. The same command as section 4.2, step 6;
run it again here when the code changed since then (RUNBOOK, section 1):

```bash
python -m pytest -m "slow or not slow" -rs
python tools/run_tests.py -m "slow or not slow"
```

- The second command runs the same tests in several processes (RUNBOOK, section 1). From here
  on it is the test run after each change: `python tools/run_tests.py` (default run) and
  `python tools/run_tests.py -m "slow or not slow"` (full run, before a commit).
- Passes: no failure, and the second command prints the same counts of passed and skipped tests
  as the first, with exit code 0. On 23 Nov the build machine printed `3419 passed, 1 skipped` for
  `main` 0c2ac24 (later commits add tests). Here one more group of tests
  can run or be skipped: some tests need a disk that tells upper and lower case apart, some one
  that does not.
- When not: a skipped test names a missing tool (`git`, `gitleaks`, `pre-commit`) or the kind of
  disk: install the tool. The test that needs Pillow stays skipped. A failed test names code that
  this machine runs in another way: repair the code, not the test (RUNBOOK, section 8, row
  "Offline tests on this machine").

**Step 3. The machine.** Free. RUNBOOK, section 2, command 1:

```bash
python -m pitchvlm.cli doctor --no-api
```

- Passes: exit code 0, and the lines of section 4.2, step 13: `[ok  ] guard hooks`,
  `[ok  ] deny-list`, `[ok  ] model SDK: google-genai 3.<x>`, and
  `[info] API key: the variable OPENROUTER_API_KEY is not set; the variable GEMINI_API_KEY is set`
  (or `[ok  ]` when both keys are typed).
- When not: `[fail] guard hooks` or `[fail] deny-list`: section 4.2, steps 8 and 10. A SDK that
  is not a version 3.x: install the version range of `pyproject.toml`.

**Step 4. The form of the key.** Free. RUNBOOK, section 8, fact 11:

```bash
printf %s "$GEMINI_API_KEY" | LC_ALL=C grep -Eqx 'AQ\.[0-9A-Za-z_-]{20,}' \
    && echo "the pattern fits" || echo "the pattern does not fit"
```

- Passes: `the pattern fits`.
- When not: tell the owner before any result file is committed: the pattern of `tools/guard.py`
  and `.gitleaks.toml` must change first (section 7, item 2).

**Step 5. The contract of the Gemini API, gate G0.** Paid. RUNBOOK, section 2, command 2:

```bash
python -m pitchvlm.cli doctor --models gemini-4-argon,gemini-3.8-flash \
    --out ../pitchvlm-results/results/metrics/doctor.json
```

- Estimate: it prints `cost estimate: 0.133 USD for 14 planned model calls (people 8, landmarks
  6)` until 30 Nov (0.230 USD for the same calls from 1 Dec), and a budget line that ends with
  `this run stops above 2.000 USD (its spend limit in the gate file)`. The owner's go-ahead is
  needed first.
- Passes: every contract check is `[ok  ]` or `[info]`, the line `[ok  ] spend` follows, and
  exit code 0. The lines `box accuracy` and `landmark accuracy` are facts without a verdict:
  write their values into `docs/DECISIONS.md`.
- When not: a `[fail]` line of a contract check is a client fault: repair the client before any
  footage is sent (RUNBOOK, section 8, row "Gemini client", also for the one exception of the
  check `boxes`). `[fail] spend` with exit code 2: RUNBOOK, section 2 (correct the token
  estimates with the owner's go-ahead, then one model at a time).

**Step 6. The data.** Free (11.2 GB of download). RUNBOOK, section 2, command 3:

```bash
python -m pitchvlm.cli data --root ~/pitchvlm-data --split valid \
    --manifest configs/splits.json --seed 20261113
```

- Passes: `extracted <about 58> clips of the split valid into .../SoccerNetGS-valid`, then
  `split list: configs/splits.json (seed 20261113; clips per group: probe 24, dev 6, fresh 3,
  reserve <the rest>)` and the line that the test split is not downloaded. Exit code 0.
- When not: run it again: it continues the download (RUNBOOK, section 8, row "Download").

**Step 7. The free label stage, gate G1.** Free. RUNBOOK, section 2, command 4 (the one clip
first, then all clips, then the copy into the results folder).

- Passes: `[ok  ] GS-HOTA of the ground truth: <at least 99.5>` on the one clip. On all clips
  every line of `gate G1 (labels only)` is `[ok  ]`, and the line `report: .../oracle.json`.
- When not: the error names the label file and the place, or shows the last lines of the scorer
  (RUNBOOK, section 8, rows "Label reader" and "Scorer wrapper"). A team-side share below 0.97
  with a ground truth of at least 99.5 is no code fault (fact 2): write the shares into
  `docs/DECISIONS.md`; the owner decides about the gate. G0 and G1 must pass before any footage
  is sent to the model.

**Step 8. Commit before the first probe.** Free. When the owner asks: commit
`configs/splits.json`, `locks/run-machine.txt` if not yet done (section 4.2, step 14) and the
notes of steps 1 to 7 in `docs/DECISIONS.md`.

- Passes: the output of `git commit` lists `repository guard (next commit)` and
  `gitleaks (staged files, installed binary)` as `Passed`, and `git status --short` prints
  nothing.
- When not: the report line names the rule and the file (RUNBOOK, section 9).

**Step 9. The cost pilot.** Paid. RUNBOOK, section 2, command 5: the three probes with
`--max-clips 2`, one after the other.

- Estimate: people `0.196 USD for 12 planned model calls (kit_legend 2, people 10)`, pitch
  `0.124 USD for 10 planned model calls (landmarks 10)`, number `0.332 USD for 36 planned model
  calls (kit_legend 2, number 34)`: 0.652 USD together until 30 Nov, 1.306 USD from 1 Dec (0.393,
  0.248 and 0.665 USD). The owner's go-ahead is needed first.
- Passes: each probe ends with `metrics file: ...` and exit code 0; the line `measured cost`
  and the lines after it give the mean tokens of each step. The ledger
  (`../pitchvlm-results/results/cache/ledger.jsonl`) has one line for each new answer.
- When not: a stop with the words "1.3 times its estimate": do not start again; go to step 11
  first. Tell the owner the measured cost against the estimate.

**Step 10. The stored answers and the paint alignment.** Free. Replay the three probes
(`--replay`, same `--max-clips 2`, another `--out`): the metrics are the same and the line is
`measured cost: 0.000 USD (replay mode: no model call was made)`. Then check the paint alignment
of the pitch mappings on the stored answers (RUNBOOK, section 3, "Paint alignment of the pitch
mappings"):

```bash
mkdir -p ~/pitchvlm-data/checks
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \
    --root ~/pitchvlm-data --max-clips 2 --replay --out ~/pitchvlm-data/checks/pilot_pitch_replay.json
python -c "import json; from pathlib import Path; a, b = (json.loads(Path(p).expanduser().read_text())['paint_alignment'] for p in ('../pitchvlm-results/results/metrics/pilot_pitch.json', '~/pitchvlm-data/checks/pilot_pitch_replay.json')); print('same scores' if a['frames'] == b['frames'] else 'scores differ'); print({k: a[k] for k in ('n_mappings', 'n_scored', 'n_cannot_judge', 'cannot_judge_reasons')}, 'median score of the good mappings:', a['separation']['score_good']['median'])"
```

- Passes: `same scores`; `n_cannot_judge` is small against `n_mappings`; the median score of the
  good mappings is high (0.87 on the synthetic frames). Write the numbers into
  `docs/DECISIONS.md`. Two clips give no AUROC that means anything: the separation is read after
  the whole pitch probe.
- When not: `scores differ`: the score is not deterministic; tell the owner. A large
  `n_cannot_judge`: the paint mask misses the lines of real frames (its reasons are in
  `cannot_judge_reasons`); tell the owner before the score is read.

**Step 11. The measured tokens into the estimates.** Free. Write the mean tokens of the lines
after "measured cost" of step 9 (all four model steps; the output count is the out tokens plus
the thinking tokens) into `DEFAULT_TOKEN_ESTIMATES` (`src/pitchvlm/vlm/cost.py`), into
`docs/ARCHITECTURE.md` (section `vlm/cost.py`) and into `docs/DECISIONS.md`, with the owner's
go-ahead. Run `python tools/run_tests.py`, and commit when the owner asks (RUNBOOK, section 2).

- Passes: the tests pass, and a probe of step 9 with `--replay` now prints a cost estimate near
  the cost that step 9 measured.
- When not: a failed test names the place: repair it, then commit. The estimates of steps 12
  and 13 below change with this step: read the new estimate line before the go-ahead.

**Step 12. The kill switch of gate G3 and its check images.** Paid, then free. RUNBOOK, section
3: the pitch probe on the first 4 clips (20 frames), then the five check images:

```bash
python -m pitchvlm.cli probe pitch --experiment configs/experiments/pitch_probe.yaml \
    --root ~/pitchvlm-data --max-clips 4 --out ../pitchvlm-results/results/metrics/pitch_probe_4_clips.json
python -m pitchvlm.cli overlay probe --experiment configs/experiments/pitch_probe.yaml --root ~/pitchvlm-data \
    --max-clips 4 --seed 20261113 --count 5 --list ../pitchvlm-results/results/overlays/pitch_probe_kill_switch.json
```

- Estimate: with the token estimates of 22 Nov it prints `0.248 USD for 20 planned model calls
  (landmarks 20)` until 30 Nov, 0.496 USD from 1 Dec. The estimate counts an empty cache: the 10
  answers of step 9 are stored, so about half of it is paid. The owner's go-ahead is needed first.
  The overlay is free.
- Passes: the probe ends with exit code 0; the overlay prints `20 of 20 frames have a mapping`
  (or fewer: the frames without a mapping are listed with their reason), one line per image, a
  line `images:` with the full path of the folder `overlays/pitch_probe-seed20261113-count5` in
  the data folder, and `list file: ../pitchvlm-results/results/overlays/pitch_probe_kill_switch.json`.
- The check by eye (the owner): open each image; write `right` or `wrong` into the field
  `by_eye` of its entry in the list file (it is `null` until then); then count:

  ```bash
  python -c "import json; from collections import Counter; d = json.load(open('../pitchvlm-results/results/overlays/pitch_probe_kill_switch.json')); print(dict(Counter(str(f['by_eye']) for f in d['frames'])), 'of', len(d['frames']), 'images')"
  ```

  Write the counts (for example "5 of 5 right") with the name of the list file, `valid_share` and
  `error_m.median` of the metrics file into `docs/DECISIONS.md`, in an entry on the kill switch.
  The kill switch (`g3_pitch_mapping.kill_switch` of `configs/gates.yaml`) is judged only then.
- When not: a frame without a stored answer stops the overlay: give it the same `--max-clips`
  as the probe. The rule of the check by eye is in RUNBOOK, section 3, "Check images of the
  pitch mappings".

**Step 13. The first doctor of an OpenRouter model, gate G0.** Paid. RUNBOOK, section 10. First
the owner types the OpenRouter key in the terminal of the assistant (the prompt block of section
10) and checks in the privacy settings of the OpenRouter account that the training of providers
on the inputs is not allowed. Then:

```bash
python -m pitchvlm.cli doctor --no-api
python -m pitchvlm.cli doctor --models openrouter:qwen/qwen3-vl-235b-a22b-instruct \
    --coords xy_1000 --thinking-level model_default \
    --out ../pitchvlm-results/results/metrics/doctor_qwen3_vl_xy_1000.json
```

- `doctor --no-api` now shows `[ok  ] API key: the variables GEMINI_API_KEY, OPENROUTER_API_KEY
  are set`.
- Estimate: with the token estimates of 22 Nov it prints `0.015 USD for 6 planned model calls
  (people 3, landmarks 3)` (the price table of OpenRouter has no change on 1 Dec), and the budget
  line ends with `this run stops above 2.000 USD`. The owner's go-ahead is needed first.
- Passes: `data policy`, `token sum` and `reported cost` are `[ok  ]`, and `boxes` and
  `landmarks` are `[ok  ]`: the convention `xy_1000` is then proven for this model.
- When not: `boxes` and `landmarks` both fail: read the answers under `contract.<model>.answers`
  of the report, and run again with the convention that they show and a new `--out` (it pays
  again). A status 503 after all attempts, a status 400 or a missing field: RUNBOOK, section 10,
  table "What was never tested before this machine, for OpenRouter". Never loosen the routing
  settings.

**Step 14. Send the text results back.** Free. RUNBOOK, section 6:

```bash
python -m pitchvlm.cli sync --results ../pitchvlm-results/results
```

- Passes: `The guard and the secret scan found no problem.`, then three git commands. The owner
  runs the `add` and the `commit` command; the output of `git commit` lists the guard and the
  secret scan as passed. No push of the branch `results` before the owner has decided where it
  goes (section 9). After a tool run (steps 16 to 23) the list holds lines `tool run: tools/...`,
  and the printed `add` takes the folder `results/tools` too: it waits for its own owner decision
  (RUNBOOK, section 3; section 11 below). Without a yes, move the folder `results/tools` to
  `~/pitchvlm-data` first.
- When not: a report line names the rule and the file. A false alarm on the text of a real
  answer needs a change of the rule with a test, not of the answer.

**Step 15. Commit and push the code branch.** Free. When the owner asks: commit the notes of the
day in `docs/DECISIONS.md`, and push the code branch: each metrics file names the commit of the
code. The output of `git push` lists the two push hooks as passed.

The steps that follow need the owner's yes to the install of the two specialist tools (it
downloads packages and about 1 GB of weights) and the licence decision (`docs/DECISIONS.md`,
last list). They are free. Do them in the first session when the owner says yes, else on the
next day.

**Step 16. The install of the two tools.** RUNBOOK, section 3, the install block. Write the
versions of `uv pip freeze` of both environments into `docs/DECISIONS.md`.

- Passes: `ls -l ~/pitchvlm-data/weights/pnlcalib` shows `SV_kp` with 264964645 bytes and
  `SV_lines` with 264857893 bytes.
- When not: RUNBOOK, section 3 (the packages without a fixed version get the package age limit
  of fact 19).

**Step 17. The first check of each tool on 2 clips.** RUNBOOK, section 3, the three commands of
RF-DETR Small and of PnLCalib, with `--max-clips 2` for `tool-job` and `tool-score`, first with
`--device mps`, then with `--device cpu` and other `--out` files.

- Passes: the checks of RUNBOOK, section 8, row "Tool script": RF-DETR recall far above 0;
  PnLCalib `axis check, median error in m: as_given <some metres at most>` as the smallest of
  its four values, `tool-score` exit code 0, `valid_share` above 0. `mps` and `cpu` give the
  same `counts`, `valid_share`, `axis_check` and errors.
- When not: the same row of section 8 says what each sign means.

**Step 18. The paint alignment of PnLCalib.** The metrics file of step 17 for PnLCalib holds
`paint_alignment`: read it as in step 10 and write the numbers of both engines into
`docs/DECISIONS.md`.

**Step 19. The pixels of a video frame in a tool environment.** Needs a video below
`~/pitchvlm-data/footage` (the demo clip, `docs/PLAN.md`, to-do 5). First the frames:

```bash
python -m pitchvlm.cli video-frames --video ~/pitchvlm-data/footage/<clip>.mp4
```

- Passes: `video <clip>.mp4: <frames> frames, <rate> frames per second, ...`, then
  `frames: <frames> PNG images, <size> MB, in ~/pitchvlm-data/videos/<clip>/img1` (about 1 GB
  for 30 s at 1920 x 1080, estimated). When not: a video whose frames are not sure is refused
  with one line; convert it to a constant frame rate (RUNBOOK, section 11).

Then the two commands of RUNBOOK, section 11, "A video file as the clip" (the check of the
pixels): frame 0 as the script reads it with Pillow, against frame 0 of `run`.

- Passes: `same pixels`.
- When not: tell the owner before a tool runs on the video.

The steps 20 to 23 came with the code of 23 Nov. Each one is optional: none of them is needed
for the first result, and the probes of M3 do not wait for them. Steps 20 and 21 are paid (owner's
go-ahead first); step 22 needs the owner's yes to an install; step 23 needs the clip runs of
RUNBOOK, section 11.

**Step 20. The doctor with the video check.** Paid. RUNBOOK, section 13:

```bash
python -m pitchvlm.cli doctor --video \
    --out ../pitchvlm-results/results/metrics/doctor_video.json
```

- Estimate: it prints `cost estimate: 0.144 USD for 8 planned model calls (people 4, landmarks
  3, people_video 1)` until 30 Nov and `0.289 USD` for the same calls from 1 Dec, and a budget
  line that ends with `this run stops above 2.000 USD (its spend limit in the gate file)`. The
  doctor stores no answers, so the 7 image checks of gemini-4-argon are paid again. The owner's
  go-ahead is needed first.
- Passes: `video boxes` and `video times` are `[ok  ]`; `video box accuracy` is a fact; the line
  `video tokens` gives the measured input tokens next to the estimate of 2,784. Exit code 0.
  Write the values into `docs/DECISIONS.md`.
- When not: RUNBOOK, section 13, table "What was never tested before this machine, for video
  input". A status 400 on the video call: tell the owner; do not change the request by hand.
- One run can check both models of step 5 with video (`doctor --video --models
  gemini-4-argon,gemini-3.8-flash`: 0.198 USD for 16 calls until 30 Nov, 0.343 USD from 1 Dec) in
  the place of steps 5 and 20. It saves the second payment of the image checks, but a failed video
  check then fails the exit code of the run that decides gate G0. Keep the two steps apart.

**Step 21. The cost pilot of the video input.** Paid. Needs step 9 (the image pilot of the same
2 clips is the baseline) and step 20 passed. RUNBOOK, section 13:

```bash
python -m pitchvlm.cli probe people --experiment configs/experiments/people_probe_video_low.yaml \
    --root ~/pitchvlm-data --max-clips 2 --out ../pitchvlm-results/results/metrics/pilot_people_video_low.json
```

- Estimate: `0.175 USD for 12 planned model calls (kit_legend_video 2, people_video 10)` until
  30 Nov, `0.350 USD` from 1 Dec. The videos are written into `~/pitchvlm-data/video-input`
  before the estimate is printed. The owner's go-ahead is needed first.
- Passes: `metrics file: ...` and exit code 0; `n_entries_wrong_time` of the metrics file is 0.
  Compare recall and precision with `pilot_people.json` of step 9: the same frames. Two clips
  decide nothing; the three arms of RUNBOOK, section 13, on 24 clips do.
- When not: a stop with the words "1.3 times its estimate": the video token rate is too low
  (RUNBOOK, section 13, line `video tokens`); tell the owner before a second try.

**Step 22. The first check of the second detector family.** Free; it downloads packages and a
checkpoint (79 MB for D-FINE Medium, estimated). It needs the owner's yes to the install and the
licence decisions of section 11. RUNBOOK, section 11, "A second detector family: D-FINE and
RT-DETRv2 (2026-11-23)": the install block, then the three commands of
`people_probe_dfine_medium.yaml` with `--max-clips 2` on `tool-job` and `tool-score`.

- Passes: `tool-job` prints `tool job people_probe_dfine_medium: 10 frames of 2 clips of the group
  probe (valid) for the tool dfine` (printed on the build machine); the script ends with
  `dfine: 10 frames of the job ...`, a line `median time per frame of each part: preprocess ...,
  model ..., postprocess ...` and `run file: ...`; `tool-score` ends with exit code 0. Write the
  versions of `uv pip freeze` into `docs/DECISIONS.md`.
- When not: a stop of the self-check of the first call means that the output of the model is
  not the one that the script was written for: tell the owner; do not loosen the check.

**Step 23. The streaming path on a dev clip.** Free. Needs the clip runs of both tools for one
clip of the group `dev` (RUNBOOK, section 11, the three commands; steps 16 and 17 first).
RUNBOOK, section 14, the second command, then the same with `--delay clip` and another `--out`
(the control).

- Passes: the first line `clip <clip>: 750 frames, 25 frames per second; streaming path with a
  delay of ...`, the lines of the times per frame of each stage (median / 95th percentile /
  largest, milliseconds), `result: .../result.json (<n> tracks)`, exit code 0. Write the 95th
  percentile of `total` with the device and the load average into `docs/DECISIONS.md`. These
  times hold no tool: a claim of real time needs the tools on the same machine at the same time
  (RUNBOOK, section 14).
- When not: an error line that names the settings: only the engines of
  `configs/engines/detector_calibration.yaml` stream (no number step).

M3 then measures both tools on all frames of the people probe and the pitch probe (RUNBOOK,
section 3), with the check images of M3 (`--count 10`). Since 22 Nov the pipeline can also use
the answers of a tool: the detector engine and the calibration engine (RUNBOOK, section 11).
Whether a tool replaces a model step stays the owner's decision under rule 8.

## 7 The first-day checks that matter most

`docs/RUNBOOK.md`, section 8, has the full list (11 parts and 21 open facts); section 10 adds
5 parts of the OpenRouter client. These decide whether the next days can start:

1. The Gemini client against the real API: step 5 of section 6 is its first real call.
2. The form of the real key (step 13 of section 4). When the pattern does not fit, tell the
   owner before any result file is committed. For a key that starts with `AQ.`, the guard
   cannot find a leaked key then. An older key that starts with `AIza` never fits this command,
   and the guard knows that form.
3. The scorer wrapper: `oracle` on one clip must give at least 99.5 for the ground truth.
4. The label reader on real files: `oracle` on all clips stops at the first value that it
   refuses, and names the file and the place.
5. The bill of the provider against the ledger after the cost pilot (fact 16).
6. The rate limits of the project on the console page of the provider (fact 18).

## 8 Money rules in one place

- Planned 120 USD, hard stop 135 USD. The budget guard knows the spend only from the cost
  ledger in the cache folder. So use one cache folder for all paid runs: do not pass `--cache`.
  The default cache folder is `../pitchvlm-results/results/cache`.
- Run one paid command at a time. Two commands do not see the spend of each other.
- Every paid command prints its estimate first and the measured cost after.
- A run stops above 1.3 times its estimate. One exception: each doctor run stops above 2 USD.
- The doctor stores no answers. A doctor that is started again pays again.
- Prices: the table `PRICES` of `src/pitchvlm/vlm/cost.py`. Each price holds from its start day
  (UTC) until the next price of the same model. The estimate takes the prices of the day on
  which the command starts (UTC); the cost of an answer takes the prices of the day of the
  answer (UTC). The prices of `gemini-4-argon` and `gemini-4-carbon` double on 1 Dec 2026; those
  of `gemini-3.8-flash` and `gemini-robotics-er-2-preview` double on 1 Jan 2027. The prices of
  the OpenRouter models (`configs/openrouter_prices.yaml`) have no change of date.
- A standard-price run that starts on 30 Nov and is still running at 00:00 UTC pays the answers
  after midnight at the new price, against an estimate at the old price: the 1.3 times stop can
  end it. Start no long run late on 30 Nov (UTC).
- The flex tier costs half of the standard price. The plan uses it for the clip runs from M5 on;
  the doctor checks it (line `flex tier`) before the first run through it.
- No paid run without the owner's go-ahead.

### Budget of the first sessions

Every number below is the estimate that the code printed on 23 Nov on the build machine for the
command of the step, with the token estimates of that day (`DEFAULT_TOKEN_ESTIMATES`), an empty
cache and the day set to 30 Nov and to 1 Dec (entry "Rehearsal of the commands of 23 Nov, the
budget at both prices and the owner decisions" of `docs/DECISIONS.md`). They are estimates, not
measurements: step 11 of section 6 replaces the token estimates with measured ones, and every
estimate after it changes. The probes were planned on synthetic clips of 750 frames: the number
of calls of a probe is that of the real split list (24 clips of the group `probe`, 5 frames and
17 tracks per clip); the video estimates depend on where the sampled frames lie in a clip. The
running total assumes an empty ledger and counts each estimate in full, so it is an upper bound:
a stored answer is not paid again (step 12 pays about half of its estimate).

| Step | Paid command | Needed | Calls | Estimate to 2026-11-30 | Estimate from 2026-12-01 | Run stops above (to / from) | Running total (to / from) | Left to the hard stop (to / from) |
|---|---|---|---|---|---|---|---|---|
| 5 | Doctor, gate G0: `doctor --models gemini-4-argon,gemini-3.8-flash` | needed | 14 | 0.133 | 0.230 | 2.000 / 2.000 | 0.133 / 0.230 | 134.867 / 134.770 |
| 9 | Cost pilot, people: `probe people`, `people_probe.yaml`, `--max-clips 2` | needed | 12 | 0.196 | 0.393 | 0.255 / 0.511 | 0.329 / 0.623 | 134.671 / 134.377 |
| 9 | Cost pilot, pitch: `probe pitch`, `pitch_probe.yaml`, `--max-clips 2` | needed | 10 | 0.124 | 0.248 | 0.161 / 0.322 | 0.453 / 0.871 | 134.547 / 134.129 |
| 9 | Cost pilot, number: `probe number`, `number_probe.yaml`, `--max-clips 2` | needed | 36 | 0.332 | 0.665 | 0.432 / 0.864 | 0.785 / 1.536 | 134.215 / 133.464 |
| 12 | Kill switch of G3: `probe pitch`, `pitch_probe.yaml`, `--max-clips 4` (10 of its 20 answers are stored by step 9) | recommended | 20 | 0.248 | 0.496 | 0.322 / 0.645 | 1.033 / 2.032 | 133.967 / 132.968 |
| 13 | Doctor of an OpenRouter model: `doctor --models openrouter:qwen/qwen3-vl-235b-a22b-instruct` | optional | 6 | 0.015 | 0.015 | 2.000 / 2.000 | 1.048 / 2.047 | 133.952 / 132.953 |
| 20 | Doctor with the video check: `doctor --video` (gemini-4-argon; it pays its 7 image checks again) | optional | 8 | 0.144 | 0.289 | 2.000 / 2.000 | 1.192 / 2.336 | 133.808 / 132.664 |
| 21 | Video pilot: `probe people`, `people_probe_video_low.yaml`, `--max-clips 2` | optional | 12 | 0.175 | 0.350 | 0.227 / 0.455 | 1.367 / 2.686 | 133.633 / 132.314 |

- The two limits of rule 6 of `CLAUDE.md`: each run stops above the value of the column "Run
  stops above" (1.3 times its estimate; 2 USD for a doctor run); all runs together stop at 135
  USD. The needed steps (5 and 9) use 0.785 USD of the 2 USD that `docs/PLAN.md` plans for the
  doctor and the cost pilot until 30 Nov, and 1.536 USD from 1 Dec.
- Worth doing before 1 Dec: every step of this table, and the full probes of M3 below. Their
  standard price until 30 Nov is the flex price after it, without the waits and the stops of a
  busy flex tier. From 1 Dec the same probes cost twice as much at the standard price.

The whole plan, one pass of each item of the budget of `docs/PLAN.md` at the printed estimates
(probes: one setting each, the number probe two settings; clips: the model engine, 179 calls of a
clip of 30 s with 50 number requests, the guess of the code):

| Item | PLAN line, USD | Standard to 2026-11-30 | Flex to 2026-11-30 | Standard from 2026-12-01 | Flex from 2026-12-01 |
|---|---|---|---|---|---|
| Doctor + cost pilot | 2 | 0.79 | 0.79 | 1.54 | 1.54 |
| People probe, images (1 setting) | 14 | 2.36 | 1.18 | 4.71 | 2.36 |
| Pitch probe (1 setting) | 15 | 1.49 | 0.74 | 2.98 | 1.49 |
| Number probe (408 tracks, 2 settings) | 11 | 7.98 | 3.99 | 15.96 | 7.98 |
| Dev clips, 6 x 2 passes, model engine | 28 | 28.14 | 14.06 | 56.28 | 28.14 |
| Fresh clips, 3 | 6 | 7.04 | 3.52 | 14.07 | 7.04 |
| Test clips, 10 | 21 | 23.45 | 11.72 | 46.90 | 23.45 |
| Demo clip, 2 passes (30 s assumed) | 6 | 4.69 | 2.34 | 9.38 | 4.69 |
| **All** | 103 of 120 (17 reserve) | 75.92 | 38.34 | 151.81 | 76.67 |
| Not in the plan: people probe, video (3 arms of RUNBOOK 13) | - | 6.75 | 3.38 | 13.51 | 6.75 |

- The row "All" sums the items of the plan only; the last row (the video arms) is not in it.
- From 1 Dec, the whole plan at the standard price (151.81 USD) does not fit under the hard stop
  of 135 USD. Through the flex tier it is 76.67 USD. With the doctor, the pilot and the probes run
  before 1 Dec at the standard price (12.61 USD) and the 27 clip passes after it through flex:
  75.92 USD.
- When the flex tier does not work after 1 Dec, the 27 clip passes at the standard price bring
  the plan to 139.24 USD with those probes: above the hard stop. The cut of `docs/PLAN.md` (one
  dev pass, 6 test clips: 17 clip passes) gives 92.34 USD.
- Not in the plan of `docs/PLAN.md`: the three video arms (last row) and the number step of the
  detector engine (0.463 USD per clip until 30 Nov, 0.926 USD from 1 Dec;
  `configs/engines/detector_calibration_numbers.yaml`). The detector engine without it is free.
- The lines of `docs/PLAN.md` for the people probe (14 USD) and the pitch probe (15 USD) hold
  several settings; one setting prints 2.36 and 1.49 USD until 30 Nov. The line of the number
  probe (11 USD for two settings) is too small at the standard price from 1 Dec (15.96 USD).

## 9 Sending results back to the build machine

Only text goes back: metrics files, stored model answers, the cost ledger. They live in
`../pitchvlm-results/results` and travel on the branch `results` (RUNBOOK, section 6). The run
files of the two specialist tools are written there too. Whether they travel is an open owner
decision (`docs/DECISIONS.md`, last list).

The remote `origin` is public since 14 Nov 2026. Do not push the branch `results` to it before
the owner has decided where that branch goes (same list).

1. On the run machine: `python -m pitchvlm.cli sync --results ../pitchvlm-results/results`. It
   checks the files and prints three git commands. It runs none of them.
2. The owner runs the `add` and the `commit` command. After a tool run, the folder
   `results/tools` waits for its own owner decision first (RUNBOOK, section 3). The `push`
   command waits for the decision above.
3. After that decision, on the build machine: `git fetch origin`. The first time, make a work
   tree for the branch:
   `git worktree add ../pitchvlm-results results`. Later: `git -C ../pitchvlm-results pull`.
   Then read the files.

Never send back: videos, images, crops, label files, prediction files, `result.json`, the key.
The guard refuses them. Do not bypass it (rule 9).

## 10 Code changes on the run machine

- Small, tested changes on the code branch. The default run `python tools/run_tests.py` passes
  after each change, and the full run `python tools/run_tests.py -m "slow or not slow"` before a
  commit (RUNBOOK, section 1; `python -m pytest` runs the same tests in one process).
  Commit when the owner asks.
- A paid command refuses to start when the git work tree has changes.
- After a change of `tools/`, `.gitleaks.toml`, `.gitignore` or `.pre-commit-config.yaml`, merge
  the code branch into `results` before the next commit of results (RUNBOOK, section 6).
- Gates are in `configs/gates.yaml`. A change of a gate needs an entry in `docs/DECISIONS.md`.

## 11 Decisions

The decisions and their reasons are in `docs/DECISIONS.md`. These five change what an assistant
does:

- Gate G0 judges the contract of the API only. Accuracy values are reported without a verdict.
- Each doctor run stops above 2 USD. Its report keeps the cost of every paid answer.
- A stored answer keeps the frame numbers of its request and no box of the request. The boxes
  that the model answers stay in the answer text. The number probe keeps the track id of the
  label file in the name of its requests.
- A commit message has no co-author line.
- The two specialist tools, RF-DETR and PnLCalib, are measured in M3 on the frames of the people
  probe and the pitch probe, without training and for free (RUNBOOK, section 3). A tool still
  replaces a Gemini step only after that step failed its gate and the tool passed the same gate
  on the same frames.

### 11.1 Open decisions of the owner, in one list (23 Nov 2026)

Every open owner decision of `docs/DECISIONS.md` is here, with its options, the recommendation of
the session that raised it, what happens when nothing is decided, and what it blocks. The source
entry is named in brackets; its text has the details. The list decides nothing: the owner does.
It is grouped by the first step that waits for it.

Before the first session on the run machine:

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 1 | Pull request 2 of the remote (open): remove the secret scan, and allow the API key in a git-ignored `.env` [the body of the pull request; owner decisions of 16 and 22 Nov that are not on `main`] | merge it; close it; make it again on the `main` of today | This session: decide it before section 4.1, step 5: it changes how the key reaches the assistant (an assistant that is not started from a terminal does not get its environment) and what the hooks run. The branch is older than five rounds of `main`: a merge needs the rehearsal of section 4 again | `main` keeps the secret scan and the key typed at the prompt; the pull request stays open | section 4.1, step 5, when the assistant is not started from a terminal |
| 2 | Text of rule 4 of `CLAUDE.md` [entries "Image models through OpenRouter" of 22 Nov, item 9; "A second detector family" of 23 Nov, item 11] | accept the proposed text (the client talks to both providers; the tool script also imports transformers for D-FINE and RT-DETRv2); change it; refuse it | Accept: the code keeps it already | The rule names only the model SDK, RF-DETR and PnLCalib, while the code has OpenRouter and transformers | nothing in the code; an assistant reads a rule that names fewer tools than the code has |
| 3 | Text of rule 5 and of the first paragraph of `CLAUDE.md` [same OpenRouter entry, item 9; "Video input for the model steps" of 23 Nov, item 7] | accept (paid models only, private routing of OpenRouter, "Video goes inline only; no file is uploaded to a provider"); change; refuse | Accept | as rule 4 | as rule 4 |
| 4 | Text of rule 6 of `CLAUDE.md` [entry "A planted-bug check of the money guards" of 23 Nov, item 4] | the proposed text says "costs more than 1.3x" and "more than 2 USD", as the code does; keep the words "stops at" | Accept: the code stops above the limit, and the difference is one exact amount of money | The rule says "at", the code "above" | nothing |
| 5 | Text of rule 8 of `CLAUDE.md` for two first-class engines [entry "A real-time detector engine next to the model engine" of 22 Nov, item 7] | accept; change; refuse | Accept after the first G2 and dev values of both engines; it assumes rules 4 and 5 above | Rule 8 names only "a specialist model may replace a Gemini step" | the choice of the engine of a step on the dev clips |
| 6 | Section "Commands" of `CLAUDE.md` [entries "A faster default run of the offline tests", item 8; "Rehearsal of the handover and the runbook", item 8; "The offline tests in several processes", item 11; this rehearsal] | the latest text, item 8 of the entry of this rehearsal; an older one; none | The latest: it names `tools/run_tests.py` and the commands of 23 Nov | The section names `python -m pytest` for the tests and 10 of the 14 sub-commands | nothing |
| 7 | Install rule of the build machine [last list of `docs/DECISIONS.md`] | keep; change | Keep: a clone on the build machine needs the rule | kept | nothing |
| 8 | Two texts that the guard refuses although they are no leak (`/home/<word>`, a bare list of 32 or more check sums) [same list] | keep both refusals; allow one | Keep both | kept | nothing |
| 9 | A tier for the process tests: a marker for the tests that start git, or `pytest-xdist` [entries "A faster default run of the offline tests", item 9; "Fewer git processes in the guard", item 9] | a marker run only before a commit; `pytest-xdist`; as now | This session: as now. Since 22 Nov `tools/run_tests.py` runs the default run in several processes (82 s on 23 Nov at a load average of 12, 3,410 tests), and the guard tests stay in it | as now | nothing |

Before the first paid step:

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 10 | New dates of the schedule [entry "Rehearsal of the handover and the runbook" of 22 Nov; section 12 of this file] | new dates; the order of section 12 without dates | Set the dates after the first session, with the measured cost of the pilot | The dates of 15 Nov stay in `docs/PLAN.md`; every one from M2 on has passed or cannot hold | planning only |
| 11 | Which paid steps run before the price change of 1 Dec [this rehearsal; section 8] | the first sessions and the full probes before 1 Dec at the standard price, the clip runs after it through flex; everything after 1 Dec through flex; everything through flex now | The first: the probes cost the same as through flex later, without its waits, and at the standard price after 1 Dec they cost twice as much (12.61 against 25.18 USD with the doctor and the pilot, estimated, without the video arms) | Each step costs the price of its day | money only |
| 12 | Requests in flight at a budget stop [last list of `docs/DECISIONS.md`] | stop after the answer that crosses a limit (now: up to `max_parallel - 1` more answers are paid); count the estimate of every call in flight before a new call | Decide after the cost pilot, when the real cost of one answer is known | now: with 4 requests in flight up to 3 answers above a limit are paid | nothing |
| 13 | A ledger line for an answer whose cost is not reported [entry "Second read of the model provider and money code" of 22 Nov, "Open, not fixed"] | as now: no ledger line, and the run stops at the first such answer; price it at the table price as an upper bound, with a ledger field that says so | This session: decide after the first OpenRouter doctor (section 6, step 13) shows whether `usage.cost` is ever missing | as now | nothing; an OpenRouter run stops at the first answer without a cost |

Before the specialist tools (section 6, steps 16 to 19 and 22; M3):

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 14 | Licence of this repository, and what the GPL of PnLCalib and the AGPL mean for `tools/run_tool.py` [entry "The code repository is public"; last list] | a licence file (which licence is open); no licence | Decide both before the tools are installed, and add a licence file; until then add only tools whose code and weights are under Apache-2.0, MIT or BSD, besides PnLCalib | The public code has no licence; no AGPL tool is added | section 6, step 16 (as recommended) and any use outside research |
| 15 | Terms of the training data of the detectors: the COCO images (Flickr Terms of Use) for every detector here, RF-DETR too, and the pretraining data of the backbones, which no card names [entry "A second detector family" of 23 Nov, items 2 and 11] | accept for research; read the terms first; use no COCO-trained weights | Decide both with the licence of the repository, before a demo is shown outside research | research use only | a demo outside research |
| 16 | Settings of the tool measurements [last list] | the settings of the tools as they come first; more settings at once | As they come first; add one setting at a time where a gate is missed | as recommended (the experiment files are fixed) | nothing |
| 17 | Run files of the tools and the clip runs on the results branch [last list, two items] | let them return (text: boxes and camera matrices); keep them on the run machine | Let them return; then add them to rule 2 of `CLAUDE.md`, RUNBOOK 6, section 9 here and `docs/PLAN.md` ("Two machines") | the folder `results/tools` is moved out before each commit of results | reading tool results on the build machine |
| 18 | Frames of gate G3 [last list] | keep the text of the gate (people-probe frames plus 3 dev clips) and add a run on the group `dev` with `--max-clips 3` for both engines when a value is near a gate line; change the gate | The first | only the group `probe` is measured | the G3 verdict near a gate line |
| 19 | Size of the detector [last list] | the smallest RF-DETR size that passes G2; Medium | The smallest that passes; Medium until then | Medium | nothing |
| 20 | The second detector family: install its environment and run its five experiment files [entry "A second detector family" of 23 Nov, item 11] | install and run with `--max-clips 2` first; not now | Install; keep RF-DETR Medium the default until G2 has numbers of both families on the same frames | not installed | section 6, step 22 |
| 21 | Check images of G3 are not blind: the caption shows the paint score and the images are numbered in score order [entry "Second read of the dissolve test, the video frames, the overlays and the guard", item 7] | keep; hide the score and shuffle the order | This session: keep for the 5 images of the kill switch (the gate asks for them as they are); decide before the 10 images of M3 | kept | nothing |

Before the clip runs (M4, M5):

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 22 | Label penalty of the tracker [last list] | growth with missed keyframes for role and team key; for the team key only | Keep now; take the team key only when G2 shows about 10% wrong roles per keyframe | kept | nothing |
| 23 | Team side with goalkeeper evidence, (a) to (c) [last list] | do (a) and (b); do (c) too; none | (a) and (b) after the label stage, (c) later | none | nothing |
| 24 | The paint score as a rule of `check_homography` [entry "Paint alignment of pitch mappings, and hard cuts", item 4] | a rule per engine that meets item 4; a diagnostic only | A rule only for an engine that meets every condition of item 4 | diagnostic only | nothing |
| 25 | Shot type in the landmark answer [same entry, item 8] | add the field (a new prompt and new cache keys); not | Decide after G3, with the cut counts of the demo clip | not added | nothing |
| 26 | Tracks and states at a cut [same entry, item 9] | keep both rules (a shown number joins tracks across a cut; no state between the last keyframe and the cut); change | Keep | kept | nothing |
| 27 | Measurement script `tools/synthetic_diagnostics.py` in `tools/` [same entry, item 9 (e)] | keep it there; move it | Keep | kept | nothing |
| 28 | G4 limit per clip or over the clips [entry "Track repairs and tracking values without labels", item 11 (a)] | per clip; over the 6 dev clips | Per clip | per clip (the RUNBOOK script) | the G4 verdict |
| 29 | Default after G4 [same entry, item 11 (b)] | a repair that passes G4 becomes the default; only when it also keeps AssA | Only when it passes G4 and does not lower AssA | no repair is the default | nothing |
| 30 | A limit for `purity` of the track repairs [same entry, item 11 (c)] | a limit before the dev runs of the arms; none | Fix it before those runs (rule 7) | no limit: a repair can pass G4 and mix people | the dev runs of the arms |
| 31 | Cost of the split arm [same entry, item 11 (d)] | team slots first, then the split arm after its estimate; both together | Team slots first | as the owner starts them | nothing |
| 32 | Video input in the pipeline (`run`) [entry "Video input for the model steps" of 23 Nov, item 7] | build it after the probes; never | Decide after the probes, with both metrics files | `run` sends images | nothing |
| 33 | Uploads of a video above 20,000,000 bytes (the Files API keeps a file for 48 hours) [same entry, item 7] | keep the refusal; an upload followed by a delete | This session: keep the refusal while only the probes send video (a 2.5 s file of 1080p frames is far below the limit) | refused before any call | nothing now |
| 34 | A video rate for an OpenRouter model [same entry, item 7] | add a `video` entry to `configs/openrouter_prices.yaml` from the model card of one model, then `doctor --video`; none | None published yet: add one only from a model card | every OpenRouter video request is refused before any call | video input through OpenRouter |
| 35 | OpenRouter: the price table on the estimate side, the token estimates per model, which models to try [last list, three items] | price table: the highest price of any provider, or that of the provider that the routing chose; token estimates: one table for all models, or per model; models: which to try first | Keep the highest price; estimates per model when the first doctor of a model differs by more than 1.5 times; Qwen3-VL first | the highest price, one table for all models, no model tried | the OpenRouter probes |

The streaming path (live):

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 36 | Default delay of `stream` [entry "A streaming path for the detector engine" of 23 Nov, item 10] | 25 frames (1 s); 5 or 12 frames | 25 frames | 25 frames (the code) | nothing |
| 37 | A dissolve within the delay [same entry, item 10] | accept that it is linked through; a delay of at least about 42 frames for clips with dissolves; a shorter window of the dissolve test for the stream (a new measurement) | none given by the entry | linked through | live runs with dissolves |
| 38 | The paint score of the keyframes in a live run [same entry, item 10] | keep it (it sets most of the 95th percentile); another thread; leave it out of `stream` | none given by the entry | kept | a claim of real time |
| 39 | The live mode of `tools/run_tool.py` and a merger of the two tool pipes [same entry, item 10] | build; not | none given by the entry | `stream --lines` reads one file or one pipe (the live mode of the script is being built in round 6; the merger is not) | a live run with both tools |

Results and publishing:

| # | Decision [source] | Options | Recommendation | If nothing is decided | Blocks |
|---|---|---|---|---|---|
| 40 | Where the branch `results` goes: the remote is public [entry "The code repository is public"; last list] | keep it on the run machine; a second remote that is private; a change of the identity rule of the guard | Do not push it to the public remote before this is decided | it stays on the run machine; the build machine reads no results | section 9, step 3 |
| 41 | A detector trained on soccer [RUNBOOK, section 12; last list] | read the terms of the SoccerNet-GSR train data, then build with a budget line for the GPU time; not | Terms first, then a budget line and a decision that research-only weights may drive the demo | not built | nothing of the first result |

Built, and waiting for a step of the owner (no decision): the check by eye of the images of the
kill switch of G3 (section 6, step 12), and the demo clip below `~/pitchvlm-data` with the check of
its pixels (section 6, step 19; `docs/PLAN.md`, to-do 5).

## 12 Schedule after the handover

These are the dates of the plan of 15 Nov, with their state on 23 Nov. No step of M2 has run, so
no date from M2 on holds. New dates are an owner decision (section 11.1, decision 10); this file
names none. The price change of 1 Dec 2026 does not move (section 8).

| Milestone | Dates of the plan (2026) | Result | State on 23 Nov |
|---|---|---|---|
| M2 | 17-18 Nov | Setup, G0, G1, cost pilot, install of the fallback tools. Spend at most 2 USD | passed; not started |
| M3 | 19-24 Nov | People, pitch and number probes. RF-DETR and PnLCalib measured on the same frames, for free. G2 and G3 decided by 24 Nov | cannot hold: it needs M2 first |
| M4 | 24-26 Nov | First end-to-end clip. A watchable video on 26 Nov. Demo clip and roster from the owner by 22 Nov | cannot hold; the date of the demo clip has passed, and whether it is there is not known on the build machine |
| M5 | 27-30 Nov | 6 dev clips run and scored through the half-price tier: identity (G4), numbers (G5b), first GS-HOTA | cannot hold before 1 Dec: it needs G2 and G3 |
| M6 | 1-5 Dec | Fixes, 3 fresh clips once, freeze tag | at risk: it needs M5 |
| M7 | 6-10 Dec | 10 test clips once, final demo clip, report | at risk: it needs the freeze tag |

The rule of `docs/PLAN.md` for a date that slips by more than 2 days cuts, in this order: extra
models, Gemini-carried IDs, the test set from 10 to 6 clips, names. Every date has slipped by more;
which cuts apply is the owner's decision.

Proposed order of the steps, without dates (the owner sets them):

1. The first session (section 6, steps 1 to 12, and 14 and 15): G0, G1, the cost pilot and the
   kill switch of G3. Before 1 Dec, at the standard price (section 8).
2. The full probes of M3 (people, pitch, number) and the two tools on the same frames (free):
   G2, G3 and G5a with numbers. Before 1 Dec if possible: from 1 Dec they cost twice as much at
   the standard price, or the same through flex with its waits.
3. The engine of each step under rule 8 (section 11.1, decision 5), then the first dev clip and
   the demo clip (M4). From 1 Dec every clip run goes through the flex tier, which the doctor
   checks first (line `flex tier`); a clip at the standard price costs about twice as much
   (section 8).
4. The dev clips (M5), then the fixes, the fresh clips and the freeze tag (M6), then the test
   clips once, the demo and the report (M7), as the plan orders them.
5. Optional, after the numbers of step 2 and only with a budget line: the video arms, the
   OpenRouter models, the second detector family, the streaming path on real clip runs (section
   6, steps 13 and 20 to 23). They are the "extra models" of the first cut of the plan.

## 13 When something fails

| Sign | Cause | Do |
|---|---|---|
| `git commit` is refused with a line that has `identity:` | The git name or e-mail of this clone is not set, or a commit has another name or e-mail | Section 4, step 9 |
| `doctor --no-api` shows `[fail] guard hooks` or `[fail] deny-list` | The hooks are not installed, or the deny-list is missing | Section 4, steps 8 and 10 |
| A paid command refuses to start and names `../pitchvlm-results` | The work tree of the branch `results` is missing | Section 4, step 11 |
| A paid command refuses to start with the words "is not clean" | The git work tree has changes | Commit when the owner asks. The option `--allow-dirty` starts it anyway, and the output file then says that the tree was dirty: use it only when the owner says so |
| The doctor stops and shows `[fail] spend` | The checks cost more than 2 USD | RUNBOOK, section 2: correct the token estimates, then one model at a time |
| A run stops with the words "1.3 times its estimate" | The estimate is too low | Do not just start again. Correct the token estimates with the owner's go-ahead (RUNBOOK, section 4) |
| A push sends only a tag or a new branch name | No hook runs for such a push | Run `python3 tools/guard.py --push` by hand before it (rule 9) |
| `pre-commit install` prints `Use -f to use only pre-commit` | A hook file was there before, for example from a git template of the account. The framework keeps it as `pre-commit.legacy` and runs it before its own hooks; `doctor --no-api` still shows `[ok  ] guard hooks` | The owner reads the old hook. To keep only the hooks of this repository: `pre-commit install -f --hook-type pre-commit --hook-type pre-push` |
