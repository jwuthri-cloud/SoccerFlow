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

## 1 State at the handover (written 15 Nov 2026, updated 22 Nov 2026)

| Topic | State |
|---|---|
| Milestone | M1 (first build) is done. M2 was planned to start on 17 Nov; on 22 Nov it has not started (`docs/PLAN.md`, "Milestones") |
| Code | Branch `main` of the remote (public since 14 Nov 2026). Package `pitchvlm` under `src/`, tests under `tests/` |
| Added on 22 Nov | A second model provider (OpenRouter, RUNBOOK section 10), the paint alignment of pitch mappings and the cut test (section 3), two track repairs as arms of G4 (`configs/arms`, section 4), a detector engine with `clip-job`, `run --detections --calibration` and `video-frames` (section 11), the check images of G3 (`overlay`, section 3), two test tiers (section 1), a faster guard |
| Offline tests | `3203 passed, 1 skipped` on the build machine (Python 3.13, no model SDK installed), full run on 22 Nov. The skipped test reads an image with Pillow, which the environment of the package does not have |
| Demo | `demo` gives a video, a radar and a valid prediction file from a synthetic clip and a fake model; `demo --engine detector` does the same with stand-ins of the two tools |
| Reviews | Two reviews tested the first build, a third one the measurement of the two specialist tools, a fourth one the fixes of the third, and a fifth one the fixes of the fourth; two more read the code of 22 Nov. Each finding was repaired, kept with its reason, or recorded as not built or as an open owner decision (`docs/DECISIONS.md`) |
| Rehearsal | On 22 Nov this file and the RUNBOOK were walked from the first line to the last on the build machine, with a synthetic dataset, the fake model, a stand-in scorer and stand-ins of the tools (`docs/DECISIONS.md`, entry "Rehearsal of the handover and the runbook") |
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
`CLAUDE.md` (rules), this file, `docs/RUNBOOK.md` (sections 1, 2 and 8 first, then 10 and 11),
`docs/PLAN.md` (gates and budget), `docs/DECISIONS.md` (why things are as they are).

Then it does section 4.2 with the owner, and then section 6. It asks before each paid command.

## 6 First session on the run machine, step by step

Do section 4.2 first. Then follow these steps in this order, one at a time. Each step names its
command, what it must print to pass, what to do when it does not, and its cost. The commands
and their details are in `docs/RUNBOOK.md` (the section is named in each step). The words to
look for were printed on 22 Nov by the same commands on the build machine, with a synthetic
dataset, the fake model, a stand-in scorer and stand-ins of the tools; the estimates are those
of the token estimates of that day.

Money: a paid step starts only after the owner's go-ahead for that step. Run one paid command at
a time, and never pass `--cache`. Before each paid step, read its estimate line first: the
command prints it before the first model call. Steps 5 to 15 together: about 0.92 USD
(estimated, not measured): 0.80 USD of the 2 USD that `docs/PLAN.md` plans for the doctor and
the cost pilot, and 0.12 USD of the line of the pitch probe. A paid command refuses a work tree
with changes: commit first, when the owner asks.

**Step 1. The guard tests on the git of this machine.** Free. Section 4.2, step 6 ran the full
run once; run the tests of the guard alone too, because the guard is the commit hook and its
tests start git many times:

```bash
git --version
python -m pytest tests/test_guard.py -rs
```

- Passes: the last line starts with `150 passed` (the count of 22 Nov), with no failure. A skipped test
  names its reason.
- When not: a failed test names a git behaviour that this machine has and the build machine did
  not. Commit nothing until it is repaired: the guard is the hook of every commit. Repair
  `tools/guard.py` with a test, never the test alone, and never bypass the hook (rule 9). Write
  the git version and the finding into `docs/DECISIONS.md`.

**Step 2. The full run of the offline tests.** Free. The same command as section 4.2, step 6;
run it again here when the code changed since then (RUNBOOK, section 1):

```bash
python -m pytest -m "slow or not slow" -rs
```

- Passes: no failure. On 22 Nov the build machine printed `3203 passed, 1 skipped`. Here one more group of tests
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

- Estimate: it prints `cost estimate: 0.133 USD for 14 planned model calls (people 8,
  landmarks 6)`, and a budget line that ends with `this run stops above 2.000 USD (its spend
  limit in the gate file)`. The owner's go-ahead is needed first.
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
  calls (kit_legend 2, number 34)`: 0.652 USD together. The owner's go-ahead is needed first.
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
go-ahead. Run `python -m pytest`, and commit when the owner asks (RUNBOOK, section 2).

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
  (landmarks 20)`. The estimate counts an empty cache: the 10 answers of step 9 are stored, so
  about half of it is paid. The owner's go-ahead is needed first. The overlay is free.
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
  (people 3, landmarks 3)`, and the budget line ends with `this run stops above 2.000 USD`. The
  owner's go-ahead is needed first.
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
  goes (section 9).
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
- Standard-price runs end on 30 Nov 2026. The prices of `gemini-4-argon` and `gemini-4-carbon`
  double on 1 Dec 2026. The half-price tier (flex) is used from M5 on (27 Nov).
- No paid run without the owner's go-ahead.

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

- Small, tested changes on the code branch. `python -m pytest` must pass before a commit.
  Commit when the owner asks.
- A paid command refuses to start when the git work tree has changes.
- After a change of `tools/`, `.gitleaks.toml`, `.gitignore` or `.pre-commit-config.yaml`, merge
  the code branch into `results` before the next commit of results (RUNBOOK, section 6).
- Gates are in `configs/gates.yaml`. A change of a gate needs an entry in `docs/DECISIONS.md`.

## 11 Decisions

The decisions and their reasons are in `docs/DECISIONS.md`. These five change what an assistant
does:

- Gate G0 judges the contract of the API only. Accuracy values are reported without a verdict.
- Each doctor run stops at 2 USD. Its report keeps the cost of every paid answer.
- A stored answer keeps the frame numbers of its request and no box of the request. The boxes
  that the model answers stay in the answer text. The number probe keeps the track id of the
  label file in the name of its requests.
- A commit message has no co-author line.
- The two specialist tools, RF-DETR and PnLCalib, are measured in M3 on the frames of the people
  probe and the pitch probe, without training and for free (RUNBOOK, section 3). A tool still
  replaces a Gemini step only after that step failed its gate and the tool passed the same gate
  on the same frames.

28 decisions of the owner are open (the list is at the end of `docs/DECISIONS.md`; three more
items of that list record work that was built on 22 Nov). Three wait for the first real answers:
the label penalty of the tracker, the team-side rule with goalkeeper evidence, and the requests
in flight at a budget stop. Two are about rules of the repository: the install rule of the build
machine and two texts that the guard refuses. Five are about the specialist tools: the settings
of their measurement, the place of their run files, the frames of gate G3, the size of the
detector, and what the AGPL or the GPL means for `tools/run_tool.py`. Two came with the public
repository: where the branch `results` goes, and the licence of the repository. Four are about
OpenRouter (the rule text, the price table, the token estimates per model, which models to try),
four about the paint alignment and the cuts, four about the track repairs, and three about the
detector engine (rule 8 for two engines, a detector trained on soccer, the clip runs on the
results branch). One came with the rehearsal of 22 Nov: the new dates of the schedule (section
12).

## 12 Schedule after the handover

These are the dates of the plan of 15 Nov. On 22 Nov no step of M2 had run, so every date from
M2 on has slipped by at least 5 days; new dates are an owner decision (`docs/DECISIONS.md`, last
list). The price change of 1 Dec 2026 does not move (section 8).

| Milestone | Dates (2026) | Result |
|---|---|---|
| M2 | 17-18 Nov | Setup, G0, G1, cost pilot, install of the fallback tools. Spend at most 2 USD |
| M3 | 19-24 Nov | People, pitch and number probes. RF-DETR and PnLCalib measured on the same frames, for free. G2 and G3 decided by 24 Nov |
| M4 | 24-26 Nov | First end-to-end clip. A watchable video on 26 Nov. Demo clip and roster from the owner by 22 Nov |
| M5 | 27-30 Nov | 6 dev clips run and scored through the half-price tier: identity (G4), numbers (G5b), first GS-HOTA |
| M6 | 1-5 Dec | Fixes, 3 fresh clips once, freeze tag |
| M7 | 6-10 Dec | 10 test clips once, final demo clip, report |

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
