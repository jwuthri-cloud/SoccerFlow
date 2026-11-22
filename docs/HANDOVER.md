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

## 1 State at the handover (15 Nov 2026)

| Topic | State |
|---|---|
| Milestone | M1 (first build) is done. M2 starts on the run machine on 17 Nov |
| Code | Branch `main` of the remote (public since 14 Nov 2026). Package `pitchvlm` under `src/`, tests under `tests/` |
| Offline tests | 2730 passed and 1 skipped on the build machine (Python 3.13, no model SDK installed). The skipped test reads an image with Pillow, which the environment of the package does not have |
| Demo | `demo` gives a video, a radar and a valid prediction file from a synthetic clip and a fake model |
| Reviews | Two reviews tested the first build, a third one the measurement of the two specialist tools, a fourth one the fixes of the third, and a fifth one the fixes of the fourth. Each finding was repaired, kept with its reason, or recorded as not built or as an open owner decision (`docs/DECISIONS.md`) |
| Real API, real scorer, real footage | Never used. Every result so far is synthetic |
| Money spent | 0 USD of the planned 120 USD |

What "never used" means in detail is in `docs/RUNBOOK.md`, section 8. Read it before day 1: the
Gemini client, the scorer wrapper, the download and the label reader run for the first time on
the run machine.

## 2 Before the code leaves the build machine

Do this section, and steps 1 to 4 of section 4.1, by 16 Nov 2026 (`docs/PLAN.md`, "Your
to-dos"). The owner does each step, or asks the assistant to do it. The assistant never commits
or pushes without being asked.

State on 14 Nov 2026: steps 1 to 3 are done. The owner chose a public repository
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
| 6 | Make the environment with Python 3.13, install the package with the extras `dev` and `gemini`, run `python -m pytest -rs`. In each new terminal: `source .venv/bin/activate` | Every test passes. A skipped test names its reason |
| 7 | Write the pinned list `locks/run-machine.txt`. Then do the check of fact 19 (RUNBOOK, section 8): it shows whether `uv pip` reads the package age limit of `pyproject.toml`. Write the second list of that check outside the repository, for example into `~/pitchvlm-data` | The file exists. The two lists of the check are equal, or the owner knows that they differ |
| 8 | Owner: install the hooks: `pre-commit install --hook-type pre-commit --hook-type pre-push` | `doctor --no-api` shows `[ok  ] guard hooks` |
| 9 | Owner: set the git identity of this clone, with the name and e-mail of the commits of the build machine (`git log -1 --no-merges --format='%an <%ae>'` prints them; a merge commit that the git host made has another e-mail, so leave it out) | `git config --local --get user.name` and `git config --local --get user.email` print the same two values |
| 10 | Owner: write the deny-list `.git/guard-deny.txt`: one name per line that must not leave this machine (RUNBOOK, section 9). Do not list a word that is a part of the git name or e-mail: the guard then refuses every commit | `doctor --no-api` shows `[ok  ] deny-list` |
| 11 | Make the work tree of the branch `results` next to the repository: `../pitchvlm-results` | `git worktree list` shows it. `git -C ../pitchvlm-results rev-parse --git-path hooks` names the hook folder of this clone |
| 12 | Make the scorer environment (Python 3.12, numpy 1.26) and clone the repository of the scorer (`sn-trackeval`) at the tag `v0.4.1` | The last command prints the commit. Write it into `docs/DECISIONS.md` |
| 13 | Run `python -m pitchvlm.cli doctor --no-api`. Then run the command "Form of the key" (RUNBOOK, section 8, fact 11): it prints no part of the key | Exit code 0. The lines `guard hooks`, `deny-list`, `model SDK` and `API key` start with `[ok  ]`, and the SDK is a version 3.x. The key command prints "the pattern fits" |
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
`CLAUDE.md` (rules), this file, `docs/RUNBOOK.md` (sections 1, 2 and 8 first), `docs/PLAN.md`
(gates and budget), `docs/DECISIONS.md` (why things are as they are).

Then it does section 4.2 with the owner. On day 1 it asks before each paid command.

## 6 Day 1 in order

Commands: `docs/RUNBOOK.md`, section 2. Steps 1 to 5 have the numbers of the commands there.
Ask the owner before each paid step. Run one paid command at a time.

| # | Step | Cost | Passes when | When it does not |
|---|---|---|---|---|
| 1 | `doctor --no-api` | Free | The lines of step 13 of section 4 | Repair the failed check (hooks, deny-list) |
| 2 | `doctor` with the models (gate G0) | About 0.13 USD for the two models of the command, an estimate that is not measured. It stops above 2 USD | Every contract check is `ok` or `info` | A `[fail]` line of a contract check is a client fault: repair it before any footage is sent. One exception for the check `boxes` is in RUNBOOK, section 8. A stop at 2 USD: RUNBOOK, section 2 |
| 3 | `data --split valid`: download 11.2 GB and write the split list `configs/splits.json` | Free | About 58 clip folders. Commit the split list when the owner asks | Run it again: it continues |
| 4 | `oracle` on one clip, then on all clips (gate G1) | Free | Ground truth scores at least 99.5. The other lines of G1 are in `configs/gates.yaml` | The error names the file or shows the last lines of the scorer (RUNBOOK, section 8). A team-side share below 0.97 with a ground truth of at least 99.5 is not a code fault (fact 2): write the shares into `docs/DECISIONS.md`, and the owner decides about the gate |
| 5 | Cost pilot: the three probes with `--max-clips 2`. Before it, commit the split list and the day-1 notes in `docs/DECISIONS.md` when the owner asks: a probe refuses a work tree with changes | About 0.7 USD | The ledger has one line for each new answer. `--replay` gives the same metrics for free | Compare the measured cost with the estimate. Tell the owner |
| 6 | Write the mean tokens that the cost pilot measured (the lines after "measured cost" of the three probes, all four model steps) into `DEFAULT_TOKEN_ESTIMATES` (`src/pitchvlm/vlm/cost.py`), into `docs/ARCHITECTURE.md` (section `vlm/cost.py`) and into `docs/DECISIONS.md`, with the owner's go-ahead. The output count is the out tokens plus the thinking tokens. Run the tests. Commit when the owner asks | Free | The tests pass. After the commit, a pilot command with `--replay` prints a cost estimate near the cost that the pilot measured | - |
| 7 | `sync`. Then the owner runs the `add` and the `commit` command of its output, and commits and pushes the code branch: each metrics file names the commit of the code. No push of the branch `results` before the owner has decided where it goes (section 9) | Free | The guard and the secret scan pass on real model answers | A report line names the rule and the file. Change the rule with a test, not the answer |

G0 and G1 must pass before any footage is sent to the model. The planned budget for the doctor
and the cost pilot together is 2 USD.

M2 also installs the two specialist tools, RF-DETR (boxes) and PnLCalib (pitch mapping).
RUNBOOK, section 3 has the commands of the install and of the measurement. Ask the owner before
this install: it downloads packages and about 1 GB of weights. Ask at the same time for the
licence decision (`docs/DECISIONS.md`, last list). The commands never ran, and the
script that runs the tools (`tools/run_tool.py`) was tested with stand-ins only (RUNBOOK,
section 8, row "Tool script"). In M3 both tools are measured on the frames of the people probe
and the pitch probe, for free (owner decision of 2026-11-14). The commands measure a tool. The
pipeline cannot use the answers of a tool yet: that work is an open owner decision.

## 7 The first-day checks that matter most

`docs/RUNBOOK.md`, section 8, has the full list (11 parts and 21 open facts). These decide
whether the next days can start:

1. The Gemini client against the real API: step 2 of day 1 is its first real call.
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

Twelve decisions of the owner are open (the list is at the end of `docs/DECISIONS.md`). Three
of them wait for the first real answers: the label penalty of the tracker, the team-side rule
with goalkeeper evidence, and the requests in flight at a budget stop. Four came with the
measurement of the two specialist tools: its settings, the place of its run files, the work
that lets the pipeline use a tool that passed its gate (not built, not budgeted), and the
frames of gate G3. Two came with the public repository: where the branch `results` goes, and
the licence of the repository. One is about the kill switch of gate G3: no command writes the
overlay images that it names.

## 12 Schedule after the handover

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
