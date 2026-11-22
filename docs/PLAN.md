# Soccer VLM radar (working name: `pitchvlm`) — plan

## Context

**Idea.** Copy the basketball demo from the LinkedIn post, for soccer. Input: a short TV broadcast
clip of European football. Output: the same clip with player boxes, team colour, shirt number and
name, plus a 2D top-down pitch "radar" with positions and trails. One vision language model (VLM),
Gemini, does the perception. Plain code does the rest.

**Your decisions (13 Nov 2026).**

| Topic | Decision |
|---|---|
| Deliverable | Working demo, private for now |
| Footage | TV broadcast, main camera |
| First result | 2-4 weeks, with measured accuracy on a small test set |
| Model | Gemini family only |
| Machines | First build on the build machine. All API runs and real footage on the run machine, with an assistant there too |
| What may come back | Text only: metrics, cached Gemini answers, cost ledger. No pixels, no label files |
| Budget | About 120 USD of API cost, hard stop at 135 USD |

**What the research found** (19 research agents, every topic fact-checked; then 3 independent
designs and 6 critical reviews; evidence strength in brackets):

- The post's "one VLM does all" is not documented. Documented: frames at 2 per second, sent as
  images (10 per request), boxes and IDs from Gemini, interpolation in plain code. Nobody published
  how the court mapping and the names are made [weak].
- Gemini finds about 9 of 10 players on broadcast soccer frames, measured on 10 frames only [weak].
  Small far-side players (about 46 x 99 px) are the risk: recall is 0.45-0.55 on small-object test
  sets [proxy].
- Nobody has measured a VLM that points at pitch landmarks. This is the biggest unknown [none].
- Shirt numbers: 87-90% on player crops, as four-option questions, older models. Not measured on
  full frames [weak].
- Test data exists: SoccerNet Game State Reconstruction (SN-GSR). 30 s clips, 1080p, main camera,
  free download. Metric GS-HOTA: best specialist pipelines 63.9-64.1, official baselines 22.26 and
  29.01. No score exists for a general VLM that does the task end to end [solid].
- Cost: about 0.01-0.03 USD per frame and 5-32 s per call [estimate from third-party token counts].
  Argon and Carbon prices double on 1 Dec 2026 [solid].

So the plan is **Gemini-first and measurement-first**. Cheap probes measure each Gemini step against
a number fixed in advance. A step that fails gets another Gemini layout first, then plain code, then
a small specialist model. The final report names what did each step.

## Success line, baseline, controls

- **Success line (the hypothesis).** GS-HOTA >= 22 on 10 unseen SN-GSR test clips, run once.
  A rough model gives 25-32 if every step sits at its gate. This is an estimate, not a measurement.
- **Baselines.** Published: 22.26 and 29.01 (official baselines), 63.9-64.1 (best). They are on the
  full split; our 10 clips make them indicative only. Internal: ground truth sampled at 2 fps and
  interpolated (the ceiling of the keyframe design).
- **Controls.** Ground truth through our own exporter must score 100. Ground truth shifted by 1 m
  must score 89.5. Oracle swaps (one step replaced by ground truth) show where points are lost.
  Each chosen setting is confirmed once on fresh frames. Repeat runs give the run-to-run noise.

## Pipeline (first result)

Gemini calls are stateless: each request has only pixels and a versioned prompt. So every answer is
cached, replays for free, and can use the half-price tier.

| # | Step | Done by | How |
|---|---|---|---|
| 1 | Keyframes | code | One every 0.48 s plus the last frame: 64 per 30 s clip at 25 fps |
| 2 | Kit legend | Gemini, 1 call per clip | 3 frames in; team A/B, goalkeeper and referee kits out. Pasted into later prompts |
| 3 | People | Gemini, 1 call per keyframe | Box, role, team key, shirt number or null |
| 4 | Pitch landmarks | Gemini, 1 call per keyframe | Our numbered pitch diagram + the frame in; visible landmark ids and points out |
| 5 | Camera motion | code | Frame-to-frame homography for all frames (feature matching + RANSAC, people masked) |
| 6 | Pitch mapping | code | RANSAC homography from landmarks, sanity checks, gaps filled through step 5 |
| 7 | Tracks | code | Match people between keyframes in motion-stabilised image space; pitch distance as a limit; join fragments |
| 8 | Shirt number | Gemini, 1 call per track | 8 enlarged crops per track in; number or null out |
| 9 | Votes | code | Role and team per track; team side per clip; names from a roster file (demo clip only, never faces) |
| 10 | Fill, export, draw | code | Positions interpolated to every frame; SN-GSR prediction file; annotated video + radar |

- Default model: `gemini-4-argon`, thinking level low (the model named in the post). Each probe also
  tests one cheaper Gemini model on the same frames. `gemini-robotics-er-2-preview` (its docs cover
  pointing) is one arm for step 4.
- Shirt number rule: the scorer counts "no number" on a numbered player as an error. So the export
  uses the most likely value per track; the video shows a number only when it is confident.
- Add-ons, measured later and not built first: IDs carried by Gemini (the recipe of the post),
  tiles or crops for small players, several queries averaged, line and circle fit for midfield.
- Fallbacks: RF-DETR for boxes (Apache-2.0), PnLCalib for pitch mapping (separate tool). Their
  adapters are written in M1 and installed in M2. A fallback replaces a step only after the gate
  of the step failed and the fallback passed the same gate on the same frames (rule 8).
- The pipeline has no switch for a fallback yet: `run` takes the people and the pitch mapping
  from the model steps only. A tool that passes its gate needs a pipeline path, and RF-DETR also
  needs a step that gives each box its role and team key. This work is not built and not
  budgeted (open owner decision, `docs/DECISIONS.md`).
- Both tools are measured in M3 on the frames of the people probe and the pitch probe, with their
  pretrained weights and without training (owner decision, 14 Nov 2026). It costs no API money.
  So the decision of 24 Nov has measured values of both tools next to those of the two model
  steps, and the time per frame of each tool on the run machine.

## Gates (numbers fixed before each run, kept in `configs/gates.yaml`)

Thresholds come from the metric formula: a position error of 1 m scores 0.89, 2 m 0.62, 3 m 0.34.
They may change once after the free stage, then they are frozen.

| Gate | Cost | Measured on | Pass | If it fails |
|---|---|---|---|---|
| G0 Contract | < 2 USD | Synthetic 1080p frames through the real API, per model | The contract verdict. The answered boxes land on the true boxes (at least half of the possible pairs, median IoU >= 0.3); the median landmark is within 5% of the image side; token sums add up; `store=False` is set; the half-price tier (Flex) answers for a model that has one, or the check reports that the tier had no capacity for 900 seconds (a fact without a verdict: run the doctor again before the first Flex run). The mean IoU (>= 0.6) and the mean point error (<= 1% of the image side) are reported and are no verdict | Fix the client before any footage is sent |
| G1 Harness | free | Labels only | Ground truth = 100 (>= 99.5); 1 m shift = 89.5 +/- 0.5; keyframes only = about 8.5; 2 fps + interpolation >= 92; team-side rule right on >= 97% of all valid clips | Fix exporter, sampling or the side rule |
| G2 People | paid | 120 frames from 24 clips | Recall >= 0.85 (lower end of its 95% interval over the clips >= 0.80), precision >= 0.85; recall for boxes up to 110 px high >= 0.70 (at least 300 such boxes); foot-point error median <= 0.7 m and 90th percentile <= 2.0 m | Higher resolution, crop, tiles, other Gemini model, then RF-DETR |
| G3 Pitch mapping | paid | Same frames + 3 of the dev clips | Position error median <= 1.2 m near goals, <= 2.0 m in midfield; 90th percentile of the position error <= 3.0 m; valid on >= 85% of keyframes. Kill-switch after 20 frames (valid < 50% or median > 3 m), counted only after five overlay images are checked by eye | One more Gemini arm, then PnLCalib. Decision on 24 Nov |
| G4 Identity | paid | 6 dev clips | >= 80% of each true track under one ID; <= 6 switches per clip | Gemini-carried IDs if better on 5 of 6 clips; else denser keyframes for step 3 only (about +1 USD per clip, test set cut to 6 clips) |
| G5a Number reading | paid | About 400 tracks cut from ground-truth boxes | Reported as an upper bound (oracle input); no gate | — |
| G5b Number, role, side | free from cache | Pipeline tracks of the 6 dev clips | Exported number equals the label (null counts) on >= 50% of player tracks; role and side >= 95%; no clip with left/right swapped | One more layout; below 40%: stop spending on numbers, report the rate |
| G6 Freeze check | paid | 6 dev clips, then 3 fresh clips once | GS-HOTA >= 22 on dev; fresh >= 0.8 x dev; <= 2.5 USD per clip | One fix round, then freeze and report as measured |

G6 only allows the freeze. The success line is tested once, on the 10 test clips.

The table names the main values of each gate. `configs/gates.yaml` holds all of them, and all
must hold.

Two gate numbers need care (SN-GSR paper, tables 1 and 2). The published baseline scores 18.05
on the validation split and 22.26 on the test split, so the 22 of G6 on validation dev clips is
not the line of the 22.26: compare a dev score with 18.05. The paper measures the mean-x
team-side rule at GS-HOTA 92.00 on the test split with ground truth for every other module. No
source gives the share of clips where the rule is right, so the 97% of G1 can fail although the
code is right. The free label stage measures this share, and it reports how often the
goalkeepers decide (`docs/RUNBOOK.md`, section 8).

Statistics: a split list is fixed with a seed before the first paid run (valid: 24 probe clips,
6 dev, 3 fresh, the rest in reserve for confirmations; test: 10 clips). Intervals use per-clip
values. A setting replaces another only if the paired difference is above 2x the run-to-run spread.
The foot-point error in G2 uses a mapping fitted from the labels, so it does not mix in Gemini's
pitch-mapping error.

## Data and scoring

- **Data.** SN-GSR from Hugging Face (`SoccerNet/SN-GSR-2025`, no login). `valid.zip` (11.2 GB, about
  58 clips) for all probes and tuning. `test.zip` (8.85 GB) is downloaded only at the freeze; the
  free G1 checks are repeated on the test clips before the paid test run. Research use only: fine
  for private evaluation, not for a public video. Data lives outside the repo folder.
- **Scoring.** Stand-alone `sn-trackeval` (MIT, pure Python) in its own environment
  (Python 3.12, numpy 1.26). The code that scores is the clone of its repository at the tag
  `v0.4.1`; the package 0.4.0 in that environment only brings the dependencies. Flags:
  `--GT_FOLDER` and `--TRACKERS_FOLDER` with the folder `SoccerNetGS-<split>` of the data and of
  the predictions, `--SKIP_SPLIT_FOL True`, `--TRACKER_SUB_FOLDER ""`, `--TRACKERS_TO_EVAL`,
  `--SPLIT_TO_EVAL` and `--SEQ_INFO`. Not the full SoccerNet baseline (old torch stack) and no
  own copy of the metric.
- **Export contract** (a mistake here gives a score of 0 without any error). One file
  `SoccerNetGS-<split>/<tracker>/<clip>.json` with `{"predictions": [...]}`. Each item:
  `supercategory` "object", `category_id` 1, `image_id` copied from the label file, `track_id` unique
  per frame, `bbox_pitch` with all six keys, `attributes` with `role`, `team`, `jersey`. Shirt number
  in the same type as the labels (read it first), `null` for unknown. A scored frame without a
  prediction is allowed: the scorer counts its people as missed. A format checker in the repo
  tests all of this.
- **Coordinates.** Metres, origin at the pitch centre, x towards the right goal, y towards the
  camera, pitch 105 x 68. We write our own landmark table of real line crossings. The Roboflow
  table is 120 x 70 m and has points that are not line crossings.

## Two machines

| | Build machine | Run machine |
|---|---|---|
| Does | First build (M1); offline tests; reading returned text results | Gemini calls; dataset; real footage; scoring; replay checks; videos; fixes with an assistant |
| Never has | API key, API call, footage, dataset, label files | — |

- One code branch, moved with a git remote (public since 14 Nov 2026). A separate `results`
  branch brings back text only: metrics, cached Gemini answers, cost ledger. Where that branch
  goes now is an open owner decision (`docs/DECISIONS.md`).
- `pyproject.toml` has version ranges and no index address. No lock file is committed: `uv.lock`
  is git-ignored from the first commit. The pinned list is made on the run machine and committed
  under `locks/`.
- Main environment: Python 3.13 on both machines. The Gemini SDK is imported in one module only.
- Commits on both machines use your personal git name and e-mail, set in the repository only.
- One guard script (Python standard library only) and a secret scan run before every commit and
  push on both machines. The guard refuses media, large files, keys, lock files, encoded pixels,
  label and prediction data, non-public host names, home-folder paths, the words of a local
  deny-list and a wrong git identity. `doctor` fails if the hooks are not installed or the
  deny-list is missing.
- The build machine needs these packages and no others: `numpy`, `opencv-python`, `scipy`,
  `pydantic`, `pyyaml`, `pytest`. If one of them is missing there, the build stops and says so,
  and the video test moves to the run machine.

## Milestones

| # | Dates | Where | Deliverable | Exit test | You do |
|---|---|---|---|---|---|
| M1 | 13-16 Nov | build | Repo: schemas, pitch table, geometry, tracking, camera motion, exporter + checker, Gemini client (real, replay, fake) with cache and ledger, fallback adapters, renderer, roster lookup, CLI, guard, handoff docs | All tests pass offline with no SDK. Synthetic clip gives video + radar + a valid prediction file | To-dos 1-4 by 16 Nov (about 1 h) |
| M2 | 17-18 Nov | run | Setup, `doctor`, data, split list, free label stage, cost pilot, fallback install | G0 and G1 pass. Real token counts replace the estimates. Spend <= 2 USD | Start the session; approve first paid calls |
| M3 | 19-24 Nov | run | People probe, pitch probe, number probe on ground-truth crops. RF-DETR and PnLCalib measured on the frames of the people probe and the pitch probe | G2 and G3 decided with numbers by 24 Nov | Check 10 overlay images; confirm the gate decisions |
| M4 | 24-26 Nov | run | If G2 or G3 failed and the tool passed: the fallback wired into the pipeline (not built, not budgeted; open owner decision). First end-to-end dev clip. Your demo clip: free dry run (decode, frame rate, keyframes), then a first paid pass | A watchable video + radar on 26 Nov | Demo clip + roster by 22 Nov; watch the video |
| M5 | 27-30 Nov | run | Identity probe; 6 dev clips scored through the half-price tier | G4, G5b decided. First GS-HOTA. Standard-price runs end 30 Nov | Read the numbers |
| M6 | 1-5 Dec | run | Fixes from G4-G6, smoothing, look of the video, 3 fresh clips once, freeze tag | G6 passes or is reported as is | Approve the freeze |
| M7 | 6-10 Dec | run | 10 test clips once; final demo clip with names; report | Report rebuilds from cached answers with one command | Watch the demo; read the report |

- Assumed: about 10 hours a week of your time. Runs go unattended (about 180 calls per clip,
  4 in parallel, throttled below the 10 USD per 10 minutes limit of the first billing tier).
- If a date slips by more than 2 days, cut in this order: extra models, Gemini-carried IDs, test set
  from 10 to 6 clips, names. The video + radar and one measured score are protected.

## Budget (your money)

Estimates until the cost pilot measures real tokens. One clip is about 2.3 USD (64 keyframes x two
calls + about 50 number calls + legend) at the standard price before 1 Dec, and the same after
1 Dec through Flex (`service_tier="flex"`, half price, slower, best effort). The 50 number calls
are a guess from the synthetic clip of 30 seconds (48 to 50 requests). The table below was made
with 25 number calls (2.1 USD per clip): the 27 clip passes of the table cost about 6 USD more,
which comes out of the reserve until the cost pilot gives real numbers.

| Item | USD |
|---|---|
| Doctor + cost pilot | 2 |
| People probe | 14 |
| Pitch probe (biggest unknown) | 15 |
| Number probe (about 400 tracks, 2 settings) | 11 |
| Dev clips (6 clips x 2 passes) + Gemini-ID arm | 28 |
| Fresh clips (3) | 6 |
| Test clips (10) | 21 |
| Demo clip (2 passes) | 6 |
| Reserve | 17 |
| **Planned total** | **120** |

- Every run prints its estimate first and the measured cost after. A run stops at 1.3x its estimate.
  A doctor run always stops at the 2 USD of gate G0 instead: its answers are the first real
  answers of a model, so its estimate is the least certain one.
  Hard stop at 135 USD. A budget alert on the Google project is the backstop.
- If Flex does not work, a clip costs about 4.7 USD after 1 Dec. Then: one dev pass instead of two,
  and 6 test clips instead of 10.
- A Flex request that gets the status 503 or 429 (no capacity) is tried again for 900 seconds
  before the run stops. These attempts are not billed.

## What I build first (M1, on the build machine)

New repository. Layout of the main files (`docs/ARCHITECTURE.md` has every module as it is
built):

- `src/pitchvlm/schemas.py` — typed records for people, landmarks, tracks, predictions.
- `src/pitchvlm/pitch.py` — landmark table in SN-GSR metres; pitch diagram for the prompt and radar.
- `src/pitchvlm/geometry.py` — RANSAC homography and its checks; `motion.py` — camera motion;
  `fill.py` — gap fill and interpolation.
- `src/pitchvlm/tracking.py` — matching between keyframes, fragment joining, votes, team side.
- `src/pitchvlm/vlm/` — `client.py` (the only file that imports the SDK; `store=False`, throttle,
  retry, resume), `cache.py`, `fake.py`, `prompts/` (versioned text), `cost.py` (dated prices, ledger).
  Cache key: clip, step, frame, item, repeat index, model, thinking level, resolutions, prompt
  hash, answer schema, request context.
- `src/pitchvlm/fallbacks/` — adapters for RF-DETR and PnLCalib output (tool output in, our records out),
  and the job file and the run file of a tool run.
- `tools/run_tool.py` — runs one of the two tools on the frames of a job, in the environment of the tool.
- `src/pitchvlm/export_gsr.py` — SN-GSR prediction writer + format checker.
- `src/pitchvlm/render.py` — boxes, labels, radar with trails, side-by-side video (OpenCV only).
- `src/pitchvlm/synth.py` — small synthetic scene with exact ground truth, for geometry and sign tests.
- `src/pitchvlm/cli.py` — `doctor`, `data`, `oracle`, `probe`, `tool-job`, `tool-score`, `run`, `eval`, `sync`.
- `tools/guard.py`, `configs/gates.yaml`, `configs/experiments/*.yaml`, `tests/`.
- Handoff docs for the session on the run machine: `CLAUDE.md` (project rules),
  `docs/HANDOVER.md` (the move, step by step), `docs/RUNBOOK.md`, `docs/PLAN.md`,
  `docs/DECISIONS.md`.

Before I write the Gemini client I read the current API pages again. The SDK had four releases in
one week, and the docs moved to a new call style (`client.interactions.create`). The client cannot
be tested on the build machine; `doctor` is its first real test.

## Verification

1. Build machine: `pytest` with network blocked and no SDK installed. The synthetic clip runs through the
   whole pipeline with the fake VLM and gives a video, a radar and a prediction file that passes the
   format checker. Sign test: a far-side player has negative y. The guard script rejects a planted
   key, a media file and a `uv.lock`.
2. Run machine: `pitchvlm doctor` (G0), then `pitchvlm oracle` (G1). These prove the API
   contract and the scoring chain before real money is spent.
3. Each run writes a metrics file with code commit, model ID, SDK version, prompt version, seed and
   cost, and refuses a dirty tree. A cached re-run on the run machine must give the same files.
4. End: GS-HOTA from `sn-trackeval` on the 10 test clips, the oracle-swap table, the table
   "step -> what did it", and a check by eye of 20 frames of your demo clip.

## Your to-dos

| # | To-do | By |
|---|---|---|
| 1 | Run machine: install `uv` and an assistant; about 45 GB free disk | 16 Nov |
| 2 | Google AI Studio: project with billing (paid tier only), a budget alert, an API key. The key stays in an environment variable on the run machine | 16 Nov |
| 3 | A git remote that both machines can reach (done on 14 Nov: a public repository) | 16 Nov |
| 4 | Tell me the personal git name and e-mail for the commits of both machines | before the first commit |
| 5 | Demo clip: one continuous main-camera shot, 1080p, 10-30 s, plus a roster file (team, number, name) | 22 Nov |

## Defaults I chose (change any of them when you review)

- Matching players between keyframes is plain code. Gemini-carried IDs are measured and used only
  if they are better.
- A specialist model replaces a Gemini step only after its gate fails, and the report says so.
- I commit locally at each milestone. I do not push anywhere unless you ask.

## Risks and limits

- Pitch mapping from Gemini landmarks may fail, mainly in midfield views with few line crossings.
  It is measured first (19-24 Nov). Its fallback, PnLCalib, has a tested adapter; the tool itself
  never ran, and the pipeline cannot use its answers yet.
- Shirt numbers are often unreadable in wide shots. The radar then shows team colours for all
  players and numbers or names for some.
- The Gemini client and the scorer wrapper are written on the build machine without a real test.
  `doctor` and `oracle` exist to catch that on day 1 on the run machine.
- Results hold for uncut 30 s main-camera clips of one league and season. Cuts, replays, the ball
  and live speed are out of scope for the first result.
- Google logs paid-tier prompts for abuse checks (up to 55 days). Its policy forbids tracking
  people without consent, with exceptions for research use. We identify players by shirt number and
  roster, never by face. Whether this private use is covered is your judgement.
- Before anything is published: use footage you have rights to, and review the licences of the
  fallback tools. The code of PnLCalib is under GPL-2.0-or-later, and `tools/run_tool.py` imports
  it when it runs; the weights of PnLCalib are under CC BY 4.0.
- The code repository is public since 14 Nov 2026 (owner decision, `docs/DECISIONS.md`). The
  licence review above is not done yet, and the repository has no licence file. No footage, no
  dataset file and no video is in the repository, and none may go there.

## Key sources

- Talk: https://www.youtube.com/watch?v=Spf5-1nk21k — recipe: https://roboflow.com/blog/gemini-4-argon-vision
- SN-GSR paper: https://arxiv.org/abs/2404.11335 — data: https://huggingface.co/datasets/SoccerNet/SN-GSR-2025
- Scorer: https://github.com/SoccerNet/sn-trackeval
- Gemini docs: https://aistudio.google.com/docs/image-understanding , https://aistudio.google.com/docs/pricing
