# Decision log

Newest entries at the bottom. Each entry: date, decision, reason, numbers if any.

## 2026-11-13 — Project decisions (owner)

| Topic | Decision |
|---|---|
| Deliverable | Working demo: annotated video + 2D pitch radar, private for now |
| Footage | TV broadcast, main camera |
| Model | Gemini family only |
| First result | 2-4 weeks, with measured accuracy on a small test set |
| Machines | First build on the build machine; all API runs and footage on the run machine |
| What may return to the build machine | Text only: metrics, cached model answers, cost ledger |
| Budget | About 120 USD of API cost, hard stop at 135 USD |

## 2026-11-13 — Design decisions

1. **Measurement first.** Every model step is measured against a gate fixed in advance
   (`configs/gates.yaml`) before the pipeline relies on it. Reason: no published measurement exists
   for a general vision language model on wide-angle soccer frames, pitch landmarks or tracking.
2. **Stateless model calls.** One request holds only pixels and a versioned prompt. Reason: answers
   can be cached, replayed for free and sent through the half-price tier.
3. **Plain-code tracking by default.** People are matched between keyframes with Hungarian matching
   in motion-stabilised image space. IDs carried by the model (the recipe of the reference post)
   are a measured alternative and are used only if they are better on 5 of 6 dev clips.
4. **Own landmark table.** 35 real pitch marks in SoccerNet-GSR metres (`src/pitchvlm/pitch.py`).
   Reason: a model can only point at visible marks; other public tables use a 120 x 70 m pitch and
   include points that are not line crossings.
5. **Shirt numbers: export differs from display.** The scorer counts "no number" on a numbered
   player as an error, so the export uses the most likely number per track. The video shows a
   number only when the reading is confident.
6. **Scorer.** Stand-alone `sn-trackeval` 0.4.0 in its own environment. No own copy of the metric.
7. **Success line.** GS-HOTA >= 22 on 10 unseen test clips, run once after a freeze tag.
8. **Fallbacks.** RF-DETR (boxes) and PnLCalib (pitch mapping) are installed early and used only
   after a failed gate, and only after they pass the same gate on the same frames.

## 2026-11-14 — Decisions of the first review (repository rules)

1. **Scorer version.** The scorer script runs the code of its clone, not the installed package.
   The clone is fixed to the tag `v0.4.1` (the tag `v0.4.0` plus one line that repairs an error
   message). This replaces "0.4.0" in decision 6 above for the code that scores; the package
   0.4.0 stays in the scorer environment for its dependencies. The commit of the clone is
   written here on day 1 of the run machine.
2. **Model SDK.** `google-genai>=3.0,<4`. Reason: the client sets one HTTP attempt per call
   (`attempts=1`). Before 3.0 the SDK read this value as the number of repeats, so each call
   could make a second, hidden attempt.
3. **Push check reads commits.** The guard has a push mode that reads every commit that the
   remote does not have and every local annotated tag (git does not record which tags a remote
   has). Reason, measured with the hook runner: the old
   push check read the working tree, and the guard let a forbidden commit pass in 5 of 5 flows
   (first push with an edited working copy, a file removed by a later commit, a push of another
   branch, a cleaned and staged file, an empty commit by another author). In 3 of the 5 the
   runner started no hook at all, so every hook now has `always_run`.
4. **Dataset files by content.** The guard refuses label data, prediction data and the result of
   a clip run by name and by the top-level keys of a JSON file. Reason: a synthetic label file
   of 0.47 MB and a prediction file of 0.45 MB passed the size rule.
5. **Ledger size.** The cost ledger is split into files of at most 1,000,000 bytes
   (`ledger.jsonl`, `ledger-0002.jsonl`, ...), and the guard keeps one size limit of 2 MB for
   every file. Reason: one line has about 370 bytes (measured on the ledgers of 3 synthetic
   clips of 750 frames, 177 to 179 lines each: 373), so one file would pass 2 MB after about
   5,350 paid answers. The hard stop of 135 USD allows about 10,300 answers with the default
   token estimates (an estimate: 179 answers of one clip for 2.34 USD); that ledger is 4 files.
   A size exception for the ledger in the guard was tried and removed: no file needs it after
   the split.
6. **Deny-list.** Private words (an organisation, an e-mail domain, a private host) are known
   only from `.git/guard-deny.txt` of each clone. The guard says when the file is missing, and
   the check `deny-list` of the `doctor` command fails without it.
7. **Key format.** The guard and the secret scan know the Google key prefix `AQ.` as well as
   `AIza`.

## 2026-11-14 — Decisions of the first review (pipeline, money, measurement)

Every number below was measured by the review on synthetic clips with the fake model (3 seeds
unless another count is given). None of it is a result on real footage.

1. **Gate G0 decides on the contract only** (confirmed by the owner on 2026-11-14). Reason: the old
   pass lines (mean IoU >= 0.6, mean point error <= 1% of the image side) failed for a client
   that follows the contract. Measured on the three contract frames: 5 px of noise on every box
   edge failed 5 of 5 seeds, 15 px of noise on every landmark coordinate failed 5 of 5 seeds,
   and two swapped landmark ids in one frame gave a mean error of 0.0402. A real contract fault
   is far away: with x and y swapped, 2 of 44 boxes have a pair and the median point error is
   0.385. New verdict: at least half of the possible pairs and a median IoU >= 0.3; a median
   point error <= 0.05. The values 0.6 and 0.01 stay in `configs/gates.yaml` and in the report
   as accuracy values without a verdict. A scale fault of about 2 percent is no longer a failed
   check; it shows in the reported mean values.
2. **Tracking never matches by a raw image position.** A track that the camera motion cannot
   reach waits, and the pipeline crosses a missing motion step through the pitch mappings of
   the two frames. Measured (1920 x 1080 pixels, pan period 4 s, 5 missing steps): 14.0 mixed
   tracks before, 0 in 9 runs after.
3. **Tracks: velocity, votes, joins.** A track is expected at 0.85 of its last step; labels are
   compared with the vote of the track; a join needs no rival and a reach of 4 m/s
   (`stitch_speed_mps`). Measured by the second review (200 trials for each speed of 2, 4, 6
   and 8 m/s, a camera that does not move, the meeting at a random time between two keyframes):
   two team mates who meet head-on on one line swapped tracks in 200 of 200 trials before and
   in 27 of 200 after; with a side offset of 1 m, in 0 of 200 after (before: 0 of 200 at 2 m/s,
   176 to 192 of 200 at 4 to 8 m/s). This replaces "58 of 64 before and 2 of 64 after" of the
   first review, whose layout is not recorded. With one false detection per keyframe 6 of 72
   joins were right before and 8 of 9 after.
4. **One person, one id over a long clip.** Fragments with the same role, team side and
   displayed shirt number become one track. Measured (750 frames, exact answers): 50.3 tracks
   for 23 people, coverage 0.606 and 28 switches before; 23.0 tracks, coverage 0.958 and 0
   switches after. The link needs confident readings: with 50% empty and 10% wrong crop
   readings the coverage is 0.78, below gate G4. The second review limited the link to a
   number that one person shows, and gave short tracks two crops (items 1 and 2 below).
5. **Team side: goalkeepers as second evidence.** The players decide alone when their two team
   means are at least 3 m apart; else the goalkeepers who stand in a goal. The 3 m is an
   estimate. Measured: one-half layouts 0.00 side accuracy in 12 of 12 runs before, 1.00 and
   0.92 after. The label stage applies the same rule to the labels and reports the gap between
   the player means and the goalkeeper counts of every clip.
6. **Landmark fit.** A fit with more than 4 pairs needs 40% of them as inliers; a keyframe
   with fewer than 7 agreeing landmarks takes the mapping that the camera motion carries over
   from a keyframe with 7 or more. Measured: answers with shuffled ids gave a valid fit in 23.7%
   (10 to 14 landmarks) and 59.0% (15 or more) of 1200 frames before, 3.4% and 0.0% after; in
   the review case with 4 landmarks at a share of the keyframes and 4 px of noise the position
   error p90 fell from 1.44 to 0.80 m (6 seeds). The second review changed how the share is
   counted and when the carried mapping is taken (items 3 and 4 below).
7. **Export number.** The reading alone is exported when it is confident or two crops agree;
   else the most frequent number over crops and frames. Measured share of player tracks with
   the right export number (50% empty and 30% wrong crops): 0.511 before, 0.960 after. The
   second review added the digits of a two-digit reading to the vote (item 7 below).
8. **Dense output.** A frame without an observation gets no state when the box of the person
   lies outside the image, and no gap above the stitch gap is filled. Measured share of states
   for people who are not in the truth, pan period 4 s: 0.274 before, 0.003 after.
9. **Run limit counts stored answers.** The limit of 1.3 times the estimate holds for new and
   stored answers together, and a paid command refuses to start when the ledger plus its
   estimate is above the hard stop. Reason: a run that the budget guard stopped got a full new
   allowance at every start.
10. **Stored answers belong to pixels.** A stored answer that was recorded with other image
    bytes stops the command before any model call; `--allow-other-images` is for the same
    footage. Reason: a clip that was replaced under the same name got the answers of the old
    pixels for free and without a word.
11. **Number requests.** The estimate assumes 50 number requests per clip (measured on the
    synthetic clip of 30 seconds: 48 to 50). The item of a number request of the pipeline names
    its crops and no track id, so the same crops are not paid again when the tracks get other
    ids.
12. **Flex tier.** `gemini-robotics-er-2-preview` has no flex tier in the documentation: no flex
    request for it is planned, estimated or sent. A flex call waits up to 900 s, a standard
    call 300 s. A lost connection is tried again; a client timeout is not, because the call can
    still be billed.
13. **Scorer wrapper.** The command names both split folders and passes `--SKIP_SPLIT_FOL True`,
    and the scores are read from `person_detailed.csv`. Read in the source of the evaluator;
    not run yet.
14. **A scored frame without a prediction is valid.** The format checker no longer reports it;
    `run` and `demo` print the number of such frames. A clip without any prediction is a failed
    run since the second review (item 11 below).
15. **`sync` runs the secret scan** of the git hooks after the guard, so that a commit of
    results is not refused later.

## 2026-11-14 — Decisions of the second review (repository rules)

Every number below was measured by the review in scratch clones with made-up values, with the
installed hooks of the hook runner where a hook is named.

1. **Names are text.** The rules `secret`, `host`, `home` and `deny` read the path of every
   checked file. The push mode also reads the names of the local branches that the remote does
   not have, the name of every local tag, and the name that the hook runner gives for the first
   branch of the push on the remote. Reason: 11 of 11 pushes with a made-up key, a deny-list
   word or the folders of a home path in a name reached the remote, with both push hooks
   passed. After: 1 of 11 (a private host name as a folder name, which only the deny-list can
   know). A push that sends only names starts no hook, so `guard.py --push` is run by hand
   before it.
2. **A tag points at a commit.** New rule `tag`. Reason: a tag at the content of a file with a
   made-up key was pushed without any hook, and the push mode by hand passed it.
3. **Identity: name and e-mail.** The clone needs a local `user.name` and a local `user.email`,
   and every author, committer and tagger must have both. Reason: with only the local e-mail
   set, a commit took the name of the user account and passed both hooks. Both machines need
   the same two values, because every local tag is checked at every push.
4. **Dataset data by content, at any depth.** The rule reads `.json` and `.jsonl` files, the two
   types that `sync` accepts: the keys of label data, of prediction data and of a run result at
   any depth, one document for each line, a byte order mark, and records of the label format.
   This replaces "the top-level keys of a JSON file" of the first review. Measured: 0 of 7
   other shapes were refused before and 7 of 7 after (6 shapes of a synthetic label file with
   434 annotations, and a run result without its outer key). 8,270 result files that the real
   commands wrote with the fake model give no report line before and after.
5. **A bare value after a secret name needs a digit.** Reason: 11 of 30 plain code lines such as
   `max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS` were refused, 0 of 30 after. On 1,609 library
   files (33.5 MB) the text rules give 39 report lines before and 34 after, and no new line.
   The secret scan has the same test in its generic rule. The cost is an estimate: a random
   value of 20 letters and digits has no digit with a chance of about 3%. A value in quotes is
   refused with or without a digit.
6. **Key forms.** A key is found after the letter of an escape (the line break of a JSON text),
   and the prefix `AQ.` after every character but a capital letter. The names `SECRET_KEY`,
   `PRIVATE_KEY`, `ACCESS_KEY` and `<word>_PASSWORD` are secret names. Measured: 5 of 5 lines
   with a made-up key after an escape, a letter or an underscore passed the guard and the
   secret scan before, 0 of 5 after.
7. **Private addresses.** An address after one equals sign is read, and the ranges 100.64 to
   100.127 and 169.254 count as private. A version after `==`, `>=` or `~=` still passes.
8. **False alarms.** The texts `~n/2`, a route `/users/me` and a rule line of dashes pass now.
   Two texts stay refused on purpose: a route `/home/<word>` with a real word (the form of a
   home folder) and a bare list of 32 or more check sums (the form of hexadecimal data that is
   wrapped at 64 characters, which the first review made the guard refuse).

## 2026-11-14 — Decisions of the second review (pipeline, money, measurement)

Every number below was measured in this review on synthetic clips with the fake model, or on a
stand-in for the model SDK with a fake clock. None of it is a result on real footage or on the
real API. "The bench" is 13 scenes x 3 seeds = 39 clips of 250 frames and 10 scenes x 3 seeds =
30 clips of 750 frames, 1920 x 1080 pixels, with one kind of answer fault per run. "Before" is
the code at the end of the first review. The closing step of the review measured every number
below again, on both code states where a number has a "before". The review did not check the
numbers of the first review that need scenes which are not recorded: items 2, 5 and 8, the
joins of item 3 and the second part of item 6.

1. **One number, one person.** The link of fragments by the displayed number (first review,
   item 4) needs a number that one person shows. Two tracks with one number are two people
   when they are further apart than a person can run plus 6 m (`tracking.TWO_PEOPLE_MARGIN_M`,
   an estimate). Such a group is not linked, and only its longest track shows the number.
   Reason: a number that was read as the number of a team mate made one track of two people,
   and the fill drew a person who ran across the pitch between them (review case: 59 states
   between the two people before, 0 after). Measured on the bench, 750 frames, one misread
   number per clip: 0.50 mixed tracks per clip before and 0.30 after, coverage 0.911 and 0.898.
   Cost with 10 px of landmark noise, 750 frames: coverage 0.886 before and 0.877 after.
   Margins of 2, 4, 6 and 10 m give a coverage of 0.815, 0.858, 0.877 and 0.885 in that cell
   (30 clips each) and change nothing in the cell with the misread number. So the margin is
   larger than the reach margin of the matching step (2 m), and tracks that overlap in time
   are not taken for two people without the distance test.
2. **Two crops for a short track.** A track with two observations or more gets two crops
   (`pipeline.crop_choice`): when the rule of one second leaves one crop, the sighting that is
   furthest in time from it is taken too. Reason: a number is displayed only when two crops
   agree, and fragments get one id again only by a displayed number, so a fragment of up to 3
   keyframes was never linked. Measured on the bench with exact answers: the switches of the
   scene with 8 s out of the image were 9, 10 and 10 before and 0, 0 and 0 after (750 frames,
   3 seeds); over all scenes 6.9 switches per clip before and 4.2 after at 750 frames, 3.2 and
   0.9 at 250 frames. 37 of 39 clips of 250 frames and 27 of 30 clips of 750 frames are inside
   the three limits of gate G4 (34 and 21 before); the misses are the scene with people who
   run through each other again and again. Not measured: whether two real crops that are less
   than one second apart agree on a wrong number (the fake model reads every crop on its own).
3. **Landmark fit: the share counts near pairs.** The share of 40% (first review, item 6) is
   counted among the pairs within 3 times the RANSAC threshold (`geometry.NEAR_LIMIT_FACTOR`),
   and a fit still needs 5 inliers. Reason: with landmark noise of the size of the threshold,
   fewer than 40% of the right points are inliers, and usable fits were thrown away. Measured
   on 4000 random camera frames (2054 with 10 to 14 landmarks and 807 with 15 or more; 2027
   and 803 in the runs with noise): answers with shuffled ids give a valid fit in 3.6% and 0.5%
   of the frames (3.4% and 0.1% before; 22.9% and 57.1% with the code before the first
   review). True ids with 10, 15 and 20 px of noise and 15 or more landmarks keep 99.8%, 93.4%
   and 75.7% of their fits (87.5%, 53.5% and 32.4% before); with 10 to 14 landmarks 99.6%,
   95.1% and 86.5% (99.0%, 87.4% and 72.3% before). Bench with 10 px of noise, 250 frames:
   1.85 frames without any state per clip before and 0 after, recall 0.919 and 0.929. Side
   effect with 15 px of noise, 250 frames: recall 0.813 and 0.854, but 10.6 and 22.9 switches
   per clip, because many more noisy fits are kept.
4. **A weaker keyframe checks the carried mapping.** A keyframe with a weaker fit takes the
   mapping that is carried over from a well-supported keyframe only when more than half of its
   own landmarks are within 3 times the RANSAC threshold of it; else it keeps its own fit.
   Reason: one well-supported but wrong fit replaced the good fits of weaker keyframes (review
   case, one keyframe shifted by 25 px: 5 of 22 keyframes more than 0.5 m off before, 4 of 22
   after, 1 of 22 before the first review). Measured with 10 px of noise on 5 scenes x 10 seeds
   = 50 clips of 250 frames: 7.5 switches per clip with the limit of 3 times, 9.4 with 2 times
   and 9.9 with 1.5 times; coverage 0.876, 0.864 and 0.862. Limit of the rule: a well-supported
   fit that is wrong by less than 24 px at the weaker keyframe is still taken.
5. **Plausibility of a well-supported fit.** Every fit may put 1 person outside the pitch.
   With more people outside, a fit with 7 or more agreeing landmarks must put two thirds of
   the people on the pitch (`pipeline.STRONG_FIT_MIN_SHARE_ON_PITCH`). Reason: a fit with 7
   agreeing landmarks was accepted with 0 of 9 and with 3 of 9 people on the pitch. The rule
   guards against a rare gross error: the bench shows no change with the rule switched off
   (250 frames, 39 clips for each kind of fault: with 10 px of noise the recall is 0.929 and
   the switches are 10.1 per clip both ways, with 15 px 0.854 and 22.9).
6. **A second box with a small shift is a repeated box.** A box with the same role and the
   same team key as an earlier box of the answer and an IoU above 0.6 is dropped
   (`pipeline.REPEATED_BOX_SAME_LABELS_MIN_IOU`, an estimate until the people probe). Measured
   on the bench with a second box for 15% of the people, shifted by 12 to 28% of the box
   width: 31.3 switches and 1.83 tracks per person before, 11.7 and 1.33 after, coverage 0.802
   and 0.878 (250 frames); 109 and 42 switches, coverage 0.754 and 0.833 (750 frames). With
   the rule switched off on the final code, the clean case and the case with one false
   detection per keyframe have the same values (250 frames).
7. **Export number: the digits of a two-digit reading.** A crop or a frame that shows one digit
   of the two-digit number of an unsure reading counts for that number in the vote (first
   review, item 7). The number 10 that was read from the crops 1 and 0 was exported as 1
   before and is 10 after. Not measured on the bench: the fake model never reads a number in
   parts.
8. **Radar trail.** The trail of a track has a gap where the track has no state in a drawn
   frame. Before, a straight line crossed the gap.
9. **Flex tier: patience.** A flex request that gets the status 503 or 429 is tried again for
   900 s after its first attempt, with waits of up to 60 s. These attempts are not billed and
   do not count as attempts. The standard tier and every other failure keep 4 attempts. The
   check `flex tier` of the doctor gives no verdict when no request found capacity. Reason:
   503 and 429 are the documented answers of a busy flex tier, and gate G0 reported them as a
   fault of the client. Measured with a stand-in for the SDK and a fake clock, 3 runs per
   case: a flex tier that is busy for 60, 120, 600 and 890 s gives its answer after 4 to 5, 5
   to 6, 16 and 22 requests. Before, a tier that was busy for 120 s or longer failed the check
   after 4 requests and 46 to 69 s (9 runs). A tier that is busy for 2000 s ends the request
   with `FlexBusyError` after 22 to 25 requests and 901 to 951 s (6 runs), and the doctor
   reports `info`. Not measured: how often the real flex tier is busy.
10. **One request per call.** The client passes `automatic_continuation=False`. Reason: by
    default the SDK continues an answer that stops at the token cap of one request with
    further billed requests, and a failure of a later request left the billed ones without a
    ledger line. Measured on the stand-in, 2 scenarios with 1 run each: an answer that stops
    at the cap and whose second request fails gave 2 HTTP requests and 0 ledger lines before,
    1 HTTP request and 1 ledger line of 0.3307 USD after. Consequence: an answer above 32,768
    tokens of thinking and output is never used. It is a billed answer that stops the run.
11. **A clip without any result is a failed run.** `run` and `demo` end with exit code 1 when
    no scored frame has a prediction (without labels: when no frame holds a person). The files
    are still written, so the scorer counts the clip as missed. Such a clip sends no
    shirt-number request, because without a pitch mapping no track gets a state. Measured on
    the synthetic clip of 50 frames with landmarks answers that name no landmark: exit code 0
    and 17 number requests (0.153 of 0.343 USD) before, exit code 1 and 0 number requests
    (0.190 USD) after. The other way, an error in `run_clip` (exit code 2 and no prediction
    file), was not taken: a clip without a prediction file can drop out of a scored set
    without a word.
12. **Stop signals.** A model call of the main thread finishes before the command ends, also
    after a second signal, and no attempt is sent after a stop signal. Measured with a fake
    model whose calls take 0.3 s: no answer lost in 30 of 30 trials (10 for the kit legend
    request of `run`, 10 for a probe, 10 for the doctor); before, 30 of 30 trials lost the
    answer of the call in flight. A request that waited for a new attempt sent 0 calls after
    the signal, and the command ended after 0.1 s, in 3 of 3 trials (before: 1 call and 4.1 to
    6.3 s). A second signal during the wait for the requests of the thread pool no longer ends
    that wait: the printed cost named 1 of the 5 paid answers before (1 run) and 5 of 5 after
    (3 runs).
13. **Cost ledger: a complete series.** The totals refuse a ledger whose files are not the
    numbers 1 to n, each one time, and a line with a negative cost. Reason, measured: a ledger
    with only its second file gave a total of 0.051 USD without an error, so a missing file
    took its paid answers out of the total of the hard stop without a word; a line with a cost
    of -100 USD lowered a total to 34.99 USD. A paid answer is still recorded in the last
    file.
14. **The evaluator gets no key.** The commands `oracle` and `eval` start the evaluator
    without the key variable of the model in its environment. The evaluator is the code of
    another project with packages that are not pinned.
15. **At most 8 requests in flight.** `max_parallel` has the upper limit 8
    (`config.MAX_PARALLEL_LIMIT`). A budget stop comes after the answer that crosses a limit,
    and the requests in flight at that time are paid too. So a run passes its limit by the
    cost of `max_parallel - 1` answers at most: 3 with the default, 7 with the upper limit.
    Before, a settings file could put a whole step in flight.
16. **Output of the command line.** The standard output is line buffered, and wrong content
    of a settings file, a roster file, a label file or an experiment file is one error line
    without a traceback. Measured with the output in a file (1 run each): 0 bytes while the
    first model call waited before, 1,147 bytes with the estimate line and the budget line
    after; the line `error:` was line 1 of 8 before and is line 8 of 8 after.

Rules that the second review measured and left as they are. Each one is a decision for the
owner after the first real answers (see the open questions). The closing step patched each
variant in at run time on the final code and ran it on the bench (39 clips of 250 frames and 30
clips of 750 frames for each kind of fault):

- **Label penalty that grows with missed keyframes.** A person who is seen again after 2 to 4
  missed keyframes with one wrong role or team key gets a second track from the matching step
  (the join of fragments repairs the simple case). In the scene with the fastest pan and wrong
  roles the coverage is 0.943 with 2.7 switches (3 clips); without the growth it is 0.969 with
  0 switches. But without the growth the clean case gets worse: 5.0 switches and 0.53 mixed
  tracks per clip against 4.2 and 0.30 (750 frames), and with one false detection per keyframe
  the share of states where nobody is rises from 0.038 to 0.050. A growth for the team key
  only gives 4.3 switches and 0.33 mixed tracks in the clean case and 8.9 against 8.4 switches
  with 20% missed people (750 frames). No variant is better in every kind of fault.
- **Velocity of a track.** Two team mates who meet head-on on one line still swap in 27 of 200
  trials (first review, item 3), and two who turn at the meeting point in 113 of 200 trials at
  6 m/s. Using the velocity only while it is fresh changes nothing in the clean case and has
  mixed signs elsewhere: 6.1 against 5.9 switches with 10% missed people, 8.3 against 8.4
  with 20% (750 frames).
- **Number guard of the joins.** One wrong per-frame number stops a right join, and two team
  mates who come back together are never joined. The fragments of the review case are now
  linked by their displayed number (item 2): with 30% wrong per-frame numbers the scene with 3
  s out of the image has 1.00 tracks per person and 0 switches (1.26 and 6 before; 250 frames,
  3 clips). A guard that needs the number in 2 frames changes no cell with one kind of fault
  and gives 42.2 against 42.0 switches in the mix of faults (750 frames).
- **Team side with goalkeeper evidence.** A track that is voted goalkeeper can turn the team
  sides of a clip when the player means are less than 3 m apart: an attacker who is voted
  goalkeeper, two goalkeepers with the same team key, a corner kick with the attackers deeper
  than the defenders (3 of the 6 hand-made review cases get the wrong sides). No change: the
  rule is the one that gate G1 measures on the labels, and each repair has a cost that only
  real label files can weigh.

## 2026-11-14 — Decisions after the second review

Two points that the second review left open follow from rules that the project has already,
so they are settled here (items 1 and 2). Items 3, 4, 7 and 8 are decisions of the owner.
Items 5 and 6 come from the checks of the change of item 4.

1. **A stored answer holds no box of the request context.** A cassette keeps the frame numbers
   of the request context and no other entry (`VLMRequest.stored_context`,
   `STORED_CONTEXT_KEYS`). Reason: the number probe cuts its crops at boxes of the label files,
   and the context of its requests went into the cassettes as it was. So label data would have
   gone to the branch `results`, against rule 2. Rule 2 now says so in words. Measured on 3
   synthetic clips with the fake model (60 number requests, 204 crop boxes, every one a label
   box): 204 of 204 boxes were in the cassettes before, 0 of 204 after. An independent check
   wrote the result files of the real commands on 4 synthetic clips (169 cassettes of all
   four steps, the ledger, 9 metrics files) with a fake model that answers from another scene,
   so that a label value in a file can only come through the code. It compared every number
   of these 179 files with 7,879 label boxes in 24 forms and with their pitch positions in 12
   forms: 0 matches outside the answer texts of the model. The request itself keeps the whole
   context, and the cache key still stands for it: a request with other frames or other boxes
   does not find the stored answer. The key of a request does not change with this decision,
   and a cassette in the old form is still read. An entry that is new is not stored until its
   name is added to `STORED_CONTEXT_KEYS`. What this decision does not remove:
   - The boxes that the model answers are part of the answer text of a people cassette. They
     are model output, not label data.
   - The boxes of the number requests of `run` are dropped too, although they are model
     output: one rule holds for every request. The people answers of the clip hold the boxes
     of the model.
   - A cassette and a ledger line of the number probe name the track id of the label file
     (the item `track-<id>-<8 hex digits>`) and the frames of the crops. So they tell that
     this track has a box in these frames, and nothing else of the label file. The script of
     fact 14 (`docs/RUNBOOK.md`, section 8) needs both values. The owner accepted this trace
     (item 7).
2. **Rule 9 names the pushes without a hook and the bypasses that were measured.** The rule now
   names a push that sends only names (a tag, or a new branch on a commit that the remote
   has). As bypasses it names the variable `SKIP` of the hook runner and another hook folder
   (`core.hooksPath`) next to `--no-verify`. It also says when the hooks run: before
   `git commit` and before a push that sends commits. Reason: each case was measured with the
   installed hooks in a scratch clone. A push of only names started no hook and reached the
   remote, `SKIP` skipped a hook, and a commit with another hook folder started no hook (the
   push hook then refused its content).
3. **A commit message has no co-author line** (owner, 2026-11-14; rule 10). Reason: a commit
   names its author and its committer, and the guard compares both with the personal identity
   of the clone. A co-author line adds a name and an address that no rule compares with it.
   Such a line can also hold a word of a local deny-list. Measured in a scratch clone with the
   installed hooks: the commit passed, because no hook reads the message at commit time, and
   the push was refused. So the fault shows only after the commit exists.
4. **The contract checks of the doctor stop at the spend limit of gate G0** (owner, 2026-11-14;
   rule 6). The limit is `g0_contract.spend_usd_max`, 2 USD, in the place of 1.3 times the
   estimate. Every other paid run keeps the 1.3 rule. Reason: the estimate of the doctor comes
   from token counts that nobody has measured, and the doctor is the run that measures them.
   Computed with the default token table at the prices of 14 Nov 2026: the day-1 command with
   two models has 14 calls and an estimate of 0.133 USD, so the old limit was 0.173 USD; the
   default model `gemini-4-argon` alone has 7 calls, 0.097 USD and 0.126 USD. Real answers
   that cost more than 1.3 times their estimate together would have ended the first check of
   the client before its report was complete. The gate allowed 2 USD already, as a check after
   the run; now the same number stops the run. What the rule means:
   - The limit holds for one doctor run with all its models together. The checks do not start
     when their estimate is above the limit, also with `--start-over-budget`.
   - The answers of the doctor are not stored. A doctor that is started again pays for every
     answer again, up to the limit at each start. Only the hard stop ends repeated starts.
     Measured with the fake model and answers that are made expensive: three starts that each
     stopped at the limit paid 7.04 USD together.
   - The plan has 2 USD for the doctor and the cost pilot together. One doctor run can use
     this amount alone. `docs/RUNBOOK.md`, section 2, says when to tell the owner.
   - The check `spend` of the report fails only together with a budget stop now.
   - A gate value that is no amount above 0 gives one error line before any other output.
   - A named limit: the meter of a run and its ledger add the same costs in another order. A
     run whose cost is equal to its limit can therefore stop by one rounding step. Measured
     with a spend limit equal to the estimate: 3 of 5 model lists stopped after their last
     answer. With the limit of 2 USD this needs a cost within one rounding step of 2 USD.
   Tested with the fake model: checks that cost 10 times their estimate finish below the
   limit; checks that pass the limit stop after the answer that passes it, and the report is
   still written; a run and a probe print the same lines as before the change (8 scenarios of
   an independent review). The stop line of a run with a spend limit has three decimals,
   because the spent amount lies less than one answer above the limit.
5. **Both budget stops name the limit with the same words.** Two checks hold the limit of a
   run: the budget guard after a paid answer, and the cost meter before the next call. With
   several requests in flight, thread timing decides which one stops the run. Their messages
   named the limit with other words, and one test asked for the words of the guard. Measured:
   that test failed in 3 of 48 runs with 8 test processes at once on a machine with a load
   average of about 106, and it passed alone. The run itself was right in both cases. Now the
   meter gets the words of the limit ("1.3 times its estimate", "its spend limit in the gate
   file"), and both messages hold them. After: 192 of 192 runs passed with 96 test processes
   at once.
6. **The signal test of the pipeline sets the SIGINT handler that it needs.** A test process
   that a shell starts in the background ignores SIGINT, and the test sent a real SIGINT.
   Measured: the test failed in 6 of 6 suites that ran as background jobs, and it failed in
   the same way alone on a machine without load; in the foreground it passed. With the handler
   set for the time of the test: 192 of 192 runs passed with 96 test processes at once, and 8
   of 8 whole suites passed at the same time (2485 passed, 1 skipped each; the skipped test
   needs the git folder of the repository).
7. **The stored answers of the number probe keep the track id of the label file** (owner,
   2026-11-14). The item of such a request stays `track-<id>-<8 hex digits>`, and the stored
   context keeps the frames of the crops (item 1). Reason: the two values name a track and
   the frames in which it has a box, and nothing else of a label file: no box, no pitch
   position, no role, team or shirt number. They let a script join a stored answer with its
   label on the run machine (`docs/RUNBOOK.md`, section 8, fact 14). The other way, a hash of
   the clip id and the track id in the item, was not taken. It would give every number request
   of the probe a new cache key, so it is free only before the first paid number probe.
8. **The report of a doctor that the budget stops stays as it is** (owner, 2026-11-14). It
   holds the checks of the models that were finished and the cost and the tokens of every paid
   answer. It does not hold the answer texts of the model that was in progress. Reason: the
   case needs checks that cost about 15 times their estimate, and the tokens that correct the
   estimate are in the report. The other way, to write these answers too, was not taken.

## 2026-11-14 — Both specialist tools are measured on the probe frames

1. **The decision** (owner, 2026-11-14). The RF-DETR detector and the PnLCalib calibration are
   measured in M3 on the frames of the people probe and of the pitch probe, with their
   pretrained weights and without training. No model is called, so it costs no API money.
   Reason: the decision of 24 Nov then has measured values of both tools next to the values of
   the two model steps (boxes for RF-DETR, pitch mappings for PnLCalib), and the time per frame
   of each tool on the run machine. Rule 8 does not change: a tool replaces a Gemini step only
   after that step failed its gate and the tool passed the same gate on the same frames. The
   two commands measure a tool. They do not switch it on: `run` takes the people and the pitch
   mapping from the model steps only, the pipeline has no setting for another engine, and
   RF-DETR gives boxes without role, team key and number. That work is an open owner decision
   (last list of this file). Not decided and not started: a detector that is trained on answers
   of the model.
2. **How.** Three steps (`docs/RUNBOOK.md`, section 3): the command `tool-job` writes the frames
   and the settings; the script `tools/run_tool.py` runs the tool in the environment of the tool
   and writes one answer per frame; the command `tool-score` measures the answers. The frames
   are those of the model probe, because the tool file and the probe file have the same group,
   frames per clip and seed; a test holds that for the files of the repository. The values come
   from the code that scores the model answers. Tested: for the same boxes and for the same
   mapping records, the tool result and the model result are equal in every shared value. A
   record of a tool has no count of agreeing landmarks, so its plausibility check is always the
   strict one (9 of 10 people on the pitch); a model fit with 7 or more agreeing landmarks
   needs 2 of 3. `tool-score` refuses a run file that was made for another experiment name or
   with other tool settings, and a run file that lacks a frame of the probe. The run file is
   read strictly: a key that comes twice is refused, a number must be written as a number (no
   text, no truth value) and a truth value as true or false. The provenance of the metrics file
   names the run file with its SHA-256 and holds the facts of the run that the script wrote,
   among them the SHA-256 of the script. `tool-score` says when that is not the script of its
   work tree: the commit of the metrics file then does not name the script that ran the tool.
3. **Settings of the five experiment files.** They are the settings that each tool documents
   for its own use, so the first values are those of the tools as they come. One setting is
   measured in both of its values, the resize of the detector:
   - RF-DETR Small and RF-DETR Large, score limit 0.5. The limit is the threshold that the tool
     uses by itself. Small reads a frame at 512 x 512 pixels and Large at 704 x 704. Both
     stretch the frame to a square: a person of 40 x 100 pixels in a 1920 x 1080 frame is about
     11 x 47 pixels at the input of Small. Each of the two has a second file with an
     antialiased resize. Reason, read in the source: the tool resized with antialiasing up to
     its version 1.8.3 and without it from 1.9.0, its weights are older than 1.9.0, and a user
     report says that 22% of the model and picture pairs change at the threshold 0.5 between
     the two resizes. The evidence about the right resize is mixed, so both are measured.
   - PnLCalib with its weights for main-camera pictures (`SV_kp`, `SV_lines`), the two
     thresholds of its script `inference.py` (0.3434 and 0.7867) and its line refinement, as in
     the command of its README. The camera is calibrated at the size of the frame, as that
     script does it.
   Settings that exist in the tools and are not in an experiment file: a frame shape for
   RF-DETR that keeps the side ratio; for PnLCalib the lower thresholds of its benchmark script
   (0.0966 and 0.3441, which need no new key), a calibration at 960 x 540 pixels, the size
   of its benchmarks, and the limit of that benchmark script for the reprojection error of a
   camera (`--max_reproj_err 100`, in pixels at 960 x 540): the script drops a camera above it.
   The measurement here drops none, so the published values of the tool come from a setting
   that refuses weak cameras. The run file keeps the error of every camera (`rep_err`, in
   pixels of the frame), so such a limit can be counted in `tool-score` without a new run of
   the tool. The frame shape and the calibration size each need a key in the experiment file
   and in the script, the error limit a key in the experiment file. They are not built, because
   no option stays unused in this repository.
4. **What was read from the source of the tools on 2026-11-14.** Nothing of it was run.
   - `rfdetr` 1.12.1 (uploaded 2026-11-12) and 1.12.0 (2026-11-06): `detr.py`, which defines
     `predict`, is the same in both. `_prediction/core.py`, which does its work, differs only in
     code that the script does not reach (float input with a value that is not a number, the
     batch size of a compiled predictor, masks). The model classes with pretrained weights for
     everyday pictures are
     `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium` and `RFDETRLarge`. `predict` takes a frame as
     red, green and blue values of shape (height, width, 3), keeps a detection when its score
     is above the threshold, and returns boxes in pixels of the frame with the class name of
     each one. An installer with a package age limit of 7 days takes 1.12.0 until 2026-11-19.
   - PnLCalib at the commit `42088864170f75ea02e76d20e9a9a1db59edb3e3` (2026-10-26). It is no
     package: the script imports files of a clone. Its world frame has the origin at the centre
     mark, x towards the right goal, y towards the main camera and z down, on a pitch of
     105 x 68 m. That is the pitch convention of this package, so the script changes no axis.
     Its calibration returns a camera or nothing. It never refuses a camera with a large error,
     so the script writes the error of each camera into the run file (`rep_err`), with the
     place of the camera and the plane that it was calibrated on. Its requirements file is for
     Linux with an NVIDIA GPU. The install of `docs/RUNBOOK.md` takes the versions of that file
     for the packages that the script needs; an install without versions would take OpenCV 5,
     and the tool was not tested with it.
   - A second check read each of these facts in the source again: 99 of 105 facts about
     PnLCalib and 98 of 100 about RF-DETR were confirmed. Four were wrong, and no sentence of
     the documents and no call of the script uses them: the transforms of the training of
     PnLCalib (the script uses the plain one of `inference.py`); which matrix the calibration
     object holds after the vote (the script builds its matrix from the camera that the vote
     returns); the side on which the line filter of PnLCalib drops a line; and the version in
     which RF-DETR moved the source picture of an answer (1.6.5, not 1.7). The 4 that cannot be
     checked without a download are about the content of the weight files.
   - Published values, not measured here: RF-DETR Small needs 20.0 ms per frame on an M4 Max
     with its plain `predict` and 11.1 ms with its compiled predictor; its box accuracy on the
     COCO pictures is 53.0 (Small) and 56.5 (Large). The paper of PnLCalib gives 439 ms per
     frame for the same voting with line refinement (the two networks on an NVIDIA RTX 2080 Ti,
     the calibration on a CPU). That time was measured with the reading of the image, at 960 x
     540 pixels, with other weights and other thresholds: it is not for the settings above.
5. **What guards against a wrong script.** The script was written without the tools, and its
   tests use stand-ins. An error of a tool on a frame stops the run and names the frame: no
   frame counts as a frame without an answer because of an error. And: (a) the script refuses
   another version of `rfdetr` and another commit of PnLCalib than those that were read, and a
   PnLCalib clone with a changed tracked file; (b) the pitch result holds `axis_check`, the
   median position error for the matrices as they are and with the sign of x, of y or of both
   changed. The pitch convention is right when the matrices as they are give the smallest value
   and that value is small (some metres at most). When another sign gives a smaller value,
   `tool-score` says so, writes the metrics file and ends with the exit code 1: a large value
   as given next to one small other value means that the script has the wrong pitch convention
   (then no frame is plausible), or that the tool mirrors the pitch on more than half of the
   frames (then some frames are plausible).
   When all four values are large (about 10 m and more), the check decides nothing: the output
   of the tool is wrong for another reason; (c) a test compares the format names, the settings
   and the model classes of the script with the package; (d) `docs/RUNBOOK.md`, section 8, has
   the first check on two clips, with the comparison of the GPU with the CPU; (e) the script
   takes the variable `GEMINI_API_KEY` out of its environment before it loads a tool, and the
   commands of the RUNBOOK start it without that variable.
6. **The adapter of the detector has no default score limit.** Before, it kept detections from
   0.3 when the caller named no limit, and the tool keeps them above 0.5 by itself. Now the
   experiment file names the limit, the script gives it to the tool, and the adapter gets the
   same value.
7. **The script binds the package names of the PnLCalib clone to the clone.** The files of the
   clone import each other as `model.<file>` and `utils.<file>`, and its folders have no
   `__init__.py`. A test showed: Python takes an installed package with such a name before
   these folders, also when the clone comes first on the module path. So the script sets both
   names to the folders of the clone before it imports from them.
8. **The run files of a tool are text results.** They hold the boxes that a detector found, or
   camera matrices, and no pixels and no value of a label file. The commands of the RUNBOOK
   write them into the results work tree. Whether they go to the results branch like stored
   model answers is an open owner decision (last list of this file): the owner decision of
   13 Nov names three kinds of files that may return, and a run file is a fourth kind. Tested:
   the command `sync`, with the guard and the secret scan, passes a detector run file of 135
   frames and 770,559 bytes and a calibration run file. The listing of `sync` names a file
   below a folder `tools` a tool run.
9. **Licences.** The code of RF-DETR and the weights of its four model classes above are under
   Apache-2.0; the model classes of its extension package have another licence and are not
   accepted by the experiment file. The code of PnLCalib is under GPL-2.0-or-later and its
   weights under CC BY 4.0. `tools/run_tool.py` imports code of the PnLCalib clone when it
   runs. The repository is public since 2026-11-14 (section "The code repository is public").
   The check of what the licence of PnLCalib means for that script is still open
   (`docs/PLAN.md`, risks).
10. **Review of this change (2026-11-14).** Five reviews, each with one view (the package code,
    the script, the command line, the measurement, the documents), and a second reading of each
    finding. 50 findings, 48 confirmed: 6 major, 19 minor and 23 small ones. Repaired, each
    code change with a test:
    - The script: a box or a score of the detector that is not a finite number stops the run
      at its frame; an import error inside an installed tool keeps its traceback; a PnLCalib clone
      with a changed tracked file is refused; the place of the run file is made before the tool
      is built; the script writes its own SHA-256 into the run file and takes the key variable
      out of its environment.
    - The package: the run file is read strictly; a number of a payload that is too large for
      a float is refused in one line; a camera matrix with very large or very small values is
      scaled before the test for a matrix that cannot be inverted; the people result counts
      the wrong boxes off the pitch (`counts.fp_off_pitch`).
    - The command line: a split list that is not JSON, an experiment file that is no text and
      a clip folder that does not fit its label file end with one line and no traceback;
      `tool-score` prints the axis check and ends with the exit code 1 when it fails; the line
      of the main metric names how many clips it used; the metrics file names the content of
      the run file and whether the script is the one of the work tree; an `--out` file that
      is the experiment file or the run file is refused; `sync` names a run file in every
      folder below `tools`.
    - The documents: what the two commands do not do (item 1), which pitch values of two
      engines are not on the same frames, the size rule of the axis check, the stricter
      plausibility rule of a tool record, and two new open owner decisions (the work that
      lets the pipeline use a tool, the frames of gate G3); the two older ones about the tools
      were made clearer.
    Kept as it is: one run of the detector serves one score limit. A run file with a lower
    limit could serve a higher one, but then the metrics file would not prove that the tool ran
    with the settings of the experiment file. Named by the review and not built: the share of
    all frames within an error limit, by-view values per clip, and the limit for the
    reprojection error of PnLCalib (item 3).
11. **Second read of these fixes (2026-11-14).** Three reviews of the fixes of item 10 (the
    code, the documents against the code, and the documents as the assistant on the run
    machine reads them), again with a second reading of each finding. 41 findings, all
    confirmed: 2 major (the same fault, found by two reviews), 10 minor and 29 small ones, and
    6 more that the second readers added.
    The major one: after the repository became public, three sentences still told the owner
    to push the branch `results`. Repaired:
    - The code: a group of the split list that names a clip twice, and a clip without a scored
      frame, end the model probe with one line too, as they do for the tool commands (before,
      the number probe took a clip without a scored frame as a clip with 0 tracks and went
      on; it stops now, like the two other probes); an
      `--out` file that is the split list, a hard link to an input file or the same file in
      another letter case is refused; the script checks the place of the run file before the
      tool is built (a folder, no write access, the job file); an error of the operating
      system inside a tool keeps its traceback. 24 test cases were added. The adapter of the
      calibration no longer repeats the singular test of `geometry.invert_homography`.
    - The documents: no push of the branch `results` before the owner decision; the identity
      command of the setup leaves out merge commits; how to tell a wrong pitch convention from
      a tool that mirrors a part of the frames (`plausible_share`); what to do when the first
      check ends in a case without a rule; which values to compare between the GPU and the
      CPU; the group `reserve` for the confirmation of a setting; every command that starts
      the interpreter of a tool environment begins with `env -u GEMINI_API_KEY`.
    Kept as it is: a guard of numpy warnings that does nothing in three places with the same
    comment (it changes no value). Measured for the documents: with matrices in another pitch
    convention, `plausible_share` is 0 for each of the three sign changes on the synthetic
    clip of the tests (5 frames); with 3 of 5 frames mirrored it is 0.4.
12. **Check of the fixes of item 11 (2026-11-15).** Two checks (the code with its tests, and
    the documents), each with a second reading of its findings. 17 findings, all confirmed:
    4 minor and 13 small ones. Repaired:
    - The code: an `--out` path through a folder that is not there yet is judged as the file
      that the command would write (before, a hard link of an input file behind such a folder
      was replaced); the line under the git commands of `sync` holds the push of the branch
      `results` back until the owner decision; three docstrings name what the code does.
      8 test cases were added, among them a clip with some frames that are not scored, the
      split list of the repository without `--manifest`, and a second run of the script on
      the same run file.
    - The documents: a copy of a tool experiment file goes into a sub-folder, because one
      offline test names the five tool files of `configs/experiments`; a second remote for
      the branch `results` needs no change of the guard when it holds the commits of `main`
      (last list of this file); the size limit of a run file is about 14,000 detections, not
      16,000; `written_at` always differs between two metrics files; the probe table names
      the two further lines of gate G2 with their keys; the folder `results/tools` is named
      at the commit step; item 10 no longer names the refusal of the split list, which is a
      repair of item 11; the merge commit of the git host is described by what was measured.
    Measured for the documents: a run file as the script writes it has about 140 bytes per
    detection. The guard in push mode judges only the commits that the remote of the push
    does not have: in a scratch repository with a merge commit of another identity, a push to
    an empty second remote was refused, and a push to a copy of the first remote passed after
    `git fetch`.

## 2026-11-14 — The code repository is public

1. **The decision** (owner, 2026-11-14). The owner had the code pushed to a public repository
   and asked for a pull request there. Before, the plan was "private for now" (first table of
   this file). The branch `main` holds the first build and the fixes of the review of the tool
   measurement: pull request 1 from the branch `m1-first-build` was merged on the page of the
   git host on 2026-11-14. That merge commit has another author e-mail than the other commits
   and the git host as its committer.
2. **What follows for every commit.** A pushed commit is public at once and cannot be taken
   back. So the rules 1 to 3 of `CLAUDE.md` now stand between a mistake and a public leak: the
   guard and the secret scan must pass before every commit and every push, and no way around
   them is allowed (rule 9).
3. **What is open** (also in the last list of this file):
   - The branch `results`. The command `sync` prints a push of the text results (metrics,
     stored model answers, the cost ledger) to the remote `origin`, which is public now. The
     line under that command and the RUNBOOK hold the push back until the owner decides.
     These files hold no pixels and no label data. They hold the answers of the model for
     frames of the dataset, and the dataset is for research use.
   - The licence. The repository has no licence file. The code of PnLCalib is under
     GPL-2.0-or-later, and `tools/run_tool.py` imports it when it runs. The check of what that
     means for the script was planned before publishing, and it is not done.

## 2026-11-22 — Image models through OpenRouter (owner decision, part of the provider)

1. **The decision** (owner, 2026-11-22). Every image or video model step must work through
   OpenRouter, not only through Gemini. The same decision takes the ideas of a classic detector
   stack and adds a real-time detector engine; other entries record those parts. This entry
   records the provider part. It changes the row "Model: Gemini family only" of the first table
   of this file, the first paragraph of `CLAUDE.md` and its rules 4 and 5 (item 9 has the
   proposed text; `CLAUDE.md` is not changed until the owner accepts it).
2. **What was built** (build machine, offline; nothing called a model or OpenRouter):
   - A step names `provider` (`gemini` or `openrouter`), `model` and `coords` in
     `configs/default.yaml` and in the `step` section of each experiment file. The doctor takes
     `--models openrouter:<slug>`, `--coords` and `--thinking-level`.
   - `OpenRouterBackend` in `src/pitchvlm/vlm/client.py`: the chat completions API over HTTPS
     with the standard library (`urllib`), no new dependency. So only `client.py` talks to a model
     service, and rule 4 keeps its meaning. `ProviderBackend` sends each request to the backend
     of its provider; `run`, `probe` and `doctor` use it.
   - Coordinate conventions per step (`src/pitchvlm/vlm/coords.py`): `yx_1000` (Gemini),
     `xy_1000`, `xy_unit`, `xy_pixel`. The people and landmarks prompts state the convention, the
     answer schema asks for integers or (for `xy_unit`) numbers, and the parsers read it. The
     contract checks of `doctor` (gate G0) are the proof per model: with the fake model, an
     answer in another convention than the one asked fails `boxes` and `landmarks` (tested for
     three wrong pairs) and the right one passes (tested for all four).
   - Money: an OpenRouter call costs what its answer reports (`usage.cost`, kept in
     `VLMResponse.reported_usd`); the ledger takes it, and its lines name the provider and the
     provider that served the call. The estimate comes from the new price table
     `configs/openrouter_prices.yaml`; a model without a price there has no estimate and is not
     sent. One ledger, the 1.3 times stop and the hard stop at 135 USD hold across both providers
     (tested: a Gemini line and an OpenRouter line together reach the hard stop).
   - The key comes from `OPENROUTER_API_KEY`. The guard and the secret scan know its form
     (`sk-or-v1-` and 64 hexadecimal digits), the assignment rule of the secret scan knows its
     name, and `openrouter.ai` is an allowed host. The scorer gets neither key.
   - The replay client and the fake client work for the new provider; the cache key holds the
     provider and the model.
3. **The Gemini path is unchanged** (measured by a test). `tests/test_vlm_openrouter.py` holds the
   cache keys, the prompt hashes and the hashes of the Interactions API arguments of five Gemini
   requests (kit legend, people, people through the flex tier, landmarks, number), computed with
   the code of commit f7d788e before any change. The new code gives the same 15 values. The
   provider enters the cache key only when it is not Gemini, and a Gemini cassette is written
   without the two new answer fields, so its content is the one of before. A Gemini ledger line
   has two new keys (`provider`, `served_by`).
4. **Documentation pages read on 2026-11-22** (public pages, no sign-in, no API call). Facts that
   the code relies on, each with its page:
   - https://openrouter.ai/docs/api-reference/overview — the endpoint
     `POST https://openrouter.ai/api/v1/chat/completions`; an image part
     `{"type": "image_url", "image_url": {"url": <data URL>, "detail": ...}}`; the fields of the
     answer (`provider`, `choices[].finish_reason`, `native_finish_reason`, `message.content`,
     `usage` with `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`, `is_byok`,
     `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`).
   - https://openrouter.ai/docs/api-reference/authentication — `Authorization: Bearer <key>`;
     a key is `sk-or-v1-` and 64 lowercase hexadecimal characters.
   - https://openrouter.ai/docs/guides/overview/multimodal/image-understanding — base64 data
     URLs; `image/png`, `image/jpeg`, `image/webp`, `image/gif`; limits per provider and model.
   - https://openrouter.ai/docs/features/multimodal/videos — video input (item 7).
   - https://openrouter.ai/docs/features/structured-outputs — `response_format` of the type
     `json_schema` with `name`, `strict` and `schema`; support per endpoint; `require_parameters`
     routes only to endpoints that support it; strict mode can restrict the keywords of a schema,
     per provider.
   - https://openrouter.ai/docs/features/provider-routing — the `provider` object:
     `require_parameters`, `data_collection` (`allow` or `deny`), `zdr`, `allow_fallbacks`;
     "If you don't want to allow any other public providers, you should disable fallbacks as well."
   - https://openrouter.ai/docs/features/zdr — "When `zdr` is set to `true`, the request will only
     be routed to ZDR endpoints"; a request setting is combined with the account settings and
     cannot switch them off.
   - https://openrouter.ai/docs/features/privacy-and-logging and
     https://openrouter.ai/docs/guides/features/input-output-logging — OpenRouter stores the
     content of a call only after the account switches input and output logging on; the
     training opt-out is an account setting.
   - https://openrouter.ai/docs/use-cases/usage-accounting — usage is always in the answer; the
     old switch `usage: {include: true}` has no effect.
   - https://openrouter.ai/docs/guides/best-practices/reasoning-tokens — `reasoning: {"effort":
     ...}` with `minimal` to `xhigh`; reasoning tokens are a part of `completion_tokens`.
   - https://openrouter.ai/docs/api-reference/parameters — a missing sampling parameter means the
     default of the provider.
   - https://openrouter.ai/docs/api-reference/errors — the error body `{"error": {"code",
     "message", "metadata"}}`; 400, 401, 402, 403, 408, 429, 502, and 503 for "no available model
     provider that meets your routing requirements"; an error during the generation comes with
     the status 200, in the body or in the choice; check for `error` in every answer.
   - https://openrouter.ai/docs/api-reference/limits — `Retry-After` may come with 429 and 503;
     free models have their own request limits.
   - https://openrouter.ai/docs/guides/routing/model-variants/free — the `:free` suffix of a slug.
   - https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation
     — the generation record, with `total_cost` "in USD".
5. **Choices made from these pages** (each one tested offline with answers written by hand in
   the shapes of the pages):
   - Routing settings of every request: `zdr: true`, `data_collection: deny`,
     `require_parameters: true`, `allow_fallbacks: false`. No page says how `allow_fallbacks:
     false` combines with the two filters, and no page says that a filter can be passed over;
     with fallbacks off, the request fails rather than going to a provider that the filters
     did not choose. The doctor line `data policy` checks every sent body.
   - Status codes 408, 409, 429 and 5xx are tried again (at most 4 attempts, the wait of
     `Retry-After` up to 300 s). The status 503 can mean that no provider meets the routing
     settings; its error line says so. A timeout is not tried again. Assumed, not documented: a
     call refused with an error status is not billed (as for Gemini).
   - A 200 answer without usage, without a cost above 0 or with `is_byok: true` raises
     `UnconfirmedCallError`: the ledger cannot price it. So does a 200 answer whose body cannot
     be read; it is not sent again. A redirect is not followed. A 200 answer with an error, a finish
     reason other than `stop` or no text raises `BilledAnswerError` with its reported cost.
   - The strict schema gets `additionalProperties: false` on every object (the strict mode of
     several providers needs it). Not known: whether every provider takes `minItems` and
     `maxItems` in strict mode.
   - `detail` takes `low` or `high`; an OpenRouter step refuses the resolutions `medium` and
     `ultra_high`. The pitch diagram goes at the resolution of the step.
   - The thinking level goes as `reasoning.effort`; the new level `model_default` sends none,
     for a model without reasoning (allowed for OpenRouter only).
   - The documentation calls `usage.cost` "credits" in one place and USD in another (the
     generation record). One credit is taken as one USD; the first doctor compares the ledger
     with the account page (RUNBOOK, section 10).
6. **Published prices** (model pages, read on 2026-11-22; not measured). The price table holds
   the highest price of any provider of a model, because the routing chooses the provider:
   `qwen/qwen3-vl-235b-a22b-instruct` 0.21 / 1.90 USD per million tokens (input / output; its
   providers 0.20 / 0.88, 0.21 / 1.90, 0.21 / 1.90), `qwen/qwen2.5-vl-72b-instruct` 0.80 / 1.00
   (one provider), `google/gemini-2.5-flash` 0.54 / 4.50 (endpoints 0.15 / 1.25, 0.30 / 2.50,
   0.54 / 4.50). The pages showed no per-image price, no `:free` entry and no mark of zero data
   retention; so it is not known whether any provider of these models meets the routing settings.
   Estimated, not measured: the first doctor of `qwen/qwen3-vl-235b-a22b-instruct` costs
   0.015 USD, the estimate that the doctor prints (6 calls at the token estimates of Gemini;
   image tokens of Qwen differ).
7. **Video input is not built.** The video page shows a `video_url` part with links (YouTube for
   Gemini on AI Studio; Vertex takes no video links) in its chat examples, and base64 video only in
   the examples of the Responses API. The footage cannot go to a public link, and no page shows a
   base64 video in chat completions. So a step that needs video would have to rest on an
   undocumented request. Build it when a page documents base64 video for chat completions.
8. **Coordinate conventions of the candidate models** (not verified). The Qwen3-VL repository
   says that it uses "relative position coordinates" for boxes and points; its scale was not
   found. The model card of Qwen2.5-VL names no convention and says that the images are resized
   to multiples of 28 pixels. So the experiment file `people_probe_openrouter.yaml` names
   `xy_1000` as a first guess, and the doctor decides (RUNBOOK, section 10).
9. **Proposed text for `CLAUDE.md`** (not applied; the owner decides):
   - First paragraph: "Soccer broadcast clip in; annotated video, 2D pitch radar and a measured
     score out. A vision language model does the perception: Gemini, or any image model through
     OpenRouter, chosen per step in the settings. Plain code does geometry, tracking, export,
     drawing."
   - Rule 4: "Only `src/pitchvlm/vlm/client.py` talks to a model service: it imports the model SDK
     of Gemini and sends the OpenRouter requests with the standard library. Everything else must
     run without the SDK and must not call a model service. Only `tools/run_tool.py` imports an external tool
     (RF-DETR, PnLCalib). It runs in the environment of that tool, and the package never imports
     it."
   - Rule 5: "Model calls: paid models only (no OpenRouter model with the suffix `:free`), no
     sampling parameters, and inputs are never stored or used for training: `store=False` for
     Gemini; for OpenRouter only endpoints with zero data retention and no data collection, with
     no fallback to another provider. A model whose providers cannot meet this is an error.
     Players are identified by shirt number plus roster, never by face."
   - The row "Model: Gemini family only" of the first table of this file: "Gemini, or any image
     model through OpenRouter (2026-11-22)". Rule 5 of `docs/RUNBOOK.md`, section 7, follows the
     text of `CLAUDE.md`.
10. **Measured on the build machine**: the offline suite (counts in the report of this session),
    the guard of the tracked files and the demo. Not verified: anything against the real
    OpenRouter API; the answers of the tests were written by hand from the pages of item 4.

## 2026-11-22 — Paint alignment of pitch mappings, and hard cuts

1. **Source of the idea** (owner decision of 2026-11-22, point 2: take the ideas of a classic
   detector stack). A public repository of such a stack claims that residual checks pass
   confidently wrong pitch mappings, and checks a mapping against the white lines of the frame
   instead. Its licence file reserves all rights, so only the idea is taken: no code, no
   configuration text and no threshold of it. Every value below comes from our own tests.
2. **Paint alignment, what it is** (`geometry.paint_alignment`, contract in
   `docs/ARCHITECTURE.md`). The pitch lines of `pitch.py` are projected through the mapping
   into the frame. A projected pixel hits when white paint (a thin bright ridge of the darkest
   colour channel, on grass) lies within 0.6% of the image width. The score is the hit share
   above chance, times a spread factor (the hits must not lie on one line or in one corner), from
   0 to 1. A frame with less than 0.2% paint, or with 50% or more of the image near paint, cannot
   judge a mapping: the score is None with a reason, not a low score. The score cannot see a
   mapping that swaps the halves or the sides of the pitch (the markings are symmetric);
   `check_homography` finds that. The constants were chosen on the synthetic clip only.
3. **Where it is reported.** Every landmark fit of `run` and `demo` (`ClipResult.paint`, before
   the plausibility check; with the pitch engine `calibration`, the camera of each keyframe:
   entry "A real-time detector engine next to the model engine", item 5), every valid mapping of the pitch probe and every PnLCalib mapping of
   `tool-score` (`paint_alignment` in the metrics file: counts, score distribution, separation,
   one number per frame). It is a diagnostic. No rule of the pipeline uses it (rule 7).
4. **Which result makes it a rule.** The run-machine measurement (`docs/RUNBOOK.md`, section 3,
   "Paint alignment of the pitch mappings"): the cached answers of the pitch probe, group
   `probe` (valid clips only, never test clips), replayed for free, and the PnLCalib run file
   scored again. The score becomes a rule of `geometry.check_homography` for an engine only when
   all of this holds for that engine, in a new dated entry of this file before G3 is frozen:
   - at least 20 good mappings (median error at most 2 m) and 20 bad ones (above 3 m);
   - AUROC at least 0.85, and AUROC minus 1.96 standard errors at least 0.75;
   - a score limit, chosen on these frames, that refuses at least half of the bad mappings and
     at most 5% of the good ones, checked once on 3 clips of the group `dev` before it is written;
   - a frame that cannot judge never refuses a mapping.
   With fewer than 20 bad mappings the result is inconclusive and the score stays a diagnostic.
5. **Measured, synthetic** (`PYTHONPATH=src python tools/synthetic_diagnostics.py`, build
   machine, synthetic clip of 250 frames, seed 0, every 5th frame: 50 frames, noise seed 7).
   Not broadcast frames: no number here says how the score works on real footage.
   - Median score of the true mapping shifted on the pitch in 8 directions (n 400 per row, 50
     for 0 m): 0 m 1.000; 0.5 m 0.920; 1 m 0.650; 2 m 0.508; 3 m 0.436; 5 m 0.238; 10 m 0.055.
   - Separation of good (at most 2 m) against bad (above 3 m) mappings: shifted mappings AUROC
     0.948, standard error 0.005 (n good 1,250, n bad 800); landmark fits from the true landmarks
     with Gaussian noise of 2, 4, 8 and 16 px AUROC 0.987, standard error 0.007 (n good 145, n
     bad 30). The standard errors count every mapping as independent; the mappings share 50
     frames, so the real uncertainty is larger. Noise 2, 4, 8, 16 px gave a median error of
     0.20, 0.60, 1.28, 3.10 m and a median score of 1.000, 0.887, 0.622, 0.323.
   - Wrong mappings (median over 50 frames): the picture mirrored left to right 0.106 (up to
     0.994 when the view is symmetric about the halfway line), top to bottom 0.000; the sign of
     one matrix entry changed: H[1,1] and H[1,2] 0.000, H[2,1] and H[2,2] 0.093, H[1,0] 0.273,
     H[0,0] 0.402, H[0,2] 0.403, H[0,1] 0.621 (a small entry). The pitch mirrored: 1.000, the
     same as the right mapping.
   - Time: about 12 ms per frame of 1280 x 720 on the build machine.
6. **Hard cuts** (`motion.is_cut`, `MotionField.cuts`). A pair of neighbouring frames is a cut
   when no camera motion (the identity, the measured motion, the last measured motion of the
   shot) moves the first frame onto the second with a median block correlation of at least 0.5
   (4 x 4 blocks of an image 160 px wide). The median of blocks keeps a static score graphic
   from joining two shots. Measured on synthetic clips (same tool): the lowest value inside a
   shot 0.881 (n 448 pairs, with a pan of up to 83 px per frame at 1280 px, with and without a
   score graphic); the highest value across a cut 0.055 (n 8 cuts of 1 to 10 s, with and
   without the graphic). Known misses: a jump in time that one camera motion explains, for
   example 0.4 s with a slow camera (0.918), or the same camera pose 6 s later. A pan faster
   than about 100 px per frame counts as a cut. The clips of SoccerNet GSR have no cut, so no
   gate measures the cut test; it guards the demo clip and other broadcast footage.
7. **Nothing crosses a cut.** Before this change, on the synthetic clip with a cut at frame 54
   (`tests/test_shot_cuts.py`, run on the code of `main` f7d788e): the referee track, which has
   no shirt number, was linked across the cut by its pitch position (`bridged_warp` and the
   joins); with a static score graphic the camera motion was measured across the cut and the
   pitch mapping of frame 54 was carried from the shot before, 47.8 m off at a foot point; and
   the frames 49 to 53 were filled across the cut. 5 of the 8 tests failed. Now the motion step
   into a cut is missing, `bridged_warp` gives None across it, `associate` and
   `stitch_fragments` run per shot (in `pipeline.py`; `tracking.py` is unchanged), and the dense
   output fills nothing across it. Only `finalize_tracks` may join two tracks across a cut, by
   one shown shirt number of one team: that is identity, not position. Cost: the frames between
   the last keyframe before a cut and the first keyframe after it get no state (less than one
   keyframe interval, 0.48 s, per cut).
8. **Proposal, not built: a shot type in the landmark answer.** One more field in the landmarks
   request (main camera, close-up, replay, other) would let the pipeline drop the landmarks of a
   frame that is not a main-camera view, and would name the shot of a cut. It changes a paid
   prompt and therefore the cache keys of every stored landmarks answer, and no label of
   SoccerNet GSR measures it. Recommendation: decide after G3, with the cut counts of the demo
   clip.
9. **Open owner decisions of this entry** (also in the last list of this file). (a) The rule of
   item 4, after the measurement. (b) The proposal of item 8. (c) Whether a shown shirt number may
   join tracks across a cut (it does now). (d) Whether the frames before a cut stay without a state
   (they do now) or get the last position. (e) `tools/synthetic_diagnostics.py` is a new file in
   `tools/`, which holds the scripts of the specialist tools; it is a measurement script of this
   entry.

## 2026-11-22 — Track repairs and tracking values without labels (session C)

1. **The owner decision of 2026-11-22, the part of this entry.** Take the ideas of a public
   detector repository (its licence file reserves all rights, so ideas only: no code, no config
   text, no weights from it; every threshold here comes from our own tests). For the tracks: repair
   identities after the tracking with plain code, read the 11-plus-11 limit of the people on the
   pitch, and add tracking values that need no labels. All of it works on the records that every
   engine gives (boxes, pitch positions, team keys per keyframe). What that repository claims, not
   measured by us: wide stitching gates weld two players; its swap fix is switched off; its sample
   output shows top speeds of 35.9 km/h, which points at position noise. The entries "Image models
   through OpenRouter", "Paint alignment of pitch mappings, and hard cuts" and "A real-time
   detector engine next to the model engine" of the same day record the other parts of the
   decision.
2. **Split of team swaps** (`tracking.split_team_swaps`, before the joins of `stitch_fragments`).
   Rule: a track of a player or a goalkeeper is cut where its team key changes and the change
   holds, and a person of the other team was in contact at the change. "Holds": runs of one key
   with at least 4 votes for their key on each side (`SWAP_MIN_VOTES`; shorter runs are wrong
   answers and join their neighbours, `team_key_runs`). "Contact": foot points at most one box
   height apart in the image at the change (`SWAP_CONTACT_BOX_HEIGHTS`), with the key of the run of
   the other track. The pieces are joined again by `stitch_fragments`, each with its own team.
   Both values are estimates. Limits: two teammates who swap show no change and stay swapped; a
   swap less than 4 votes before the end of a track is not seen; a model that names the wrong
   team for seconds next to an opponent gives a wrong cut whose pieces are not joined again.
3. **Team slots** (`tracking.link_team_slots`, at the end of `finalize_tracks`). Rule over time,
   not per frame (the camera shows a part of the pitch): a team side has at most 10 players and 1
   goalkeeper in a clip (`TEAM_SLOTS`). A group (side, role) with `n` tracks and `s` slots holds at
   least `n - s` later parts of its other tracks, so the `n - s` shortest links (pitch distance
   from the end of one track to the start of a later one) are made, among links that do not
   overlap in time, show no different numbers, have both positions and lie within the reach of
   `max_speed_mps`. No more links than the slots need. Not in `stitch_fragments`: the joins run
   before the numbers are read, and a slot link there could weld two players whose numbers
   would tell them apart. Limits: the distance is the only evidence (two players who come back
   where the other one left are linked the wrong way); a substitution or a red card adds people.
4. **Stitching gates.** `stitch_fragments` was reviewed against "a wide gate welds two players"
   with two new test cases. Two teammates who cross while both are hidden are not welded: the
   rival rule refuses both joins. A teammate who enters where a player left is joined when the
   player does not come back within 3 s (nothing in the tracks tells the two apart: a limit of
   the data), and not joined when the player comes back while the teammate is in view or when
   the numbers differ. No test showed a gate to tighten, so none was changed.
5. **The defaults do not change.** Both repairs are arms of gate G4, off by default
   (`tracking.split_team_swaps`, `tracking.team_slots`; settings files in `configs/arms`). The
   current behaviour stays the default until G4 numbers on the 6 dev clips exist for each arm
   (RUNBOOK, section 4, "The track repairs against gate G4"), with AssA of `eval` next to them.
6. **A value for wrong joins.** `identity_scores` gives `purity` too: the mean share of a predicted
   id that one true track covers. Coverage, switches and track ratio (the values of G4) do not
   fall when one id follows two people one after the other, so a repair that links tracks could
   pass G4 and still mix people. No gate uses `purity`.
7. **Tracking values without labels** (`evalx.diagnostics.track_diagnostics`): births and ends of
   tracks away from the image border, steps faster than `max_speed_mps` between observed
   positions and the top speed of each track, team-key changes inside tracks, tracks per team
   key alive and seen per keyframe. `run` and `demo` write them under `diagnostics` and print
   them. No gate uses them.
8. **Synthetic numbers, before and after** (measured on the build machine; `python -m
   pitchvlm.evalx.trackbench --seeds 20`; synthetic, no real footage and no real model answer).
   100 cases: 5 kinds of fault times 20 seeds, 30 s clips of the synthetic scene, exact answers
   without per-frame shirt numbers, a confident number reading for a random half of the people
   (about the share of G5b), true camera motion and pitch mappings. Faults: `none`; `swap` (the
   tracks of two players of different teams at most one box height apart exchange their later
   observations, 39 swaps in 20 clips); `team_noise` (10 % of the answers name the other team,
   10 % no team); `occlusion` (4 players missing for 2.4 to 4.8 s each); `mixed` (all three,
   39 swaps). Means over the 20 cases of each kind, then the mean change against the current
   behaviour on the same cases with its 95 % t interval:

   | fault | arm | coverage | switches | track ratio | purity | tracks |
   |---|---|---|---|---|---|---|
   | none | current | 0.788 | 13.55 | 1.589 | 1.000 | 36.5 |
   | none | split of swaps | 0.788 | 13.55 | 1.589 | 1.000 | 36.5 |
   | none | team slots | 0.922 | 3.90 | 1.085 | 1.000 | 24.9 |
   | swap | current | 0.762 | 19.35 | 1.589 | 0.975 | 36.5 |
   | swap | split of swaps | 0.788 | 13.35 | 1.580 | 1.000 | 36.4 |
   | swap | team slots | 0.895 | 10.35 | 1.089 | 0.969 | 25.1 |
   | swap | both | 0.926 | 3.50 | 1.076 | 1.000 | 24.8 |
   | team_noise | current | 0.782 | 13.65 | 1.574 | 0.999 | 36.2 |
   | team_noise | split of swaps | 0.782 | 13.65 | 1.574 | 0.999 | 36.2 |
   | team_noise | team slots | 0.928 | 3.55 | 1.070 | 0.996 | 24.6 |
   | occlusion | current | 0.764 | 14.45 | 1.626 | 1.000 | 37.4 |
   | occlusion | team slots | 0.896 | 4.95 | 1.104 | 1.000 | 25.4 |
   | mixed | current | 0.732 | 20.75 | 1.663 | 0.972 | 38.2 |
   | mixed | split of swaps | 0.744 | 19.30 | 1.685 | 0.989 | 38.8 |
   | mixed | team slots | 0.857 | 12.15 | 1.135 | 0.966 | 26.1 |
   | mixed | both | 0.871 | 10.40 | 1.152 | 0.984 | 26.5 |

   | fault | arm | switches vs current | coverage vs current | track ratio vs current |
   |---|---|---|---|---|
   | swap | split of swaps | -6.00 [-6.66, -5.34] | +0.026 [+0.019, +0.033] | -0.009 [-0.023, +0.005] |
   | team_noise | split of swaps | 0.00 [0.00, 0.00] | 0.000 | 0.000 |
   | mixed | split of swaps | -1.45 [-2.49, -0.42] | +0.012 [+0.005, +0.018] | +0.022 [+0.000, +0.043] |
   | none | team slots | -9.65 [-10.57, -8.74] | +0.134 [+0.124, +0.144] | -0.504 [-0.536, -0.472] |
   | mixed | both | -10.35 [-12.35, -8.35] | +0.139 [+0.125, +0.153] | -0.511 [-0.554, -0.468] |

   The split arm changes no value on the clips without swaps, also with 10 % wrong team keys
   (every one of the 60 cases of `none`, `team_noise` and `occlusion` has the values of the
   current behaviour). With the swaps alone it takes back
   all 6 switches that the 39 swaps cost (19.35 -> 13.35, the 13.55 of `none`) and gives purity
   1.000. With wrong team keys and occlusions together it takes back 1.45 switches, where the
   swaps alone cost about 6 (inferred from the `swap` rows): the noise hides the contact or
   shortens the runs, and a cut that is not joined again adds tracks (track ratio +0.022). The team slots link the fragments of people who left the image and came
   back: with half of the numbers read and no per-frame number, the current behaviour misses three
   of the four G4 limits on the clean synthetic clips (coverage 0.788, 13.55 switches, track
   ratio 1.589), the slots meet them (0.922, 3.90, 1.085), and purity stays at 1.000.
9. **What the bench cannot say.** The players of the synthetic scene drift a few metres around
   fixed places, so a link by distance is easy there; real players run across the pitch, and the
   purity of the slots on real clips is not known. The swaps are injected only between people at
   most one box height apart, the same assumption as the contact rule, so the bench measures the
   cut and the joins, not the contact rule. Two players of one team never come that close in the
   scene, so the swap of teammates is not in the bench. How often swaps, wrong team keys and
   returns happen in real clips is not known. Nothing ran on real footage or on real model
   answers: G4 on the dev clips decides.
10. **Hard cuts** (after the merge of the entry "Paint alignment of pitch mappings, and hard
    cuts"). Only a shown shirt number of one team joins two tracks across a cut, and both repairs
    keep that rule.
    - The split of team swaps runs in each shot on its own, between `associate` and
      `stitch_fragments` of that shot (`pipeline._tracks`). Why per shot: it cuts a track where
      the key of the track changes, the tracks of `associate` never cross a cut, and the joins of
      the same shot link the pieces again.
    - The team slots run over the clip, at the end of `finalize_tracks`, after the joins by shirt
      number: a track that a shown number joined across a cut is one person in both shots. But no
      slot link crosses a cut (the earlier track must end in the shot where the later one starts,
      `finalize_tracks(cuts=...)`), and the slots are counted per shot: a group with `n` tracks
      seen in a shot and `s` slots gets at most `n - s` links in that shot. Why: the people seen
      in two shots are two tracks each, so the count of the whole clip would ask for links that
      join different people inside a shot. A clip without a cut is one shot and keeps the count
      of the clip, so the code path of the synthetic numbers of item 8 is the same (inferred from
      the code; the bench was not run again).
    - The tracking values without labels: the first and the last observation of each shot bound
      the births and ends inside the image, and no step crosses a cut (a track joined across a
      cut by its number stands somewhere else after it).
    - Tests: `tests/test_tracking.py` (a slot link refused across a cut; the slots counted per
      shot: with the count of the clip 10 tracks of that case would cross the cut, now none
      does); `tests/test_shot_cuts.py` (the split runs once per shot; with both repairs only a
      shown shirt number joins tracks across the cut, and no track joins two people);
      `tests/test_diagnostics.py` (births, ends and steps at a cut).
11. **Open owner decisions** (also in the last list of this file). (a) Is `track_ratio_max` of G4 per clip or over the 6 clips? The
    RUNBOOK script reads it per clip. (b) Does a repair that passes G4 but lowers AssA of `eval`
    become the default? This entry proposes: no; a repair becomes the default when it passes G4
    and does not lower AssA on the same clips. (c) Should `purity` get a limit in G4 before the
    dev runs (rule 7: gates are fixed before a run)? (d) The split arm pays new number requests
    for the tracks that it changes (about 0.009 USD each, estimated); the team slots are free on
    stored answers.

## 2026-11-22 — A real-time detector engine next to the model engine

1. **The decision** (owner, 2026-11-22). (1) Any image or video LLM must work through
   OpenRouter, not only Gemini (the entry before this one). (2) Take the ideas of a classic
   detector stack that a public repository shows (`Mhmdwael77/ai-football-analytics`): ideas
   only. Its licence file reserves all rights (its README badge says MIT; the file wins), and
   its detector is an Ultralytics model (AGPL-3.0). No code, no config text and no weights of it
   are used; every threshold here is new. (3) A real-time detector is a first-class engine next
   to the model engine. This entry records point 3 and the two ideas of point 2 that belong to
   it: team and role from kit colours, and a separate ball tracker.
2. **Detectors: published values, not measured here.** Read on 2026-11-22 from the pages and
   licence files of each project by an agent of this session (one reader). AP is COCO val2017
   AP50:95. Latency is batch 1 on an NVIDIA T4.

   | Family (pip package) | Sizes: COCO AP | Latency, ms | Code licence | Weights licence | Source |
   |---|---|---|---|---|---|
   | RF-DETR (`rfdetr`, 1.12.1 of 2026-11-12) | N 48.4, S 53.0, M 54.7, L 56.5 | 2.3, 3.5, 4.4, 6.8 (TensorRT, FP16) | Apache-2.0 | Apache-2.0 | README and LICENSE of https://github.com/roboflow/rf-detr at tag 1.12.1 |
   | RF-DETR XL, 2XL (`rfdetr[plus]`) | 58.6, 60.1 | 11.5, 17.2 (TensorRT, FP16) | Apache-2.0 | Roboflow Platform Model License 1.0 | same |
   | Ultralytics YOLO26 (`ultralytics` 8.4.216) | n 40.9, s 48.6, m 53.1, l 55.0, x 57.5 | 1.7, 2.5, 4.7, 6.2, 11.8 (TensorRT 10; precision not stated) | AGPL-3.0 or paid licence | AGPL-3.0; the licence page says every trained model is AGPL-3.0 | Ultralytics documentation, pages of YOLO26 and of the licence |
   | Ultralytics YOLO11 | n 39.5, s 47.0, m 51.5, l 53.4, x 54.7 | 1.5, 2.5, 4.7, 6.2, 11.3 (as above) | AGPL-3.0 or paid licence | AGPL-3.0 | Ultralytics documentation, page of YOLO11 |
   | RT-DETRv2 (no package; in Hugging Face transformers) | S 48.1, M 51.9, L 53.4, X 54.3 | 4.6, 6.9, 9.3, 13.5 (FP16, TensorRT; given as frames per second, converted by the reader) | Apache-2.0 | Apache-2.0 | https://github.com/lyuwenyu/RT-DETR |
   | D-FINE (no package; in transformers) | COCO only: N 42.8, S 48.5, M 52.3, L 54.0, X 55.8; with Objects365: S 50.7 to X 59.3 | 2.12, 3.49, 5.62, 8.07, 12.89 (FP16, TensorRT 10.4) | Apache-2.0 | Apache-2.0; the README warns that the Objects365 weights may carry the terms of Objects365 | https://github.com/Peterande/D-FINE |
   | DEIM (no package) | N 43.0 to X 56.5 | 2.12 to 12.89 (conditions not stated) | Apache-2.0 | Apache-2.0 | https://github.com/ShihuaHuang95/DEIM |
   | DEIMv2 | S 50.9 to X 57.8 | 1.10 to 13.75 (conditions not stated) | non-commercial licence since 2026-08-24 | non-commercial | https://github.com/Intellindust-AI-Lab/DEIMv2 |
   | RT-DETRv4 | S 49.8 to X 57.0 | 3.66 to 12.90 (precision not stated) | Apache-2.0 | not stated | https://github.com/RT-DETRs/RT-DETRv4 |
   | LW-DETR | T 42.6 to X 58.3 | 2.0 to 19.1 (conditions not stated) | Apache-2.0 | Apache-2.0 (Objects365 pretraining) | https://github.com/Atten4Vis/LW-DETR |

   Soccer-trained weights: none was found with a clear licence for both the weights and their
   training data. The Roboflow `sports` repository trains Ultralytics YOLOv8 (AGPL-3.0). A
   RF-DETR Large model on Hugging Face (`julianzu9612/RFDETR-Soccernet`, tagged Apache-2.0,
   claims 49.8 mAP on SoccerNet tracking; third-party, unverified) is trained on SoccerNet data
   whose terms are research-only as far as read. The licence page of the Roboflow football
   dataset could not be read.
3. **Recommendation: RF-DETR, size by gate G2.** It is the only candidate with an Apache-2.0
   licence for code and weights, a pip package, a documented fine-tune, and an adapter and four
   experiment files in this repository already. Its published speed on a T4 with TensorRT is
   2.3 to 6.8 ms per frame; the script uses plain `predict` without TensorRT, so the speed on
   the run machine must be measured. The default of the detector engine is RF-DETR Medium
   (`configs/default.yaml`, section `detector`), the middle of the two sizes that the probe
   measures; one more experiment file measures it on the frames of the people probe
   (`people_probe_rfdetr_medium.yaml`). The owner sets the size that passes G2 (Small, Medium or
   Large; XL and 2XL are not Apache-2.0). No second family is added now. The candidate with a
   clear licence path is D-FINE with its COCO-only weights through the Apache-2.0 code of
   Hugging Face transformers; it is added only after the licence of each weight file is read.
   Ultralytics YOLO is not taken: its code and every model trained with it are AGPL-3.0.
4. **Licence, said plainly.** This repository is public and has no licence file. Code under the
   AGPL or the GPL may run only inside the environment of its tool, started by
   `tools/run_tool.py`, as PnLCalib (GPL-2.0-or-later) does now; the package never imports a
   tool (rule 4). What the AGPL or the GPL means for the script itself, which imports such code
   when it runs, is not settled: it is an open owner decision, together with the licence of the
   repository (last list of this file).
5. **What was built** (build machine, offline, stand-ins only):
   - Settings: `engines` names the engine of the people step (`model`, `detector`), of the pitch
     mapping (`model`, `calibration`) and of the shirt numbers (`model`, `none`); `detector` and
     `calibration` hold the tool settings that a clip run must have. Two settings files:
     `configs/engines/detector_calibration.yaml` (no model step, free) and
     `detector_calibration_numbers.yaml` (adds the paid number step).
   - Clip job and clip run (`fallbacks/runs.py`): a tool runs on every frame of one clip. The
     run is cut into parts of at most 1,000,000 bytes, named like the ledger, with compact rows
     for the detector. Measured with the part writer of the script on random boxes: 37.9 bytes
     per detection; 750 frames with 26 detections each give one part of 738,558 bytes, and the
     guard passed on that part. The format of the probe runs (about 140 bytes per detection)
     would give about 2.7 MB for the same clip. What may go back into the repository does not
     change: clip runs are tool runs, held back until the owner decides on tool run files.
   - `tools/run_tool.py` reads a clip job, writes the parts, refuses to start when later parts
     of an older run lie next to its output, and takes `GEMINI_API_KEY` and
     `OPENROUTER_API_KEY` out of its environment.
   - Command `clip-job`; `run --detections --calibration`; `demo --engine detector`. A run
     whose engines call no model opens no model session and costs nothing.
   - People engine `detector` (`pipeline.py`): people of every frame, kit-colour team keys
     (`kitcolour.py`: torso histogram, 2-cluster k-means in numpy, outliers beyond the median
     plus 3 scaled median absolute deviations), tracks on every frame with the time limits of
     the keyframe design, goalkeeper or referee for tracks of outlier kits by their mean pitch
     position (a goal zone and inside the width of the penalty area), then the joins of
     `tracking.py`, which is used unchanged. The team side comes from `assign_team_sides`.
   - Shirt numbers stay the optional paid number step on track crops. Its request names no
     team, because a kit cluster is not a team of the kit legend. Without it no player has a
     number, and `run` says so.
   - Ball (`ball.py`): the class `sports ball` of the detector, a tracker that takes the camera
     motion out and gates by the speed limit of a ball (40 m/s, an estimate) at the scale of
     its box (0.22 m), fills gaps up to 0.5 s and drops runs of fewer than 3 sightings. The
     record types take it cleanly: `FrameState.ball` (`BallState`, optional, default None). It
     goes to the video and the radar. The prediction file has no ball, and `docs/PLAN.md` keeps
     the ball out of the scored result.
   - Pitch engine `calibration`: the camera of every frame from PnLCalib, checked with
     `fit_problems` and the people of the frame, gaps filled through the camera motion.
   - Timing: `run` and `demo` write the measured time of each stage (milliseconds per frame,
     frames per second) and print it; the stages `detector` and `calibration` are the times
     that the script measured per frame. Measured on the build machine, synthetic clip of 100
     frames of 1280 x 720 pixels, detector demo with stand-ins (n = 1 run): motion 9.18 ms per
     frame, kit colour 3.40 ms, tracks 0.67 ms, all package stages 13.50 ms (74 frames per
     second). This is no claim of real time: it holds no detector, no calibration, no decoding
     of real frames and not the run machine.
   - Hard cuts and paint alignment (added when the entry "Paint alignment of pitch mappings,
     and hard cuts" came in): the detector engine tracks each shot on its own, as the model
     engine does, so no track crosses a cut; the ball tracker ends a run at a cut and fills
     nothing across it (`track_ball(..., cuts=)`); the calibration cannot carry a camera across
     a cut, because the motion step into it is missing. The camera of the calibration at each
     keyframe gets its paint-alignment score before the plausibility check, like a landmark
     fit, so both pitch engines are scored on the same frames (`ClipResult.paint`, stage
     `paint` of the timing). Measured on the synthetic clip with a cut at frame 54 (detector
     and calibration stand-ins, seed 3): without the tracking per shot 3 of 31 tracks crossed
     the cut; with it 0 of 33. The paint scores of the 10 exact stand-in cameras were 0.996 to
     1.000.
   - Track repairs (added when the entry "Track repairs and tracking values without labels"
     came in). The detector engine takes both arms of gate G4 with the same settings as the model
     engine, and neither is refused:
     - `tracking.split_team_swaps`: in each shot, after `kit_roles` and before the joins
       (`_detector_tracks`). It holds: the team key of an observation is the kit cluster of its
       torso, a per-frame reading like the team that the model names, and a track that took over
       a person of the other kit flips cluster and stays flipped. A goalkeeper or a referee has the
       key `none` after `kit_roles`, so the split never cuts it. Risk, not measured: when two
       players overlap, the torso window of one box can show the shirt of the other; a run needs 4
       votes on each side (`SWAP_MIN_VOTES`), 4 frames here (0.16 s at 25 fps) against 4
       keyframes (1.9 s) for the model engine, so a long overlap can give a wrong cut. G4 on the
       dev clips decides, per engine.
     - `tracking.team_slots`: in `finalize_tracks`, shared by both engines, with the cuts of the
       clip. It holds: the side comes from the kit clusters and `assign_team_sides`, and the
       goalkeeper from `kit_roles`; the number engine `none` gives no numbers, so the distance is
       the only evidence, the limit that the repairs entry names. Measured on the synthetic clip
       of 250 frames (seed 3, exact stand-in boxes): without the slots 27 tracks, 4 identity
       switches, coverage 0.972, track ratio 1.174; with them 23 tracks, 0 switches, 1.000,
       1.000, and 10 player tracks per side.
     - The tracking values without labels are written and printed for a detector run too. They
       count every observed frame where the model engine counts keyframes: a step is 0.04 s, not
       0.48 s, so box noise weighs 12 times as much in a speed, and "per keyframe" reads "per
       observed frame". The values of the two engines are therefore not comparable (open
       question in the list "Open questions").
   - The engine table of every result names each step: `model:<model>`, `detector:rfdetr:<class>`,
     `tool:pnlcalib`, `code`, `code:kit-colour`, `code:ball-tracker` or `none`. A model step is
     written as the command line reads it: the bare name for Gemini, `openrouter:<slug>` for
     OpenRouter. The engine and the provider live side by side: the engine of a step says
     whether a model runs it, and `provider`, `model` and `coords` of the step say which model.
     A run whose engines call no model needs no key of any provider and builds no model backend
     (tested).
6. **Measurement, fixed before any run** (RUNBOOK, section 11): gate G2 for the detector on the
   frames of the people probe (the five RF-DETR files), and end to end on the dev clips with the
   same scorer, each engine with its own tracker name, compared clip by clip together with the
   role and team-side accuracy of G5b. The plan for a detector trained on soccer is RUNBOOK
   section 12; nothing was downloaded or trained.
7. **Proposed new text of rule 8 of `CLAUDE.md`** (not applied; the owner decides). It fits the
   proposed rules 4 and 5 of the entry before this one:
   "8. Two engines are first-class for the people step and for the pitch mapping: a model
   (`model`: the provider and the model of the step, under rule 5) and a specialist tool
   (`detector`, `calibration`). The shirt numbers come from a model or from no step (`none`).
   The settings of a run name the engine of each step, and every result file and report names
   the engine of every step, with the provider and the model of each model step. A comparison
   of two engines, or of two models, counts only on the same frames, against the same gate of
   `configs/gates.yaml`, with the same scorer. A tool runs only through `tools/run_tool.py` in
   its own environment (rule 4); code under the AGPL or the GPL runs nowhere else."
   Until the owner applies it, rule 8 stands: the docstring of `fallbacks/__init__.py` and
   section 3 of the RUNBOOK still describe the tools as fallbacks.
8. **Not verified.** No real detector, no real calibration, no real footage and no timing on
   the run machine. The script was tested with stand-ins of the tools; the kit colours with flat
   synthetic kits; the ball with a stand-in ball that the synthetic picture does not show. The
   values of the table above were read by one agent and not checked by a second reader.

## 2026-11-22 — Second read of the model provider and money code

The second read of the code that the OpenRouter entry above brought in (`vlm/client.py`,
`vlm/cost.py`, `vlm/types.py`, `vlm/coords.py`, `vlm/steps.py`, the prompts, the doctor and its
command line, the price table, the guard and the secret scan, and the OpenRouter parts of the
docs). Four readers read separate parts (rules 4 and 5 with the secrets; the money; errors,
coordinates and cache keys; the docs). Each claim then went to a second reader whose job was to
refute it with a test or an exact trace. Every test ran offline, with fake HTTP layers.

Counts: 24 distinct claims. 12 confirmed and fixed: 1 major, 5 minor, 6 small. 11 refuted. 1
not sent to a second reader, because the docs already describe it (a `Retry-After` header with a
date is ignored, and the normal wait is used). Each code fix has one test that fails on the code
before the fix (run on a copy of the old code with the new tests: 9 of 9 fail for the reason
that the fix removes).

1. **Major. A success answer whose body breaks off was sent again.** `UrllibTransport` turned a
   failed read after the status 200 into a lost connection, which is tried again: up to 4
   billed calls for one request, none of them in the ledger. Now it raises
   `HttpUnreadAnswerError`, and the backend ends the request with `UnconfirmedCallError` at
   once, as for a timeout.
2. **Minor. A redirect carried the key to another host.** `urlopen` follows 301, 302 and 303 on
   a POST as a GET with every header, also `Authorization`. The transport now follows no
   redirect: a 3xx answer is an error status that is not tried again.
3. **Minor. A reported cost of 0 counted as a paid answer.** The ledger, the spend window and the
   stops then counted 0 USD for the call. A cost of 0 now raises `UnconfirmedCallError` (after
   the BYOK case, which keeps its own message), and the doctor line `reported cost` fails for it.
4. **Minor. The guard and the secret scan missed an OpenRouter key after a letter, a digit or an
   underscore** (`MY_KEY_sk-or-v1-...`), where they find a Google key. The `sk-or-v1-` prefix is
   long enough to need no word start: both rules dropped it.
5. **Minor. Doc: proposed rule 4** said that everything else runs "without a network", but
   `datasets.py` downloads the dataset. It now says "must not call a model service".
6. **Minor. Doc: RUNBOOK section 10** said that a 503 that ends the doctor means that no provider
   meets the privacy settings. A 503 is retried like every 5xx, can be a busy service, and a
   failed request fails one check while the doctor goes on. The text now says so.
7. **Small. An error status whose body read failed** escaped as a raw `IncompleteRead`, not
   retried and not logged. The body is now left out with a warning, and the status decides.
8. **Small. Error text from a 200 body** went into the error unchanged: no key removal, no cut
   to 500 characters. `_error_message` now cleans every body text.
9. **Small. `"error": null`** in a 200 body rejected a usable, billed answer. Only an error that
   is not null counts now.
10. **Small. Docs:** the first Qwen3-VL doctor costs 0.015 USD as estimated by the code (6
    calls, `DEFAULT_TOKEN_ESTIMATES` at 0.21 / 1.90 USD per million), not "about 0.02";
    `--thinking-level` applies to every model of a doctor run, not only to the OpenRouter
    models; the Gemini thinking levels are `minimal` to `xhigh`, by model.

Refuted (behaviour reproduced, but not a defect of this change):
- An answer with readable tokens but no usable cost gets no ledger line: the documented policy
  for a call whose cost is unknown, as for Gemini; the run stops at the first one.
- One spend window per provider: documented; the window protects the limit of each account,
  and the ledger stops count both providers together.
- The doctor goes on after a call whose cost is unknown: unchanged since f7d788e, and the line
  `reported cost` fails.
- `:FREE`, a trailing space or `:free:beta` pass the free-model check: a request is sent only
  for a slug with an exact entry in the tracked price table, which is paid only.
- An upper-case key followed by `,` or `)` escapes gitleaks: real keys are lower-case (auth
  page), and the guard's general `sk-` rule finds them.
- A model that answers in another coordinate convention is not detected in a run: by design,
  gate G0 proves the convention before a settings file names it.
- The routing settings and the image type are not in the cache key; the prompt names stay `v1`;
  "data policy checks every sent body"; "the doctor compares the ledger with the account page";
  reasoning tokens counted twice without `total_tokens` (logged, and the money uses the
  reported cost): intended, harmless, or only loose wording.

Open, not fixed (inferred, not measured):
- Unknown-cost answers with readable tokens could be priced at the table price as an upper
  bound, instead of having no ledger line. That needs a ledger field that says the cost is
  not reported; owner decision.
- `run` and `probe` do not check that a G0 pass exists for the model and the convention of a
  step, and nothing counts coordinates outside the bounds of the convention.
- Not checkable offline: whether OpenRouter ever redirects, reports a cost of 0, treats slugs
  case-insensitively, or bills a 524; `urllib` applies the timeout to each read, so a server
  that sends bytes slowly can hold a call longer than 300 s.

## 2026-11-22 — Second read of the tracking, cuts, paint alignment and detector engine

A second read of the code that pull requests 5, 6 and 7 brought into `main` (`652c82b`; base
`f7d788e`), with the hand-resolved merges. Five readers read separate parts (merge seams, the cut
rule, the two track repairs, kit colours and numbers, texts and rules). Each finding was checked
by a second reader with a test that fails on the old code or with an exact trace; one more
reader then tried to refute the fixes. Every number below is synthetic: build machine, synthetic
clips, stand-ins of the model, the detector and the calibration. None is a result on real
footage. "Before" is `652c82b`; "after" is the commit of this entry.

1. **Counts.** Major: 4 found, 4 confirmed, 3 fixed, 1 open (item 5). Minor: 8 found, 8
   confirmed, 7 fixed, 1 kept as a written limit (item 3, last point). Small: 13 found, 9
   confirmed and fixed, 4 refuted or taken back (item 6). The read of the fixes found 1 small
   regression of this review (fixed, item 4) and 1 change of this review that it argued
   against (taken back, item 6).
2. **Major, fixed.**
   - `tracking.link_team_slots` checked the numbers of each link alone, so a chain 1, (no
     number), 13 of one goalkeeper slot became one track and number 13 was lost. A link is now
     refused when the chain it makes shows two displayed numbers or two most frequent per-frame
     numbers. Test: `test_team_slots_do_not_chain_two_numbers_through_a_track_without_one`.
   - The detector engine counted two keyframe limits in frames: a team-key change of
     `split_team_swaps` needed 4 votes = 4 frames (0.16 s at 25 fps) instead of 1.9 s, and the
     label penalty of `associate` grew per missed frame instead of per missed keyframe (a
     detection with another kit key could not match after 4 missed frames). Found by three
     readers. `pipeline.keyframe_frames` (12 frames at 25 fps; 1 for the model engine) now
     scales both: `associate(keyframe_frames=)`, `split_team_swaps(min_votes=4 * 12)`, and the
     team-key changes of the tracking values (`track_diagnostics(team_min_votes=)`). The model
     engine is unchanged (1 and 4). Tests in `tests/test_tracking.py` (penalty per interval,
     a 4-frame flip next to an opponent is no change, a 120-frame swap is cut) and
     `tests/test_detector_engine.py` (the split gets 48 votes). Measured: see item 7.
   - `kitcolour.fit_kit_clusters` ran one k-means++ start. On the synthetic scene (250 frames,
     seeds 0 to 9) it split the referee against both teams in 4 of 10 seeds (1, 5, 6, 8): the
     referee got the key `A` and 41 % to 42 % of the player boxes got `none`. Now 10 starts
     (`KMEANS_STARTS`), the fit with the smallest sum of squared distances is kept: 0 of 10
     seeds, players in their own cluster 0.933 to 0.998 (before, in the 6 good seeds, 0.960 to
     0.994). Test: `test_the_kit_clusters_are_the_two_teams_and_the_referee_is_an_outlier`
     (50 frames, seed 0, fails before). The test of two kits with 40 noisy copies each was
     changed: with the new fit one blue copy of 40 lies beyond the outlier limit (0.119 against
     0.114), which the rule (median plus 3 scaled deviations of a skewed distance) allows; the
     old fit passed because an outlier inside the cluster widened the limit. The test now
     asks for distinct team keys, no copy with the key of the other team, and at most one
     outlier per team.
3. **Minor, fixed.**
   - A shown number joined two tracks across a cut only when their positions on the two sides
     were within the reach of a runner (`_one_person_per_number` compared positions across the
     cut), so after a replay or another camera the number was taken from all but the longest
     track. Across a cut the number alone decides now; inside a shot the test of the way stays.
   - Without kit clusters (one torso feature, or none) every track of the detector engine
     became a referee or a goalkeeper (`kit_roles` read every `none` as an outlier kit). Now
     `kit_roles` runs only when a person of the clip has the key `A` or `B`.
   - Noisy crops of one kit were split into two "teams" (the check asked only for two
     different features). `KitClusters.two_kits` (each centre beyond the outlier limit of the
     other) now refuses such clusters with a warning.
   - The comment of `SWAP_MIN_VOTES` said a false change has a chance of 0.6 % per track of 60
     observations. The weak runs join their neighbours, so 4 votes need not come in a row.
     Monte Carlo (10 % wrong keys, 10 % no key, one by one; 2,000 tracks per length, seed 0):
     2.15 % at 60 observations, 9.30 % at 250, 26.6 % at 750; with 48 votes 0 of 2,000 at 250
     and at 750. The comment now says so.
   - RUNBOOK section 12: step 1 ran `data --split train`, which the command refuses; step 3
     imported `rfdetr` in a `python -c` line, against rule 4. Both are now written as "to
     build", the training as a job of `tools/run_tool.py`.
   - "A pan above about 100 px per frame counts as a cut" (ARCHITECTURE; the paint-and-cuts
     entry, item 6) was not measured. Measured now: at 100 px per frame no pair is below 0.5
     (lowest 0.873), at 125 px 7 of 39 pairs are, at 150 px 15 of 39. ARCHITECTURE says so.
   - Kept as a written limit, not fixed: `link_team_slots` counts every track of a group as a
     person, so a second box of one person, a wrong-team piece or a referee with a team key
     can force one link between two people. Two of these causes are fixed above; the limit is
     in the docstring.
4. **Small, fixed.** `run` printed no ball count (`demo` did). The radar drew trails across a
   cut (`RadarRenderer(cuts=)`, `render_clip` passes `result.cuts`). OpenCV hue 2 and hue 178
   (one red) fell in the first and the last hue bin, distance sqrt(2); the bins now start half
   a bin before hue 0. `shot_correlation` raised `ZeroDivisionError` on an image whose 160 px
   copy has fewer rows than the grid (empty blocks). `load_clip_run` looked for an older part
   only at `parts + 1`; it now looks at every 4-digit part above the run, with the name escaped
   for `glob` (the escape also in `tools/run_tool.later_parts`; a stem with `[` was the
   regression that the read of the fixes found). `track_diagnostics` gave an infinite speed for
   two positions of one track in one frame; it raises `ValueError` now. Texts: ARCHITECTURE
   on the keyframe check of `run` and `demo` (model engine only), on `BallState` defaults, on
   the chain rule of the slots, and the contracts of every changed function.
5. **Major, open: a dissolve is not a cut.** `is_cut` compares neighbouring frames; in a
   dissolve each frame differs from the next by a small blend step. Measured
   (`tools/synthetic_diagnostics.py`, seed 0, two shots of the synthetic match 100 frames apart):
   a hard cut 0.033, a dissolve of 5 frames lowest pair 0.900, of 10 frames 0.929; so the motion
   chain, the tracks and the fill run through it, and a mapping carried across a 10-frame
   dissolve was 39 m off at the frames after it (one reader, scratch script). No fix is small:
   a test over a window of frames needs its own measurement, and a fast pan must not become a
   cut. ARCHITECTURE names the limit. Owner decision: build a window test before the demo
   clip, or accept the limit (the clips of SoccerNet GSR have no cut).
   Update: a window test was built, see the entry "A dissolve is a transition, not a shot" of
   2026-11-22.
6. **Refuted or taken back.** (a) Flat or tiny frames are always a cut: by design and written.
   (b) The change point of `team_key_runs` after merges (for example A10 B3 A3 B10): the contact
   is tested from the last vote of the merged run to the first vote of the next, the right
   window. (c) The engine table names `model` for the number step when no number request is
   sent (no mapping): older than these pull requests, and `run` prints that no player has a
   number; the ball step names the tracker, and the detector of its candidates is named in
   `people`. (d) Taken back: this review first refused slot links between different export
   numbers; an export number can come from one low-confidence crop, so one misread crop would
   block a link. The display number stays the evidence. (e) `min_observations` of the detector
   engine stays 2, of frames (0.08 s): a noise filter whose value G2 and G4 decide; written in
   `every_frame_tracking`.
7. **Measured, before and after** (build machine; commands from the top folder with
   `PYTHONPATH=src`):
   - `python -m pitchvlm.evalx.trackbench --seeds 20` (seeds 0 to 19): before, every number of
     the table of the track-repairs entry, item 8, again to the last digit. This also checks the
     claim of that entry, item 10, that the cut changes leave the clips without a cut alone.
     After: the same table to the last digit (the chain rule of the slots changes no case of the
     bench). A first run with the export-number check of item 6 (d) changed one row (mixed,
     both: track ratio 1.154, purity 0.985, 26.6 tracks).
   - `python tools/synthetic_diagnostics.py`: before, every number of the paint-and-cuts entry,
     items 5 and 6, again (median scores per shift, AUROC 0.948 and 0.987, the mirrored
     mappings, 448 pairs lowest 0.881, 8 cuts highest 0.055, the jump 0.918). The script now
     also prints the time of one score (17.5 ms median, n 2,450, on the build machine while
     other sessions ran; the earlier "about 12 ms" was not measured by a script), the limits of
     items 3 and 5, and the detector-engine numbers of the detector entry, item 5: 250 frames,
     seed 3, without slots 27 tracks, 4 switches, coverage 0.972, track ratio 1.174; with slots
     23, 0, 1.000, 1.000; a cut at frame 54: 33 tracks, 0 across the cut, paint of 10 keyframe
     cameras 0.996 to 1.000. The same numbers before and after.
   - Detector engine, split of team swaps (scratch scripts of the review, kept in the notes of
     the session, not in the repository; stand-in detector, true camera, 10 s clips, seeds 0 to
     9, a team-A player passes in front of a team-B player, depth gap 0.5 m):

     | case | far player wrongly cut, before | after | switches without split; with split, before; after |
     |---|---|---|---|
     | pass 0.5 m/s | 10 of 10 | 0 of 10 | 38; 184; 38 |
     | pass 1 m/s | 10 of 10 | 0 of 10 | 38; 80; 38 |
     | pass 2 m/s | 10 of 10 | 0 of 10 | 38; 76; 38 |
     | pass 4 m/s | 0 of 10 | 0 of 10 | 38; 38; 38 |
     | mark 1 s | 10 of 10 | 0 of 10 | 38; 156; 38 |
     | mark 2 s | 10 of 10 | 10 of 10 | 38; 260; 260 |
     | depth gap 1 m: pass 0.5, 1 m/s; mark 1 s | 2, 1, 4 of 10 | 0, 0, 0 of 10 | 38; 62, 44, 72; 38 |

     The before column was measured with a best-of-20 k-means fit in the script, so that the
     cluster failure of item 2 did not hide the split; the after column with the code of this
     entry. A mark of 2 s shows the shirt of the other kit for longer than the 1.9 s of the
     keyframe design, so it is still cut: the limit that the track-repairs entry, item 2, names
     for the model engine.
   - Detector engine, injected swaps (the swap of `trackbench` applied to the output of
     `associate`, 15 s clips, seeds 0 to 9): switches over 10 clips (19 swaps
     after, 18 before), before then after: clean without repair 90 and 90, clean with the split
     92 and 90; with the swaps, no repair 126 and 128, split 106 and 118, slots 41 and 46, both
     18 and 36. The split now takes back 10 of the 38 switches that the swaps cost (20 of 36
     before): a swap must hold for 48 frames on each side, and on these clips the tracks of the
     detector engine are often shorter after a swap. This is the cost of the fix. Whether the
     detector engine should use a shorter time than the keyframe design is for G4 on the dev
     clips.
   - Natural scene (30 s, seeds 0 to 9): tracks of a team with a run of 4 or more wrong kit
     votes: 1 of 532 (5 frames, seed 4); wrong cuts of the split 2 before and 0 after; switches
     263 without the split, with it 265 before and 263 after.
8. **Not verified.** Nothing ran on real footage, a real detector or a real model. The new
   `two_kits` check has no measurement on real crops: on simulated two-kit features its margin
   was 0.79 to 0.99 against limits of 0.53 to 0.66 (4 cases, one reader); similar real kits
   could fail it, and the clip then has no team keys. The 10 k-means starts were measured on
   flat synthetic kits only. The `keyframe_frames` scaling keeps the time of the keyframe
   design; whether 1.9 s is the right time for a kit-colour swap is for G4. The swap of two
   teammates and the dissolve stay unseen.

## 2026-11-22 — A faster default run of the offline tests

1. **Goal and result.** The owner called the wait for the offline tests unusable (3,090 tests,
   181 s on `main` 652c82b, measured by the orchestrator). The goal was a default run of about
   30 s after each change, with every test still in one full run. **The goal is missed.** The
   default run went from 258 s to 150 s (mean of two runs each, below), 42 % shorter. Most of
   what is left is the tests of the guard and of the command line, which start git, the guard,
   the secret scanner and the hook runner as processes. They check hard rules and stay in the
   default run. All numbers here are measured on the build machine while other sessions ran
   (load average 3 to 17 on 14 cores); the absolute seconds depend on that load.

2. **Before and after.** Wall time of `python -m pytest`, base and new checkout run one after the
   other with the same machine load (the series of item 9).

   | Run | Tests | Run 1, s | Run 2, s | Load average at the start (1 min) |
   |---|---|---|---|---|
   | Before: `main` 652c82b, every test | 3,090 (1 skipped) | 255.6 | 261.0 | 3.0, 6.4 |
   | After: default run, `python -m pytest` | 3,081 (1 skipped) | 150.0 | 150.3 | 7.4, 6.2 |
   | After: slow tier, `python -m pytest -m slow` | 9 | 18.2 | 20.4 | 7.0, 10.5 |
   | After: full run, `python -m pytest -m "slow or not slow"` | 3,090 (1 skipped) | 164.3 | 177.0 | 6.3, 8.9 |
   | After: full run, test files in reversed order | 3,090 (1 skipped) | 169.9 | - | 16.6 |
   | After: full run without the caches (`PITCHVLM_TEST_CACHE=0`) | 3,090 (1 skipped) | 230.6 | - | (with the fault check beside it) |

   Two earlier runs of the base, before any change, under a load of 5 to 10: 264.7 s and 306.3 s.
   The skipped test is `tests/test_run_tool.py:528` (Pillow is not installed), in every run.

   Seconds per test file (sum of the test times, mean of the two runs of each row above):

   | File | Tests before | Before, s | Tests in the default run | After, s |
   |---|---|---|---|---|
   | `test_guard.py` | 143 | 82.7 | 143 | 64.3 |
   | `test_cli.py` | 303 | 62.0 | 303 | 35.6 |
   | `test_detector_engine.py` | 33 | 42.3 | 32 | 12.3 |
   | `test_pipeline.py` | 90 | 26.4 | 89 | 10.4 |
   | `test_shot_cuts.py` | 12 | 8.1 | 12 | 3.5 |
   | `test_tool_probe.py` | 126 | 7.2 | 126 | 3.0 |
   | `test_trackbench.py` | 6 | 4.3 | 0 | 0 |
   | `test_paint_alignment.py` | 36 | 4.7 | 35 | 2.4 |
   | `test_probes.py`, `test_run_tool.py` | 192 | 6.2 | 192 | 4.4 |
   | the other 31 files | 2,149 | 8.6 | 2,149 | 8.2 |

   Collection takes 2.6 to 4.4 s (imports of cv2 and scipy; `import pitchvlm.cli` alone takes
   1.2 s, half of it `scipy.stats`, imported by `evalx/metrics.py`).

3. **Causes, before the change.** One run of the base under `cProfile` (311 s profiled; the
   profiler adds time, so the seconds are larger than in a plain run), and a count of every
   child process with its wall time:
   - Processes started by tests: 120 s for 1,958 processes. The guard as a process 52 s (most
     of it in `test_guard.py`; each run starts Python, imports the guard and starts its own git
     calls), git 49 s, the hook runner 11 s (8 runs), other Python scripts 6 s, the command line
     as a process 1.4 s (1 run). Starting a process costs 12 ms (`/usr/bin/true`), git 26 to 30
     ms and `python -c pass` 48 ms on this machine.
   - Pipeline runs (`run_clip`, 118 runs, inside many tests): 120 s. In it the measured camera
     motion 57 s (64 measurements), the frames of the synthetic scenes 39 s (6,627 frames drawn,
     7 ms each, often the same frames again), the paint alignment 21 s (1,030 scores), reading
     frame images from files 15 s (1,694 reads), JPEG encoding 6 s (6,180 crops).
   - Real waits: 6.4 s (93 sleeps of the tests of signals and of answers in flight).
   - Repeated parsing of settings: 1.5 s (463 YAML loads); not changed.

4. **Changes.** Only the tests and the test settings changed; nothing under `src/` or `tools/`.
   - `tests/conftest.py` (new): caches of six pure results, each computed once per test process:
     the frames of a synthetic scene (key: the settings of the scene), frame images read from
     files (key: the SHA-256 of the file), the measured camera motion (key: the source, the
     people boxes and the scale), the paint alignment (key: the SHA-256 of the frame and the
     matrix), the JPEG bytes of a request image (key: the SHA-256 of the pixels and the quality)
     and the git state of this checkout (key: the git variables of the environment). A cached
     frame is read-only and every caller gets a copy; a cached motion is built again as a new
     `MotionField` and writes its log lines again. The caches are off in the modules that test
     these functions (`test_synth`, `test_motion`, `test_frames`, `test_geometry`,
     `test_vlm_steps`) and while a test has replaced a function of `motion`, `geometry` or `cv2`.
     `PITCHVLM_TEST_CACHE=0` turns them off; the full run passes that way too (table above).
   - `tests/test_guard.py`: the guard of the repository runs inside the test process, with the
     environment of the test, its output streams in the encoding of a guard process and its
     exit code as the process would give it. It still runs as a process in the hook tests
     (`HookSettingsTest`, through the hook runner), in `test_paths_mode_accepts_relative_paths`,
     for the copies of the guard with another host list, and in `test_vlm_cost.py`. Each test
     copies the git folder of one `git init` with the local identity (made once per process)
     instead of three git calls, and a commit reads its name from `.git/HEAD` and the branch
     file instead of one more `git rev-parse` (git is asked when the files do not hold it).
   - `pyproject.toml`: the marker `slow`, left out by `addopts = "-q -m 'not slow'"`.
   - Saved per cause (difference of the per-file table, mean of two runs): the caches about 75 s
     (`test_cli.py` 26 s, `test_detector_engine.py` 25 s, `test_pipeline.py` 13 s, the rest 11 s);
     the guard inside the test process and the copied git folder 18 s; the slow tier 14 s.

5. **The slow tier (9 tests).** Each needs a long synthetic clip for its claim, and none checks a
   hard rule. Times of the two slow runs:

   | Test | s | Why it stays slow |
   |---|---|---|
   | `test_detector_engine.py::test_team_slots_link_the_fragments_of_the_detector_engine` | 5.3, 6.7 | two pipeline runs on 250 frames with measured motion; the claim (4 people break into two tracks) needs that length |
   | `test_pipeline.py::test_clip_of_thirty_seconds_keeps_one_track_per_person` | 2.9, 3.0 | 750 frames: people must leave the picture and come back |
   | `test_paint_alignment.py::test_the_score_separates_good_from_bad_mappings_on_the_synthetic_clip` | 1.8, 2.0 | 20 mappings on each of its frames; a smaller sample changes the AUROC claim |
   | `test_trackbench.py` (6 tests, `pytestmark`) | 3.7, 3.9 together | each bench case runs every arm on 375 frames; the repairs are also checked in `test_tracking.py` and `test_detector_engine.py` |

   The cut tests (`test_shot_cuts.py`, 3.5 s) stay in the default run: the cut rule is checked in
   every place that links or fills over time.

6. **What stays slow in the default run.** `test_guard.py` (64 s): about 1,900 git processes,
   at 25 to 35 ms each under load; the guard itself starts most of them (repository root,
   identity, index, objects, remotes), and the hook tests start the hook runner, which starts
   the guard and the secret scanner. `test_cli.py` (36 s): 21 s are about 480 child processes
   that the commands under test start (the doctor checks of the hooks, the stand-in scorer
   scripts, the secret scanner, the guard of `sync`, git for the results work tree, two runs of
   the command line as a process that test its output buffering). These check rules 6, 9 and the
   secret scan, and they stay.

7. **Proof that no check was lost.**
   - Test ids: the full run collects the 3,090 ids of `main` 652c82b, the same list (no test
     split or renamed). The default run collects 3,081, the slow tier 9.
   - Order: every changed test file alone passes (`test_guard.py`, `test_detector_engine.py`,
     `test_pipeline.py`, `test_paint_alignment.py`, `test_trackbench.py`); the full run in
     reversed file order passes; the default run passes two times in a row (item 2).
   - Fault check: 36 hand-made faults, one at a time, undone after each: 34 in 24 modules under
     `src/` and 2 in `tools/guard.py`. Eight of them are in the cached functions (frames, motion,
     paint alignment, JPEG, git state). A control run without a fault passes. **The default run
     catches 33 of 36** (31 of 34 in `src/`, 2 of 2 in the guard), every cached-function fault
     but one included. The slow tier catches none of the other 3; the full run of `main`
     652c82b does not catch them either (the old suite has the same gaps):
     - F01, `synth.py`: the people are drawn far over near. No test looks at the order.
     - F31, `fallbacks/fake.py`: the stand-in ball moves at half speed. The tests take the true
       ball from the same function, so the fault changes the truth and the answer alike.
     - F33, `pipeline.py`: a guarded `import google.genai` at the top of the module. Rule 4 has
       one test (`test_vlm_client.py::test_sdk_is_imported_only_inside_a_function`); it checks
       `client.py` only, so an SDK import in another module is not found. A test over every
       module of the package would close this gap (a follow-up for the review of rule 4).

   | Fault | Module | Change | Caught by the default run | First failing file |
   |---|---|---|---|---|
   | F01 | `synth.py` | far people drawn over near people | **no** | - |
   | F02 | `synth.py` | no grass texture | yes | `test_cuts.py` |
   | F03 | `motion.py` | measured motion never finds a cut | yes | `test_cuts.py` |
   | F04 | `motion.py` | step converted back to full size the wrong way | yes | `test_detector_engine.py` |
   | F05 | `motion.py` | a pair without a correlation is no cut | yes | `test_cuts.py` |
   | F06 | `geometry.py` | score without the spread | yes | `test_paint_alignment.py` |
   | F07 | `vlm/steps.py` | quality ignored | yes | `test_vlm_steps.py` |
   | F08 | `cli.py` | a changed work tree is clean | yes | `test_cli.py` |
   | F09 | `vlm/client.py` | Gemini call stored by the provider | yes | `test_cli.py` |
   | F10 | `vlm/client.py` | OpenRouter routing allows data collection | yes | `test_vlm_openrouter.py` |
   | F11 | `vlm/cost.py` | hard stop 1 USD late | yes | `test_cli.py` |
   | F12 | `vlm/cost.py` | run stops at 2x its limit | yes | `test_cli.py` |
   | F13 | `vlm/types.py` | prompt left out | yes | `test_cli.py` |
   | F14 | `vlm/types.py` | provider left out | yes | `test_vlm_openrouter.py` |
   | F15 | `vlm/cache.py` | stored answer of other pixels is used | yes | `test_cli.py` |
   | F16 | `pipeline.py` | repeated boxes kept as two people | yes | `test_pipeline.py` |
   | F17 | `pipeline.py` | crops half a second apart | yes | `test_pipeline.py` |
   | F18 | `frames.py` | last frame not a keyframe | yes | `test_cli.py` |
   | F19 | `fill.py` | states filled across a cut | yes | `test_fill.py` |
   | F20 | `vlm/coords.py` | model point read as [x, y] | yes | `test_cli.py` |
   | F21 | `evalx/metrics.py` | identity switches counted wrong | yes | `test_metrics.py` |
   | F22 | `pitch.py` | width taken as length | yes | `test_geometry.py` |
   | F23 | `export_gsr.py` | goalkeeper gets a jersey | yes | `test_cli.py` |
   | F24 | `export_gsr.py` | jersey of a goalkeeper not refused | yes | `test_export_gsr.py` |
   | F25 | `roster.py` | left team reads the right roster | yes | `test_pipeline.py` |
   | F26 | `timing.py` | ms per frame off by 10 | yes | `test_detector_engine.py` |
   | F27 | `ball.py` | gate does not grow with the frames | yes | `test_ball.py` |
   | F28 | `kitcolour.py` | outlier limit without the margin | yes | `test_kitcolour.py` |
   | F29 | `datasets.py` | archive member with '..' accepted | yes | `test_datasets.py` |
   | F30 | `config.py` | other keyframe interval | yes | `test_cli.py` |
   | F31 | `fallbacks/fake.py` | ball moves at half speed | **no** | - |
   | F32 | `vlm/client.py` | privacy check of the routing passes anything | yes | `test_cli.py` |
   | F33 | `pipeline.py` | the pipeline imports the model SDK | **no** | - |
   | F34 | `schemas.py` | union counts the overlap two times | yes | `test_cli.py` |
   | F35 | `tools/guard.py` | Google key pattern one character too long | yes | `test_cli.py` |
   | F36 | `tools/guard.py` | home folders in names not found | yes | `test_guard.py` |

8. **Proposed text for `CLAUDE.md`** (section "Commands"; the owner applies it):

   ```bash
   python -m pytest                          # offline tests, default run: all but the slow tests
   python -m pytest -m "slow or not slow"    # full run, before a commit: every test
   python -m pitchvlm.cli demo --out outputs/demo   # synthetic clip, fake model, no network
   python3 tools/guard.py --tracked      # leak check of all tracked files
   python3 tools/guard.py --push         # leak check of what a push can send: commits, tags, new names
   ```

   and in "How to work": "Small, tested changes. The default run of `pytest` passes after each
   change, and the full run before a commit, offline and with no SDK installed."

9. **Open: an owner decision for about 30 s.** With the guard tests in the default run the
   default cannot reach 30 s on this machine: `test_guard.py` alone takes 64 s, because of the
   git processes. Two ways, neither applied: (a) a third marker `git` for the tests that start
   git, the guard, the hook runner or the secret scanner as processes, run with the full run
   before each commit. The default run would then take roughly 60 s, estimated from the
   per-file table (150 s less 64 s of `test_guard.py` and about 25 s of the process tests of
   `test_cli.py`); the hooks still run the guard and the scan on every commit and push. It
   takes checks of hard rules out of the default run, which this task did not allow. (b) Run in
   several processes with `pytest-xdist` (`-n auto`) on a machine that has it. It is not a
   dependency and not installed here; it was not tried.

10. **Not verified.** No run on a quiet machine: every time here is under the load of other
    sessions (base 256 to 306 s against 181 s on the orchestrator's earlier run). The caches
    were tested on this machine and Python 3.13 only; a run with `pytest-xdist` was never tried.

11. **Measured again after the merge of `main` c1ae26a** (the reviews of the provider and of the
    tracking, merged into 344a851; 3,117 tests, the same ids as `main`; the 27 new tests are in
    the default run). Items 2 to 10 above are the runs on 344a851. New runs, one after the other,
    load average 2.5 to 4.3:

    | Run | Tests | Wall, s |
    |---|---|---|
    | Default run, two times | 3,108 (1 skipped) | 155.9, 160.1 |
    | Full run with the caches | 3,117 (1 skipped) | 171.1 |
    | Full run without the caches (`PITCHVLM_TEST_CACHE=0`) | 3,117 (1 skipped) | 243.1 |
    | Slow tier | 9 | 18.3 |

    `test_guard.py` took 69.8 and 71.8 s, `test_cli.py` 36.2 and 36.5 s of these default runs.
    The merge changed no cached function and no call of one (`compute_motion_field`,
    `paint_alignment`, `encode_jpeg`, `code_state`, the scene frames and the image reads); the
    change in `motion.py` is inside the cut test and changes no input of the cache key. The
    change in `tools/guard.py` is one pattern; the guard inside the test process runs its `main`
    as before. One change of `tests/conftest.py`: the caches of the motion, the paint alignment
    and the JPEG bytes now also stop while a test has replaced a setting (not only a function)
    of `motion`, `geometry` or `cv2`; no test does that today. The faults F01 to F08 of the
    cached functions, planted again one at a time: the default run catches 7 of 8, as before
    (F01 again caught by no run).

## 2026-11-22 — A dissolve is a transition, not a shot

Closes the open major of the entry "Second read of the tracking, cuts, paint alignment and
detector engine", item 5. Every number here is synthetic: build machine, synthetic clips of
`synth.py`, no model, no footage. None is a result on broadcast frames. Base `main` 81bb2d9.

1. **Design.** The pair test of neighbouring frames (`motion.is_cut`) stays as it was; its cuts
   and motion steps are identical to `main` on 40 synthetic clips (controls, dissolves and hard
   cuts of seeds 0, 1, 2 and 9, checked against the `motion.py` of 81bb2d9). A second test runs
   over windows (`motion.window_correlations`): frame `t - 30` is moved onto frame `t` with the
   chained motion of the 30 steps between them (no identity), and the median block correlation
   of the 160-pixel test images is taken. A pan that the chain follows explains its window; a
   dissolve does not, because no frame in it shows the change of view from one shot to the
   other. The chain takes the measured motion of a pair when it passes the pair test alone,
   else the motion of the pair test that explains the pair best (identity or last motion): in
   fast pans the optical flow was wrong by 9 to 250 px on some pairs that the last motion
   explained. A window with a missing step, a hard cut, or too little cover is not judged.
2. **Where the transition lies.** Each run of windows below 0.18 holds one transition
   (`motion.find_transitions`). The first frame of the first window shows the old shot alone,
   the last frame of the last window the new shot alone, for a dissolve shorter than the
   window. The motion inside a dissolve follows the shot that dominates the blend: the old one
   in the first half, the new one in the second. So the transition starts at the first frame
   that the old anchor, moved with the chain, no longer explains with a regression slope of
   0.95 (a blend frame with weight `w` has a slope near `1 - w`), and ends after the last frame
   that the new anchor does not explain. A change without a blend frame gives a cut at one
   frame.
3. **Representation: every frame of a transition is a cut, and so is the frame after it.**
   Each blend frame is then a shot of one frame. This is the smallest form that every consumer
   of `MotionField.cuts` already handles: `bridged_warp`, `associate`, `stitch_fragments` and
   `split_team_swaps` (per shot in `pipeline`), `link_team_slots` and `finalize_tracks`, the
   dense fill, the ball tracker, the propagation of landmark fits and calibration cameras, the
   tracking values and the radar trails all stop at a cut, so nothing crosses the transition
   and none of them changed. `MotionField`, its JSON and `ClipResult` keep their form. One
   change was needed: a blend frame must get no state at all, and a cut alone does not stop a
   frame's own answers. `MotionField.transition_frames` names the shots of one frame next to a
   cut, and `pipeline.run_clip` drops the people (before the kit colours and the tracks), the
   pitch mapping (landmark fit or camera) and the ball candidates of these frames, for both
   engines. A lone frame between two hard cuts is treated the same way: one frame cannot be
   tracked. A model keyframe that falls on a blend frame is still paid for (the motion is
   measured after the people requests), and its answer is dropped. A separate list of spans
   was the other option: it needed a change in every consumer and in the JSON of the field.
4. **Measurement.** Hypothesis: the window test finds dissolves of 5 to 25 frames and raises no
   cut on clips without one. Baseline: the pair test of `main`. Command:
   `PYTHONPATH=src python tools/synthetic_diagnostics.py dissolve-fit dissolve-held-out`.
   Clips: two shots of the synthetic match of one seed (shot B 75 to 249 scene frames after
   shot A, each panning up to 25 px per frame), 40 frames each, blended over the dissolve
   (frame `40 + i` has the weight `(i + 1) / (length + 1)` of shot B); controls of 60 frames:
   pans at about 25, 50, 75 and 100 px per frame at 1280 px, a zoom by 1.5 within 8 to 20
   frames, a shake (sd 0.15 degrees in pan and tilt, 0.1 in roll), a goal kick (fast pan,
   tilt up 5 degrees and back over 25 frames, zoom out by 12 %), and 100 frames without a cut.
   The zoom, the shake and the goal kick come from a virtual camera that turns inside a
   synthetic picture drawn at 2560 x 1440. The thresholds were fixed on seeds 0 to 9; seeds 10
   to 19 were run once after that, and no threshold changed afterwards.
5. **Held-out seeds 10 to 19** (one clip per seed and case, n = 10 each). "False cuts" counts
   blocks of cut frames that do not touch the true change. "Lowest window value" is the lowest
   judged window of each clip, as a range over the 10 clips; the found transitions need a value
   below 0.18. Span error: found minus true first and last blend frame, in frames.

   | Case | Fastest motion, px per frame | Hits | Misses | False cuts | Pair test alone: hits, false cuts | Lowest window value | Span error start / end |
   |---|---|---|---|---|---|---|---|
   | Dissolve, 5 frames | 8 to 25 | 10 | 0 | 0 | 0, 0 | -0.042 to 0.018 | -1 to 0 / 0 to +1 |
   | Dissolve, 10 frames | 8 to 25 | 10 | 0 | 0 | 0, 0 | -0.037 to 0.017 | 0 to +1 / -1 to 0 |
   | Dissolve, 15 frames | 8 to 25 | 10 | 0 | 0 | 0, 0 | -0.037 to 0.012 | 0 to +2 / -2 to -1 |
   | Dissolve, 20 frames | 8 to 25 | 10 | 0 | 0 | 0, 0 | -0.041 to 0.034 | +1 to +2 / -4 to -1 |
   | Dissolve, 25 frames | 8 to 25 | 10 | 0 | 0 | 0, 0 | -0.062 to 0.075 | +2 to +3 / -3 to -2 |
   | Hard cut | 8 to 25 | 8 | 2 | 0 | 8, 0 | 0.807 to 0.869 | 0 / 0 |
   | Pan, scene step 3 | 24 to 25 | - | - | 0 | -, 0 | 0.811 to 0.858 | - |
   | Pan, scene step 6 | 47 to 50 | - | - | 0 | -, 0 | 0.828 to 0.876 | - |
   | Pan, scene step 9 | 71 to 75 | - | - | 0 | -, 0 | 0.747 to 0.855 | - |
   | Pan, scene step 12 | 96 to 100 | - | - | 0 | -, 0 | 0.201 to 0.860 | - |
   | Zoom x1.5 | 9 to 25 | - | - | 0 | -, 0 | 0.750 to 0.889 | - |
   | Shake | 27 to 49 | - | - | 0 | -, 0 | 0.801 to 0.870 | - |
   | Goal kick | 31 to 67 | - | - | 0 | -, 0 | 0.776 to 0.828 | - |
   | No cut, 100 frames | 6 to 8 | - | - | 0 | -, 0 | 0.839 to 0.871 | - |

   Dissolves: 50 of 50 found (the pair test alone: 0 of 50). Controls: 0 false cuts in 80
   clips, as the pair test alone. The 2 missed hard cuts are missed by the pair test of `main`
   too (lowest pair value of these clips up to 0.913): shot B is the same synthetic camera some
   seconds later in a pose the optical flow reaches, the known limit "a jump in time that one
   camera motion explains". Not one window was judged across a hard cut, so the window test
   cannot change a hard cut. The span errors leave the faint ends of a dissolve (weight up to
   about 0.1) in their shots. The closest control is a pan at 100 px per frame with a lowest
   window value of 0.201, 0.021 above the threshold: the margin at the top of the pan range is
   thin.
6. **Fit seeds 0 to 9**, last run with the final code (same layout, n = 10 per case):
   dissolves 49 of 50 found (5, 10, 15 and 20 frames: 10 of 10; 25 frames: 9 of 10), highest
   lowest window value of a found dissolve 0.087; the miss (seed 6, 25 frames) has 0.611: its
   shot B is the same camera in a pose the chain of shot A happens to align. Hard cuts 10 of
   10. Controls 0 false cuts from the window test; the pans at 96 to 99 px per frame have 3
   false cuts of the pair test (as `main`, lowest pair value 0.141), and the lowest window value
   of a control is 0.268 (pan at 99 px per frame). Span error start 0 to +4, end -3 to +4.
   Shot shares (slopes) between the anchors: frames of the anchor's own shot 1.009 to 1.147
   (n 1954, 5 not measurable), frames with a blend weight in (0.2, 0.3] 0.747 to 0.898, in
   (0.1, 0.2] 0.835 to 1.001.
7. **How the thresholds were fixed (seeds 0 to 9 only).** The window length, 30 frames, is
   the design: a dissolve of up to 29 frames lies inside one window. 0.18 is halfway between
   0.087 and 0.268 (item 6); 0.95 is halfway between 0.898 and 1.009. The first design had
   other values and was changed on the fit seeds, never after the held-out run: (a) with the
   measured motion as the chain and the pair threshold 0.5 as window threshold, the window
   test added false cuts in fast pans (1 of 10 at 75 px per frame, 6 of 10 at 100 px against 3
   of the pair test), from wrong flow steps; (b) a refinement of the chained motion by image
   alignment (`cv2.findTransformECC`) removed those, but it also aligned shot A with shot B of
   same-camera dissolves (2 of 50 missed); the corrections it made overlap (2 to 14 test pixels
   in pans, 3 to 18 in dissolves), so no bound separates them, and it was dropped; (c) the
   chain rule of item 1 and the threshold 0.18 then separated the fit seeds; (d) 0.9 was the
   first share threshold, 0.95 the midpoint. The goal-kick control first panned at up to 132 px
   per frame, beyond the 100 px of the hypothesis (the pair test alone had 8 false cuts in 10);
   its pan was reduced to the stated range before the held-out run.
8. **Cost.** Time of the cut test per frame on the build machine (seeds 0 clips, n = 135 frames,
   `time.perf_counter`): the pair test of `main` median 1.12 ms; the new pair test plus the
   window test median 1.42 ms, so about 0.3 ms more per frame. The localisation runs only on a
   window run below 0.18 and was not timed. The motion measurement itself (optical flow) is
   unchanged. Memory: the 160 x 90 test image of every frame is kept until the field is built
   (57.6 KB per frame).
9. **Out of scope, measured once on seeds 10 to 19 (n = 10 each).** A fade through black over
   10 frames: the darkest frames have too little contrast, so the pair test makes them hard
   cuts (10 of 10; a block of 1 or 2 cut frames in the middle of the fade, frames 45 to 46 of
   40 to 49); the dimmed frames stay in their shots. A wipe over 10 frames (shot B covers the picture from left to right): found by the
   window test as a transition in 10 of 10 (span error start 0 to +4, end -3 to 0), though not
   designed for it. A picture-in-picture box of shot B (a ninth of the picture) for 25 frames
   over shot A: the box is not found in 10 of 10 (the clip then cuts to shot B, found in 9 of 10
   as a hard cut, the miss as in item 5). A dissolve during a pan that the chain cannot follow
   for a whole window is not judged (the windows of the 50 px pans were judged 171 of 300
   times), and not measured as a case. A dissolve between two moments of one camera can be
   explained by the chain (item 6, 1 of 100 dissolves over both seed sets).
10. **Open.** Nothing ran on broadcast frames: real dissolves (often into and out of a replay,
    with a logo wipe), real chain drift and real people motion over 1.2 s may move both
    thresholds; the margin at 100 px per frame (item 5) is thin. The 30-frame window was
    measured at 25 frames per second only. The faint ends of a dissolve keep their states.

## 2026-11-22 — A video file into the detector engine, and the check images of gate G3

Two tools for the first real clip runs, from the last list below ("Video files for the detector
engine", "Overlay images of the kill switch of G3"). Everything here was measured on the build
machine with synthetic clips and stand-ins of the tools (OpenCV 5.0.0, Python 3.13); nothing ran
on real footage.

1. **Design of the video path: one command writes the frames as images.** `video-frames` writes
   every frame of a video file as PNG into `~/pitchvlm-data/videos/<clip id>/img1` with the record
   `video.json` (SHA-256 of the video, frame count, rate, size). `clip-job` reads that folder as it
   reads a clip of the dataset, and puts the SHA-256 into the job (`video_sha256`); the script
   copies it into the head of the clip run; `run` reads a clip run only with the video of that
   check sum (null: only with a folder of images). The other way, a clip job with a video as its
   source, was not taken: the script would decode the video in each tool environment with the
   OpenCV and FFmpeg build of that environment, and two builds need not give the same pixels; the
   script reads images with Pillow, which reads no video; the two tools would decode the video two
   times. With images the clip job, the script and both tools stay as they were; the cost is disk.
2. **Same frames, proved on synthetic videos.** A video of 20 random frames (more than the 8
   frames that the video source keeps) and the synthetic clip of 50 frames (1280 x 720): every
   image read back by `ImageFolderSource` is equal, pixel for pixel, to the frame that
   `VideoFileSource` decodes, in the same order and count (`tests/test_frames.py`,
   `tests/test_video_overlay.py`). The script reads the images with Pillow, which is not installed
   here: that read of PNG was not compared (open item a).
3. **A video whose frames are not sure is refused, with one line** (`frames.check_frame_times`,
   in `VideoFileSource`, so `run` refuses it too). Measured: an mp4 of 10 frames whose second half
   was given double frame times (variable rate) decodes in OpenCV as 8 frames at 16.7 frames per
   second, without any error: two frames were dropped in silence. The source now refuses a header
   without a frame count, a frame more than half a frame away from its time at the constant rate
   (checked first, so this video gets the line about the rate), and a header count that differs
   from the decoded frames (a Motion JPEG file cut by 15 % decodes 8 or 9 of its 10 frames).
   Not found: the same file cut by 400 bytes decodes all 10 frames without an error (FFmpeg
   prints "overread" on the error stream); the last frame is then damaged (open item b).
4. **Rule 2.** `video-frames` and `overlay` refuse an image folder inside the repository or any
   work tree of it (`git worktree list`, so the results work tree too), compared as places on the
   disk. The defaults are `~/pitchvlm-data/videos` and `~/pitchvlm-data/overlays`. The list file
   of `overlay` is text: clip ids, frame numbers, the engine, how each mapping was made and its
   paint score or reason; the guard passes on it (measured with `--paths`).
5. **Check images: one command, three sources.** `overlay probe` replays the stored landmark
   answers of the pitch probe (no model call; a missing answer stops it) and fits them as the
   probe does (`probes.pitch_probe_fits`, the same code path): its count of frames with a mapping
   equals `n_frames_valid` of the probe (test). `overlay tool` reads the PnLCalib run file of
   `tool-score` on the frames of its probe. `overlay run` draws the mapping that a run used on
   each keyframe (`result.json`): the landmark fit or tool camera of that frame, or a mapping
   carried with the camera motion; the caption says which. The image is the frame under a caption
   band (clip, frame, engine, mapping, paint score or the reason why the frame cannot judge) with
   the pixels of `geometry.projected_line_pixels` in 70 % magenta, 1 pixel per 1280 of width, so
   the white paint stays visible beside a line. A first version with opaque lines of 3 pixels
   covered the paint of a right mapping completely, and the eye could not judge it.
6. **The choice cannot be steered.** The frames with a mapping are ordered by paint score
   (frames that cannot judge first), cut into `count` equal parts, and from each part the frame of
   the clip with the fewest picks so far is taken; ties by a SHA-256 of the seed, the clip and the
   frame. The RUNBOOK fixes the seed (20261113, the seed of the probe experiment files) and the
   counts (5 for the kill switch, 10 for M3). The owner writes `right` or `wrong` into `by_eye` of
   each entry of the list file and the counts into this file with the decision on G3 (RUNBOOK,
   section 3, "Check images of the pitch mappings").
7. **Measured on the synthetic demo clip** (100 frames, 1280 x 720, mp4v): `video-frames` 2.7 s
   and 63.5 MB (0.64 MB per frame); clip jobs of both tools named the SHA-256; the stand-in clip
   runs carried it; `run --clip <video>` read them (exit 0). `overlay run` on that result: 10
   keyframes, all with a mapping, 5 drawn in 1.2 s (3.3 MB of images); the true camera scores
   1.000 on every keyframe. The size per frame of real 1920 x 1080 footage is not measured
   (estimate 1 to 2 MB, about 1 GB per clip of 30 s).
8. **Formats.** `video_sha256` is a new key of the clip job and of the head of a clip run, null for
   images; a file without the key reads as null, so the format names stay `-1` (no file of the old
   form exists outside tests). The script refuses a value that is not null or 64 small hexadecimal
   digits.
9. **Open.**
   (a) The script reads the PNG frames with Pillow: compare one image of a real clip read by
   Pillow (converted to BGR) with `run`'s frame on the run machine before the first tool run.
   (b) A damaged last frame that still decodes is not found; look at the end of each video.
   (c) `overlay run` draws used mappings only: a refused landmark fit that no frame uses is not
   drawn (its score is in `result.json`). For a check of the raw fits of a run, a replay of the
   landmark answers of the run would be needed; not built.
   (d) The colour and width of the lines were judged on synthetic frames only.
   (e) No real video, real tool or real footage ran through any of this.
10. **After the merge of the dissolve test** (`main` 9b63a02, entry "A dissolve is a
    transition, not a shot"). A run drops the mapping of every transition frame. `overlay` now
    lists every frame without a mapping with its reason (printed, and `frames_without_mapping` in
    the list file) and draws no image of it; a keyframe of a run on a transition frame gets the
    reason "a frame of a transition", found from the cuts of the result with
    `MotionField.transition_frames`, the rule of the pipeline. Measured on a synthetic mp4 of 90
    frames with a dissolve of 10 frames (`tests/test_video_overlay.py`): `video-frames` gives the
    frames that `run` decodes, pixel for pixel; the run with stand-in clip runs of that video put
    at least one keyframe on a transition frame (asserted), and the overlay of every keyframe
    named it and drew it not. The cut test runs on the decoded (lossy) frames, as `run` reads
    them; the tools never see the cuts.

## 2026-11-22 — Fewer git processes in the guard, and three test gaps closed

1. **Goal and result.** The guard starts most of the git processes of `tests/test_guard.py`
   (item 6 of the entry "A faster default run of the offline tests"). Goal: the same
   behaviour with fewer processes, and tests for the gaps F01, F31 and F33 of that entry.
   **Result:** the guard starts 16 % fewer git
   processes in that file (1,625 to 1,362), and the file runs 8 % faster with the same tests
   (45.2 s to 41.5 s). The new guard tests cost about 3.6 s, so the whole file is not
   faster (item 5). The default run is about 2 s faster with 17 more tests. No output, message
   or exit code changed in 480 generated repository states (item 6). Every time here is
   measured on the build machine while other sessions ran; the load average goes with each.
   The raw output of every measurement is kept outside the repository.

2. **Where the processes came from (before).** A stand-in `git` first on `PATH` counted the
   processes of one run of `tests/test_guard.py` on `main` 81bb2d9: 2,300, of them 1,625 started
   by the guard and 675 by the tests (add, commit, push, init, the hook runner). The guard's by
   command: `var` 200, `rev-parse --show-toplevel` 187, `rev-parse --git-common-dir` 185,
   `cat-file --batch-check` 176, `cat-file --batch` 176, `config --local` 160, `check-ignore`
   150, `diff --cached` 103, `rev-parse --verify` 64, `remote`, `for-each-ref` and `rev-list`
   57 each, `diff-tree` 41, `ls-files` 12. A trivial git process takes about 12 ms here, a real
   one 25 to 35 ms. How Python starts it (fork or `posix_spawn`) makes no difference (measured,
   12 to 13 ms either way, also from a process with NumPy and OpenCV loaded).

3. **Changes in `tools/guard.py`.**
   - One `git rev-parse --show-toplevel --git-common-dir` gives the top folder and the git
     folder (was two processes). When that process fails, or a path holds a line break (git
     prints one path per line, so the two cannot be told apart), the guard runs the two old
     commands one at a time, so every error message is the old one, word for word.
   - One `git cat-file --batch-check` process stays open for the whole run and answers every
     question of the type and size of an object (was one process per question): the sizes of
     the commits and tags of a push, the sizes of the files, and whether `HEAD` names a commit
     (was `git rev-parse --verify --quiet HEAD^{commit}`; both give the same answer for 30 forms
     of a revision, tried one by one: unborn, detached, branch, tag, tag at a blob, tree, blob,
     missing object, short and full object names, a leading dash, spaces, `@`, `HEAD~1`,
     `HEAD@{1}`, `:/text` and more). When that process cannot answer the `HEAD` question (git
     dies on a damaged commit), the guard asks `git rev-parse --verify` as before, so the error
     names that command. Its exit code is checked at the end of the run. A result lives for one
     run only: the reader is made in `find_violations` and closed at its end.
   - The commit that the hook runner names (`PRE_COMMIT_TO_REF`) is still tested with
     `git rev-parse --verify`: it comes from the environment and can hold a line break, which
     the line protocol of `cat-file` cannot carry.
   - Tried and dropped: an open `git cat-file --batch` process for the contents. It **hangs**
     on a damaged object whose header gives more bytes than the object has: git prints the
     short content and waits for the next name. The contents are still read by one
     `cat-file --batch` per call, with its input closed. `--batch-check` prints one line for
     each name or ends: 10 kinds of damage tried (short and long content, bad header, no zlib,
     cut zlib, empty file, size overflow, bad type, unreadable file, a folder in place of the
     file); none hangs.
   - Not changed, and why: the two `git var` calls (author and committer of the next commit):
     `git var -l` gives both in one process but does not refuse an empty name, so an error of
     today (exit 2) would become a report (exit 1). `git remote` and `git config --local` read
     other sources than one call could replace exactly.

4. **Git processes of one guard run, before and after** (counted by the stand-in `git`).

   | Mode | Repository | Before | After |
   |---|---|---|---|
   | `--paths` (2 or 3 files) | this one, a small one | 3 | 2 |
   | `--tracked` | this one, a small one | 4 | 3 |
   | `--staged`, nothing staged | this one | 6 | 5 |
   | `--staged`, 3 files staged | a small one | 9 | 8 |
   | `--push`, nothing new | this one | 7 | 6 |
   | `--push`, a new commit with 3 files and an annotated tag | a small one | 13 | 10 |
   | `--push`, the same, from the hook runner (`PRE_COMMIT_TO_REF` set) | a small one | 14 | 11 |

   `test_each_mode_starts_a_known_number_of_git_processes` fixes the "After" column.

5. **Times.** Back to back, old against new, both orders; the load average (1 min) before each.
   The old tree is a clone of 81bb2d9; the new tree is the same with the change of this entry.

   | What | Old, s | New, s | Load |
   |---|---|---|---|
   | `tests/test_guard.py` of 81bb2d9 (143 tests), only `tools/guard.py` swapped, 3 runs per arm | 45.1, 45.4, 45.2 | 41.5, 41.7, 41.3 | 3.5 to 4.1 |
   | `tests/test_guard.py` of each tree (old 143 tests, new 148), 2 runs per arm | 43.3, 46.3 | 45.0, 45.2 | 4.8 to 5.5 |
   | Default run of each tree (old 3,107 tests, new 3,124), 2 runs per arm | 111.1, 111.3 | 109.3, 108.7 | 4.6 to 5.6 |

   The new tree of the second and third rows had 5 of the 7 new guard tests; the other two
   (a damaged commit of a detached `HEAD`, a `cat-file` that ends before it answers) came
   after these runs and take less than 1 s together. The
   machine was faster than in that entry (70 s for `test_guard.py` then), so the seconds
   of the two entries cannot be compared; only the back-to-back runs here count. The guard runs
   in the test process in most tests, so a test pays for its git processes and not for Python.

   One guard run as a process, median of 7 runs, two series per arm (old, new, new, old; load
   3.6 to 3.9):

   | Mode | Repository | Old, ms | New, ms |
   |---|---|---|---|
   | `--staged`, nothing staged | this one | 137, 137 | 129, 128 |
   | `--push`, nothing new | this one | 156, 157 | 144, 140 |
   | `--tracked` | this one | 490, 487 | 476, 477 |
   | `--paths` (3 files) | this one | 129, 136 | 117, 121 |
   | `--staged` | a small one | 173, 181 | 161, 169 |
   | `--push` | a small one | 225, 231 | 191, 193 |
   | `--tracked` | a small one | 105, 105 | 95, 96 |
   | `--paths` (2 files) | a small one | 96, 90 | 82, 80 |

   The hooks, through the hook runner, in a clone of this repository with one staged file and
   one new commit, median of 8 runs per arm in alternating order (load 4.0 to 4.4); the output
   of both arms is the same: commit hook 451 ms to 435 ms, push hook 503 ms to 463 ms.

6. **Differential check: no difference.** A script builds seeded repository states, runs the
   guard of 81bb2d9 and the new guard on each, in every mode, and compares the output, the
   error output and the exit code byte for byte. Final run on the final guard: seeds 2000 to
   2479, **480 states, 1,964 runs, 0 differences**. Exit codes of the runs: 0 in 366, 1 in
   1,161, 2 in 437 (errors of git or of use). The states mix, at random: planted media and
   dataset files (by name and by content), keys of six services, home paths, private and
   unknown hosts, base64 runs, deny-list words in files, messages, branch and tag names, binary
   and non-UTF-8 content, path names that are not UTF-8 (in the index), spaces, tabs, line
   breaks, a leading dash and non-ASCII letters in file names, links, submodule entries,
   renames, deletions, empty commits, annotated, plain and blob tags, new branches on old
   commits, a bare remote with some commits pushed, the variables of the hook runner (also a
   pushed commit with a line break or a leading dash), a missing or partial local identity,
   other authors; and per state: 100 merge commits, 126 detached heads, 67 with
   `core.hooksPath`, 29 shallow clones, 22 work trees, 10 runs from a subfolder, 50 damaged or
   deleted objects (32 of them the commit of `HEAD`), and runs in a folder that is no
   repository and in a bare repository (240 runs end with the same `rev-parse` error). An
   earlier run on the first version of the change (seeds 2000 to 2239 of an older generator,
   240 states, 987 runs) also had 0 differences. A difference found by hand after it: a
   detached `HEAD` on a damaged commit that no branch names gave the same exit code and the
   same text of git, but named `cat-file --batch-check` instead of `rev-parse --verify`. The
   change now asks `rev-parse` in that case (item 3), and a test fixes the old message.

7. **Fault check: the default run catches 28 of 30.** Thirty faults planted in the new guard
   one at a time, each undone after its run: 16 in the new code and 14 in the old rules.

   | Fault | Change | Caught |
   |---|---|---|
   | G01 | two folders: more than two lines taken | yes |
   | G02 | two folders: git folder of a subfolder joined to the top folder | yes |
   | G03 | two folders, slow way: `.git` under the top folder, not what git says | yes |
   | G04 | two folders: the slow way never taken | yes |
   | G05 | `HEAD`: any answer of three fields is a commit | yes |
   | G06 | `HEAD`: never a commit | yes |
   | G07 | sizes: the object name of the answer not compared | **no** |
   | G08 | sizes: one byte too small | yes |
   | G09 | reader: exit code of git not checked | yes |
   | G10 | reader: exit code not checked at the end of the run | yes |
   | G11 | reader: an answer without a line break taken | **no** |
   | G12 | contents: one git process per object | yes |
   | G13 | push: the commit of the hook runner not read | yes |
   | G14 | push: `HEAD` never a start | yes |
   | G15 | `HEAD`: a failed question is no commit, `rev-parse` not asked | yes |
   | G16 | `HEAD`: the stopped process kept after a failed question | yes |
   | R01 to R14 | lock files pass; size limit off by one; deny text not masked; e-mail of an identity not compared; remote branch name not read; a deleted file read as a change; the ignore file of the user read; a tag at a tag passes; base64 limit 201; every remote counts; no byte order mark removed; tagger not read; secret not redacted; merge commit against its first parent only | yes, 14 of 14 |

   G03, G10 and G16 were missed by the first round of this check and are caught by three of the
   new tests (a work tree in a folder with a line break; a `cat-file --batch-check` that ends
   with an error after its answers, and one that ends before). G07 and G11 need a git that
   breaks its own line protocol; `tests/test_guard.py` and the whole default run (run with each
   of the two planted) pass. A control run without a fault passes.

8. **The three gaps of the entry "A faster default run of the offline tests", closed with
   tests.** Each new test is in the default run. No gap showed a defect of the code: nothing
   under `src/` changed.
   - **F33, rule 4 over the whole package.**
     `tests/test_vlm_client.py::test_only_the_client_imports_the_sdk_and_only_the_tool_runner_imports_a_tool`
     reads the source of every module of `src/pitchvlm/` and of every other script of `tools/`
     (it imports none of them and needs no SDK and no tool). It fails for an import of the
     model SDK (`google`) outside `vlm/client.py`, and for an import of an external tool
     (`rfdetr`, and the packages of PnLCalib, read from `tools/run_tool.py`) anywhere: at the
     top, inside a function or a class, inside `try`, and through `importlib.import_module` or
     `__import__` with a text name. An import whose name is computed fails too, because no
     check can read it; the package has none. A second test runs the check on 8 small sources
     (each form above, and a clean one). Shown on the real tree: the guarded import of F33
     (`try: import google.genai`) at the top of `pipeline.py`, no SDK installed: the old test of
     rule 4 passes, the new one fails. An unguarded `from google import genai` there, with a
     stand-in `google.genai` on the path as on the run machine: the old test passes, the new
     one fails (without the stand-in, the import already stops the test run). `import rfdetr`
     inside `_draw_person` of `synth.py`: the new test fails.
   - **F01, draw order of `synth.py`.**
     `tests/test_synth.py::test_near_person_covers_a_far_person_where_their_boxes_overlap` takes
     pairs of whole boxes of different kit colours that overlap and that no third box reaches,
     in the clips of seeds 1 and 14 (seed 7, the clip of the other tests, has no overlap). Inside
     the overlap, one pixel in from its edges, no pixel may have the kit colour of the far person
     (the one whose feet are higher in the image), and some must have the near person's. The
     expectation comes from the boxes and the kit colours, not from the drawing code. 86 pairs
     are checked: in 22 the near person comes first in the fixed order of the people, in 64
     later (seed 1 alone has none of the first kind, so a drawing in the fixed order would pass
     there). Faults planted one at a time: far over near (two forms), the fixed order of the
     people, the order of x: the new test fails for each; the 52 other tests of `test_synth.py`
     pass for each. It takes 0.3 s.
   - **F31, the stand-in ball.**
     `tests/test_detector_engine.py::test_stand_in_ball_rolls_from_its_start_at_its_speed` maps
     the bottom centre of each ball box of `fake_detector_answers` to the pitch with the true
     mapping of the scene and compares it with the numbers of the design written in the test
     (start (-10, 5) m, speed (6, -2) m/s, 25 frames per second), to 1e-6 m; it also checks the
     frames without a ball. It does not call `fake_ball_position`. Faults: the time halved, the
     speed constant halved, the start moved by 1 m: the new test fails for each, the old
     `test_ball_follows_the_stand_in_ball` passes for each. It takes 0.01 s.

9. **Open.**
   - What the guard still starts is what it needs: per staged commit `diff --cached`,
     `check-ignore`, one `cat-file --batch`, `config --local` and two `git var` (item 3 says
     why they stay); per push `remote`, `for-each-ref`, `rev-list`, `diff-tree` and up to two
     `cat-file --batch`. The tests start the other 700 processes of `test_guard.py`.
   - `test_guard.py` still takes about 41 to 45 s of the default run here. The decision of the
     entry "A faster default run of the offline tests" (a marker for the process tests, or
     `pytest-xdist`) stays with the owner.
   - Faults G07 and G11 (item 7) are found by no test: both need a git that breaks its own
     line protocol.
   - Differences that the generated states cannot show, by reading the code: when two things
     fail in one run, the first error can be another one than before (for example, git answers
     "missing" for one object and then dies on a later one in the same question: before, the
     exit code was reported; now the missing object). The error text of a failed
     `cat-file --batch-check` can hold warnings of earlier questions of the same run.
   - Not verified: any git other than 2.56 on macOS, the run machine, and Linux. The features
     used (`rev-parse` with two options, `cat-file --batch-check` reading one name at a time)
     are old, but the equality of the `HEAD` answers was tried on this git only.

## 2026-11-22 — Rehearsal of the handover and the runbook

Nobody had walked `docs/HANDOVER.md` and `docs/RUNBOOK.md` from the first line to the last since
ten pull requests of 22 Nov changed the package and each edited its own part of the documents.
This entry records that walk on the build machine (base `main` fe4f316) and what it changed.

1. **How.** A scratch clone of the repository, with a home folder of its own, so that
   `~/pitchvlm-data` was a scratch folder; hooks, identity, deny-list (comments only) and the
   work tree `../pitchvlm-results` were made there with the commands of the documents. The
   environment of the main checkout (Python 3.13, no SDK, no Pillow), `PYTHONPATH=src`. The
   data: a synthetic `valid.zip` of 34 clips (50 frames, 1280 x 720, JPEG, made from three
   synthetic scenes) put into `~/pitchvlm-data/archives`, so the real `data` command only
   extracted it and wrote the split list. The model: the fake model of the tests in the place of
   the real backend. The scorer: the stand-in evaluator of `tests/test_cli.py` at the path of the
   scorer script. The tools: `tools/run_tool.py` with only its two tool engines replaced by the
   truth of the scene (its reading of the job, its checks of `--out` and its writers ran). The
   estimates of the two doctor commands: the environment checks passed by hand and a backend
   that stops at the first call. Every command that needs no key, no data download and no tool
   ran this way; the others were parsed. The walk log (73 steps, each with the command, what the
   document says, what happened and a verdict) is in the notes of this group of sessions, outside
   the repository.
2. **Counts** (78 verdicts on 73 steps; 5 steps have two): agrees 49, document wrong 18, code
   wrong 1, cannot be checked on the build machine 10.
3. **Measured on the synthetic data** (the commands print the same numbers on the run machine
   for the same plans; the costs are estimates of the token table of 22 Nov, not measured):
   doctor of the two Gemini models 0.133 USD for 14 calls; doctor of one OpenRouter model
   0.015 USD for 6 calls; cost pilot 0.196 + 0.124 + 0.332 = 0.652 USD (12, 10 and 36 calls); the
   pitch probe on 4 clips 0.248 USD for 20 calls; the whole pitch probe 1.488 USD for 120 calls;
   a clip of 750 frames 2.345 USD (64 keyframes, 50 number requests), its run limit crossed at
   the 129th number request; one number request 0.0090 USD; the detector engine with the number
   step 0.463 USD per clip of 750 frames. The ledger had one line per new answer (56 after the
   pilot), and a replay gave the same metrics for free.
4. **Fixes of the documents** (one line each):
   - HANDOVER 1: the state of 22 Nov (what was added, the full run of the offline tests, the
     rehearsal).
   - HANDOVER 2: marked as done; it stays as the record of the move.
   - HANDOVER 4.2, step 13, and RUNBOOK 2: `doctor --no-api` checks the keys of both providers,
     so with the Gemini key alone its key line is `[info]`, not `[ok  ]`.
   - HANDOVER 6: replaced by the first session on the run machine, step by step (19 steps:
     command, what it prints to pass, what to do when not, cost, estimate and go-ahead).
   - HANDOVER 6 (old text): "the pipeline cannot use the answers of a tool yet" removed; the
     detector engine is built.
   - HANDOVER 7, 11: the OpenRouter parts of RUNBOOK 10; 28 open owner decisions, not twelve.
   - HANDOVER 12 and PLAN "Milestones": a note that no date from M2 on holds.
   - HANDOVER 13: a row for `pre-commit install` that finds a hook of a git template.
   - RUNBOOK 2, command 4: `mkdir -p ../pitchvlm-results/results/metrics` before the copy of
     `oracle.json` (the copy failed when no doctor report had made the folder).
   - RUNBOOK 3: six tool experiment files, not five.
   - RUNBOOK 3, paint alignment: every metrics file on the run machine already has the key; a
     replay needs the `--max-clips` of the probe (without it, it stopped at the first frame
     without a stored answer).
   - RUNBOOK 3, `overlay run`: an `--out` per run folder (finding F1 below).
   - RUNBOOK 11: the commands of the check of the pixels of a video frame read by Pillow in a
     tool environment against the frame of `run` (it was only an open item of an entry).
   - PLAN, M4: the tool engines are built; the switch stays an owner decision under rule 8.
   - ARCHITECTURE 5: the new test.
   - This file: a stray line of a merge conflict (`||||||| 81bb2d9`) before "Open questions"
     removed.
5. **A test keeps this from drifting again.** `tests/test_documented_commands.py` (default run)
   reads every fenced block of `CLAUDE.md`, `docs/HANDOVER.md` and `docs/RUNBOOK.md`: every
   shell command that calls the package, `tools/run_tool.py` or `tools/guard.py` must parse with
   that parser; every file below `configs/` that a call of the package names must exist, an
   experiment or settings file must load, and a probe must name the experiment file of its
   kind; every name that a Python block imports from the package must exist; each sub-command
   of the package must be documented at least once; and no document holds a line of a merge
   conflict. It reads 60 calls today. Three drifts planted in the real documents (the
   sub-command `tool-measure` in RUNBOOK 3, the file `configs/arms/g4_team_slot.yaml` in RUNBOOK
   4, the option `--images` in the place of `--count` in HANDOVER 6) made it fail with one line
   each; the same three kinds are kept as a unit test of the reader. The stray conflict line
   above was found by it.
6. **Finding for the review of the code, not fixed** (only a help text or an option name was
   in the scope of this work):
   - F1. `overlay run` names its image folder by the clip only
     (`cli._cmd_overlay`: `folder = args.out / f"{mappings.name}-seed{args.seed}-count{args.count}"`,
     where `mappings.name` is the clip id for the source `run`). Trace: `overlay run --result
     ~/pitchvlm-data/runs/dev-detector/SNGS-116/result.json --clip .../SNGS-116 --seed 20261113
     --count 5 --list .../SNGS-116_dev-detector.json` (exit 0), then the same with
     `runs/dev/SNGS-116/result.json` and another `--list`: `error: .../overlays/SNGS-116-seed20261113-count5
     is already there: remove it, or give another --out`, exit 2. Two engines on one clip are
     exactly the comparison of RUNBOOK 11. The documents now give an `--out` per run folder; a
     name with the run folder (or the engine) would remove the trap.
7. **Not checkable on the build machine** (each is a step of HANDOVER 6 or of section 4): the
   install with the extra `gemini` and the SDK version line; the pinned list and fact 19; the
   prompt of the key and the form of the real key; the scorer environment and its clone; every
   real model call (both doctors, the pilot) and the real costs; the download; the label reader
   and the scorer on real files; the guard tests on the git of the run machine; the install of
   the tools, their first run, `mps` against `cpu`; the Pillow read of a real video frame (the
   check ran with OpenCV in the place of Pillow on a synthetic video: same pixels); the paint
   alignment on real frames; the check images of real mappings.
8. **Proposed text for `CLAUDE.md`** (section "Commands", last paragraph; the owner applies it):
   "The run-machine commands (`doctor`, `data`, `oracle`, `probe`, `tool-job`, `tool-score`,
   `clip-job`, `video-frames`, `overlay`, `run`, `eval`, `sync`) are in `docs/RUNBOOK.md`; the
   order of the first session is in `docs/HANDOVER.md`, section 6."
9. **Owner decision added to the list below:** new dates of the schedule.
10. **Checks of this work.** Full run of the offline tests: 3203 passed, 1 skipped (Pillow), 191 s,
    exit code 0 (8 of them new). `python3 tools/guard.py --tracked`: exit code 0.

## 2026-11-22 — Second read of the dissolve test, the video frames, the overlays and the guard

A second read of the code that pull requests 11, 12 and 13 brought into `main` (`fe4f316`; base
`81bb2d9`), plus one code finding of the rehearsal of the runbook. Four readers read separate
parts (the guard; video frames and the image folder check; overlays and the rules; dissolves).
Each finding went to a second reader whose job was to refute it, with a test that fails on the
old code or an exact trace; only confirmed findings are fixed. One more reader then tried to
break the fixes. Every number below is synthetic or made on the build machine: synthetic clips,
stand-ins of the model and the tools, videos written by OpenCV. None is a result on real footage.
"Before" is `fe4f316` (the guard and the documents: `638863a`, which has the same guard);
"after" is the commit of this entry.

1. **Counts.** Major: 2 proposed, 1 confirmed and fixed (item 2), 1 confirmed and graded minor
   by its second reader (the frame rate, item 3). Minor: 10 confirmed, 7 fixed, 3 kept as written
   limits (item 7). Small: 15 confirmed, 9 fixed, 6 kept as written limits (item 7). Refuted: 2,
   and 1 graded down to an owner item (item 8). No fail-open of the guard was found (item 5).
   The read of the fixes found 1 major regression of this review (the dissolve hidden by one
   failed pair, item 2), 1 minor and 4 small holes (item 4): all fixed with a test that fails
   before.
2. **Major, fixed: a failed optical flow inside a shot made up a dissolve.** When
   `estimate_motion` gave no motion for a pair that passed the pair test, `measure_shots` put the
   identity or the last motion into the window chain (`ShotTests.test_steps`), so a pan was
   judged with a wrong chain, while the entry "A dissolve is a transition, not a shot" and
   ARCHITECTURE said that a window with a missing step is not judged. Such a pair is now a
   missing step: the windows over it are not judged; the pair test and the hard cuts are
   unchanged. The reader of the fixes then found the other side: one such pair inside a
   dissolve hid the whole dissolve, because every window that holds it holds the missing step.
   So a lone such pair between two pairs with a step takes the last measured motion when that
   motion explains the pair alone; two or more in a row stay missing. Tests: `test_a_pan_whose_flow_fails_for_some_pairs_has_no_transition` (seed 21,
   flow forced off for the pairs into frames 1 to 10; before: frames 2 to 9 dropped as a
   transition in a clip without a cut) and
   `test_one_pair_of_a_dissolve_without_a_measured_motion_does_not_hide_it` (one and two lone
   pairs; the dissolve was not found without the bridge). Measured on new seeds 20 to 29 (n = 10 per case, the clip
   families of `tools/synthetic_diagnostics.py`; the old `motion.py` against the new one):
   - Without forcing, the flow failed on 0 of 9,230 pairs outside cut pairs and the pans of
     100 px per frame, on 10 of 790 hard-cut pairs and on 17 of 590 pairs of the 100 px pans.
     Seed 29, pan at 100 px per frame: before, a natural false transition (frames 26 to 40
     dropped); after, none. Every other case: the same hits and false cuts (dissolves 49 of 50,
     hard cuts 10 of 10; control false blocks 3 before, 2 after, all in the 100 px pans; windows
     judged in those pans 270 of 300 before, 240 after).
   - Flow forced off (pan 25 px per frame, pairs 1 to 10 / 20 to 40): clips with a made-up
     transition 4 of 10 (26 frames dropped) / 8 of 10 (114) before, 0 of 10 after; pan 50 px,
     pairs 20 to 40: 3 of 10 (26) before, 0 after.
   - Lone pairs forced off inside a dissolve (10 frames: pair 45, or pairs 43 and 47; 25 frames:
     pair 52, or pairs 45, 50 and 55): found 10 of 10 / 10 of 10 / 9 of 10 / 9 of 10, as before;
     without the bridge 0 of 10 each. Lone pairs forced off in pans (every 5th pair, or pair 30;
     25 and 50 px per frame): no transition before or after.
   - The cost: a dissolve whose flow fails for several pairs in a row inside the blend is no
     longer found by the window test. Pairs 41 to 48 of a 10-frame dissolve: found 10 of 10
     before, 5 of 10 after; pairs 50 to 55 of a 25-frame dissolve: 9 of 10 before, 3 of 10 after.
     Without forcing the flow failed on 0 of 4,700 dissolve pairs. On real blends it may fail
     more often (inferred): an open question below.
   - Command: `PYTHONPATH=src python d1.py natural|forced` of the scratch folder of this read
     (not in the repository; it loads the old `motion.py` next to the new one).
   - `tools/synthetic_diagnostics.py dissolve-fit dissolve-held-out` run again with the fix,
     only to see whether the numbers of the entry still hold (no threshold changed, nothing
     tuned): every hit, miss and false cut on seeds 0 to 19 is unchanged. Changed: the lowest
     window value of the pans at 100 px per frame, fit seeds 0.268 to 0.848 (247 of 300 windows
     judged) -> 0.281 to 0.848 (218 of 300); held-out seeds 0.201 to 0.860 (300 of 300) -> 0.358
     to 0.860 (275 of 300); the fade through black 0.572 -> 0.807 (295 -> 280 of 600 windows).
     So the thin margin of item 5 of that entry (0.201 against 0.18) came from a made-up chain;
     the cost is that 25 of 300 windows of those pans are no longer judged.
3. **Minor, fixed.**
   - **The frame rate of a video comes from its frame times** (`frames.check_frame_times`,
     reader's grade major, second reader's minor: the refusal is loud and only the video path is
     hit). OpenCV gives `CAP_PROP_FPS` as the frame count over the length of the stream, and
     that length holds the length of the last frame. Measured: an mp4 of 750 frames, every frame
     40 ms after the one before, whose last frame lasts 1/40 of a frame, reads 25.0326 frames per
     second and was refused at frame 384 as "a variable frame rate"; with a last frame of 2
     frames, 24.9667, refused at frame 376. Any last frame shorter than 0.5 or longer than 1.5
     frames was refused, for any clip length. Now the rate is the time from the first to the last
     frame over the count less one (`info.fps`, so `video.json` too); a frame is refused when its
     step from the frame before is more than half the median step away from it (a frame dropped
     or doubled anywhere: the mean step alone hides one lost frame in the middle of a long clip)
     or when it is more than half the mean step away from its place (a slowly changing rate).
     The header count check is unchanged. Tests: `test_video_with_a_short_last_frame_keeps_its_rate`,
     `test_frame_times_give_the_rate_of_the_frames_whatever_the_header_rate` (25, 30000/1001,
     60000/1001, 24000/1001 with times rounded to the millisecond; header rates exact, 0.13 %
     off, and rounded to 2 decimals), `test_a_frame_missing_in_the_middle_of_a_long_clip_is_refused`.
     The docstring said "in decoding order": OpenCV gives the frames in the order they are shown
     (measured on an H.264 file with B-frames and an edit list: 120 of 120 frames, times rising).
   - **`overlay run` checked only the clip id, the frame count and the size of `--clip`.** A
     video of the same name with other pixels got the mappings of the run drawn on other frames.
     `check_result_video` now compares the SHA-256 of every clip run of the result
     (`provenance.tool_runs.*.video_sha256`) with the video of `--clip` (the file, or the record
     of a frame folder). Test: `test_overlay_of_a_run_refuses_a_video_with_other_pixels` (the
     frames of the run in reverse order). A run of the model engine records no clip hash, so it
     is not checked (item 7).
   - **The rule-4 test left `tools/run_tool.py` out**, so an SDK import there passed. It is
     scanned now, with the SDK forbidden and the tools and computed names allowed there. Test:
     `test_rule_4_check_lets_the_tool_runner_import_a_tool_but_not_the_sdk`.
   - **A paid model answer of a keyframe on a blend frame was dropped without a word**, and the
     run printed the frames of a dissolve as "hard cuts". `run` and `demo` now print the cuts
     that start a shot apart from the transition frames, with the keyframes among them, whose
     answers are dropped. Test: `test_the_cli_names_the_transition_frames_and_the_keyframes_whose_answers_are_dropped`
     (the synthetic dissolve clip: keyframe 48 lies on a blend frame).
   - **The guard: a broken pipe at the close of the input of `git cat-file --batch-check`** left
     `main` as an error it does not catch: exit code 1 with a traceback (the code of
     "violations", with no violation line), where the guard of `81bb2d9` exited 2 with the
     exit code of git. It needs a git that ends before the guard writes its first name. Still a
     refusal, not a leak. `ObjectInfo.finish` now takes that second broken pipe; the exit code of
     git, or `_ended`, still raise. Test: `test_git_that_ends_before_the_first_name_is_written_is_a_git_error`.
   - **The guard: no test held the `resolve()` of the start folder.** With `start.absolute()`
     (a planted fault), `--repo` naming a link to a subfolder missed the deny-list and exited 0.
     The code was right; test: `test_deny_list_is_found_through_a_link_to_a_subfolder`.
   - **`overlay run` named its image folder by the clip only** (found by the rehearsal of the
     runbook): two engines on one clip, the comparison of RUNBOOK section 11, refused the second.
     The folder is now `<clip>-result<first 8 digits of the SHA-256 of the result file>-seed..-count..`;
     the RUNBOOK lost its way around. Test: `test_overlays_of_two_runs_of_one_clip_go_to_two_folders`.
4. **Small, fixed.** `overlay` refuses an experiment name with a folder separator (the image
   folder is named after it, a text of the experiment file: `../` put the images outside `--out`,
   into a work tree when it pointed there). The list file of `overlay` is written before the
   image folder gets its name, so a failed write leaves neither and the next run is not refused.
   `video-frames` names the hidden `.<clip>.partial-*` folders that a killed run left (SIGKILL
   runs no clean-up; about 1 to 2 GB each). `overlay probe` gives an answer that cannot be parsed
   its own reason (`probes.pitch_probe_fits` returns the flag). The rule-4 check now flags every
   use of a dynamic import function (`import_module` under another name, handed on without a
   call, `pkgutil.resolve_name`, `find_spec`, `run_module`, pydantic `ImportString`, a relative
   name in `import_module`); the package uses none. A clip of 30 frames or fewer logs that it
   has no dissolve test. Texts: the docstring of `measure_shots` (it repeats the pair test of
   `is_cut`, it does not call it), the comment of `ClipResult.cuts` (blend frames are cuts too),
   the module text of `overlays.py` (what the list file holds), ARCHITECTURE.
   Found by the reader of the fixes, fixed: a list file written half way (a full disk) blocked
   the next `overlay` (minor; the list was not there at the start, so any file there now is
   removed on a failure); the pattern of the hidden folders of `video-frames` was not escaped (a
   clip id with `[` named the folders of another clip); `check_result_video` let a video pass
   for clip runs of images of no video; the step test let a step of exactly half the usual step
   pass (a variable rate of 60 % 80 ms and 40 % 40 ms steps was taken as 15.6 frames per second;
   the check before this read let it pass too); the rule-4 check read
   `__import__('genai', {...}, None, [], 1)` (relative by its last argument) as `genai` and took a
   variable named `run_path` for an import. Still not found by the rule-4 check, and only by
   deliberate evasion: `getattr(importlib, "import_module")`, `exec` of a text,
   `vars(builtins)[...]`, a `SourceFileLoader`.
5. **The guard: differential and fault check.** The generator is new (written in this read,
   kept in the scratch folder, not in the repository): plain repositories, subfolders, linked
   work trees and their subfolders, submodules, a symbolic link as the start folder, folder names
   with a line break, a trailing line break or spaces; HEAD unborn, detached, detached on a
   damaged commit, a branch or a tag named `HEAD`; annotated, lightweight and blob tags; a
   remote with tracking branches; missing blobs; file names with a line break; files above
   2 MB; tracked ignored files; deny words in files, paths, messages and branch names; the
   variables `GIT_DIR` (absolute and relative), `GIT_WORK_TREE`, `GIT_INDEX_FILE` (absolute and
   relative), `GIT_COMMON_DIR` and those of the hook runner, one with a line break. Each state
   ran in `--staged`, `--push`, `--tracked` and `--paths`; exit code, violation lines and the
   class of the error were compared.
   - `fe4f316` against `81bb2d9`, seeds 9100 to 9419: 312 states (8 not built: damaged objects
     of the generator break `git worktree add`), 1,205 mode pairs, 2,410 runs, 0 differences.
   - Final code against `81bb2d9`, seeds 9500 to 9759: 253 states (7 not built), 978 mode pairs,
     1,956 runs, 0 differences.
   - Real `git commit` and `git push` hooks in a plain repository, a linked work tree and a
     submodule: both guards refuse. No hang and no process left behind in any run.
   - A folder name that is not UTF-8 could not be made (the file system refuses it); by the code
     it takes the old commands, which fail with exit code 2 as before (inferred).
   - Faults planted in the new code paths, `tests/test_guard.py` run on each: on `fe4f316` 16 of
     21 caught (one only as a hang of 900 s: `stop` without `kill`). On the final code, with one more fault (the catch of item 3 taken out) and without the
     one that hangs: 17 of 21 caught; both faults that the new tests of item 3 aim at are caught.
     Not caught and judged harmless: the check of the object name in an answer (git answers one
     line per question), no `stop` after an error (git ends at the end of its input when the
     guard ends), `HEAD` asked without `^{commit}` (differs only for a HEAD on a tag object), a
     decode with "replace" (not testable here).
   - Error texts: with a relative `GIT_DIR` and `GIT_WORK_TREE` from a subfolder, both guards exit
     2, but the new one names `git ls-files` or `git remote` where the old one named
     `git rev-parse --git-common-dir`. The entry of the faster guard promises the old texts only
     on the path back; kept.
6. **Measured again: the dissolve entry** (`tools/synthetic_diagnostics.py dissolve-fit
   dissolve-held-out` on `fe4f316`): every number of items 5 and 6 of that entry matches, also
   the shares, the out-of-scope rows and the midpoints 0.1775 and 0.9535. Timing differs:
   pair test 1.17 ms, pair and window test 1.48 ms per frame (n 135; the entry: 1.12 and 1.42 ms);
   1.31 and 1.65 ms with the fixes. Timing noise of this machine; not compared further.
7. **Kept as written limits, not fixed** (each confirmed by a test or a trace; seeds 20 and up).
   - Minor: two dissolves less than about 25 frames apart become one transition and the clean
     shot between them is dropped (gaps of 10 and 15 frames: 5 of 5 merged; 20, 25 and 30: split;
     seeds 20 to 22). `find_transitions` has no upper bound on the length of a transition.
   - Minor: a hard cut in the middle of a dissolve: only the hard cut is found in 4 of 5 clips
     (seeds 20 to 24); the blend frames next to it keep their states.
   - Minor: the window is 30 frames at any frame rate. A dissolve of 1 s at 50 frames per second
     leaves frames with up to 0.137 of the other shot in the shots, at 75 frames up to 0.28 (the
     dissolve entry: "up to about 0.1", for 5 to 25 frames at 25 frames per second). A window
     in seconds needs a new fit, which rule 7 forbids here.
   - Small: the test images are kept as `float32`, 57.6 KB per frame: 43 MB for a clip of 30 s,
     7.8 GB for a whole match (out of scope).
   - Small: the kit legend is asked before the motion is measured, so its keyframe can be a blend
     frame.
   - Small: `refuse_pixels_in_repository` knows only the work trees that `git worktree list`
     names: a work tree moved by hand without `git worktree repair`, and a separate clone of the
     repository, pass (`.gitignore` still covers `*.png`). `run` and `demo` write their video
     into `--out` without this check (older code).
   - Small: `overlay run` compares no video for a result of the model engine (it records no
     hash of its clip) and for a result file older than the key `video_sha256`.
   - Small: the SHA-256 ties a clip run to the bytes of a video, not to its decoded pixels:
     another OpenCV or FFmpeg build could decode other pixels (inferred).
   - Small, for the owner: the check images of G3 are not blind: the caption shows the paint
     score and the images are numbered in score order, which can anchor the eye. G3 and the
     RUNBOOK define the caption so; a change is the owner's.
8. **Refuted, and checked and found right.** Refuted: a NaN paint score (no source: the matrices
   are finite, the score is a mean of booleans); an unmeasurable shot share counted as a blend
   frame (no case on seeds 20 to 24: 0 frames); the eye check against the definition of G3 (G3
   asks only for 5 images checked; kept as an owner item, item 7). Found right: the steering of the choice of overlay frames: the same choice under
   `PYTHONHASHSEED` 0, 1 and 12345 and with numpy integers, 0 of 500 shuffles of the input
   changed it. A text in the overlay list against rule 2 or 3: every field traced, none. A model
   call by `overlay probe`: none (a missing answer stops it). Frame numbering: `000001.png` is frame 0
   everywhere, no off-by-one. The refusal of an image folder in the repository held against a
   symbolic link, a dangling link, `..`, another letter case, a linked work tree, a link to it,
   the git folder and a relative path.
9. **Estimated, not measured on real frames:** a PNG frame of 1080p at compression level 1 is
   about 1.1 to 3.0 MB on synthetic frames with noise (the video entry: 1 to 2 MB), so 0.8 to
   2.3 GB per clip of 30 s at 25 frames per second.
10. **Only the run machine can settle:** a broadcast TS or MKV whose frame count OpenCV estimates
    from the length (an off-by-one refusal); interlaced 1080i where FFmpeg may report 50 for 25
    frames per second (the header count would then be refused); how often a real cut file has a
    last frame of odd length; edit lists that drop leading frames of a `-c copy` cut; how often
    the optical flow fails inside a real dissolve (item 2).
11. **Tests.** Every fix has a test that fails on the code before it (checked by putting the old
    file back). Full run (`-m "slow or not slow"`) on the commit of this entry: 3,232 passed,
    1 skipped (Pillow is not installed), exit code 0. `python3 tools/guard.py --tracked`: exit
    code 0. `demo --engine model` and `demo --engine detector`: exit code 0.

## 2026-11-22 — The offline tests in several processes

1. **Goal and result.** The default run of the offline tests took 110 to 155 s in one process.
   Hypothesis: the test files can run in N processes with the same collected tests and the same
   outcomes as one process, and the default run then takes at most 40 s on the build machine.
   Baseline: `python -m pytest` in one process on the same tree. **Result: the same tests and
   outcomes always; the 40 s were met in the first window of measurements and missed in the
   last one.** In the first window the new runner `tools/run_tests.py` ran the default run in
   27.4 to 37.2 s (14 runs, load average 5.7 to 27 on 14 cores), against 142.5 to 144.5 s in one
   process (4 runs), and the full run in 28.3 to 36.4 s against 164.3 s. Two hours later, after
   the merge of `main` f2057df (item 13), the default run took 47.5 to 62.4 s and one process
   200 to 203 s; the tree of the first window took 47 to 53 s then too, back to back with the
   merged tree, so the machine changed, not the code. The runner was 3.8 to 4.5 times faster
   than one process in both windows. Every time here is measured on the build machine while
   other sessions ran; the load average (1 min) is given with each run. Nothing ran on the run
   machine.

2. **Design.** `tools/run_tests.py` imports only the standard library and starts plain
   `python -m pytest` processes; no package and no plugin was added. One
   `python -m pytest --collect-only` with the options of the run lists the test ids, as one
   process collects them. The ids are split over one worker per core (`-j N`): a file whose time
   is more than half the share of one worker is cut into parts in collection order, and the parts
   are given out longest first, each to the worker with the least time so far; each worker runs
   its ids in collection order. The time of each test comes from the last run in this checkout
   (`.pytest_cache/run-tests-times.json`, ignored by git, written by the runner). Without that file
   the runner balances by test count and says so; the run is correct, only slower (first run of
   the default tier in a new checkout: 63.8 s). Each worker gets its own base temporary folder
   (`--basetemp`), its own pytest cache folder (`-o cache_dir=`), its own JUnit XML file, its own
   process group, and the options `-m`, `-k`, `-x` and the test ids. Any other option of pytest
   is refused.

3. **Checks in every run.** The run fails when:
   - a test fails or has an error (exit code 1; the full failure text of each failed test is
     printed with its id, then one summary line: passed, failed, errors, skipped, deselected,
     wall time);
   - a worker ends in another way than pytest's normal end (exit code not 0 or 1, no JUnit XML
     file, a count of collected tests other than the count of its ids, an exit code that does not
     fit its JUnit XML file): exit code 2, the output of that worker is printed;
   - a worker writes nothing for 120 s (`--silence-limit`): its process group is killed, and the
     run names the file and the last test that started (exit code 2; pytest's `faulthandler`
     prints the stack of the hanging test after half the limit);
   - the ids that the workers ran differ from the collected ids: a test that ran twice, a test
     that the collection does not list, a collected test that did not run (exit code 2). With
     `-x`, the tests after the first failure are allowed to be missing.
   Ctrl-C (and SIGTERM, SIGHUP) stops every worker group: SIGTERM, 5 s, then SIGKILL (exit code
   130). The JUnit XML name of a test (`classname`, `name`) is mapped back to its id with the rule
   of pytest 9; a change of that rule fails the id check instead of losing tests.

4. **What is shared beside other processes, and what was found.**
   - pytest's base temporary folder and cache folder: each worker gets its own, inside one
     folder of the system's temporary folder per run, removed after a run that can be trusted
     and kept (with the output of each worker) after exit code 2.
   - Fixed file names under `/tmp`, fixed ports, `HOME`, environment variables: no test writes a
     fixed name under `/tmp` (every test uses `tmp_path` or `tempfile`), none opens a port, the
     tests that start git set a private `HOME` and `GIT_CONFIG_NOSYSTEM` (`private_environment` of
     `test_guard.py`, the fixture `private_git` of `test_cli.py`), and environment variables
     belong to one process.
   - The git folder of this checkout: only read (`git rev-parse`, `git status`, `git ls-files`).
     `git status` may refresh the index; a second process that finds the lock skips the refresh.
     The checkout stays unchanged after a run (`git status --porcelain --ignored` before and
     after: only `.pytest_cache/`), and no `~/pitchvlm-data` was made.
   - The caches of `tests/conftest.py` and the git template of `test_guard.py` are per process;
     each worker builds its own. `tests/conftest.py` needed no change.
   - `tests/test_guard.py` and `tests/test_cli.py` are split by test ids over 5 to 10 workers; the
     same outcomes in every run show that their tests are safe apart.
   - In 43 runs with the runner and no planted fault, 3 had a failed test: 2 were the runner
     test of (a) below; the output of the third (item 6) was not kept, so its test is not known.
     Both flaky tests that were found are tests of the runner itself, and each showed a defect of
     the runner, fixed with a test:
     (a) `test_first_failure_stops_the_other_workers` failed in 2 of 13 runs (and probably in 1
     of 8 runs of item 6, whose output was not kept): with `-x`, when the other worker ended by
     itself before the failing one, the tests skipped by pytest's own `-x` counted as missing
     (exit code 2 instead of 1). The new test
     `test_first_failure_ends_its_worker_after_the_other_workers_ended` forces that order and
     failed before the fix. (b) `test_ctrl_c_stops_every_worker_and_their_children` failed once
     in 17 one-process runs: `os.killpg` on macOS gave `PermissionError` for a worker that was
     ending at the moment of the Ctrl-C (read from the code of macOS's `killpg`: no process of the
     group could take the signal; not proved), and the cleanup stopped. The runner now takes that
     answer as "nothing left to signal" and kills the worker itself before it waits for it; 8
     runs of the runner tests in a row passed after the fix.
   - Process starts are a shared limit of the machine: about 180 starts of `/usr/bin/true` per
     second and 250 of `git --version` per second, the same from 2 to 14 parallel streams. The
     suite starts about 3,000 processes (mostly git in `test_guard.py`), so about 12 to 15 s of
     any run are process starts that more workers cannot shorten.

5. **Same outcomes as one process** (full run, `-m "slow or not slow"`, back to back, load 15.4
   and 14.3). One process: 3,224 passed, 1 skipped, 164.3 s. Runner: 3,224 passed, 1 skipped,
   36.4 s. The outcome of each of the 3,225 tests (from the JUnit XML file of the one process,
   and `--results` of the runner) is the same: `diff` of the two sorted lists is empty.

6. **Time per number of workers** (default run, 3,215 tests, order 2, 4, 8, 14, 14, 8, 4, 2):

   | Workers | Wall, s (load before) |
   |---|---|
   | 2 | 74.1 (13.1), 75.6 (19.5) |
   | 4 | 43.9 (7.7), 42.9 (20.0) |
   | 8 | 30.6 (9.6), 33.5 (19.6) |
   | 14 | 33.9 (14.1), 34.0 (18.5) |

   It stops paying between 8 and 14 workers on this machine (process starts, item 4). The
   default stays one worker per core: no number of this machine is written into the code.

7. **A/B, back to back, both orders** (default run, 3,216 tests; order one, runner, runner, one,
   one, runner, runner, one; load before each run in brackets):

   | Arm | Wall, s | Mean |
   |---|---|---|
   | `python -m pytest` | 142.5 (21.0), 144.4 (17.3), 143.1 (6.4), 144.5 (18.2) | 143.6 |
   | `python tools/run_tests.py` | 30.3 (8.0), 30.5 (12.9), 29.2 (5.7), 37.2 (13.8) | 31.8 |

   The runner uses about 24 % more processor time than one process (user + system 185 s against
   149 s for the full run): 14 Python starts with NumPy, OpenCV and SciPy, and caches built in
   each worker.

8. **Stability.** After the fix of item 4 (a): 10 default runs in a row, all green, each 3,216
   passed, 1 skipped, 9 deselected, 27.4 to 32.7 s (load 15 to 27); then 3 full runs, all green,
   each 3,225 passed, 1 skipped, 28.3 to 30.1 s. Before the fix: the same series, 11 of 13 green;
   the 2 failures are item 4 (a).

9. **Fault check.** Twelve faults planted under `src/`, one at a time, each undone after its
   runs; for each, the default run in one process and with the runner. The runner exits with
   code 1 for each fault the one process finds, and names the same failing tests:

   | Fault | Module | Change | Failing tests, one process / runner |
   |---|---|---|---|
   | P01 | `ball.py` | gate without the noise of one box width | 1 / 1 |
   | P02 | `datasets.py` | archive member with `..` accepted | 3 / 3 |
   | P03 | `export_gsr.py` | goalkeeper gets a jersey | 84 / 84 |
   | P04 | `vlm/client.py` | Gemini call stored by the provider | 20 / 20 |
   | P05 | `vlm/cost.py` | default hard stop 136 USD, not 135 | 0 / 0 |
   | P06 | `vlm/cost.py` | run stops at 2x its estimate | 3 / 3 (and item 4 (b) in the one process) |
   | P07 | `motion.py` | a pair without a correlation is no cut | 1 / 1 |
   | P08 | `pitch.py` | pitch 0.5 m too wide | 23 / 23 |
   | P09 | `schemas.py` | union counts the overlap two times | 44 / 44 |
   | P10 | `frames.py` | other keyframe interval | 1 / 1 |
   | P11 | `cli.py` | a changed work tree is clean | 1 / 1 |
   | P12 | `datasets.py` | absolute archive member accepted | 2 / 2 |

   P05 is a gap of the suite, not of the runner: `test_default_hard_stop_is_135_usd` spends 134
   and then 136 USD, so a hard stop of 136 passes too. The command line takes the hard stop from
   `configs/gates.yaml` (135), not from this default. A control run without a fault passes (item
   8). Three faults of the run itself, on the real suite: a worker killed from outside with
   SIGKILL (exit code 2, the worker and its files named, no process of the run left); a test file
   with `time.sleep(10**6)` and `--silence-limit 20` (exit code 2 after 79 s, the file and the
   test named, no process left); a test file with a syntax error (exit code 2, the `SyntaxError`
   of the collection printed; the one process ends with exit code 2 too).

10. **The hooks.** The commit hook and the push hook run the guard and the secret scan only
    (`.pre-commit-config.yaml`); they never ran the tests, and that stays so. The hook setup did
    not change.

11. **Proposed text for `CLAUDE.md`** (section "Commands"; the owner applies it):

    ```bash
    python tools/run_tests.py                          # offline tests, default run, in several processes
    python tools/run_tests.py -m "slow or not slow"    # full run, before a commit
    python -m pytest                                   # the default run in one process
    python -m pitchvlm.cli demo --out outputs/demo     # synthetic clip, fake model, no network
    python3 tools/guard.py --tracked      # leak check of all tracked files
    python3 tools/guard.py --push         # leak check of what a push can send: commits, tags, new names
    ```

    and in "How to work": "`pytest` must pass offline, with no SDK installed, in one process
    and with `tools/run_tests.py`."

12. **Open and not verified.**
    - No run on the run machine, on Linux, or on a quiet machine. The target of 40 s held in the
      first window here and was missed in the second (items 1 and 13); the time of a run on this
      machine moves with the state of the machine by half or more.
    - The reading of macOS's `PermissionError` in item 4 (b) is inferred from the code of the
      system, not proved by a test that forces it.
    - The mapping of JUnit XML names to test ids follows pytest 9.2; another pytest version may
      fail the id check of every run (it cannot pass a wrong one).
    - The gap of fault P05 (item 9) is not closed here.
    - An earlier version of the Ctrl-C test, before fix (b), left two idle test processes on the
      build machine; they ended by themselves after one hour.

13. **After the merge of `main` f2057df** (the rehearsal of the runbook and the second read of
    round 3: new tests in seven files and `tests/test_documented_commands.py`). The new tests
    write only below `tmp_path`, and the new guard tests keep the private git environment of
    `test_guard.py`: nothing new is shared beside other processes. `test_documented_commands.py`
    now also parses each documented call of `tools/run_tests.py` with the runner's own parser
    (a new test plants an option that the runner refuses). Measured on the merged tree, in this
    order:
    - Same outcomes (full run, back to back, load 8.3 and 6.8): one process 3,263 passed,
      1 skipped, 218.4 s; runner 3,263 passed, 1 skipped, 59.7 s; the outcome of each of the
      3,264 tests is the same (`diff` empty).
    - Stability: 10 default runs, all green, each 3,254 passed, 1 skipped, 9 deselected, 48.6 to
      58.3 s (load 22 to 30); 3 full runs, all green, each 3,263 passed, 1 skipped, 53.6 to
      55.8 s.
    - A/B (default run; order runner, one, one, runner, runner, one): one process 202.7 (23.7),
      204.3 (8.0), 202.6 (24.8) s, mean 203.2 s; runner 53.7 (23.7), 62.4 (4.9), 55.7 (21.2) s,
      mean 57.3 s.
    - The cause of the slower times: the tree of item 1 (commit of this entry before the merge,
      3,216 tests) and the merged tree (3,255), runner, back to back, order old, new, new, old,
      old, new: old 47.3 (4.0), 51.0 (29.5), 52.9 (26.0) s; new 52.1 (16.5), 53.7 (23.5), 52.5
      (25.1) s. The machine was slower for both. Process starts were not the cause: 450 starts of
      `/usr/bin/true` per second in this window, against 180 in the first.

## 2026-11-23 — Video input for the model steps (session A of round 5)

1. **The goal** (owner request: "any img/video llm" must work). A model step can send a video
   file, or a time range of it, in the place of a list of images, through Gemini and through
   OpenRouter. Built on the build machine, offline: no model call, no key, no footage, no
   install. Nothing here ran against a real model or a real video of the dataset.
2. **Documentation pages read on 2026-11-23** (public pages, no sign-in, no API call; every value
   below is published, not measured). The pages of `ai.google.dev/gemini-api/docs/...` answered
   with a redirect to the same paths on `aistudio.google.com/docs/...`.
   - https://aistudio.google.com/docs/video-understanding — inline data for "shorter videos
     under 20MB total request size"; "Always use the Files API when the total request size ... is
     larger than 20 MB"; File API 20 GB (paid). Video part of the Interactions API:
     `{"type": "video", "data": <base64>, "mime_type": "video/mp4"}`. Clipping and rate in
     `"processing": {"type": "static", "start_offset": "1200s", "end_offset": "1500s"}`, `fps` in
     `(0, 24]`; "extracting frames at 1 FPS" by default. Types `video/mp4`, `video/mpeg`,
     `video/mov`, `video/webm` (the type list and the `fps` inside `processing` were read from a
     cut copy of the page: to confirm). Up to 10 videos per request (Gemini 2.5 and later).
   - https://aistudio.google.com/docs/files — "Files are automatically deleted after 48 hours";
     the Files API is for requests "larger than 100 MB" (this conflicts with the 20 MB above).
   - https://aistudio.google.com/docs/media-resolution — Gemini 4 and 3 models: video tokens per
     frame 70 at unspecified, low and medium, 280 at high; sound 25 tokens per second; the
     resolution can be set "for individual media objects (content items)" (the example is an
     image part). The video page gives 66 and 258 per frame; the tokens page
     (https://aistudio.google.com/docs/tokens) gives "263 tokens per second", "~100 (default low
     resolution) / ~300 (high resolution)" per second and "32 tokens per second" for sound. Three
     sets of numbers that do not agree.
   - https://aistudio.google.com/docs/pricing — one input price for all inputs of gemini-4-argon,
     gemini-4-carbon, gemini-4-hydrogen and gemini-3.8-flash (no separate video or sound price);
     the robotics model: the same price for text, image, video and sound (read from a cut copy).
   - https://openrouter.ai/docs/features/multimodal/videos — chat completions part
     `{"type": "video_url", "video_url": {"url": ...}}`; "The video_url can be either a URL or a
     base64-encoded data URL"; "Google Gemini (AI Studio): Only supports YouTube links"; "Google
     Gemini (Vertex AI): ... Use base64-encoded data URLs instead"; other providers: check the
     model; `processing` is read by Gemini only ("Other providers ignore the `processing`
     field"). No size limit is stated.
   - https://openrouter.ai/docs/features/multimodal/overview — "Video: Priced as input tokens
     based on duration and resolution". https://openrouter.ai/docs/guides/overview/models — the
     models list names the input types under `architecture.input_modalities`; no video price key.
   - https://openrouter.ai/api/v1/models/google/gemini-2.5-flash/endpoints — `input_modalities`
     holds `video`; its three endpoints are all "Google AI Studio" (which takes YouTube links
     only, by the video page), with an `audio` price above the `prompt` price (1.0 against 0.3
     USD per million at the standard endpoint) and no video price. The model pages of
     `qwen/qwen3-vl-235b-a22b-instruct` and `qwen/qwen2.5-vl-72b-instruct` name no input types.
     The public list https://openrouter.ai/api/v1/models answered in another shape than its page
     documents (no `architecture`, no `pricing`), so the video models could not be listed.
   - Item 7 of the entry "Image models through OpenRouter" (2026-11-22) said that no page showed
     base64 video in chat completions. The video page now says it in words (above), so the part is
     built; no example on the page shows it.
3. **What was built** (each with tests; `tests/test_vlm_video.py` and new tests in
   `tests/test_cli.py`):
   - `VideoPart` (`vlm/types.py`): path, SHA-256, type, length, optional start, end and rate,
     resolution (`low`, `medium`, `high`). No bytes. `identity()` (no path) goes into the cache
     key and into the cassette (`videos`). A request without video parts has neither entry: its
     key and cassette are those of before (the 15 frozen values of `tests/test_vlm_openrouter.py`
     still pass).
   - Gemini (`vlm/client.py`): inline base64 with `resolution`, and `processing` of the type
     `static` when the part names a range or a rate. The bytes are checked against the hash
     before the call. No upload: the Files API keeps a file at the provider for 48 hours, against
     the intent of `store=False`. A request with a video whose inline data is above 20,000,000
     bytes (the smaller published limit) is refused before any call (`VideoInputError`). A request
     of images only has no such limit, as before.
   - OpenRouter: a `video_url` part with a base64 data URL through the `urllib` transport, the
     same private routing; a part with a range or a rate is refused (the provider would ignore
     it), and so is inline data above the same 20,000,000 bytes (chosen, not published).
   - Money (`vlm/cost.py`): `VideoRate` and `GEMINI_VIDEO_RATES` (4 Gemini models; frame tokens
     70 / 70 / 280, the larger of each published pair; 32 sound tokens per second, the larger
     value; default rate 1 per second; billed at the input price). The OpenRouter rates are the
     optional `video` entries of `configs/openrouter_prices.yaml` (`load_video_rates`); none is
     filled (item 7). A model without a rate takes no video: the estimate raises `KeyError`
     before a run, and both backends raise `VideoInputError` before any call. The estimate of a
     video request is the text estimate of its step plus `frames x tokens_per_frame + seconds x
     sound tokens` per part; `people_video` counts its output once per target frame. The billed
     cost comes from the measured usage, so the 1.3 times stop and the hard stop work as for
     images. `Ledger` and `BudgetGuard` are not changed, and session D needs to change nothing for
     this: a Gemini video line is priced from its measured tokens at the input price; an
     OpenRouter line takes the reported cost.
   - Video files (`vlm/video.py`): the chosen frames of a frame source, written with OpenCV at the
     sampling rate (H.264 `avc1`, else MPEG-4 Part 2 `mp4v`), so frame `k` is at `k / fps`
     seconds and the model samples each written frame. A file that is there is kept, because its
     hash is in the key; its name holds a hash of the decoded pixels of its frames, so new footage
     under an old clip name gets new files (the image path has `OtherImagesError` for this case). Measured on the build machine (OpenCV 5.0.0): two writes of the same 5
     frames, 1 s apart, gave the same bytes, for `avc1` and for `mp4v` (1 trial each); OpenCV
     reads a written file back with its frame count, size and a rate of 2.083 for 2.0833.
   - Steps (`vlm/steps.py`, prompts `people_video/v1`, `kit_legend_video/v1`): the people prompt
     lists the target frames by their time in the file and asks one entry per target with
     `time_s`; `parse_people_video` takes an entry only when its time is within half a frame
     interval of its target and counts the others. Both video prompts keep rule 5: kit colours
     and shirt numbers only, "Do not use faces and do not give names".
   - `FakeBackend` answers both video steps from the request context; it never opens the file.
     The image answers draw the same random numbers as before (the tests of the fake pass
     unchanged).
   - Probe (`probes.VideoProbe`, the section `video` of an experiment file: `fps`,
     `context_frames`, `targets_per_request`; option `probe --video-dir`, default
     `~/pitchvlm-data/video-input`): the frames, the labels and the scorer of the image probe;
     the videos are written before the estimate; the result adds `n_people_requests`,
     `n_entries_wrong_time` and the video settings. Three experiment files:
     `people_probe_video.yaml`, `people_probe_video_low.yaml`, `people_probe_video_batch.yaml`.
   - Doctor (`doctor --video`): one video request per model, a 7 s video of 7 frames of the
     synthetic scene, 3 targets; checks `video boxes`, `video box accuracy`, `video times`,
     `video tokens` (measured input against the estimate). A model without a rate gets a failed
     `video boxes` line and no call. Estimated 0.048 USD for gemini-4-argon until 2026-11-30 and
     0.095 USD from 2026-12-01, inside the 2 USD of gate G0.
   - No default changed: `StepSettings`, `configs/default.yaml`, the pipeline and `demo` are as
     before; only the probe and the doctor send video.
4. **Measured offline with the fake model.** The video people probe on the synthetic clip of the
   probe tests (100 frames, 5 frames, seed 1) gives the same recall, precision, counts and foot
   errors as the image probe with one request per frame and with 5 frames per request
   (`test_video_people_probe_measures_the_frames_of_the_image_probe`). This proves the plumbing
   (frames, times, parsing, scoring), not the model: the fake answers from the truth.
5. **The experiment for the run machine.** Hypothesis: video input reaches gate G2 at a lower
   estimated cost per keyframe than image input. Baseline: `people_probe.yaml` (images, the same
   model). Controls: the same 24 clips of the group `probe`, the same 5 frames per clip (seed
   20261113), the same labels, the same scorer, the same model and thinking level; each video
   arm changes one thing (resolution `high` / `low`; one or five frames per request). Sample: 120
   frames per arm, the size of gate G2. Estimated costs (printed by the code, prices of the
   table, not measured):

   | Arm | Calls | To 2026-11-30 | From 2026-12-01 | Per keyframe |
   |---|---|---|---|---|
   | images (`people_probe.yaml`) | 144 | 2.357 USD | 4.714 USD | 0.0196 / 0.0393 |
   | video, high, 1 frame per request | 144 | 2.390 USD | 4.780 USD | 0.0199 / 0.0398 |
   | video, low, 1 frame per request | 144 | 2.108 USD | 4.215 USD | 0.0176 / 0.0351 |
   | video, high, 5 frames per request | 48 | 2.275 USD | 4.549 USD | 0.0190 / 0.0379 |

   The estimates differ by at most 11 % because the output estimate of a video frame is set to
   the 1,400 tokens of an image frame, and output is about 80 % of the cost. So the hypothesis
   is a question about measured output and thinking tokens and about recall at 70 tokens per
   frame, not about the estimate. The measured cost per keyframe is the measured cost of a run
   divided by its `n_frames`.
6. **Risks** (inferred, not measured):
   - Frame sampling: the mapping assumes that the provider samples at `0, 1/fps, 2/fps, ...` of
     the file. If it samples elsewhere, an answer describes another frame; `video times` and
     `n_entries_wrong_time` show a model that names other times, but not a model that names the
     right time for the wrong picture. The boxes of the doctor would then fail on the moving
     camera of the synthetic scene.
   - Timestamps: the model must copy the listed times; an answer that writes `1.4` for `1.44`
     is within the tolerance (0.25 s at 2 per second).
   - Coordinates: a box is given in the frame of its time; the convention is that of the step.
     With 5 frames per request the model can mix the boxes of two windows.
   - Pixels: the video arm sees H.264 (or MPEG-4 Part 2) frames, the image arm JPEG at quality
     90 of the same frames. A difference in recall can come from the codec, not from the
     input type.
   - Token rate: the three published sets of numbers differ by up to 3.7 times at low resolution
     (70 + 32 against 263 per second); a rate that is too low lets the 1.3 times stop end a probe
     early (it never lets it pay more than 1.3 times the estimate).
   - Codec: the run machine may write `mp4v` only, which a provider may refuse.
7. **Owner decisions (open).**
   - Uploads: a video above 20,000,000 bytes of inline data would need the Files API, which keeps
     the file for 48 hours. Not built; the client refuses such a request. Decide whether an upload
     followed by a delete is acceptable, or keep the refusal.
   - OpenRouter video rates: no page publishes a video token rate for an OpenRouter model, and the
     only Gemini model checked there serves through AI Studio (YouTube links only). So no
     `video` entry is in `configs/openrouter_prices.yaml`, and every OpenRouter video request is
     refused before any call. To enable one model, the owner adds, for example:
     `video: [{valid_from: <day>, tokens_per_frame: <n>, audio_tokens_per_second: <n>,
     frames_per_second: <n>, video_usd_per_million: <price>, audio_usd_per_million: <price>}]`
     from its model card, and runs `doctor --video` with it.
   - Video in the pipeline (`run`): not built. Decide after the probes, with both metrics files.
   - Rule 5 text of `CLAUDE.md` (not applied): add "Video goes inline only; no file is uploaded
     to a provider." after the routing sentence of the text proposed in the entry of 2026-11-22,
     item 9.
8. **Not verified (only the run machine can):** every request shape against the real APIs (the
   Interactions video part with `resolution` and `processing`; the OpenRouter `video_url` data
   URL); the token counts of video; the cost of a video call against its estimate; whether
   OpenCV of the run machine writes H.264; whether the same frames give the same bytes there;
   the sampling times of the provider; recall and precision of video input on real frames.
9. **Second read** (a subagent, before the commit): no high finding; 6 medium or low, all
   confirmed and fixed with a test where code changed. `probe --video-dir` is refused inside the
   repository and inside or above the cache folder (medium, rule 2). The privacy check of the
   doctor builds its call without the video parts, so a video that failed before its call no
   longer stops the doctor with a traceback. An estimate that raises `ValueError` (flex for a
   model without flex) gives one error line. The runbook says that `video times` shows only that
   the model copies the listed times; `video boxes` on the moving synthetic camera is the check of
   the sampling times. A plain doctor run no longer reads the video rates. Dead code and one test
   that passed without its fix were removed. Its recomputed numbers match items 3 and 5.

## 2026-11-23 — A second detector family: D-FINE and RT-DETRv2 through transformers

1. **Goal.** The owner can run the same gate (G2 on the frames of the people probe, then the dev
   clips) with a second detector family and choose by measurement (rule 8: same frames, same
   gate, same scorer). Point 3 of the entry "A real-time detector engine next to the model
   engine" named D-FINE with its COCO-only weights through the Apache-2.0 code of Hugging Face
   transformers, "added only after the licence of each weight file is read". Ultralytics YOLO is
   not added (AGPL-3.0; open owner decision). No code, config text or weights of any project
   were copied. Hypothesis for the run machine: a family of the same published latency as an
   RF-DETR size has a G2 recall within the noise of the 5-frame probe; baseline: the five
   RF-DETR files; nothing was measured here.
2. **Licences, read on 2026-11-23** (one reader, this session; pages read as text with `curl`,
   not checked by a second reader):

   | What | Page | What it says |
   |---|---|---|
   | Code path: transformers 5.23.0 | `LICENSE` and `src/transformers/models/rt_detr/image_processing_rt_detr.py` of https://github.com/huggingface/transformers at tag v5.23.0; PyPI page of `transformers` | Apache License 2.0; the file headers say Apache-2.0; PyPI: "Apache 2.0 License" |
   | D-FINE upstream code and weights | `LICENSE` and `README.md` of https://github.com/Peterande/D-FINE (master, commit `956d1709`) | Apache-2.0. The README: checkpoints trained or pretrained on Objects365 (`*_obj365`, `*_obj2coco`) "may be subject to the Objects365 dataset terms" and are not to be assumed commercially cleared |
   | RT-DETRv2 upstream code and weights | `LICENSE` and `rtdetrv2_pytorch/README.md` of https://github.com/lyuwenyu/RT-DETR (main, commit `df2a96c8`) | Apache-2.0; the COCO+Objects365 weights are a separate table |
   | 5 D-FINE checkpoints `ustc-community/dfine-{nano,small,medium,large,xlarge}-coco` | model card (README.md) and API page of each on huggingface.co, at the commits of `TRANSFORMERS_CHECKPOINTS` in `tools/run_tool.py` | licence `apache-2.0`, dataset `coco`; the card: "_coco -> model trained on COCO", "trained on COCO (Lin et al. [2014]) train2017"; weights as `model.safetensors` |
   | 4 RT-DETRv2 checkpoints `PekingU/rtdetr_v2_{r18vd,r34vd,r50vd,r101vd}` | same, at the commits of the script | licence `apache-2.0`, dataset `coco`; the card: "trained on COCO (Lin et al. [2014]) train2017"; no mention of Objects365; `model.safetensors` |
   | COCO, the training data of the 9 checkpoints | https://cocodataset.org/#termsofuse (text of `dataset/termsofuse.htm`) | annotations CC BY 4.0; the consortium does not own the images, "Use of the images must abide by the Flickr Terms of Use", and users "accept full responsibility" |

   Not usable until read: the 8 D-FINE checkpoints `*-obj365` and `*-obj2coco` of
   `ustc-community` (Objects365 terms not read; the hub tags `dataset: coco` on the `obj2coco`
   cards, but their README says Objects365 then COCO: the tag is not enough), the RT-DETRv2
   COCO+Objects365 weights of the upstream README, and the RT-DETR (v1) checkpoints
   `PekingU/rtdetr_*_coco_o365`. Not read at all: the data on which the backbones were
   pretrained before COCO (HGNetv2 for D-FINE, ResNet-vd for RT-DETRv2); the cards do not name
   it. The same question holds for the backbone of RF-DETR, which the entry of 2026-11-22 did not
   ask.
3. **Published values, not measured here** (COCO val2017 AP50:95; latency on one NVIDIA T4,
   batch 1, FP16; D-FINE with TensorRT 10.4.0, RT-DETRv2 with TensorRT 8.5.1 or newer, given as
   frames per second and converted here as 1000 / fps):

   | Checkpoint (size key) | Upstream model | AP | Latency, ms | Parameters (hub page) |
   |---|---|---|---|---|
   | dfine nano | D-FINE-N | 42.8 | 2.12 | 3.80 M |
   | dfine small | D-FINE-S | 48.5 | 3.49 | 10.35 M |
   | dfine medium | D-FINE-M | 52.3 | 5.62 | 19.64 M |
   | dfine large | D-FINE-L | 54.0 | 8.07 | 31.36 M |
   | dfine xlarge | D-FINE-X | 55.8 | 12.89 | 62.85 M |
   | rtdetrv2 r18vd | RT-DETRv2-S | 48.1 | 4.61 (217 fps) | 20.21 M |
   | rtdetrv2 r34vd | RT-DETRv2-M* | 49.9 | 6.21 (161 fps) | 31.47 M |
   | rtdetrv2 r50vd | RT-DETRv2-L (inferred) | 53.4 | 9.26 (108 fps) | 43.02 M |
   | rtdetrv2 r101vd | RT-DETRv2-X | 54.3 | 13.51 (74 fps) | 76.80 M |

   The card of `rtdetr_v2_r50vd` does not say which of the two upstream r50vd models it is
   (M 51.9 AP, 36 M parameters; L 53.4 AP, 42 M). Its configuration (encoder expansion 1.0, 6
   decoder layers) and its 43.0 M parameters fit L: inferred, not stated. The entry of
   2026-11-22 gave RT-DETRv2 M as 6.9 ms: that is the upstream r50vd-M, which has no checkpoint
   in transformers.
4. **The interface, read from the source of transformers 5.23.0** (tag v5.23.0, read on
   2026-11-23) and from the `config.json` and `preprocessor_config.json` of each of the 9
   checkpoints at its commit:
   - All 9 use `RTDetrImageProcessor`: resize to 640 x 640 pixels without keeping the aspect
     ratio (`size: {height, width}`), no padding, values times 1/255, no normalisation. In 5.23.0
     the processor runs on torchvision (`TorchvisionBackend`).
   - `post_process_object_detection(outputs, threshold, target_sizes, use_focal_loss=True)`:
     the corners from the centre format, times (width, height, width, height) of each target
     size given as (height, width); the logistic function of the logits, the best `queries`
     values over all pairs of query and class (`topk` on the flattened scores), class = index
     modulo the classes, and the scores above the threshold (strictly).
   - The output classes `DFineObjectDetectionOutput` and `RTDetrV2ObjectDetectionOutput` hold
     `logits` and `pred_boxes` (normalised centre x, centre y, width, height). Their docstring
     says `num_classes + 1` for the logits; every configuration has 80 labels and the focal
     loss, and the post-processing takes the last dimension as the classes: the script reads the
     shape and refuses one that does not fit `id2label`.
   - All 9 configurations: `num_queries` 300, `use_focal_loss` true, `id2label` 0 `person`,
     32 `sports ball`, `eval_size` and `anchor_image_size` null. With both null the models build
     their position embeddings and anchors for the size of the input (`modeling_d_fine.py`,
     `modeling_rt_detr_v2.py`), so another `input_size` runs; how well the weights do at
     another size is not known.
   - `from_pretrained(..., dtype=...)`: `torch_dtype` is deprecated in 5.23.0.
5. **What was built** (build machine, offline, stand-ins only):
   - Settings: the `name` of the section `detector` is the family: `rfdetr` (default, unchanged),
     `dfine` or `rtdetrv2` (`fallbacks.runs.DETECTOR_FAMILIES`). The two new families share
     `TransformersDetectorSettings`: `model` (a size of `TRANSFORMERS_MODELS`), `min_score`,
     `input_size` (320 to 1280, a multiple of 32), `batch` (1 to 64), `half`. One class serves
     RT-DETRv2 without an extra code path: the same processor, the same output, the same
     post-processing; only the checkpoint table and the model type differ. A settings file that
     names a family is laid over the defaults of that family (`config.DEFAULT_DETECTORS`: D-FINE
     Medium and RT-DETRv2 r18vd, 0.5, 640, batch 1, float32), because the families have other
     keys. The device stays an option of the script (`--device`), as for RF-DETR: in the
     settings it would tie the clip run to one machine, and the run facts name it.
   - Script (`tools/run_tool.py`, sub-commands `dfine` and `rtdetrv2`): lazy import of `torch` and
     `transformers`, the version check (`TRANSFORMERS_VERSIONS`), the checkpoint and its commit
     by size (`TRANSFORMERS_CHECKPOINTS`), safetensors only, float16 when `half` (refused on the
     device `cpu`), the checks of the configuration (model type of the family, focal loss, the
     classes `person` and `sports ball`). `detections_of_queries` converts the output in NumPy
     like the post-processing of the tool, from `min_score` on (the tool: above it), and keeps
     only `person` and `sports ball` (RF-DETR keeps every class; the package reads only these
     two). On the warm-up call the script compares its boxes with the post-processing of the
     installed tool and stops when they differ by more than 0.01 pixel or 1e-5 in score. The
     detections are matched by these tolerances, not by their order (item 9).
   - Time per frame: frames go to a tool in batches (`answer_frames(job, root, answer, batch)`;
     RF-DETR and PnLCalib through `one_at_a_time`, batch 1, as before). `seconds` of a frame is
     the whole call (image processor, model, conversion) divided by the frames of the batch:
     the same meaning as the `predict` time of RF-DETR. A transformers run adds
     `runner.stage_ms_per_frame` (median per frame of `preprocess`, `model` after a device
     synchronisation, `postprocess`) and prints it. The per-frame loop of `answer_frames` is the
     place for a line per frame of a live feed (session B): each entry is complete when it is
     appended.
   - Package: the clip run format does not change; `answers.detector_answers`, `tool-score` and
     `clip-job` take every family (`runs.is_detector`); `clip-job --tool <family>` refuses a family
     other than that of the settings file. The engine name is `detector:<family>:<size>`
     (`detector:rfdetr:RFDETRMedium` as before, `detector:dfine:medium`); `pipeline._engines` is
     the only place that writes it, and nothing parses it.
   - Experiment files of G2 with the frames of the five RF-DETR files (group `probe`, 5 frames
     per clip, seed 20261113), `min_score` 0.5, 640 pixels, batch 1, float32: D-FINE Small,
     Medium, Large and RT-DETRv2 r18vd, r50vd. They bracket RF-DETR Small to Large by published
     latency (3.5 to 6.8 ms): D-FINE 3.49, 5.62, 8.07 ms; RT-DETRv2 4.61, 9.26 ms (RT-DETRv2 has
     no size below 4.61 ms).
   - RUNBOOK section 11 (subsection "A second detector family"), the install proposal with
     pinned versions, all at least 7 days old on 2026-11-23: transformers 5.23.0 (2026-11-13),
     torch 2.15.0 (2026-10-28), torchvision 0.29.2 (2026-10-28, needs torch 2.14.0 or newer),
     huggingface-hub 2.4.0 (2026-10-29), safetensors 0.8.0 (2026-06-09), numpy 2.5.4
     (2026-10-10), pillow 13.0.0 (2026-10-16); dates of the PyPI pages. The requirement file of
     `locks/` is written on the run machine. Section 12 says how the plan applies to a fine-tune
     of these families.
6. **Measured on the build machine** (offline, stand-ins of `torch` and `transformers`):
   - `tests/test_detector_families.py`: 51 tests (the conversion by hand: centre format against
     corners, parts of the input against pixels on a frame of 200 x 100 pixels; the class filter
     and the limit; the best pairs; values that are not finite; the script with the stand-ins:
     checkpoint, commit, dtype, safetensors, input size, synchronisation, batches, the self-check,
     the refusals; the clip run read by the package; `tool-score` of a D-FINE run on the 5 probe
     frames of one synthetic clip: recall 1.0, precision 1.0; the settings, the defaults, the engine
     name, the experiment files, `clip-job`).
   - Each fix has a test that fails without it: 15 changes of the code were made one at a time
     and each made a test fail (width and height swapped in the scale; no limit on the pairs;
     strict limit; boxes read as corners; no class filter; the target size as (width, height);
     no compact rows for the new families; batch time not divided; no synchronisation; no
     family defaults in `load_config`; no family check in `clip-job`; `tool-score` only for
     RF-DETR; and the three fixes of the second read in item 9).
   - Test runs: see item 9.
7. **Assumptions that only the run machine can check** (each one is a line of RUNBOOK section 8
   to add after the first run):
   (a) `AutoImageProcessor.from_pretrained` and `AutoModelForObjectDetection.from_pretrained`
   take `revision=`, and the model takes `dtype=` and `use_safetensors=True` as in the source
   read; (b) the processor takes a list of NumPy arrays of shape (height, width, 3) in RGB and
   the keyword `size={"height": s, "width": s}` in its call, and gives `pixel_values` of shape
   (batch, 3, s, s); (c) the model takes `pixel_values` alone (no `pixel_mask` without padding)
   and gives `logits` of shape (batch, 300, 80) and `pred_boxes` (batch, 300, 4); (d)
   `post_process_object_detection` takes `target_sizes` as a list of (height, width) and a plain
   object with `logits` and `pred_boxes`; (e) the self-check passes on the first frame; (f)
   `torch.cuda.synchronize(device)` and `torch.mps.synchronize()` exist in torch 2.15.0; (g)
   half precision runs on the device of the run machine (`mps` or `cuda`); (h) the pinned
   versions install together for the Python and the platform of the run machine; (i) the hub
   still serves the 9 commits; (j) the time per frame: nothing here says how fast either family
   is on the run machine, and the published values are TensorRT, which the script does not use.
8. **Not verified.** No real model, no weights, no footage, no timing. The licence pages were
   read by one reader. The terms of Objects365, of the Flickr images of COCO and of the backbone
   pretraining data were not read.
9. **Second read** (a subagent of this session that was given the code, not these conclusions).
   It confirmed the conversion against the source of `post_process_object_detection` of
   5.23.0 (box format, the order (height, width), the best pairs, the threshold) and the batch
   loop (RF-DETR and PnLCalib unchanged at batch 1), and read the configuration of
   `dfine-medium-coco` and two of the 9 commits on the hub again. Its findings, each fixed with a
   test that fails without the fix:
   - Medium, confirmed: the self-check paired the detections by sorted order. Two scores that
     are equal in float32 (the tool) and differ in float64 (the script) sort differently, so a
     correct run could stop (reproduced with two logits one float32 step apart near 4.0). Fix:
     each detection of the tool is matched to its own detection of the script by the
     tolerances.
   - Low, confirmed: `detector: {name: [dfine]}` raised `TypeError` in `load_config` instead of a
     settings error. Fix: the family must be text before it names defaults.
   - Low, confirmed: the batch test could not see a warm-up call counted in the stage times (the
     median hid it). Fix: a test of one frame whose warm-up call is 100 times slower.
   - Low: one assertion compared a value with itself; a branch for an empty list of times could
     not run; an import out of order. Fixed.
   - Not changed: the script does not check the types of `batch` and `input_size` again (the
     package validates both, as for `min_score` of RF-DETR); a float32 score that rounds onto
     the threshold could change the count of the self-check (chance estimated at about 1e-5 per
     detection, not measured).
10. **Test runs** (build machine, `tools/run_tests.py`, 14 workers, after the fixes of item 9):
    default run 3,310 passed, 1 skipped, 9 deselected, exit 0, 111.5 s (load 28.8 at the start);
    full run (`-m "slow or not slow"`) 3,319 passed, 1 skipped, exit 0, 102.2 s (load 61.8). An
    earlier full run at load 50 had one failure in `tests/test_run_tests.py` (a worker
    watchdog of 1 s killed a second, passing worker as well); that test passed alone, and the
    file is not part of this change.
11. **Owner decisions, proposed.**
    - Install the transformers environment (RUNBOOK section 11) and run the five new files with
      the five RF-DETR files, `--max-clips 2` first. Recommendation: keep RF-DETR Medium as the
      default until G2 has numbers of both families on the same frames.
    - Licence of the training data: the COCO images carry the Flickr Terms of Use for every
      detector here, RF-DETR too; the backbone pretraining data is not named on any card.
      Recommendation: decide both with the licence of the repository (last list), before a
      demo is shown outside research.
    - Proposed text for rule 4 of `CLAUDE.md` (not applied): "Only `tools/run_tool.py` imports an
      external tool (RF-DETR, Hugging Face transformers for D-FINE and RT-DETRv2, PnLCalib). It
      runs in the environment of that tool, and the package never imports it."

## 2026-11-23 — A streaming path for the detector engine (session B of round 5)

1. **Goal and result.** The detector engine worked on a whole clip: several of its stages look at
   later frames, and `timing.py` gave a mean per stage only, so nothing could say whether it keeps
   up with a live feed. Built: `stream.py`, a path that takes one frame at a time (its detections,
   its camera, its picture) and gives the state of frame `t` once frame `t + L` is in, for a fixed
   delay `L`; the command `stream`; per-frame times (median, 95th percentile, largest). Everything
   here is measured on the build machine with synthetic clips and the stand-ins of the tools
   (`fallbacks.fake`): no real detector, no real calibration, no real footage, no pipe.
   **Result: the hypothesis holds for clips without a cut and with a hard cut from L = 5 on, and
   fails for L = 0 and for clips with a dissolve at every L up to 25.** Recommended delay: 25 frames
   (1 s at 25 fps), the default of `stream` now. The time per frame of the package without the tools, at L = 25: median 22.3 to 23.5 ms, 95th percentile 39.8 to 50.2 ms, largest 90 to 114 ms (3 runs, load average 22.5 to 30.6); the clip path takes a mean of 10.0 to 10.4 ms per frame on the same clip. Whether the stream keeps up with 25 fps on the run machine with the tools is not measured.
2. **Design** (contract: `docs/ARCHITECTURE.md`, section "`stream.py`").
   - On arrival of frame `t`, once: the camera motion and the cut tests of the pair `t - 1, t`
     (`motion.ShotMeter`); the camera of the calibration checked with `pipeline.fit_problems`, and
     without one the nearest earlier plausible camera moved with the motion; the people off the
     pitch dropped; the team key of each person from the kit clusters; the matching
     (`tracking.Associator`; a new one at each hard cut).
   - Decision of frame `f = t - L`: the steps of the clip path that look at a whole clip run on a
     window of frames `f - H` to `t`, with the same functions: cuts known at `t`, camera fill (the
     rule of `propagate_homographies` among the cameras up to `t`), `kit_roles`, the split of team
     swaps (when set), `stitch_fragments`, `finalize_tracks`, the fill of the people and of the ball.
     `H` = 123 frames by default: the longest join (3 s) plus the votes of a team-key change on its
     earlier side (48 frames).
   - A given state is never changed. A window track keeps the id of its newest given observation;
     two tracks that ask for one id get it in the order of how recently they had it. So a late
     join or a late split changes nothing that was given and costs no extra switch.
   - Kit clusters: fitted again on every frame of a warm-up of 25 frames, on all torso features so
     far, then fixed; a fit that does not show two kits is tried again on every frame, on the last
     25 frames, until one does. Fixed and not slowly updated: the kits do not change in a match, a
     refit can swap the names `A` and `B` of the two clusters, and a slow update drifts towards a
     mix of the kits when the light changes. The warm-up length was set on the fit seeds 100 to
     109 (plain clip, 250 frames, n = 10 per cell; team-side accuracy of the matched people, mean
     and lowest seed):

     | detector | L | warm-up 1 | 5 | 25 | 75 |
     |---|---|---|---|---|---|
     | exact | 0 | 0.963 (0.894) | 0.893 (0.358) | 0.973 (0.931) | 0.973 (0.931) |
     | exact | 12 | 0.982 (0.925) | 0.907 (0.424) | 0.990 (0.959) | 0.990 (0.959) |
     | noisy | 0 | 0.801 (0.101) | 0.780 (0.113) | 0.984 (0.931) | 0.984 (0.931) |
     | noisy | 12 | 0.812 (0.114) | 0.793 (0.125) | 0.993 (0.949) | 0.993 (0.950) |

     A fit on 1 to 5 frames of 12 to 14 people can pass the two-kit check with a wrong split and is
     then fixed for the whole clip (lowest seed 0.10). 25 and 75 frames gave the same numbers; 25
     is kept (load average 27 to 56 during the run).
   - Shared code, results of the clip path unchanged: `tracking.associate` runs the new
     `Associator`, `motion.measure_shots` runs the new `ShotMeter`, `fill.players_in_frame` gives
     one frame of `dense_frame_states`, `ball.track_ball` takes `first_frame`, `pipeline` has
     `ball_state` and `clip_kit_clusters` taken out of its functions. Proof: every existing test
     passes, and `tests/test_stream.py` checks each shared piece against the clip function
     (the matching with forgetting, the shot tests on a dissolve, every frame of the fill, the
     ball from a first frame) and the whole-clip control below.
3. **Input.** Rule 4 forbids importing a tool, so the stream reads lines (`STREAM_FORMAT =
   "pitchvlm-stream-1"`): a header line `{"format": "pitchvlm-stream-1", "clip": <clip id>,
   "detector": <detector settings of any family>, "calibration": <PnLCalib settings>}`, then one
   JSON object per line and per frame from frame 0, `{"frame": i, "detector": {"seconds": s,
   "payload": <the payload of a clip run frame>}, "calibration": {"seconds": s, "payload": <the
   payload of fallbacks.pnlcalib>}}`. `clip_run_lines` writes these lines from two clip runs and
   `read_stream_lines` reads them one at a time (a file, or a pipe with `--lines -`). Not built
   (session C owns `tools/run_tool.py`): a live mode of the script. Session C names the place: the
   batch loop of `answer_frames`, where each entry is complete when it is appended. What it must
   print, to its standard output only (log lines to the error stream), flushed after each line:
   the header line once, then one line per frame in frame order, with `seconds` as the script
   measures it now. The two tools run in two environments, so one process cannot fill both parts
   of a line: either each tool prints its own lines (`{"frame", "seconds", "payload"}` after a
   header with its settings) and a small merger, to build, zips the two pipes by frame, or the
   reader takes two pipes. The package reads the picture of each frame from `--clip`, so a live
   pipe also needs a frame source that gives the same frames as the tool reads (not built).
4. **Measurement, fixed before the report run.** Hypothesis: with `L` at most 25 frames the stream
   reaches the offline detector engine on identity switches, coverage, track ratio, role accuracy
   and team-side accuracy. Baseline: `run_clip` (detector engine, calibration engine, no number)
   on the same frames, detections and cameras. Controls: `L = 0`, and `L` and `H` of the whole
   clip (`clip`; and `clip-kits`, the same with the kit clusters of the clip path, which must give
   the result of the clip path frame for frame). Arms: `L0`, `L5`, `L12`, `L25` (with `H` = 123),
   `clip`, `clip-kits`. Clips (`evalx/streambench.py`): 250 frames (10 s); `plain` (no cut),
   `cut` (a hard cut at frame 125 to a moment 250 frames later), `dissolve` (a dissolve of 10
   frames from frame 120 to the same moment; the stand-ins answer with the truth of a hard cut at
   frame 125). A jump of 100 frames (4 s), as in the dissolve test of the video path, is not
   seen by the window test at frame 120 on this scene (lowest window value 0.73), so 250 frames
   were taken; at 500 frames the camera pan (period 16 s) is back near the same view. Detector:
   `exact` boxes, and `noisy` (each person missed with a chance of 0.1, box edges with 2 px of
   noise), fixed before the first run. Report seeds 200 to 209 (seeds 0 to 19 were used by earlier
   sessions, 100 to 109 for the warm-up above). Values: `evalx.metrics.identity_scores` against
   the synthetic truth, and the share of matched people (within 5 m, per frame) with the true role
   and the true team side. Command: `PYTHONPATH=src python -m pitchvlm.evalx.streambench --seeds
   200-209 --workers 6` on commit `25c09c0` plus the merge of `28f06ad` (load average 21 to 41).
5. **Measured, per arm: mean (lowest to highest seed), n = 10 seeds each.**

   | kind | detector | arm | cases | coverage | switches | track ratio | purity | role acc. | side acc. | equal to offline |
   |---|---|---|---|---|---|---|---|---|---|---|
   | plain | exact | offline | 10 | 0.964 (0.940 to 0.977) | 5.0 (4.0 to 7.0) | 1.213 (1.174 to 1.261) | 0.999 (0.990 to 1.000) | 0.977 (0.936 to 1.000) | 0.974 (0.932 to 1.000) | 10 of 10 |
   | plain | exact | L0 | 10 | 0.960 (0.948 to 0.970) | 4.7 (4.0 to 6.0) | 1.204 (1.174 to 1.261) | 1.000 (1.000 to 1.000) | 0.974 (0.910 to 0.999) | 0.967 (0.864 to 0.999) | 0 of 10 |
   | plain | exact | L5 | 10 | 0.968 (0.956 to 0.977) | 4.6 (4.0 to 6.0) | 1.200 (1.174 to 1.261) | 1.000 (1.000 to 1.000) | 0.979 (0.916 to 1.000) | 0.975 (0.884 to 1.000) | 0 of 10 |
   | plain | exact | L12 | 10 | 0.967 (0.956 to 0.977) | 4.7 (4.0 to 6.0) | 1.204 (1.174 to 1.261) | 1.000 (1.000 to 1.000) | 0.987 (0.923 to 1.000) | 0.986 (0.914 to 1.000) | 0 of 10 |
   | plain | exact | L25 | 10 | 0.967 (0.956 to 0.977) | 4.7 (4.0 to 6.0) | 1.204 (1.174 to 1.261) | 1.000 (1.000 to 1.000) | 0.996 (0.962 to 1.000) | 0.996 (0.959 to 1.000) | 0 of 10 |
   | plain | exact | clip | 10 | 0.967 (0.956 to 0.977) | 4.7 (4.0 to 6.0) | 1.204 (1.174 to 1.261) | 1.000 (1.000 to 1.000) | 0.996 (0.963 to 1.000) | 0.996 (0.961 to 1.000) | 5 of 10 |
   | plain | exact | clip-kits | 10 | 0.964 (0.940 to 0.977) | 5.0 (4.0 to 7.0) | 1.213 (1.174 to 1.261) | 0.999 (0.990 to 1.000) | 0.977 (0.936 to 1.000) | 0.974 (0.932 to 1.000) | 10 of 10 |
   | plain | noisy | offline | 10 | 0.907 (0.886 to 0.928) | 97.2 (47.0 to 162.0) | 1.457 (1.391 to 1.609) | 0.999 (0.994 to 1.000) | 1.000 (0.998 to 1.000) | 1.000 (0.998 to 1.000) | 10 of 10 |
   | plain | noisy | L0 | 10 | 0.809 (0.784 to 0.827) | 111.5 (63.0 to 158.0) | 1.461 (1.391 to 1.565) | 0.999 (0.993 to 1.000) | 0.989 (0.953 to 0.999) | 0.986 (0.935 to 0.999) | 0 of 10 |
   | plain | noisy | L5 | 10 | 0.908 (0.885 to 0.928) | 78.9 (48.0 to 105.0) | 1.543 (1.391 to 1.826) | 0.999 (0.994 to 1.000) | 0.992 (0.953 to 1.000) | 0.991 (0.944 to 1.000) | 0 of 10 |
   | plain | noisy | L12 | 10 | 0.908 (0.886 to 0.928) | 83.7 (48.0 to 117.0) | 1.561 (1.391 to 2.043) | 0.999 (0.994 to 1.000) | 0.996 (0.969 to 1.000) | 0.995 (0.967 to 1.000) | 0 of 10 |
   | plain | noisy | L25 | 10 | 0.907 (0.886 to 0.928) | 90.4 (47.0 to 132.0) | 1.457 (1.391 to 1.609) | 0.999 (0.994 to 1.000) | 1.000 (0.997 to 1.000) | 1.000 (0.997 to 1.000) | 0 of 10 |
   | plain | noisy | clip | 10 | 0.907 (0.886 to 0.928) | 97.2 (47.0 to 162.0) | 1.457 (1.391 to 1.609) | 0.999 (0.994 to 1.000) | 1.000 (0.998 to 1.000) | 1.000 (0.998 to 1.000) | 10 of 10 |
   | plain | noisy | clip-kits | 10 | 0.907 (0.886 to 0.928) | 97.2 (47.0 to 162.0) | 1.457 (1.391 to 1.609) | 0.999 (0.994 to 1.000) | 1.000 (0.998 to 1.000) | 1.000 (0.998 to 1.000) | 10 of 10 |
   | cut | exact | offline | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 10 of 10 |
   | cut | exact | L0 | 10 | 0.569 (0.560 to 0.577) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.973 (0.909 to 0.999) | 0.966 (0.866 to 0.999) | 0 of 10 |
   | cut | exact | L5 | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.977 (0.916 to 1.000) | 0.974 (0.886 to 1.000) | 0 of 10 |
   | cut | exact | L12 | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.985 (0.923 to 1.000) | 0.984 (0.914 to 1.000) | 4 of 10 |
   | cut | exact | L25 | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.994 (0.962 to 1.000) | 0.994 (0.959 to 1.000) | 7 of 10 |
   | cut | exact | clip | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.997 (0.970 to 1.000) | 0.997 (0.968 to 1.000) | 9 of 10 |
   | cut | exact | clip-kits | 10 | 0.577 (0.568 to 0.585) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 10 of 10 |
   | cut | noisy | offline | 10 | 0.557 (0.548 to 0.566) | 102.5 (40.0 to 168.0) | 2.287 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 10 of 10 |
   | cut | noisy | L0 | 10 | 0.497 (0.488 to 0.507) | 117.1 (48.0 to 167.0) | 2.287 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.986 (0.955 to 0.999) | 0.984 (0.938 to 0.999) | 0 of 10 |
   | cut | noisy | L5 | 10 | 0.557 (0.548 to 0.566) | 90.4 (40.0 to 125.0) | 2.396 (2.087 to 2.870) | 1.000 (1.000 to 1.000) | 0.989 (0.955 to 1.000) | 0.988 (0.947 to 1.000) | 0 of 10 |
   | cut | noisy | L12 | 10 | 0.557 (0.548 to 0.566) | 95.3 (40.0 to 135.0) | 2.391 (2.087 to 3.348) | 1.000 (1.000 to 1.000) | 0.993 (0.971 to 1.000) | 0.993 (0.969 to 1.000) | 0 of 10 |
   | cut | noisy | L25 | 10 | 0.557 (0.548 to 0.566) | 98.9 (40.0 to 148.0) | 2.287 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.997 (0.981 to 1.000) | 0.997 (0.979 to 1.000) | 0 of 10 |
   | cut | noisy | clip | 10 | 0.557 (0.548 to 0.566) | 102.5 (40.0 to 168.0) | 2.287 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 10 of 10 |
   | cut | noisy | clip-kits | 10 | 0.557 (0.548 to 0.566) | 102.5 (40.0 to 168.0) | 2.287 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 10 of 10 |
   | dissolve | exact | offline | 10 | 0.545 (0.535 to 0.552) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 10 of 10 |
   | dissolve | exact | L0 | 10 | 0.621 (0.591 to 0.650) | 19.5 (18.0 to 21.0) | 1.848 (1.783 to 1.913) | 1.000 (1.000 to 1.000) | 0.962 (0.898 to 0.990) | 0.954 (0.854 to 0.989) | 0 of 10 |
   | dissolve | exact | L5 | 10 | 0.649 (0.598 to 0.689) | 18.5 (17.0 to 21.0) | 1.804 (1.739 to 1.913) | 1.000 (1.000 to 1.000) | 0.975 (0.913 to 0.999) | 0.971 (0.883 to 0.999) | 0 of 10 |
   | dissolve | exact | L12 | 10 | 0.736 (0.698 to 0.790) | 14.1 (12.0 to 16.0) | 1.613 (1.522 to 1.696) | 1.000 (1.000 to 1.000) | 0.985 (0.924 to 1.000) | 0.984 (0.915 to 1.000) | 0 of 10 |
   | dissolve | exact | L25 | 10 | 0.572 (0.562 to 0.583) | 26.7 (24.0 to 28.0) | 2.161 (2.043 to 2.217) | 1.000 (1.000 to 1.000) | 0.994 (0.962 to 1.000) | 0.994 (0.959 to 1.000) | 0 of 10 |
   | dissolve | exact | clip | 10 | 0.545 (0.535 to 0.552) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 0.997 (0.970 to 1.000) | 0.997 (0.968 to 1.000) | 9 of 10 |
   | dissolve | exact | clip-kits | 10 | 0.545 (0.535 to 0.552) | 22.0 (22.0 to 22.0) | 1.957 (1.957 to 1.957) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 10 of 10 |
   | dissolve | noisy | offline | 10 | 0.527 (0.519 to 0.540) | 100.0 (39.0 to 164.0) | 2.278 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 10 of 10 |
   | dissolve | noisy | L0 | 10 | 0.537 (0.502 to 0.564) | 115.1 (45.0 to 165.0) | 2.187 (1.957 to 2.348) | 1.000 (1.000 to 1.000) | 0.977 (0.947 to 0.991) | 0.973 (0.929 to 0.991) | 0 of 10 |
   | dissolve | noisy | L5 | 10 | 0.650 (0.568 to 0.746) | 86.2 (31.0 to 119.0) | 2.191 (1.739 to 2.652) | 1.000 (1.000 to 1.000) | 0.986 (0.954 to 0.998) | 0.985 (0.946 to 0.998) | 0 of 10 |
   | dissolve | noisy | L12 | 10 | 0.685 (0.629 to 0.769) | 89.2 (30.0 to 128.0) | 2.104 (1.696 to 3.130) | 1.000 (1.000 to 1.000) | 0.993 (0.971 to 1.000) | 0.992 (0.969 to 1.000) | 0 of 10 |
   | dissolve | noisy | L25 | 10 | 0.552 (0.544 to 0.563) | 105.0 (44.0 to 155.0) | 2.543 (2.304 to 2.739) | 1.000 (1.000 to 1.000) | 0.997 (0.980 to 1.000) | 0.997 (0.978 to 1.000) | 0 of 10 |
   | dissolve | noisy | clip | 10 | 0.527 (0.519 to 0.540) | 100.6 (39.0 to 164.0) | 2.283 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 4 of 10 |
   | dissolve | noisy | clip-kits | 10 | 0.527 (0.519 to 0.540) | 100.6 (39.0 to 164.0) | 2.283 (2.087 to 2.435) | 1.000 (1.000 to 1.000) | 0.999 (0.992 to 1.000) | 0.999 (0.991 to 1.000) | 4 of 10 |

   Paired difference against `offline` on the same seed, mean and 95 % Student t interval
   ("all cases" when every seed gave the same difference):

   | kind | detector | arm | switches vs offline | coverage vs offline | track ratio vs offline | side acc. vs offline |
   |---|---|---|---|---|---|---|
   | plain | exact | L0 | -0.300 [-0.783, +0.183] | -0.004 [-0.010, +0.002] | -0.009 [-0.022, +0.004] | -0.007 [-0.037, +0.022] |
   | plain | exact | L5 | -0.400 [-0.900, +0.100] | +0.005 [-0.001, +0.011] | -0.013 [-0.028, +0.002] | +0.001 [-0.027, +0.029] |
   | plain | exact | L12 | -0.300 [-0.783, +0.183] | +0.004 [-0.002, +0.009] | -0.009 [-0.022, +0.004] | +0.012 [-0.013, +0.037] |
   | plain | exact | L25 | -0.300 [-0.783, +0.183] | +0.004 [-0.002, +0.009] | -0.009 [-0.022, +0.004] | +0.022 [+0.000, +0.043] |
   | plain | exact | clip | -0.300 [-0.783, +0.183] | +0.004 [-0.002, +0.009] | -0.009 [-0.022, +0.004] | +0.022 [+0.001, +0.043] |
   | plain | exact | clip-kits | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | plain | noisy | L0 | +14.300 [-0.611, +29.211] | -0.098 [-0.101, -0.094] | +0.004 [-0.013, +0.022] | -0.014 [-0.027, +0.000] |
   | plain | noisy | L5 | -18.300 [-33.161, -3.439] | +0.001 [-0.001, +0.003] | +0.087 [+0.007, +0.167] | -0.009 [-0.022, +0.003] |
   | plain | noisy | L12 | -13.500 [-24.106, -2.894] | +0.001 [-0.001, +0.003] | +0.104 [-0.053, +0.262] | -0.005 [-0.012, +0.003] |
   | plain | noisy | L25 | -6.800 [-13.585, -0.015] | +0.001 [-0.000, +0.001] | +0.000 (all cases) | -0.000 [-0.001, +0.000] |
   | plain | noisy | clip | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | plain | noisy | clip-kits | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | cut | exact | L0 | +0.000 (all cases) | -0.008 [-0.008, -0.008] | +0.000 (all cases) | -0.034 [-0.063, -0.005] |
   | cut | exact | L5 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.026 [-0.051, -0.002] |
   | cut | exact | L12 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.016 [-0.035, +0.003] |
   | cut | exact | L25 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.006 [-0.016, +0.004] |
   | cut | exact | clip | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.003 [-0.010, +0.004] |
   | cut | exact | clip-kits | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | cut | noisy | L0 | +14.600 [+3.808, +25.392] | -0.061 [-0.064, -0.058] | +0.000 (all cases) | -0.016 [-0.029, -0.002] |
   | cut | noisy | L5 | -12.100 [-24.547, +0.347] | +0.000 [-0.000, +0.000] | +0.109 [-0.023, +0.241] | -0.011 [-0.023, +0.001] |
   | cut | noisy | L12 | -7.200 [-15.842, +1.442] | +0.000 [-0.000, +0.000] | +0.104 [-0.132, +0.340] | -0.006 [-0.015, +0.002] |
   | cut | noisy | L25 | -3.600 [-8.501, +1.301] | +0.000 [-0.000, +0.000] | +0.000 (all cases) | -0.002 [-0.007, +0.002] |
   | cut | noisy | clip | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | cut | noisy | clip-kits | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | dissolve | exact | L0 | -2.500 [-3.108, -1.892] | +0.076 [+0.063, +0.089] | -0.109 [-0.135, -0.082] | -0.046 [-0.075, -0.017] |
   | dissolve | exact | L5 | -3.500 [-4.408, -2.592] | +0.104 [+0.086, +0.122] | -0.152 [-0.192, -0.113] | -0.029 [-0.054, -0.004] |
   | dissolve | exact | L12 | -7.900 [-8.937, -6.863] | +0.191 [+0.169, +0.213] | -0.343 [-0.389, -0.298] | -0.016 [-0.035, +0.002] |
   | dissolve | exact | L25 | +4.700 [+3.805, +5.595] | +0.027 [+0.026, +0.029] | +0.204 [+0.165, +0.243] | -0.006 [-0.016, +0.004] |
   | dissolve | exact | clip | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.003 [-0.010, +0.004] |
   | dissolve | exact | clip-kits | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
   | dissolve | noisy | L0 | +15.100 [+4.098, +26.102] | +0.010 [-0.003, +0.023] | -0.091 [-0.119, -0.064] | -0.026 [-0.040, -0.011] |
   | dissolve | noisy | L5 | -13.800 [-25.309, -2.291] | +0.123 [+0.089, +0.156] | -0.087 [-0.251, +0.077] | -0.014 [-0.026, -0.001] |
   | dissolve | noisy | L12 | -10.800 [-19.086, -2.514] | +0.158 [+0.131, +0.185] | -0.174 [-0.430, +0.082] | -0.007 [-0.015, +0.001] |
   | dissolve | noisy | L25 | +5.000 [+0.291, +9.709] | +0.025 [+0.024, +0.027] | +0.265 [+0.234, +0.296] | -0.002 [-0.007, +0.002] |
   | dissolve | noisy | clip | +0.600 [-0.529, +1.729] | +0.000 (all cases) | +0.004 [-0.005, +0.014] | +0.000 (all cases) |
   | dissolve | noisy | clip-kits | +0.600 [-0.529, +1.729] | +0.000 (all cases) | +0.004 [-0.005, +0.014] | +0.000 (all cases) |

6. **What the tables say.**
   - Control `clip-kits`: equal to the clip path frame for frame in all 40 cases without a dissolve
     and in 14 of 20 with a dissolve (exact 10 of 10, noisy 4 of 10). On the 6 noisy dissolve cases
     that differ, the switches differ by +0.6 on average (CI -0.5 to +1.7) and the coverage not at
     all. Inferred, not traced case by case: the matching learns the dissolve about 30 frames after it starts, so it links
     through the blend frames before the window cuts the fragments at the cut; the new shot then
     starts from fragments that the clip path never makes (it matches each shot on its own). This
     is the one difference that remains with the kit clusters given. A second difference was found
     by the second reader and fixed: a frame without a camera at arrival kept its people off the
     pitch, where the clip path takes the camera of a later frame and drops them; the window now
     drops them (test `test_the_control_drops_the_people_off_the_pitch_of_frames_that_had_no_camera_at_arrival`;
     before the fix 48 of 60 frames differed).
   - Control `clip` (kit warm-up instead of the clusters of the clip): the same as the clip path on
     noisy clips, and on exact clips the team side is right more often (plain +0.022, CI +0.001 to
     +0.043): a fit on the first 25 frames splits the kits better than a fit on all 250 frames in
     some seeds. Not a claim about real kits.
   - `L0` fails: noisy coverage -0.098 (plain) and -0.061 (cut), +14.3 and +14.6 switches. A track
     needs 2 observations (`min_observations`), so the first frame of every track and every frame
     after a miss, whose gap the fill closes only with a later observation, have no state.
   - `L5` to `L25` on `plain` and `cut`: coverage within +0.005 of the baseline, team side within
     0.03 (one interval excludes 0: cut, exact, L5, -0.026, CI -0.051 to -0.002), and fewer
     switches than the baseline on noisy clips (L5 -18.3, L12 -13.5, L25 -6.8 on plain). Fewer
     switches come with more ids at L5 and L12 (track ratio +0.087, CI +0.007 to +0.167, at L5 on
     plain; +0.104, CI -0.053 to +0.262, at L12): a join that the window sees late does not take
     back an id that was given, so a person keeps two ids where the clip path would join or swap
     them. At L25 the track ratio is that of the baseline in every case.
   - `dissolve`: the stream at L <= 25 is not the clip path. At L5 and L12 it carries the ids
     across the dissolve, which the clip path never does: coverage +0.10 to +0.19 and fewer ids,
     because the synthetic truth keeps its ids across the shot change and the synthetic people of
     both moments match by place (purity 1.000). On real footage a dissolve leads mostly to a
     replay or another camera, where such links are wrong. At L25 the cut is known for part of the
     frames: +4.7 switches and track ratio +0.20 (exact). The window test knows a dissolve only
     when its run of low windows has ended (about `DISSOLVE_WINDOW_FRAMES` = 30 frames after it
     starts; the test bounds it by 42 frames on seed 0). An arm added after the report run, not fixed before it
     (same seeds 200 to 209, dissolve clips only, a scratch script over `run_case`, load average 26 to
     34): with L = 50 frames the stream equals the baseline on the dissolve clips: switches +0.000
     (exact, all cases; noisy CI -1.6 to +1.6), coverage +0.000, track ratio +0.000 (exact) and
     +0.004 (noisy, CI -0.005 to +0.014), team side -0.005 (CI -0.012 to +0.003) and -0.001. So the
     gap is the lag of the window test, not the window of the decision.
   - Hypothesis: holds on `plain` and `cut` from L = 5 (L25 closest to the baseline); fails at
     L = 0; fails on `dissolve` for every L up to 25.
7. **Time per frame** (`python -m pitchvlm.evalx.streambench --time --seeds 200`: plain clip, exact
   boxes, 250 frames, 3 rounds, the order of the arms turned in each round, images drawn before the
   runs; build machine while other sessions ran, load average given per line). The stream: median
   / 95th percentile / largest, milliseconds per frame. The clip path has no frame samples: its
   mean per frame.

   | arm | round (order) | load | total | decide | motion | paint |
   |---|---|---|---|---|---|---|
   | offline | 1 | 23.2 | mean 10.35 | - | mean 7.93 | mean 1.10 |
   | offline | 2 | 23.8 | mean 9.98 | - | mean 7.62 | mean 1.05 |
   | offline | 3 | 23.8 | mean 10.07 | - | mean 7.49 | mean 1.07 |
   | L0 | 1 (forward) | 23.2 | 15.82/26.82/32.95 | 7.26/8.63/24.72 | 7.47/8.37/8.78 | 0.00/11.79/13.08 |
   | L0 | 2 (reverse) | 25.2 | 16.61/27.77/30.69 | 7.80/9.45/11.12 | 7.66/8.65/9.44 | 0.00/12.07/12.58 |
   | L0 | 3 (forward) | 22.3 | 24.57/73.07/137.12 | 9.04/28.95/108.45 | 11.46/32.46/102.10 | 0.00/14.55/35.58 |
   | L5 | 1 (forward) | 22.0 | 17.62/29.71/39.62 | 8.35/12.56/25.97 | 7.61/9.80/17.59 | 0.00/12.49/16.71 |
   | L5 | 2 (reverse) | 26.9 | 18.36/29.85/34.94 | 8.88/11.29/14.38 | 7.74/10.00/12.66 | 0.00/12.58/14.86 |
   | L5 | 3 (forward) | 24.6 | 25.24/67.87/164.24 | 11.31/28.75/91.63 | 10.24/30.12/141.67 | 0.00/15.68/92.90 |
   | L12 | 1 (forward) | 21.0 | 20.80/47.92/156.38 | 9.62/20.60/136.17 | 8.99/26.94/55.85 | 0.00/14.05/57.24 |
   | L12 | 2 (reverse) | 28.0 | 20.56/32.76/40.08 | 10.48/14.74/16.56 | 8.15/11.28/24.79 | 0.00/13.50/20.22 |
   | L12 | 3 (forward) | 28.9 | 20.55/37.74/63.68 | 9.90/14.59/29.78 | 8.75/18.66/50.33 | 0.00/13.49/35.31 |
   | L25 | 1 (forward) | 22.5 | 22.26/45.02/90.41 | 10.35/19.01/34.99 | 9.68/23.77/56.06 | 0.00/14.01/35.30 |
   | L25 | 2 (reverse) | 27.3 | 23.54/50.17/114.43 | 10.47/20.72/103.28 | 10.15/23.46/61.39 | 0.00/14.54/27.99 |
   | L25 | 3 (forward) | 30.6 | 22.80/39.84/104.03 | 10.30/20.35/93.92 | 9.27/15.78/32.94 | 0.00/13.88/25.73 |

   These times hold no detector and no calibration (the stand-ins take no time), and the load
   average of the machine was 21.0 to 30.6 (1 minute, 14 cores, other sessions running). Reading, with the times as measured (n = 3 runs per arm, 250 frames each):
   - The median of the whole frame is 15.8 to 25.2 ms at every delay (40 to 63 frames per second);
     at L = 25 it is 22.3 to 23.5 ms. The decision costs 7.3 to 11.3 ms at the median: the window
     of 123 + L frames is run again for each frame. The stream costs about twice the mean of the
     clip path per frame.
   - The 95th percentile is above 40 ms in 5 of 12 runs (L0 round 3, L5 round 3, L12 round 1, L25
     rounds 1 and 2), and the largest frame takes 31 to 164 ms. The rounds of one arm differ by up
     to a factor of 2.7 at the 95th percentile, so the stalls follow the load of the machine more
     than the delay. A claim of real time needs the run machine without other load.
   - The paint score of every 12th frame (a diagnostic) costs 11.8 to 15.7 ms at the 95th
     percentile: it alone sets most of the 95th percentile of the frame. It could run on another
     thread or be left out of a live run (owner decision, item 10).
8. **Not verified.** A real detector, a real calibration, real frames, a live pipe, the run machine;
   the kit warm-up on real kits (stripes, numbers, shade, a first frame that is a close-up or a
   replay); the warm-up against a match whose first 25 frames show few people; the time of the
   tools on the run machine together with the stream; a dissolve on real footage; the number step
   (it works on whole tracks and does not stream). The second reader read the code (findings in
   item 9); the tables were read by one session.
9. **Second read** (one subagent, before the commit): 1 major (item 6, the camera of a frame
   without one at arrival), 2 minor (the forgetting of a short window dropped what the matching
   needs when `H + L + max_gap_frames` is below the time a track stays live, fixed: the state of
   the matching is kept for that time; this entry was missing), 4 small (`--history abc` gave a
   traceback, fixed to a usage error; line numbers of a stream skipped blank lines, fixed; `track_ball`
   on 0 frames raised, fixed; a frame `1.0` passed, refused now). Each fix has a test. Refuted by the
   reader: the matching, the shot tests and the fill equal the old code on its own scripts; no
   causal leak; ids unique per frame. Open small points: the `--lines` path checks the clip id but
   not the SHA-256 of the video (a line has none), and `--frames`, `--seed` and `--fps` are ignored
   where they do not apply.
10. **Proposed owner decisions.**
    - The default delay of `stream`: 25 frames (1 s), as recommended above; 5 or 12 frames are
      closer to live with about one more id per 10 people per 10 s on noisy boxes.
    - The dissolve: either accept that a dissolve within the delay is linked through (the stream
      of a live match sees few dissolves outside replays), or a delay of at least the lag of the
      window test (about 42 frames) for clips with dissolves, or a shorter window of the dissolve
      test for the stream (a new measurement).
    - The paint score of the keyframes in a live run: keep it (it sets most of the 95th percentile,
      item 7), move it to another thread, or leave it out of `stream`.
    - Whether to build the live mode of `tools/run_tool.py` (item 3) and the merger of the two tool
      pipes. No text of `CLAUDE.md` is proposed: rule 4 stands as it is (the stream reads lines and
      imports no tool).

## 2026-11-23 — A planted-bug check of the money guards, and a second read of the test runner

1. **Question.** Which small bugs in the money code (rule 6 of `CLAUDE.md`: estimate first,
   measured cost after, a run stops at 1.3 times its estimate, the doctor at 2 USD, hard stop at
   135 USD) pass every offline test? In round 4 one bug did: a default hard stop of 136 USD (fault
   P05 of the entry "The offline tests in several processes"). No hypothesis: the measure is the
   share of planted bugs that the tests find, and the list of the others.

2. **Method.** 89 bugs were written down, one line each, before any test file of the money code
   was read (a list kept in the scratch folder of the session, not committed; one later edit fixed
   the indentation of the old text of M22, the bug itself unchanged). Each bug was planted alone in
   a git clone of `main` 3587a58, and the money test files ran with `tools/run_tests.py -j 12`:
   `test_vlm_cost.py`, `test_vlm_client.py`, `test_vlm_openrouter.py`, `test_vlm_cache.py`,
   `test_cli.py`, `test_pipeline.py`, `test_probes.py`, `test_fake_backend.py` (973 tests). A bug
   is killed when the run ends with exit code 1. Control: the clone without a bug, 973 passed,
   exit 0. Every survivor ran again on the whole default run (3,254 tests): it survived there too.
   Then each survivor ran with the new tests of this entry. Scope: `vlm/cost.py` (guard, ledger,
   prices, estimates: 42 bugs, M01 to M42), the spend window and the price checks of
   `vlm/client.py` (12, K01 to K12), the start of a run, the cost meter and the session of
   `cli.py` (26, L01 to L26), the spend check and the plan of `doctor.py` (4, D01 to D04), the cost
   report of `vlm/cache.py` (4, H01 to H04), and the stop after a failure in `pipeline.py` (1, P01).

3. **Result.** 83 of 89 killed by the old tests; the 6 others are killed by 5 new tests (each
   survivor: exactly 1 failed test with the new tests). By kind:

   | Kind | Planted | Killed, old tests | Killed, with the new tests |
   |---|---|---|---|
   | removed check | 25 | 24 | 25 |
   | constant changed | 13 | 11 | 13 |
   | boundary (`>` against `>=`, plus one cent) | 13 | 10 | 13 |
   | estimate part missing | 10 | 10 | 10 |
   | removed call | 7 | 7 | 7 |
   | skipped part of a sum | 5 | 5 | 5 |
   | swapped (min/max, sign, factor) | 5 | 5 | 5 |
   | wrong run id | 4 | 4 | 4 |
   | skipped ledger write | 4 | 4 | 4 |
   | check after the call, not before | 2 | 2 | 2 |
   | estimate part counted twice | 1 | 1 | 1 |
   | **All** | **89** | **83** | **89** |

   The survivors, why each matters, and the test that kills it now:

   | Bug | Change | Rule 6? | New test |
   |---|---|---|---|
   | M01 | default hard stop of `BudgetGuard` 136 USD | the default only: the command line takes 135 from `configs/gates.yaml`, and a test pins that file | `test_default_hard_stop_stops_at_exactly_135_usd_and_not_one_cent_before` |
   | M02 | default hard stop 135.01 USD | as M01 | the same test |
   | M39 | estimate of a number request: 40 output tokens, not 400 | yes: the printed estimate of a clip falls from 2.3448 to 2.1648 USD, and the run limit with it | `test_default_token_estimates_give_the_costs_that_the_runbook_names` |
   | K02 | the spend window ignores a wait of 1 s or less | the limit of 8 USD per 10 minutes, not rule 6: a call goes out while the window is full | `test_spend_limiter_waits_also_when_the_window_has_room_in_less_than_a_second` |
   | K10 | a negative cost reported by OpenRouter is read as a cost | yes: it lowers the ledger total that the hard stop reads (the spend window refuses it, with a `ValueError` and no ledger line) | a case of `test_an_answer_without_a_cost_is_a_call_whose_cost_is_unknown` |
   | L02 | a run starts when the ledger and its estimate pass the hard stop by up to 1 cent | weak: the guard still stops at 135 during the run | half a cent over the hard stop in `test_run_whose_estimate_fits_exactly_under_the_hard_stop_starts` |

   The test of M39 does not pin the default token table (the module says why it is not pinned).
   It ties the table to the two costs that `docs/RUNBOOK.md` names for the owner before a paid run
   (about 2.3 USD for a clip, about 0.009 USD for one number request, gemini-4-argon at the November
   price): a new table must change those two numbers too. 18 of the 83 bugs killed by the old tests
   are killed by exactly one test (M16, M18, M26, M27, M28, M36, M37, K04, K07, K12, L10, L18, L19,
   L20, L24, D01, H03, H04): no gap, but thin. A second reader planted each survivor again in a
   copy, confirmed that the new test fails for it and passes on the real code, and found no float
   trap (134.99 plus 0.01 is 135.0 exactly in the ledger: 67,495,000 and 5,000 tokens at 2 USD per
   million, `math.fsum`).

4. **Which side of "at 135" stops.** The guard stops when the ledger total reaches the hard stop
   (`total >= 135`): 134.99 USD goes on, 135.00 USD stops. This agrees with rule 6 ("Hard stop at
   135 USD in total") and with the code's own text. Two other limits do not agree with the words
   of rule 6; the code is not changed here:
   - The run limit stops when a run spent **more than** 1.3 times its estimate (`spent > limit`,
     and the meter's `paid + stored > limit`); a run at exactly 1.3 times goes on. Rule 6 says "A
     run stops at 1.3x its estimate", and so does the list of rules in `docs/RUNBOOK.md`; the
     code, the budget line ("this run stops above"), `ARCHITECTURE.md` and
     `test_run_stops_above_1_3_times_its_estimate` say "above".
   - The doctor stops when it spent more than 2 USD, and its spend check passes at 2 USD or less.
     Rule 6 says it "stops at the spend limit of gate G0 (2 USD)"; `configs/gates.yaml` says "it
     stops above".
   The difference is one exact amount of money, and a stop comes after the answer that crosses a
   limit, so it costs at most nothing in practice. Owner decision (proposed text for rule 6, the
   behaviour stays): "6. Money: every paid run prints its estimate first and the measured cost
   after. A run stops when it costs more than 1.3x its estimate. The contract checks of `doctor`
   are the one exception: each doctor run stops when it costs more than the spend limit of gate G0
   (2 USD). Hard stop when all runs together reach 135 USD. Do not start a paid run without the
   owner's go-ahead."

5. **Second read of `tools/run_tests.py`.** Each finding was reproduced on a small project in a
   temporary folder, fixed, and given a test that fails without the fix:
   - **R1. A test that fails and then has an error in its teardown made the run untrusted.**
     pytest writes two JUnit entries for it (the failure, then the error). The runner counted two
     results for one id: "ran 2 tests of its 1", "-1 not run", exit code 2, where one process ends
     with exit code 1. Now entries with the same id inside one worker are one result (the error
     wins, the texts are joined). A test that runs in two workers still fails the id check.
     Test: `test_a_test_that_fails_and_then_errors_in_teardown_counts_once`.
   - **R2. A file that skips itself at collection disappeared from the totals.** A module-level
     `pytest.importorskip` gives no test id, so the runner said "0 skipped" where one process says
     "1 skipped". The collection now runs with `-rs`; the summary line adds "1 file skipped at
     collection" and a line names the files. The exit code does not change (it is 0 in both).
     This matters on the run machine, where a missing package can skip a whole file. Test:
     `test_a_file_skipped_at_collection_is_counted`.
   - **R3. A Ctrl-C while the workers start left a worker running.** A `KeyboardInterrupt` inside
     `subprocess.Popen` comes after the process was made but before the runner holds it, so the
     clean-up does not know it, and the run still prints "every worker was stopped". Measured: 12
     test files, `-j 12`, SIGINT at a random time of 0 to 30 ms after the collection line: a
     worker was left in 20 of 25 runs before the fix, 0 of 25 after. The runner now holds the stop
     signals while it makes each worker (`_SignalsHeld`: Python handlers only, so the workers keep
     their signal mask) and raises the held signal right after. Test:
     `test_ctrl_c_while_the_workers_start_stops_the_worker_that_just_started`.
   - Repeated Ctrl-C during the clean-up (2 to 4 signals, gaps of 0 to 1 s, 7 patterns): no
     process left, exit code 130 each time.
   - Refuted, no change: a worker that ends early with exit code 0 (`pytest.exit(returncode=0)`)
     makes the run untrusted (exit code 2), where one process passes; parametrised ids with spaces,
     `::`, `/`, `[`, non-ASCII or an empty value, and two equal ids, run and match; stored test
     times of tests that no longer exist only weigh the plan, they never select a test; a
     selection that leaves no test ends the collection with pytest's exit code 5, and the run with
     exit code 2; a hook that deselects by position gives the same ids, or a collection count that
     the worker check refuses.
   - The 120 s silence limit against a slow test: the slowest stored test time of the full run is
     51.0 s, measured at a high load. A worker is killed only after 60 s of silence (pytest's
     `faulthandler` prints the stack of the test then, which counts as output) plus 120 s, so a
     single test must run about 180 s without output: 3.5 times the slowest one.

6. **Flaky tests of the runner.** The runner's own tests ran 20 times in one process and 20 times
   with the runner (33 tests each; load average 18 to 61, other sessions running). The `-x` test,
   its order test and the Ctrl-C test failed 0 times in 40 runs. One test failed: in one process 2
   of 20 runs, with the runner 11 of 20 runs, always
   `test_a_silent_worker_is_killed_and_named_with_its_file`. Its silence limit of 1 s was shorter
   than the start of pytest on the loaded machine: both workers were killed ("no test started"),
   not only the silent one. The same failure was reported by session C in a full run at a load
   average near 50. The runner did what the limit said; the test was too tight. It now uses a
   limit of 10 s with 2 workers, so it still shows that the healthy worker survives; it takes
   about 15 s.

7. **To check on the run machine (Linux; not possible here, read only).**
   - `python tools/run_tests.py tests/test_run_tests.py`: the process group kill, the Ctrl-C test
     and the new start test on Linux. The `PermissionError` branch of `_signal_group` is a macOS
     case; on Linux `killpg` of a group of exiting processes succeeds.
   - `os.cpu_count()` counts every processor of the host, not the share of a container or a CPU
     quota: compare `nproc` with `python -c "import os; print(os.cpu_count())"`, and pass `-j`
     when they differ.
   - The size of the system's temporary folder (`df -h /tmp`): every worker writes its pytest
     folders there.
   - The pytest version: `pyproject.toml` allows `pytest>=8.0`; the JUnit names, the collection
     output and the `-rs` lines are read in the format of pytest 9.2. Another version fails the run
     with exit code 2 (it cannot pass a wrong one).
   - A full run with the runner and one in one process: the same totals.

8. **Not verified.** The kill test of R3 forces the signal inside `Popen` by a stand-in of
   `Popen`; the race itself was measured with real signals (item 5). No run on Linux.

9. **The video money paths of `main` af02f53** (native video input, merged during this work).
   16 more bugs were written down from the code of af02f53 before any of its video tests was read
   (one edit before the first run: the quoting of V16), planted one at a time in a clone of the
   merge commit of this branch, and each ran with the whole default run (3,381 tests; control
   without a bug: 3,381 passed, 1 skipped, exit 0). 13 of 16 were killed by the tests of af02f53;
   3 new tests kill the 3 others (each: exactly 1 failed test).

   | Bug | Change | Old tests |
   |---|---|---|
   | V01 | sampled frames of a video part rounded down, not up | killed |
   | V02 | one sampled frame fewer | killed |
   | V03 | sound tokens not counted | killed |
   | V04 | a Gemini frame at high resolution counted at 70 tokens, not 280 | killed |
   | V05 | default sampling rate of Gemini 0.5 frames per second, not 1 | killed |
   | V06 | the tier factor not applied to the video part of a Gemini estimate | killed |
   | V07 | the output of `people_video` counted once, not once per target frame | killed |
   | V08 | the video part of an OpenRouter request not estimated | killed |
   | V09 | OpenRouter sound priced at the input price, not at its own price | killed |
   | V10 | OpenRouter video priced at the input price, not at its own price | killed |
   | V11 | Gemini sends a video for a model without a video rate | killed |
   | V12 | a range or a sampling rate sent to a provider that reads whole files | killed |
   | V13 | the estimate of the doctor leaves out its video request | **survived** |
   | V14 | the estimate of a session (the video probe) leaves out its built video requests | killed |
   | V15 | a new video rate holds from the day after its start day | **survived** |
   | V16 | the doctor looks up the video rate of every model in the Gemini table | **survived** |

   - V13 matters for rule 6: the doctor printed 0.097 USD where its checks with video cost
     0.144 USD by the estimate (one Gemini model; the video request alone 0.048 USD), and the
     doctor refuses to start only when its estimate is above 2 USD. Killed by new lines in
     `test_doctor_video_check_with_a_fake_model`: the printed estimate must hold the video request.
   - V15 matters on the day a new rate starts (no table has a second dated rate today): the
     estimate would use the old rate for one day. Test:
     `test_a_new_video_rate_holds_from_its_first_day`.
   - V16 drops the video check of an OpenRouter model that has a video rate in the price file
     (it spends less, but the check is silently not done). Test:
     `test_doctor_plans_a_video_request_for_an_openrouter_model_from_the_rates_of_the_price_file`.

10. **Owner decisions.**
   - The text of rule 6 (item 4).
   - `docs/RUNBOOK.md` names "about 2.3 USD for a clip of 30 seconds", at the November price of
     gemini-4-argon. From 1 December 2026 that model costs twice as much, so the printed estimate
     of a clip will be about 4.7 USD (2.3448 x 2 = 4.6896, computed, not measured). The new test of
     M39 is fixed to a November day, so it does not fail then; the runbook sentence should say
     both prices.

## 2026-11-23 — Second read of the video input and the detector families (session A of round 6)

1. **What and how.** An independent second read of two pieces of round 5: the video input of the
   model steps (entry "Video input for the model steps") and the second detector family (entry
   "A second detector family: D-FINE and RT-DETRv2 through transformers"). The reader did not
   write them. Method: list findings, reproduce each one with a failing test or a script,
   confirm it with a second reader, fix it with a test, and undo each fix once to see its test
   fail. Build machine only: no model call, no key, no footage, no install, no weights. Public
   pages were read with `curl` on 2026-11-23 (published, not measured).
2. **Findings.** No major defect was found in either piece. Ranks: major (wrong numbers, money
   or privacy at risk on a normal run), minor (wrong on a plausible edge), small (wrong only on
   input that no pinned file has).

   | Id | Rank | Finding | Reproduction (build machine) | Status | Test |
   |---|---|---|---|---|---|
   | V1 | minor | The 20,000,000-byte limit of a request with a video counted only the base64 inline data. The video page of Gemini names the "total request size". A request with its inline data just under the limit and a prompt, schema and settings on top was sent. | inline data 100 bytes under a lowered limit, 1,000 characters of text: no refusal | fixed: `client.request_size_bytes` (the whole request as JSON text) for both providers; a request of exactly the limit is sent, one byte more is refused | `test_size_limit_counts_the_whole_request_and_not_only_its_inline_data` |
   | V2 | minor | `refuse_video_folder` compared the video folder and the cache folder as text. On a disk that does not tell letter cases apart (the default disk of macOS, the build machine and a likely run machine), `--video-dir <data>/CACHE/videos` with the cache `<data>/cache` passed, so the MP4 files would go into the cache folder, which goes to the text-only results branch (rule 2). | `refuse_video_folder(tmp/"CACHE", tmp/"cache")`: no refusal | fixed: `cli._lies_in` compares folders that exist as places on the disk (`os.path.samefile`), and paths that do not exist yet as text without letter case (this refuses more than it must on a disk that tells cases apart, never less) | `test_probe_refuses_a_video_folder_in_the_cache_typed_in_another_letter_case` |
   | V3 | minor | A video written again from the same frames can get other bytes: same file name (the name holds the pixel hash), other SHA-256, so another cache key. A paid session then pays the request again without an error, where an image request stops with `OtherImagesError`. The docstring of `VLMRequest.cache_key` said the opposite ("its hash does not change with a library version"); `docs/RUNBOOK.md` already says not to delete the video folder between a paid run and its replay. | 3 frames of the contract scene written with `avc1`, removed, written with `mp4v`: same name, other hash (1 trial; written again with `avc1`: same hash) | written as a limit; the docstring now says it. `--replay` refuses a miss, so a replay costs nothing. Owner decision below | none (no code change) |
   | D1 | small | `detections_of_queries` checked only that the classes and the names are as many. With names at the places 1 to 80 (the COCO ids start at 1), column 0 among the best pairs above the limit stopped the script with a bare `KeyError: 0` (the second reader reproduced it at the limit 0.001); the other columns took the name that the tool also gives them. All 9 pinned configurations have the places 0 to 79 (read again, item 4), so no pinned file is hit. | names `{1: person, 2: car, 3: sports ball}`: no `RunnerError` | fixed: the places must be 0 to classes minus 1, else `RunnerError` with a line that says why | `test_class_places_that_are_not_0_to_classes_are_refused` |
   | D2 | minor | The warm-up self-check compared the script with `post_process_object_detection` of the tool only above `min_score`. A first frame with no detection above it (a replay graphic, a black frame) gave an empty comparison, and a changed tool passed it. | stand-in tool that shifts its boxes by 0.5 pixel, first frame with a best score of 0.3: exit 0, run file written | fixed: the check compares from `SELF_CHECK_MIN_SCORE` = 0.001 whatever `min_score`, and the run facts count the compared detections (`self_check_detections`; the runbook says that 0 means that it compared nothing) | `test_self_check_also_compares_a_first_frame_without_a_detection_above_the_limit`, `test_self_check_counts_the_detections_that_it_compared` |
   | N1 | minor | Found by the second reader in the fix of D2. From 0.001 the cut to the best 300 pairs often decides. `torch.topk` gives no order for equal scores, the script sorts stably, so at the cut the two sides can keep other tied pairs and a correct run stops. Half precision gives many ties: the second reader simulated 2,000 frames of 300 x 80 logits (normal(-9, 2), person column + normal(3, 2)): ties at the cut in 893 frames in float16, 266 of them (about 13 %) with a person or a ball; 0 in float32. Not measured on a real model; the five files of G2 run in float32. | stand-in tool that breaks ties from the last place: the run stopped | fixed: when the tool kept as many pairs as queries, the check compares only the pairs more than `SELF_CHECK_SCORE` above the lowest kept score (`self_check_floor`) | `test_self_check_leaves_out_the_pairs_that_tie_at_the_cut_to_the_best_pairs` |

   Each fix was undone once: each of its tests then failed (V1, V2, D1: "DID NOT RAISE"; D2:
   exit 0 against 2, and another count; N1: exit 2 against 0).
3. **Checked and found right** (read, and run where a check could run here):
   - Rule 2 on the files that a fake run really writes: `probe people` with
     `people_probe_video_batch.yaml`, the fake model, one synthetic clip. Written: `ledger.jsonl`,
     2 cassettes, 1 metrics file. No file holds the temporary path, the word `.mp4`, a base64 run
     of 200 characters or more, or a box of the labels; the cassettes hold the SHA-256 of each
     video (the identity, by design). Error texts of the video paths name the type of an
     `OSError` and a 12-digit hash prefix, never a path. The standard output names the video
     folder, as it names the cache folder and the ledger.
   - Cache key: the identity of a video part holds the hash, the range, the rate, the
     resolution and the length; a people request also names the hash in its item; the replay of
     the CLI test hits all 6 answers.
   - Money: a model without a video rate is refused before any call from the probe (the estimate
     raises, `CliError`), the doctor (no request, a failed line) and both backends
     (`VideoInputError` before the price check and the call); every date is the date in UTC; a
     video rate holds from its first day (inclusive); the doctor estimate includes the video
     request, and the meter of the doctor stops the run above 2 USD, counting the checks before
     the video check (a call that starts under the limit can end above it, as for every call of
     the doctor).
   - Time: frame `k` of a written file is at `k / fps`; the prompt writes times with 2 decimals,
     at most 0.005 s from the true time, against a tolerance of half a frame interval (0.0208 s at
     the largest rate, 24); the mapping goes by place in the file, so a clip rate that is not 25
     (step `round(clip fps / video fps)`) and a range after zero (no step sends one) do not move
     it.
   - Rule 5: `store=False` and no sampling parameter for Gemini, the private routing for
     OpenRouter (the privacy check leaves the video parts out); no upload anywhere; both video
     prompts ask for kit colours and shirt numbers and say not to use faces or names.
   - Detector: the box conversion (item 5), class mapping by name from the configuration, the
     last short batch, the score limit (the script keeps a score equal to `min_score`, the tool
     drops it; the package keeps it), half precision (scores and boxes go to float32 before the
     conversion), the pinned revisions of both the processor and the model, safetensors only,
     no remote code; a settings file that names a family and one value takes the defaults of
     that family; an unknown family or a size of another family is a settings error; the engine
     name is written in one place and parsed nowhere.
4. **Licences, read again on 2026-11-23** (this reader, `curl`, the pinned commits of the script).
   Every row of the table of the round-5 entry matches:
   - `LICENSE` of transformers at tag v5.23.0: Apache License 2.0. `LICENSE` of D-FINE at
     `956d1709` and of RT-DETR at `df2a96c8`: Apache License 2.0. The D-FINE README: checkpoints
     trained or pretrained on Objects365 "should not be assumed to be commercially cleared"
     (`*_obj365`, `*_obj2coco`); the RT-DETRv2 README lists COCO+Objects365 weights apart.
   - The 9 checkpoints at the 9 commits of `TRANSFORMERS_CHECKPOINTS` (hub API and README at the
     commit; the commit of the API answer equals the pinned one for all 9): licence `apache-2.0`,
     dataset `coco`, `train2017` named, no word of Objects365, one weight file
     `model.safetensors`, 80 classes at the places 0 to 79 with `person` at 0 and `sports ball`
     at 32, 300 queries, focal loss, `eval_size` and `anchor_image_size` null.
   - COCO terms (`dataset/termsofuse.htm`): annotations under CC BY 4.0; the consortium does not
     own the copyright of the images; use "must abide by the Flickr Terms of Use"; users accept
     full responsibility.
   - New, not in the round-5 table: the D-FINE configuration `configs/dfine/include/dfine_hgnetv2.yml`
     at `956d1709` sets `pretrained: True` for its backbone HGNetv2, and
     `src/nn/backbone/hgnetv2.py` downloads `PPHGNetV2_B*_stage1.pth` from a release of the
     author; neither file names the data of that pretraining. That the hub checkpoints were
     trained from this configuration is inferred, not stated on their cards.
5. **Differential check of the box conversion.** A NumPy reference of
   `post_process_object_detection` (transformers v5.23.0, `RTDetrImageProcessor`) was written by a
   reader who had not seen the script, from the public source of the tag and the configuration
   files of two checkpoints. It is `independent_post_process` in
   `tests/test_detector_families.py`. Its facts: the corners from the centre format, scaled by
   (width, height, width, height) of a target size given as (height, width), no clipping; the
   logistic function; `topk` with k = queries over the flattened pairs; class = place modulo
   classes; scores above the threshold, strictly; both checkpoints resize to 640 x 640 without
   keeping the aspect ratio and without padding (`do_pad: false`), so the plain scale is right.
   Measured (seed 300, which no earlier entry used; 1,000 random outputs: frame sizes from 1 x 1
   to 3840 x 2160, square and not, 1 to 300 queries, 80 classes, limits 0 to 0.9, centres on the
   edges and corners in 1 of 5 outputs):

   | Measure | Value |
   |---|---|
   | Detections compared | 41,361 |
   | Detections with a box that reaches outside the frame | 38,911 |
   | Detections within 0.01 above their limit | 362 |
   | Outputs in which the cut to the best pairs removes a pair above the limit | 732 |
   | Outputs whose detections differ (classes, order, count) | 0 |
   | Largest box difference | 0.0 pixels |
   | Largest score difference | 1.1e-16 |

   A first version of the test drew scores whose best pairs all lay far above the limit: a
   limit moved by 0.01 then passed it. The scores of the person, the ball and one other class
   are now spread over 0 to 1. Six planted bugs were tried once each against the final test,
   and each made it fail: width and height swapped in the scale, boxes read as corners, no cut
   to the best pairs, query and class places swapped, and the limit moved up or down by 0.01.
   The test runs in about 2 s.
6. **Limits written, not fixed.**
   - V3 (above).
   - The fake model takes every video request: it does not refuse a model without a video rate,
     a range for OpenRouter or a request above the size limit, it never opens the file, and its
     usage is the same for every video length (1,200 input tokens for `people_video`). The real
     backends refuse before any call, so no money is at risk, but a fake run cannot show that a
     setting would be refused, and the 1.3 times stop is never tested here against video tokens
     that grow with the frames.
   - The five detector files of G2 use `min_score` 0.5 for every family. The score of each
     family has its own calibration, so the recall of two families at one limit compares
     limits as well as detectors. Rule 7 fixes the limit before the run, which is right; the
     report of G2 should name the limit next to each recall.
7. **Not verified (only the run machine can):** the size of a real request on the wire against
   its JSON text (the SDK adds its own fields); the output of a real checkpoint and of the real
   `post_process_object_detection`; how often a score of the tool (float32) and the same score of
   the script (float64) fall on two sides of 0.001 and stop a correct run (the same question as
   for `min_score` in the round-5 entry, item 9; not measured, expected to be rare because both
   come from the same float32 logits); the letter-case behaviour of the disk of the run machine.
8. **Owner decisions (proposed).**
   - V3: keep the written videos with the data and run every replay with `--replay` (no code
     change), or put the pixel hash of the frames in the key in the place of the file hash and
     stop a hit whose file has other bytes, as for images (a change of the key of video
     requests; no video answer is stored yet, so no answer would be lost). Recommendation: the
     second, before the first paid video probe.
   - No text of `CLAUDE.md` is proposed.
9. **Second read** (a subagent given the diff and the list of findings, run on a copy of the
   commit before the fixes): V1, V2, V3, D2 confirmed with the tests failing before the fixes and
   passing after them; D1 confirmed, with the description corrected above (a bare `KeyError`,
   not a wrong name); it probed the fixed `refuse_video_folder` with a link to the cache, a cache
   reached through a link, `..`, a relative path in capitals and a cache that does not exist yet
   (all refused) and a sibling, `cache2` and a sibling that does not exist yet (all taken). New:
   N1 (above, fixed); two sentences of the docs that still said "inline data" (fixed); blank
   lines (fixed). It found that the first version of the differential test did not see a limit
   moved to `<=` (a test of round 5 does); the final version sees a limit moved by 0.01 either
   way (item 5).
10. **Test runs** (build machine, `tools/run_tests.py`, 14 workers, after every fix):
    default run 3,417 passed, 1 skipped, 9 deselected, exit 0, 58.3 s (load 7.6 at the start);
    full run (`-m "slow or not slow"`) 3,426 passed, 1 skipped, exit 0, 52.3 s (load 32.0).

## 2026-11-23 — The live feed of the detector engine (session B of round 6)

1. **Goal and result.** Round 5 built a stream that decides frame `t` once frame `t + L` is in,
   but it read finished clip runs: no tool wrote a line per frame, nothing joined the feeds of the
   two tools, and two results stood against real time (dissolves known about 30 frames late; the
   95th percentile of the frame time above 40 ms on a loaded machine). Built: a live mode of
   `tools/run_tool.py`, a merger of the two tool feeds (`feed.py`, command `merge-feeds`), a queue
   with back-pressure in `stream --lines -`, the paint score on its own thread, an early dissolve
   test, a latency log, a bench of the whole chain as real processes (`evalx/livebench.py`), and
   the fixes of an independent second read of `stream.py`. Everything is measured on the build
   machine with synthetic clips and stand-ins of the tools; nothing ran with a real tool.
   **Results:** (a) dissolve hypothesis **holds** on the held-out seeds: with the early test the
   stream at L = 25 equals the offline engine on 10- and 20-frame dissolves (exact boxes: every
   paired difference 0; noisy boxes: every interval holds 0), where without it +4.3 switches;
   (b) paint hypothesis **fails**: off the frame path the 95th percentile drops by about 5 to 10 ms
   at a load average of 12 to 17, but under a load of 12 spinning processes it stays above 40 ms
   in 2 of 3 runs (`main` in 3 of 3); (c) the chain of four processes gives, frame for frame, the
   result of the clip-run path; (d) latency: table in item 8.
2. **Live mode of the tool script** (`tools/run_tool.py`, new class `LiveFeed`, function
   `run_live`, a hook in the per-frame loop `answer_frames`; the transformers adapter functions
   are untouched). `--feed FILE|-` (a clip job only) writes the tool feed: a header
   `{"format": "pitchvlm-tool-feed-1", "clip", "n_frames", "video_sha256", "job_sha256", "tool"}`,
   then one line per frame, written and flushed as soon as its batch is answered,
   `{"frame", "seconds", "arrived_at", "payload"}` (the compact detector payload of a clip run).
   `--out` gets a copy of the feed instead of a clip run, and `<stem>-runner.json` the facts of
   the run (those of a clip run, with `self_check_detections` of a transformers detector, added
   at the merge of PR 21). `--pace FPS` reads frame `i` not before
   the start plus `i / FPS` (a recorded clip played as a camera sends it); `--start-at` gives two
   tools one start (seconds since the epoch). `--skip-behind` (pace, batches of 1) takes the
   newest frame that has come and writes `{"frame", "arrived_at", "skipped": true}` for each frame
   it leaves out. The script still imports neither the package nor a tool at module level and
   takes the API keys out of its environment first (rule 4). A test checks that the line of frame
   `i` is in the feed before the tool gets frame `i + 1`.
3. **The merger** (`feed.FeedMerger`, `feed.merge_feeds`, `merge-feeds --detector --calibration
   [--wait 1] [--queue 250]`). Policy, as written in the module docstring and tested case by case
   (`tests/test_feed.py`, 25 tests):
   - The stream header goes out when both headers are in; another clip, number of frames, video
     or tool stops the merger before any line (exit 2).
   - Frames go out in order, each once, as soon as both parts are in. The wait is bounded: the
     clock of frame `n` starts when a line of frame `n` or later first comes from either tool (not
     before both headers); after `--wait` seconds a part that is not in goes out as
     `{"missing": "late"}`. Each frame has its own clock, so a stalled tool delays the stream by
     one wait, not one wait per frame (test: a stall of 2 frames costs 1.0 and 1.0 s from 12.0 and
     12.1 s, not 2 s).
   - A slower tool: its parts are taken while they come within the wait. A tool that dies (its
     pipe closes): its parts go out as `ended` with no wait; when both feeds have ended the stream
     stops early and `stream` says so. A skipped frame goes out as `skipped` with no wait.
   - Lines out of order wait for their frame. A line of a frame that went out (`late lines`), a
     second line of a frame (`repeated`), a line that is not a frame line or names a frame outside
     the clip (`broken`: its frame waits as if it had not come), and a last line without its line
     end (`cut-off`) are dropped and counted. Report on the error stream; exit 1 when anything was
     missing or dropped.
   - Frames are decided one at a time, and the lines that came while the merger wrote a line are
     applied before the next frame is decided, so a slow reader makes no part late. After the last
     frame a feed that is still open is read to its end, so the pipe of a slow tool does not break.
   - `stream` reads a missing detector part as a frame without people and a missing calibration
     part as a frame without a camera (the camera then comes from the nearest earlier one, moved
     with the motion, as in round 5). The format of the stream lines is now `pitchvlm-stream-2`:
     the header names `n_frames` and `video_sha256`, a part may be `{"missing": reason}`, and a
     part may carry `arrived_at`.
4. **Back-pressure policy: delay, never drop.** Every frame carries a step of the camera motion,
   of the cut tests and of the matching, so a frame cannot be left out of the stream without
   breaking them. `stream --lines` reads on a thread into a queue of `--queue` lines (default 50,
   2 s at 25 fps; `stream.QueuedLines`), and the merger reads each feed into a queue of 250 lines.
   A full queue stops the reading of its pipe; the pipe fills and its writer waits. So a slow
   stream slows the merger and then the tools; nothing is dropped; the delay shows in the latency
   log (the reader thread notes when each line was read, so the time in the queue counts). Where
   dropping is wanted, it is at the source: `--skip-behind` of a tool that cannot keep the pace
   (the published time of PnLCalib is 439 ms per frame, item 8); the stream then gets frames
   without a camera. Tests: a reader of 3 lines stops a producer of 20 after 4 lines and gives all
   20 in order; a merged writer of 20 ms per line with queues of 2 lines gives 30 of 30 frames
   whole with a wait of 50 ms (no part declared late while its tool was held back). Not built:
   deciding several waiting frames on one window to catch up (it would make the result depend
   on the timing).
5. **The paint score off the frame path** (`stream._PaintWorker`): one thread; a frame waits only
   when 4 scores are open (`PAINT_MAX_PENDING`; test with a held score: the push of keyframe 48
   waits while 0, 12, 24 and 36 are open). Equal results: the control (`--delay clip`) still
   equals the clip path in its paint scores (existing test), and a test compares each score with
   `paint_alignment` on the same frame. Measurement (fixed before the run): hypothesis "the 95th
   percentile of the frame time stays under 40 ms at a load where `main` is above"; baseline
   `main` 0c2ac24 (a `git archive` of that commit); the same bench call in both trees
   (`streambench.time_lines(250, 600, (25,), 1)`: plain clip, seed 600, exact boxes, L = 25, the
   default window); order main, branch, branch, main, main, branch per block; block 1 as the
   machine was, block 2 with 12 extra processes that spin (14 cores). Milliseconds per frame,
   median / 95th percentile / largest, n = 250 frames per run:

   | block | arm | run | load before | total | paint stage | decide | motion |
   |---|---|---|---|---|---|---|---|
   | as is | main | 1 | 17.3 | 18.5 / 29.6 / 34.3 | 0.00 / 12.4 / 14.0 | 9.5 / 13.5 / 24.9 | 7.6 / 8.4 / 8.9 |
   | as is | branch | 1 | 16.1 | 18.1 / 23.1 / 37.2 | 0.00 / 0.02 / 0.18 | 9.6 / 13.7 / 29.3 | 7.5 / 8.8 / 11.3 |
   | as is | branch | 2 | 14.7 | 18.3 / 25.5 / 37.6 | 0.00 / 0.02 / 0.11 | 9.6 / 15.8 / 29.2 | 7.6 / 9.1 / 13.4 |
   | as is | main | 2 | 15.1 | 17.7 / 29.3 / 35.1 | 0.00 / 12.3 / 13.7 | 9.0 / 13.1 / 25.0 | 7.5 / 8.4 / 9.7 |
   | as is | main | 3 | 14.3 | 18.5 / 30.5 / 44.4 | 0.00 / 12.6 / 16.9 | 9.3 / 16.1 / 36.3 | 7.7 / 9.7 / 12.9 |
   | as is | branch | 3 | 13.5 | 16.9 / 19.2 / 34.1 | 0.00 / 0.02 / 0.11 | 8.4 / 10.3 / 26.0 | 7.3 / 8.7 / 11.0 |
   | loaded | main | 1 | 13.2 | 23.9 / 44.3 / 57.7 | 0.00 / 20.3 / 27.6 | 9.1 / 15.9 / 40.1 | 12.8 / 17.6 / 38.5 |
   | loaded | branch | 1 | 15.3 | 23.7 / 41.2 / 79.4 | 0.00 / 0.04 / 1.72 | 9.2 / 21.3 / 57.3 | 12.9 / 22.0 / 31.7 |
   | loaded | branch | 2 | 16.1 | 27.0 / 45.3 / 136.5 | 0.00 / 0.04 / 5.73 | 9.4 / 23.0 / 109.3 | 14.5 / 24.8 / 46.2 |
   | loaded | main | 2 | 21.2 | 24.3 / 44.8 / 134.0 | 0.00 / 21.6 / 37.7 | 9.1 / 16.7 / 27.2 | 12.6 / 18.9 / 87.4 |
   | loaded | main | 3 | 24.7 | 24.3 / 47.4 / 73.3 | 0.00 / 21.2 / 27.9 | 9.2 / 19.7 / 45.4 | 12.5 / 17.6 / 43.5 |
   | loaded | branch | 3 | 24.6 | 24.7 / 39.6 / 63.2 | 0.00 / 0.04 / 0.21 | 9.4 / 20.2 / 50.2 | 13.0 / 20.5 / 27.7 |

   Reading (n = 3 runs per arm and block, so weak): as is, the 95th percentile is 29.3 to 30.5 ms
   on `main` and 19.2 to 25.5 ms here, all under 40; loaded, `main` is above 40 in 3 of 3 runs
   (44.3 to 47.4) and this branch in 2 of 3 (39.6 to 45.3, mean 42.0 against 45.5). **The
   hypothesis fails.** The paint stage of the frame falls from about 12 (as is) and 21 ms (loaded)
   at its 95th percentile to 0.04 ms, but under load the score still takes a core, and the motion
   and the decision grow (their 95th percentiles 17.6 to 24.8 and 15.9 to 23.0 ms). The largest
   frame is not lower (63 to 136 ms against 57 to 134 ms). The load average of the second block
   rose from 13 to 25 during the block (other sessions ran too), so the two arms did not see the
   same load: the order turned, but the load is not controlled. What would help under load: the
   decision runs the window of 123 + L frames again for every frame (item 7 of the entry of round
   5); this was not changed here.
6. **Dissolves: an early test.** `motion.ShotMeter.open_transition(K)`: while a run of low window
   values is open, `transition_span` is taken with the newest judged frame of the run as the
   anchor of the new shot; while that anchor is still a blend, the end of the span moves with it.
   The span is taken once it came out the same for `K` judged windows in a row and its new shot
   starts at or before the anchor; it is kept until the run ends, and then the final span of the
   window test replaces it. `measure_shots` never calls it: the offline result is unchanged (the
   demo predictions of both engines are byte-equal to `main`; the existing dissolve tests pass).
   `StreamSettings.early_dissolve_windows` = 4 (`None` = the round-5 behaviour).
   - Fit of `K` (seeds 300 to 309, clips of `tools/synthetic_diagnostics.py dissolve-early-fit`:
     dissolves of 5 to 25 frames between two shots of 40 frames, the 8 controls of the entry "A
     dissolve is a transition, not a shot" with pans up to 100 px per frame, zoom, shake, goal
     kick, 100 frames without a cut; 10 clips per case). Rule fixed before the held-out run: the
     smallest `K` whose early span has the start of the final span and a stop within 1 frame of
     it, known at most 25 frames after the start, for dissolves of up to 15 frames, with no early
     transition on a control. `K` = 1, 2, 3 fail the stop (by up to 6, 3 and 2 frames on 15-frame
     dissolves); `K` = 4: stops 0 to -1, known 9 to 22 frames after the start (the final span: 31
     to 38). Controls: 0 early transitions in 80 clips at every `K` (no control has a low window).
   - Held-out seeds 400 to 409 (`dissolve-early-held-out`, run once after the fit; `K` = 4):

     | case | clips | taken early | equal to the final span | early minus final (start; stop) | known after the start, frames | final known after the start |
     |---|---|---|---|---|---|---|
     | dissolve, 5 frames | 10 | 10 | 8 | 0; -2 to 0 | 9 to 14 | 31 to 33 |
     | dissolve, 10 frames | 10 | 10 | 6 | 0; -2 to 0 | 13 to 17 | 32 to 36 |
     | dissolve, 15 frames | 10 | 10 | 6 | 0; -4 to 0 | 13 to 21 | 32 to 38 |
     | dissolve, 20 frames | 10 | 9 | 2 | 0; -1 to 0 | 24 to 25 | 37 to 41 |
     | dissolve, 25 frames | 10 | 10 | 4 | 0; -2 to 0 | 28 to 33 | 37 to 44 |
     | 8 controls | 80 | 0 | - | - | - | - |

     The stop rule of the fit (within 1 frame) does not hold on the held-out seeds: the early span
     ends up to 4 frames before the final one, on blend frames with a weight below about 0.15 (the
     offline test itself leaves such faint ends out, entry of 2026-11-22). Inferred: these frames
     get states of the new shot under the early span and none under the final span. Statement:
     a dissolve of up to 15 frames is known within 21 frames of its start (n = 30 held-out clips),
     one of 20 frames within 25 (9 of 10 taken early), one of 25 frames within 33; so the stream
     needs a delay of at least the length of the dissolve plus about 8 to 10 frames, and at least
     35 frames for dissolves of 25 frames.
   - The stream against the offline engine (`streambench`, held-out seeds 400 to 409, 250 frames,
     kinds plain, cut, dissolve (10 frames) and dissolve-20 (20 frames, new), detectors exact and
     noisy, n = 10 per cell; arms offline, L5, L25 (early test), L25-late (without it), L50;
     command `PYTHONPATH=src python -m pitchvlm.evalx.streambench --seeds 400-409 --workers 6
     --kinds plain,cut,dissolve,dissolve-20 --arms offline,L5,L25,L25-late,L50`, load average 13
     to 30). Criterion fixed before the run: on dissolve clips the paired differences of L25 to
     offline in switches, coverage and track ratio are 0 in every case with exact boxes, and their
     95 % intervals hold 0 with noisy boxes. Paired difference to offline, mean [95 % t interval]:

     | kind | detector | arm | switches | coverage | track ratio | side acc. |
     |---|---|---|---|---|---|---|
     | dissolve | exact | L25 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.001 [-0.002, +0.000] |
     | dissolve | exact | L25-late | +4.300 [+3.621, +4.979] | +0.028 [+0.027, +0.029] | +0.187 [+0.157, +0.216] | -0.001 [-0.002, +0.000] |
     | dissolve | noisy | L25 | -4.200 [-9.941, +1.541] | -0.000 [-0.000, +0.000] | +0.000 (all cases) | +0.000 [-0.002, +0.003] |
     | dissolve | noisy | L25-late | +2.400 [-3.920, +8.720] | +0.037 [+0.014, +0.060] | +0.200 [+0.091, +0.309] | +0.000 [-0.002, +0.003] |
     | dissolve-20 | exact | L25 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | -0.001 [-0.002, +0.000] |
     | dissolve-20 | exact | L25-late | +3.900 [+3.113, +4.687] | +0.049 [+0.047, +0.050] | +0.170 [+0.135, +0.204] | -0.001 [-0.002, +0.000] |
     | dissolve-20 | noisy | L25 | +2.900 [-3.449, +9.249] | -0.001 [-0.002, +0.000] | +0.009 [-0.004, +0.022] | -0.001 [-0.002, +0.000] |
     | dissolve-20 | noisy | L25-late | +11.000 [+4.837, +17.163] | +0.044 [+0.042, +0.046] | +0.261 [+0.215, +0.307] | -0.001 [-0.002, +0.000] |
     | dissolve | exact | L5 | +5.200 [+1.073, +9.327] | +0.040 [+0.023, +0.057] | +0.226 [+0.047, +0.406] | -0.020 [-0.034, -0.005] |
     | dissolve | exact | L50 | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) | +0.000 (all cases) |
     | plain | noisy | L25 | -6.400 [-11.347, -1.453] | +0.000 [-0.001, +0.001] | +0.000 (all cases) | -0.000 [-0.002, +0.001] |
     | cut | noisy | L25 | -5.600 [-11.847, +0.647] | +0.000 [-0.000, +0.001] | +0.000 (all cases) | +0.000 [-0.002, +0.003] |

     **The hypothesis holds** on these seeds for dissolves of 10 and 20 frames. On plain and cut
     clips L25 and L25-late are equal in every value (the early test changes nothing without a
     dissolve: no false cut). L25 on noisy plain clips has fewer switches than offline (-6.4, as
     in round 5: a late join does not take back a given id). The full tables (all arms, means and
     spreads) are in the output of the command; the clip-level cut count on the controls is item 6
     above (0 of 80).
7. **Second read of `stream.py` as a whole** (one independent subagent on a copy of 0c2ac24, every
   finding with a reproduction it ran; repro scripts not kept in the repository):
   - Minor, fixed: the whole-clip control differed from the clip path when the first frame of a
     shot had no camera: at arrival that frame was matched without pitch positions, while the
     clip path takes the camera of a later frame (cut clip, 60 frames, only frame 30 without a
     camera, miss 0.3 and 4 px noise: 29 or 30 of 60 frames differed on 4 of 4 seeds). Fix: with a
     history of the whole clip, `finish` matches the whole clip again with the cameras of the
     whole clip (people of blend frames and people off the pitch dropped as `run_clip` does).
     After the fix: 0 of 60 frames differ in all 12 cases of the reproduction; test
     `test_the_control_matches_the_first_frame_of_a_shot_without_a_camera_with_a_later_camera`.
   - Small, fixed: an infinite tool time passed the reader and broke `result.json` at the end
     (now refused with its line number, also NaN and infinity anywhere in a line); a deeply
     nested line raised `RecursionError` past the command (now "no JSON" with its line number); a
     ball candidate of another frame was accepted by `push` and failed 5 frames later naming the
     wrong frame, leaving the stream broken (now refused at its push, before any change). Each has
     a test.
   - Refuted by the reader (by running): the forgetting of `_forget` (96 combinations against a
     stream that keeps everything: identical), the control on cuts at the clip edges and on plain
     clips with half the cameras dropped, `_StreamIds` (no duplicate id), `_PitchMemo`, `finish`
     with a delay longer than the clip, and 12 more kinds of wrong lines.
   - Second read of this session's own work: item 10.
8. **End-to-end latency** (`PYTHONPATH=src python -m pitchvlm.evalx.livebench --runs 3 --delays
   5,25`): four processes as on the run machine (two `tools/stand_in_tool.py` in live mode with
   `--pace 25` and one `--start-at`, `merge-feeds` between two named pipes, `stream --lines -`
   through a pipe), a synthetic clip of 250 frames (seed 500) written as PNG files. The tools are
   stand-ins that answer from a file after an added delay per frame. **The delays are a
   simulation, taken from published values, not measured:** `fast` 4.4 ms per frame for both
   tools (RF-DETR Medium, T4, TensorRT FP16, entry of 2026-11-22); `rfdetr-small-m4` 20.0 ms for
   both (RF-DETR Small, plain `predict`, M4 Max, entry of 2026-11-14); `pnlcalib-paper` 20.0 ms
   for the detector and 439 ms for the calibration (the PnLCalib paper, other hardware and
   settings, entry of 2026-11-14), with `--skip-behind` for the calibration because 439 ms is
   above the 40 ms of a frame. Milliseconds, median / 95th percentile / largest over the frames
   decided while the stream ran (n = 250 - L each):

   | profile (simulated ms: detector, calibration) | L | run | load | exit codes | missing parts | arrival to decision | arrival of f + L to decision | line of f + L to decision | stream frame total | largest queue |
   |---|---|---|---|---|---|---|---|---|---|---|
   | fast (4.4, 4.4) | 5 | 1 | 3.3 | all 0 | none | 241 / 244 / 247 | 41 / 44 / 47 | 20 / 22 / 26 | 20 / 22 / 25 | 0 |
   | fast (4.4, 4.4) | 5 | 2 | 3.0 | all 0 | none | 241 / 244 / 246 | 41 / 44 / 46 | 20 / 22 / 24 | 20 / 21 / 23 | 0 |
   | fast (4.4, 4.4) | 5 | 3 | 3.2 | all 0 | none | 241 / 244 / 246 | 41 / 44 / 46 | 20 / 22 / 23 | 20 / 22 / 23 | 0 |
   | fast (4.4, 4.4) | 25 | 1 | 3.5 | all 0 | none | 1042 / 1045 / 1047 | 42 / 45 / 47 | 21 / 23 / 24 | 20 / 22 / 24 | 0 |
   | fast (4.4, 4.4) | 25 | 2 | 3.0 | all 0 | none | 1042 / 1045 / 1047 | 42 / 45 / 47 | 21 / 23 / 25 | 20 / 22 / 24 | 0 |
   | fast (4.4, 4.4) | 25 | 3 | 2.9 | all 0 | none | 1042 / 1046 / 1048 | 42 / 46 / 48 | 21 / 23 / 24 | 21 / 23 / 24 | 0 |
   | rfdetr-small-m4 (20, 20) | 5 | 1 | 3.1 | all 0 | none | 258 / 261 / 264 | 58 / 61 / 64 | 21 / 22 / 24 | 20 / 22 / 24 | 0 |
   | rfdetr-small-m4 (20, 20) | 5 | 2 | 2.9 | all 0 | none | 258 / 262 / 278 | 58 / 62 / 78 | 21 / 23 / 32 | 20 / 22 / 31 | 0 |
   | rfdetr-small-m4 (20, 20) | 5 | 3 | 3.0 | all 0 | none | 258 / 262 / 265 | 58 / 62 / 65 | 21 / 23 / 25 | 21 / 22 / 25 | 0 |
   | rfdetr-small-m4 (20, 20) | 25 | 1 | 3.6 | all 0 | none | 1059 / 1063 / 1065 | 59 / 63 / 65 | 22 / 24 / 25 | 21 / 23 / 25 | 0 |
   | rfdetr-small-m4 (20, 20) | 25 | 2 | 2.8 | all 0 | none | 1059 / 1064 / 1068 | 59 / 64 / 68 | 22 / 23 / 24 | 21 / 23 / 24 | 0 |
   | rfdetr-small-m4 (20, 20) | 25 | 3 | 3.1 | all 0 | none | 1059 / 1063 / 1066 | 59 / 63 / 66 | 22 / 24 / 25 | 21 / 23 / 25 | 0 |
   | pnlcalib-paper (20, 439) | 5 | 1 | 3.4 | merge-feeds 1, others 0 | calibration 227 skipped | 646 / 708 / 725 | 446 / 508 / 525 | 157 / 335 / 381 | 30 / 33 / 48 | 10 |
   | pnlcalib-paper (20, 439) | 5 | 2 | 2.9 | merge-feeds 1, others 0 | calibration 227 skipped | 648 / 710 / 733 | 448 / 510 / 533 | 157 / 332 / 394 | 29 / 33 / 44 | 10 |
   | pnlcalib-paper (20, 439) | 5 | 3 | 2.9 | merge-feeds 1, others 0 | calibration 227 skipped | 648 / 710 / 726 | 448 / 510 / 526 | 157 / 337 / 393 | 29 / 33 / 45 | 10 |
   | pnlcalib-paper (20, 439) | 25 | 1 | 2.6 | merge-feeds 1, others 0 | calibration 227 skipped | 1461 / 1514 / 1559 | 461 / 514 / 559 | 168 / 364 / 406 | 31 / 34 / 47 | 10 |
   | pnlcalib-paper (20, 439) | 25 | 2 | 3.3 | merge-feeds 1, others 0 | calibration 227 skipped | 1461 / 1511 / 1550 | 461 / 511 / 550 | 168 / 363 / 411 | 31 / 35 / 48 | 10 |
   | pnlcalib-paper (20, 439) | 25 | 3 | 2.5 | merge-feeds 1, others 0 | calibration 227 skipped | 1467 / 1514 / 1531 | 467 / 514 / 531 | 173 / 374 / 443 | 32 / 36 / 59 | 10 |

   Reading (3 runs per cell, load average 2.5 to 3.6, the machine otherwise quiet; the order of
   the 6 arms turned each run): every run ended with exit code 0 for the stream and the tools,
   and no part was late. With tools of 4.4 and 20 ms the time from the arrival of frame
   `f + L` to the decision of `f` is 41 to 59 ms at the median and at most 78 ms; the stream
   alone (from the reading of the line) 20 to 22 ms at the median and at most 32 ms, and the queue
   held no line: the stream kept the pace. So a decision comes `L` / 25 s plus about 45 to 60 ms
   after its frame came (241 to 262 ms at L = 5, 1042 to 1064 ms at L = 25). The gap of about 20
   ms between the arrival of a frame and its line at the stream is the reading of the PNG file by
   the tool, its time, the merger and the pipes (not split further). With a calibration of 439 ms
   per frame and `--skip-behind`, the calibration answers 23 of 250 frames (227 skipped) and every
   other frame gets the camera moved with the motion; the merger holds each frame until the
   calibration line of a frame at or after it came, so the decision comes 446 to 467 ms (median)
   and up to 559 ms after frame `f + L` came, the lines come in bursts (largest queue 10 lines),
   and `merge-feeds` ends with exit code 1 (skipped parts). An earlier run of the same bench, on
   the merger before the fix of item 10 and at a load of 6 to 29, had the same values except in 2
   of 6 runs of this profile, where the 95th percentile grew to 3.4 and 4.4 s and the queue to 49
   lines: under load the bursts piled up. Not repeated after the fix under that load.

9. **What this does and does not show about real time.** It shows that the chain of four
   processes runs at a 25 fps pace on the build machine with these simulated tool times, what
   the queues and the merger add, and how the delay `L` and a slow calibration show in the time
   from a frame to its decision. It does not show real time: the tools are stand-ins that sleep
   (a real detector takes a GPU and CPU time beside the stream), the frames are PNG files of a
   synthetic clip read at a pace (no camera, no decoder), the delays are published values from
   other hardware, and the build machine ran other sessions (load given per run). The latency of
   a frame includes the delay of `L` frames by design (`L` / 25 s).
10. **Second read of this work** (one subagent, before the commit, every finding reproduced by it):
    - Major, fixed: `merge-feeds` returned once the last frame went out while a slower tool still
      wrote; the next write of that tool broke its pipe (exit 2) and its copy at `--out` stopped
      (calibration 300 ms per frame, 20 frames: the copy held 1 of 21 lines). The documented
      commands also left out `--start-at`, so the tool that was ready later had most of its parts
      late (120 of 150). Fix: the merger reads every feed to its end; the copy gets each line before
      the feed; `START` and `--start-at` are in RUNBOOK section 14. After the fix: 21 of 21 lines,
      no broken pipe; tests `test_the_merger_reads_a_slow_tool_to_its_end_so_that_its_pipe_does_not_break`
      and `test_the_copy_gets_each_line_before_the_feed`.
    - Minor, fixed: the merger decided every frame whose wait had run out in one go, so a part
      that came while a slow reader held a write went out late (detector every 30 ms, reader 200
      ms per line: 6 to 10 of 12 parts late). Fix: one frame at a time, the waiting lines applied
      before each; after the fix 0 of 12 late (test
      `test_a_slow_reader_does_not_make_a_part_late_that_came_while_a_line_was_written`).
    - Small, fixed: `stream --synthetic --tracker X` was refused although the synthetic clip uses
      the tracker name; `--skip-behind` with batches above 1 was refused only after the header was
      in the feed (now before the tool is built). Each has a test.
    - Refuted by the reader (by running): merger deadlines and header races, both feeds ending,
      the whole-clip control against `run_clip` (seeds 0 to 3, 30 % missed people, about 15 % of
      cameras dropped: 0 frames differ), the kit keys before the pitch filter, the paint worker,
      `open_transition` across runs, the latency pairing of `f` with `f + L`, the pace of
      `LiveFeed`, and the RUNBOOK commands against `check_stream_options`.
11. **Other changes.** `--lines` checks the video: the header names `video_sha256` (null for a
    folder of images) and the number of frames, and `stream` refuses lines of other pixels than
    `--clip` or of another number of frames (round-5 open point). Options that do not apply are
    refused, not ignored: `--frames` and `--seed` without `--synthetic`; `--clip`, `--lines`,
    `--detections`, `--calibration`, `--labels`, `--fps`, `--split`, `--tracker`, `--queue` with
    it; `--detections`/`--calibration` with `--lines`; `--queue` without `--lines`; `--split` and
    `--tracker` without `--labels`; `--fps` with a video file (round-5 open point). `result.json`
    of `stream` adds `paint_timing`, `latency`, `missing_parts` and `transitions_known_at`. The
    offline `demo` predictions of both engines are byte-equal to `main` 0c2ac24 (SHA-256
    `bb50efd43536dd71...` for the model engine, `1f97af18358d652b...` for the detector engine);
    the summaries differ in timing values and the time of writing only.
12. **Not verified.** A real detector or calibration in live mode, the run machine, a camera or
    a decoder as the source of frames (the frames here are files), named pipes on Linux (built
    and run on macOS only), a real dissolve, the early dissolve test on footage (the fit and the
    held-out clips are synthetic two-shot clips), the latency with real tool times, the paint
    thread on the run machine, a broadcast clip whose stream is longer than 10 s.
13. **Proposed owner decisions.**
    - Keep the early dissolve test on by default (`early_dissolve_windows` 4) with the delay of 25
      frames: on the synthetic clips it removes the dissolve gap for dissolves up to 20 frames at
      no cost on clips without a dissolve; a delay of at least 35 frames for longer dissolves.
    - The calibration in a live run: `--skip-behind` when its time is above 40 ms per frame on the
      run machine (its published time is 439 ms); else the stream waits for it.
    - The wait of the merger (default 1 s): it bounds what a stalled tool costs; it must be above
      the time of the slower tool per frame, or most of its parts go out late.
    - The paint score in a live run: on its own thread now; under load it still competes for a
      core. Leave it out of live runs if the run machine shows the 95th percentile above 40 ms.
    - No text of `CLAUDE.md` is proposed: the package still imports no tool; the stand-in
      launcher `tools/stand_in_tool.py` imports `run_tool.py` by its name and no tool.

## 2026-11-23 — Rehearsal of the commands of 23 Nov, the budget at both prices and the owner decisions

Session C of round 6. Base `main` 0c2ac24. The owner's first session on the run machine follows
`docs/HANDOVER.md`, section 6, and `docs/RUNBOOK.md`. Both were rehearsed on 22 Nov (entry
"Rehearsal of the handover and the runbook"), before the code of 23 Nov added video input,
a second detector family, the streaming path and new tests of the money guards. This entry
records the walk of the new commands, the budget at the two prices, the list of open owner
decisions, and what changed in the documents.

1. **Method** (the one of 22 Nov). A scratch clone of 0c2ac24 outside the repository, with a home
   folder of its own (so `~/pitchvlm-data` was a scratch folder), its hooks installed, a
   deny-list of comments only, and the work tree `../pitchvlm-results` on the branch `results`.
   The environment of the main checkout (Python 3.13, no SDK, no Pillow, no torch). Data: 34
   synthetic clips (`synth.SyntheticScene`, seeds 300 to 333, not used before), JPEG, first with
   50 frames, then with 750 frames (30 s at 25 frames per second); the real `data` command
   extracted them from a local archive and wrote the split list (probe 24, dev 6, fresh 3,
   reserve 1), with no request. Paid commands ran through a wrapper that sets the day that the
   code sees (30 Nov or 1 Dec 2026, by replacing `datetime` in `cli` and `doctor`) and replaces
   the model backend by one that raises at its first call; for the doctor it also takes the
   checks of the SDK and of the key as passed. So no request could leave the machine. The tools:
   clip runs written from the truth of the scene with the part writer of `tools/run_tool.py`.
   The wrapper, the data script and the tables script are in the notes of this group of
   sessions, not in the repository.
2. **Walk of the commands** (verdicts: A agrees, D document wrong and fixed, N cannot be checked
   on the build machine).

   | Command | What happened | Verdict |
   |---|---|---|
   | `data --split valid --manifest --seed` (local archive) | as HANDOVER step 6 says | A |
   | first commit in the clone | a hook of a git template of the build machine ran first and failed; `pre-commit install -f`, the row of HANDOVER 13, repaired it | A |
   | `doctor --no-api` | `[ok  ] guard hooks`, `[ok  ] deny-list`, `[info]` for SDK and keys, exit 0 | A |
   | `demo`, `demo --engine detector` | valid prediction files, exit 0 | A |
   | `stream --synthetic --frames 250 --delay 25` (with and without `--allow-dirty`), `--delay clip` | exit 0, times per frame printed | A |
   | `clip-job` x 2, clip runs, `run --detections --calibration` on a dev clip of 750 frames | "this run makes no model call and costs nothing", exit 0 | A |
   | `stream` on those clip runs, then `--delay clip` | exit 0; the control line "a delay of whole clip" | A |
   | `stream --lines` with a missing file; `stream` with the number settings | one error line each, exit 2 | A |
   | `tool-job` of the 5 files of the second family; `--max-clips 2` | "120 frames of 24 clips", "10 frames of 2 clips" | A |
   | `tools/run_tool.py dfine`, `rtdetrv2` without torch | "the package torch is not installed here", exit 2 | A |
   | `tool-score` without a run file; `probe --video-dir` inside the clone; `doctor --video --no-api` | one error line each, exit 2 | A |
   | `sync` | lists the clip runs as tool runs; HANDOVER step 14 did not say that its `add` takes `results/tools` | D |
   | `guard.py --tracked`, `--push` | exit 0 | A |
   | `pytest tests/test_guard.py -rs` | 152 passed; HANDOVER step 1 said 150 | D |
   | `tools/run_tests.py`, then `-m "slow or not slow"` | 3,410 passed, 1 skipped, 9 deselected, exit 0 (82.3 s, load 11.8); 3,419 passed, 1 skipped, exit 0 (62.3 s, load 14.8); HANDOVER said 3203 | D |
   | `doctor --models gemini-4-argon,gemini-3.8-flash` without key | `[fail] API key`, "The contract checks did not run", exit 1 | A |
   | `probe people --max-clips 2` with the real backend, no SDK, no key | estimate and budget line, then `error: The environment variable GEMINI_API_KEY is not set or is empty. ...`, exit 2 | A |
   | estimates of doctor, pilot, kill switch, OpenRouter doctor | equal to HANDOVER 6 on 30 Nov; no December value was written | D |
   | `doctor --video` | 0.144 USD for 8 calls; RUNBOOK 13 gave only the video part | D |
   | `doctor --video --models openrouter:...` | no video request planned, `[fail] video boxes` without a call | A |
   | the three video probes | 2.382 / 4.764, 2.105 / 4.211, 2.267 / 4.534 USD; RUNBOOK 13 says 2.390 / 4.780, 2.108 / 4.215, 2.275 / 4.549 (another set of synthetic clips); 1 to 4 % lower on clips of 50 frames | D (note) |
   | `run --replay` and the paid run of a dev clip | 2.345 USD for 179 calls, stops above 3.048; 4.690 USD from 1 Dec; RUNBOOK 4 named November only | D |
   | install of the transformers environment, every real call, the download, the scorer, real clips and real videos | not run | N |

   Counts: 23 rows; agrees 15, document wrong 7, cannot be checked 1. No defect of the code was
   found (item 6 names one behaviour).
3. **Fixes of the documents** (one line each):
   - HANDOVER 1: the state of 23 Nov (what was added, the counts of the tests, this rehearsal).
   - HANDOVER 5: the reading order names RUNBOOK 13 and 14, and the list of decisions.
   - HANDOVER 6: a table of the 23 steps (paid or free, estimate at both prices, needed or
     optional); the smallest path to a first result (steps 1 to 11, 14, 15: 0.785 USD until 30
     Nov, 1.536 USD from 1 Dec); the December estimate in steps 5, 9, 12 and 13; 152 and 3,419 in
     steps 1 and 2; the folder `results/tools` in step 14; new optional steps 20 (`doctor
     --video`), 21 (video pilot), 22 (second detector family), 23 (`stream` on a dev clip).
   - HANDOVER 8: how the price is chosen (start day of the command for the estimate, day of the
     answer for the cost, both UTC), the run across midnight of 30 Nov, the price change of the
     flash and robotics models on 1 Jan 2027, and the budget tables (item 4).
   - HANDOVER 11: "each doctor run stops above 2 USD" (the code); the list of 41 open owner
     decisions in place of the sentence "28 decisions of the owner are open" (item 5).
   - HANDOVER 12: the state of each date and a proposed order without dates (item 7).
   - RUNBOOK 2 and 4: both prices for the doctor, the pilot, a clip and a number request; the
     rule "tell the owner when the doctor runs cost more than about 1.3 USD" now reads "when the
     measured doctor runs and the estimate of the pilot pass 2 USD together", because the pilot
     costs 1.3 USD from 1 Dec. The phrases that `test_vlm_cost.py` reads ("About 2.3 USD for a
     clip of 30 seconds", "about 0.009 USD each with the default token estimates") stay.
   - RUNBOOK 13: the printed total of `doctor --video`, the estimate of the video pilot, and the
     note that the estimate of a video probe depends on the sampled frames.
   - PLAN: the state of the milestones and of the to-dos on 23 Nov; the printed estimates of a
     clip and of the budget table at both prices; PnLCalib can be used by the pipeline since 22
     Nov (the risk text said it could not).
   - `cli.py`, help strings only: `stream --frames`, `--split`, `--tracker` now name their
     defaults; `stream --history` names its default in words, not by a function name; `stream
     --config` names the file whose engines it takes; `stream --fps` says "when no label file is
     given", as `run` does; `clip-job --tool` says that the family must be that of the settings
     file; `clip-job --clip` says that it must lie below `--root` and that a video needs
     `video-frames` first.
4. **Budget at both prices** (printed by the code, estimates, not measured; the tables are in
   HANDOVER 8 and were generated from the printed lines, not typed). The code takes the price of
   the start day of a command (`cli.py`, `datetime.now(UTC).date()`) for the estimate and of the
   day of each answer for its cost (`vlm/cost.py`, `response.created_at`).
   - First session: needed steps 0.785 USD until 30 Nov, 1.536 USD from 1 Dec; with every
     optional paid step of section 6, 1.367 and 2.686 USD. Every run stays below its own limit
     by construction (1.3 times its estimate; 2 USD for a doctor); the hard stop is 132 USD or
     more away.
   - One clip of 30 s, model engine: 2.345 USD until 30 Nov, 4.690 USD from 1 Dec, 1.172 and
     2.345 USD through flex. The doctor, the pilot and the full probes (one setting, number two):
     12.61 USD until 30 Nov at the standard price, 25.18 USD after it.
   - The whole plan of `docs/PLAN.md`, one pass of each item: 75.92 USD at the standard price
     until 30 Nov, 38.34 USD through flex until 30 Nov, 151.81 USD at the standard price from 1
     Dec (above the hard stop), 76.67 USD through flex from 1 Dec. Probes before 1 Dec and the 27
     clip passes after it through flex: 75.92 USD. Flex failing after 1 Dec: 139.24 USD, or
     92.34 USD with the cut of the plan (17 clip passes). The three video arms are not in the
     plan: 6.75 USD until 30 Nov, 13.51 USD at the standard price from 1 Dec. A first version of
     these totals held the video arms (found by the second reader, item 11); they are out now.
   - Not verified: every number above uses the token estimates of 22 Nov; the cost pilot
     replaces them. The number requests of a clip are a guess (50).
5. **Owner decisions.** HANDOVER 11.1 lists 41 open decisions from every entry of this file and
   from pull request 2, grouped by the first step that waits for each, with options, the
   recommendation of the session that raised it, the default and what it blocks. Raised by this
   session (each a recommendation, not a decision): which paid steps run before 1 Dec (the first
   sessions and the full probes; decision 11); the tier of the process tests (as now; 9); the
   ledger line of an answer without a cost (after the first OpenRouter doctor; 13); the
   captions of the check images (keep for the kill switch; 21); uploads above 20 MB (keep the
   refusal; 33); pull request 2 (decide before the key prompt of section 4.1; 1).
6. **Behaviour to know, not changed** (inferred from the code, not run): a standard-price run
   that starts on 30 Nov and runs past 00:00 UTC pays its later answers at the new price
   against an estimate at the old one, so the 1.3 times stop can end it. HANDOVER 8 says so. A
   guard that refuses a start when the estimate day and the next day differ in price would be a
   change of the money code (sessions of the money code, not this one).
7. **The test** (`tests/test_documented_commands.py`, default run, 18 new tests, 27 in all). It now also
   reads the text around the blocks: every file of the repository that inline code names
   (paths, modules below `src/pitchvlm`, experiment files by their bare name, which must load);
   every command in inline code (`doctor --video`, `clip-job --tool dfine`, `tools/run_tool.py
   dfine`), with its options and values; every inline option of RUNBOOK 11 to 14 and HANDOVER 6
   (`--video-dir`, `--delay clip`); every reference "RUNBOOK, section N" or "HANDOVER, section
   N.M", with a quoted title that must be a heading of that section; the numbering of the RUNBOOK
   without a gap; that sections 11, 13 and 14 hold commands that the reader sees; and that the
   help of every option with a default names it. Four drifts planted in the real documents (the
   experiment file `people_probe_video_lo.yaml` in the table of RUNBOOK 13, the option
   `--videos-dir` in its text, `stream --delays N` in RUNBOOK 14, the heading "A video as the
   clip" in a reference of HANDOVER 6) made it fail with one line each; the test before this
   change found none of the four. The help test fails on the `cli.py` of 0c2ac24 (three
   options). The same kinds are kept as unit tests of the reader.
8. **Proposed text for `CLAUDE.md`** (section "Commands"; the owner applies it; it replaces the
   proposals of the entries "A faster default run of the offline tests", item 8, "Rehearsal of
   the handover and the runbook", item 8, and "The offline tests in several processes", item
   11): the commands block of item 11 of the last of these, and as the last paragraph: "The
   run-machine commands (`doctor`, `data`, `oracle`, `probe`, `tool-job`, `tool-score`,
   `clip-job`, `video-frames`, `overlay`, `run`, `stream`, `eval`, `sync`) are in
   `docs/RUNBOOK.md`. The order of the first session on the run machine is in
   `docs/HANDOVER.md`, section 6, and the open decisions of the owner are in its section 11.1."
9. **What the parallel sessions of round 6 may change** (written for `main` 0c2ac24): session A
   (review of round 5: video input, the transformers adapter, `fallbacks/`, `config.py`) may
   change the lines of `doctor --video`, the estimates of the video probes and the commands of
   the second family (HANDOVER 6, steps 20 to 22, and the rows 20 and 21 of the budget table);
   session B (live feed) may change `stream`, its help, RUNBOOK 14 and the live mode of the tool
   script (HANDOVER 6, step 23, and decision 39). The test of item 7 fails when a command,
   option or file that these documents name goes away.
10. **Not verified.** Every real call, real cost, real clip, real video and the install of every
    tool environment; the token rate of video; the frame counts of real clips (750 assumed). The
    test counts are those of the build machine.
11. **Second read** (one subagent, given the diff and not these conclusions). Medium, confirmed
    and fixed: the totals of the whole plan held the three video arms, which the same text called
    "not in the plan" (82.68 / 41.72 / 165.32 / 83.43 USD; now 75.92 / 38.34 / 151.81 / 76.67 USD,
    the arms in a row of their own); the conclusions hold (151.81 and 139.24 USD are above the hard
    stop). Low, confirmed and fixed: the reader of the text skipped commands written as `python -m
    pitchvlm.cli ...` or `python tools/run_tool.py ...` and spans whose second word is not an
    option; references "section N of the RUNBOOK", "RUNBOOK.md, section N" and the second number
    of "sections N to M"; files below `locks/`; and it read a heading "## 5.1" as section 5. Each
    now has a case in the planted text of the test. Low, fixed: decision 39 put a state in the
    column of the recommendation, decision 35 its recommendations in the column of the options,
    and decision 1 named an application. Kept: the help test accepts a help text that has the
    word "default" without its value. Confirmed right: every other number (recomputed), the help
    edits change no behaviour, the print claims of steps 20 to 23 (each traced to its `print`),
    no owner decision is missing, and none is decided.
12. **Merge of `main` 7e89517** (the review fixes of round 5, entry "Second read of the video
    input and the detector families"). The documents now hold: owner decision 42 (V3: a video
    written again with other bytes is paid again; the review recommends a key on the pixel hash
    of the frames before the first paid video probe) in HANDOVER 11.1 and in step 21; the limit of
    20,000,000 bytes that counts the whole request (decision 33, step 20); the count of the
    self-check of the second family (`self_check_detections` above 0, step 22). 42 open owner
    decisions. Test runs after the merge (14 workers): default run 3,435 passed, 1 skipped, 9
    deselected, exit 0, 39.5 s (load 4.3); full run 3,444 passed, 1 skipped, exit 0, 43.3 s (load
    17.8); `tests/test_guard.py` 152 passed.
13. **Merge of `main` 6b6a803** (the live feed, entry "The live feed of the detector engine").
    The conflict of `cli.py` kept the code of `main` (the options of `stream` without a default
    value, so that an option that does not apply is refused) and the help text of `--fps` of this
    entry; the help test still passes. The reader of the text first refused the documented span
    `tools/run_tool.py --feed`: an option of the sub-commands of the script; it now looks for
    such an option in the sub-commands, and a planted `tools/run_tool.py --fed` still fails.
    Rehearsed on the build machine, as in item 1: the three terminals of RUNBOOK 14, "Live run",
    with `tools/stand_in_tool.py` in the place of both tools (named pipes, `--pace 25`, one
    `--start-at`, `--skip-behind` for the calibration, a simulated 20 ms per frame) on the dev clip
    of 750 frames, then the control from the copies of the feeds: each tool printed `live run: 0
    frames skipped` with its feed copy and its `<stem>-runner.json`; `merge-feeds` printed
    `merge-feeds: 750 of 750 frames went out`, exit 0, live and from the files; `stream --lines -`
    exit 0 both times (one run, not a timing measurement). The four live-feed decisions are in
    HANDOVER 11.1 as decisions 37, 38, 39 (each now with its live-feed recommendation) and 43 (new):
    43 open owner decisions. HANDOVER step 23 adds the live run with the words that it printed.
    Test runs after the merge (14 workers): default run 3,491 passed, 1 skipped, 10 deselected,
    exit 0, 50.9 s (load 3.7); full run 3,501 passed, 1 skipped, exit 0, 62.2 s (load 21.2);
    `tests/test_guard.py` 152 passed.


## 2026-11-23 — Second read of the live feed, and a cache key of video requests that survives a new write

1. **What and how.** An independent second read of the live feed of round 6 (pull request 22,
   `git diff 7e89517 6b6a803`: `feed.py`, `merge-feeds`, the live mode of `tools/run_tool.py`,
   the live parts of `stream.py`, `evalx/livebench.py`, `tools/stand_in_tool.py`), a planted-bug
   check of the same code, the latency run of round 6 repeated under a load made on purpose, and
   the cache key of video requests (finding V3 of round 6; owner's yes). The reader did not write
   the live feed. Each finding was reproduced before it was called one, fixed with a test, and
   the test was seen to fail with the fix undone. Build machine only: no model call, no key, no
   footage, no install. Held-out seeds of this entry: 700 to 719 (not used before).
2. **Findings.** Ranks: major (a normal run hangs, gives wrong numbers or loses money), minor
   (wrong on a plausible edge), small (wrong only on unusual input).

   | Id | Rank | Finding | Reproduction (build machine) | Status | Test |
   |---|---|---|---|---|---|
   | L0 | major | At a low load the live chain hung: a tool that waited in the open of its named pipe was paired with a process that opened the pipe for reading and closed it again before `merge-feeds` opened it. Its first write broke (exit 2, nothing written), and `merge-feeds` waited for ever in its own `os.open`, the other tool with it. Reported by the session "eng - linux ci" (3 tests of `tests/test_live_feed.py` failed at load 3 to 4). | On 72fad94 at load 3 to 6: the 3 chain tests hung in 3 of 3 runs (200 s limit); one chain run timed out at 20 s. The tool's open returned 0.7 s before `merge-feeds` reached its open. A writer probe that opens without blocking saw a reader on the pipe 8 to 27 ms after the start of every `python -m <module> <pipe>` process (3 of 3, also another Python and a module that does not exist) and never for `python -c ... <pipe>`, `python <script> <pipe>`, `sleep` or `sh -c` (0 of 3 each): a program outside the project. At a high load the tools reach their open after that reader is gone, so the tests passed there. | fixed: `run_tool.NamedPipeFeed`: on a broken pipe the feed waits up to `FEED_REOPEN_WAIT_S` = 10 s for a reader (an open that does not block, every 50 ms) and sends the whole feed again from its header; it says so on the error stream and counts it (`feed_reopened` in `<stem>-runner.json`); no reader in time: the broken pipe as before | `test_a_reader_that_opens_the_pipe_and_goes_away_does_not_stop_the_tool` (without the fix the reader waits for ever: timeout), `test_a_feed_whose_reader_does_not_come_back_stops_after_its_wait`; repeats in item 3 |
   | L1 | minor | A tool that stopped before it opened its feed (a wrong job file, a tool that cannot be built: the feed was opened after the tool was built) left `merge-feeds` waiting for ever in the open of that pipe, and the other tool with it. | detector job file missing: after 15 s `merge-feeds` and the calibration still running | fixed: `open_feed` opens a named pipe (and `-`) first, before the job is read; any later stop closes it and `merge-feeds` stops with "the feed of the detector ended before its header" (exit 2). A plain file is opened as before, when the frames start (second read: opened first, `--feed` on the job file emptied the job) | `test_a_tool_that_stops_before_its_first_frame_ends_the_chain` (codes 2, 2, 2) |
   | L2 | minor | With `--feed -` the feed is the standard output, but what a tool or a library prints while it is built or per frame went into it too: the merger refuses a feed whose first line is not its header (exit 2), and a print without a line end breaks the next frame line. | a stand-in that prints while the tool is built: the first line of the standard output was the print | fixed: `StandardOutputFeed` keeps the standard output for the feed alone and points the descriptor 1 of the process at the error stream while the feed is open (and back when it is closed: second read) | `test_with_the_feed_on_the_standard_output_a_print_of_the_tool_goes_to_the_error_stream` |
   | L3 | small | `--feed` and `--out` on one file wrote every line twice through two handles into one file (the second reader ran it: the same bytes at the same offsets, so the file stayed whole; the refusal is mostly for clarity), and `--feed` on the job file is refused too. | (by reading, and the run of the second reader) | fixed: refused before the feed is opened, by `os.path.samefile` when both exist | `test_the_feed_and_its_copy_cannot_be_one_file`, `test_a_feed_that_names_the_job_file_is_refused_and_the_job_is_kept` |
   | L4 | minor (docs) | The owner decision of round 6 said the wait of the merger "must be above the time of the slower tool per frame". The wait bounds the lag of a tool, which grows with every frame when the tool is slower than the pace without `--skip-behind`, and it never catches up. | `FeedMerger` with simulated arrivals, 25 fps, wait 1 s, 1,500 frames: a calibration of 41 ms per frame lost 521 parts (from about frame 980), 45 ms 1,305, 60 ms 1,452 | fixed in RUNBOOK section 14 (`--skip-behind` at 40 ms per frame or more at 25 fps) and in item 9 | `test_a_tool_a_little_slower_than_the_pace_loses_every_part_once_its_lag_passes_the_wait` |
   | L5 | small (test) | `test_the_chain_ends_when_the_stream_stops_before_it_reads` depended on the load: a tool that started after the shared start had every frame due at once and, without a time per frame, put all 40 lines into the pipe before the merger stopped (exit 0, not 2). | 72fad94 at load 66 to 84: 2 of 6 runs | fixed: the stand-ins of that test take 100 ms per frame | the same test, repeats in item 3 |
   | L7 | minor | Found by the planted-bug check: a tool that hung after the last frame (alive, its pipe open, nothing written) kept `merge-feeds` reading it for ever (policy 7), so the merged lines never ended, and `stream` read to the end of its input: it never reached `finish`, the last `delay` frames were never decided and no result was written. The module docstring said that a stalled tool delays the stream by at most one wait. | `merge_feeds` with a calibration that writes every frame and then waits: all 5 frames out at 0.21 s, `merge_feeds` still running after 5 s | fixed: the merged lines end once the last frame went out (`end`: `merge-feeds` points its standard output at nothing); a feed silent for `drain_s` (`DEFAULT_DRAIN_S` = 10 s) after the last frame is left and counted (`left_open`, exit 1); `stream` reads no line after frame `n_frames - 1` | `test_a_tool_that_hangs_after_the_last_frame_does_not_hold_the_merger`, `test_the_stream_reads_no_line_after_the_last_frame` |
   | L6 | small (docs) | RUNBOOK section 14 called the merge of the two copies "the same lines" as the live run; a part that was late in the live run is whole in them. | (by reading) | fixed in the comment of the command | none (text) |

3. **Repeats of the three chain tests after the fixes** (`test_the_live_chain_gives_the_result_of_the_clip_runs`,
   `test_a_tool_that_dies_leaves_its_part_missing_and_the_stream_goes_on`,
   `test_the_chain_ends_when_the_stream_stops_before_it_reads`; one pytest call per run, the load
   average before each run):

   | tree | load average before the runs | runs | runs with 3 of 3 passed |
   |---|---|---|---|
   | 72fad94 (before the fixes), first session, stray reader present | 3 to 6 | 3 | 0 (each hung until the 200 s limit) |
   | fixed, stray reader present then (probe 3 of 3 at the same hour) | 4.8 to 5.9 | 6 | 6 |
   | fixed, machine growing busier | 15.8 to 32.1 | 14 | 14 |
   | fixed, 12 processes that spin | 36 to 61 | 20 | 20 |
   | fixed, quiet again, stray reader absent (probe 0 of 10) | 5.1 to 12.3 | 14 | 14 |
   | 72fad94, quiet, stray reader absent | 5.6 to 6.0 | 3 | 3 |

   So 20 runs at a load under 13 and 20 runs at a load of 36 to 61 passed on the fixed tree (60
   of 60 tests each); 18 of the low ones at a load under 7. The program that opened the pipes
   came and went: probes of `python -m nonexistent_mod <pipe>` saw a reader 3 of 3 times at a
   load of about 5 (and 2 of 2 for `merge-feeds`), 0 of 9 at a load of 28 and later 0 of 10 at a
   load of 8.6, when 72fad94 passed its 3 runs and the by-hand chain of RUNBOOK section 14 (two
   stand-in tools on named pipes started first, then `merge-feeds | stream --lines -`) ended
   with exit codes 0 in 3 of 3 runs. So the low-load runs of the fixed tree after that hour show
   that nothing broke, not that the fix works; that is shown by the test that plays the stray
   reader (`test_a_reader_that_opens_the_pipe_and_goes_away_does_not_stop_the_tool`: the reader
   waits for ever without the fix). Why round 6 did not see it: its chain runs at a load of 8 to
   29 passed, and its bench at 2.5 to 3.6 too; whether the program ran then is not known, and how
   often it runs was not measured (inferred from the probes above: not all the time, and not
   only at a low load).

4. **Checked and found right** (read, and run where a check could run here):
   - The bounded wait: the clock of frame `n` starts at the read time of the first line of frame
     `n` or later (the reader thread stamps it, `time.monotonic`); the deadline is
     `max(clock, both headers) + wait`; at exactly the deadline the frame goes out. After a
     `Queue.get` timeout the monotonic clock is at or past the deadline, so the loop does not spin.
   - Back-pressure: the merger applies every waiting line before it decides a frame, so the
     queues of the two feeds are drained together; no deadlock between the bounded queues (the
     main thread only blocks in the write to its reader). The reader threads are daemons; at
     the end both have put their end mark before `merge_feeds` returns.
   - The paint thread: `paint_alignment` gets the image and a new matrix and nothing writes into
     a frame image after it is pushed; the scores go only to `result.paint`, never to a decision,
     so two runs of the same input cannot differ through it.
   - The early dissolve test of the fit (`synthetic_diagnostics._early_take`) is a copy of the
     rule of `ShotMeter.open_transition`: on the held-out clips below both gave the same spans
     in 380 of 380 clips (the 19 cases of item 5).
   - "Delay, never drop": the stream and the merger hold back; only `--skip-behind` (at the
     source) and the wait of the merger drop parts, as written; L4 makes the second explicit.

5. **The early dissolve test on held-out seeds 700 to 719** (`ShotMeter.open_transition(4)` as
   the stream calls it, 20 clips per case; the 8 controls of round 6 and 5 new hard cases: a
   white flash of 1 frame, of 3 frames, three flashes 3 and 4 frames apart, a logo sting of 12
   frames over the picture, slow motion with each frame shown twice; transitions of
   `TwoShotSource`):

   | case | clips | taken early | final | early without a final | early minus final (start; stop) |
   |---|---|---|---|---|---|
   | 8 controls of round 6 | 160 | 0 | 0 | 0 | - |
   | flashes (3 cases) | 60 | 0 | 0 | 0 | - |
   | logo sting, 12 frames | 20 | 0 | 1 | 0 | - |
   | slow motion | 20 | 0 | 0 | 0 | - |
   | dissolve, 5 frames | 20 | 19 | 19 | 0 | 0; -2 to 0 |
   | dissolve, 15 frames | 20 | 17 | 17 | 0 | 0; -1 to 0 |
   | dissolve, 25 frames | 20 | 18 | 18 | 0 | 0; -9 to 0 |
   | fade through black, 10 frames | 20 | 0 | 0 | 0 | - |
   | wipe, 10 frames | 20 | 19 | 19 | 0 | 0; -1 to +2 |
   | picture in picture, 10 frames | 20 | 0 | 0 | 0 | - |

   No false early transition in 260 clips without a cut (0 of 260; the 95 % upper bound of the
   rate is about 1.1 %). The early span of a 25-frame dissolve ended up to 9 frames before the
   final one on these seeds (round 6: up to 4), and a wipe up to 2 frames after it.

   What the early span that ends before the final one does to the decided frames and the ids
   (stream at L = 25 with the early test, L25-late without it, and the offline engine, on the
   dissolve clips of `streambench` with dissolves of 5 to 25 frames, held-out seeds 700 to 709,
   exact and noisy boxes; n = 10 clips per row; "gap" = the frames between the early stop and
   the final stop, decided under the early span; paired differences to offline):

   | dissolve | boxes | gap frames (all clips; most in one) | gap frames whose state differs | switches L25 - offline, mean [min, max] | switches L25-late - offline |
   |---|---|---|---|---|---|
   | 5 | exact | 0; 0 | 0 | +0.0 [0, 0] | -7.7 [-9, -5] |
   | 10 | exact | 0; 0 | 0 | +0.0 [0, 0] | +4.5 [2, 7] |
   | 15 | exact | 8; 1 | 2 | +0.0 [0, 0] | +4.4 [2, 7] |
   | 20 | exact | 8; 1 | 3 | +0.0 [0, 0] | +4.5 [2, 7] |
   | 25 | exact | 9; 2 | 5 | +0.0 [0, 0] | +4.2 [2, 6] |
   | 5 | noisy | 0; 0 | 0 | +0.1 [-10, 11] | -3.7 [-15, 5] |
   | 10 | noisy | 0; 0 | 0 | +0.8 [-8, 9] | +10.1 [-3, 29] |
   | 15 | noisy | 5; 1 | 3 | -0.4 [-8, 6] | +8.9 [-3, 19] |
   | 20 | noisy | 8; 1 | 6 | -0.8 [-8, 2] | +9.9 [-1, 28] |
   | 25 | noisy | 9; 2 | 7 | +5.5 [-7, 50] | +15.0 [0, 57] |

   On these clips the early span ends at most 2 frames before the final one (the clips of item 5
   up to 9: other camera steps). A gap frame is decided as a frame of the new shot: its people
   are in the stream and not in the offline result (26 such frames in 100 clips). With exact
   boxes the ids are those of offline in 50 of 50 clips (switches and number of ids equal).
   With noisy boxes the paired differences spread both ways; the largest (+50, seed 700, 25
   frames) has +57 without the early test too, so it is the matching of a noisy late join, not
   the gap. So: the stream gives up to 2 frames (here) of people that offline leaves out as
   blend frames, and no id change was traced to it; on 25-frame dissolves with the gaps of item
   5 (up to 9 frames) this was not measured. The -7.7 of the late arm on 5-frame dissolves (fewer
   switches than offline) is not explained here; it is an arm without the early test.

6. **Planted bugs.** Written before the matching tests were read (a subagent; list and table kept with the
   session notes, not in the repository), on the code after the fixes of item 2: 37 bugs planted one
   at a time in `feed.py` (14), the live mode of `tools/run_tool.py` (12), the live parts of
   `stream.py` (9) and `merge-feeds` with `stream --lines` in `cli.py` (2), each run against
   `tests/test_feed.py`, `test_live_feed.py`, `test_stream.py`, `test_detector_families.py` and
   `test_documented_commands.py` (`-m "slow or not slow"`; 173 passed on the clean code).

   | Outcome | Bugs |
   |---|---|
   | caught by the existing tests | 28 (one of them by a hang: the test runner stopped the silent worker after 120 s) |
   | missed | 8 |
   | equivalent (no change of behaviour) | 1: the report of `run_tool.py` sent to the standard output with `--feed -`, which already points at the error stream |

   The 8 missed bugs, each now caught by its own test in `tests/test_live_feed_planted.py` (8 tests,
   about 1 s, no process; each fails with its bug and passes without it):

   | Id | Place | Planted change | New test |
   |---|---|---|---|
   | F3 | `FeedMerger.line` | a frame line of frame `n_frames` taken (`<=` for `<`) | `test_a_line_of_the_frame_just_past_the_clip_is_broken` |
   | F13 | `FeedMerger.line` | the repeated-line check ignores skipped frames | `test_a_part_after_a_skip_line_of_the_same_frame_is_a_repeated_line` |
   | R2 | `NamedPipeFeed._reopen` | only the last line sent again (no header) | `test_a_reader_that_comes_back_after_some_lines_gets_the_whole_feed_from_its_header` |
   | S1 | `stream._inputs` | the later of the two arrivals (`max` for `min`) | `test_the_arrival_of_a_frame_is_the_earlier_arrival_of_its_two_parts` |
   | S3 | `stream._part` | the reason of a missing part not checked | `test_a_missing_part_with_an_unknown_reason_is_refused` |
   | S9 | `QueuedLines.__iter__` | `last_read_at` the time the line was taken | `test_the_queue_gives_the_time_at_which_a_line_was_read_not_the_time_at_which_it_was_taken` |
   | S12 | `DetectorStream._log_transitions` | one log entry per span, whatever its kind | `test_a_span_known_early_is_logged_again_when_it_is_final` |
   | C3 | `cli.stream_inputs` | `--queue` not used for `--lines FILE` | `test_stream_lines_take_the_queue_size_of_the_command_line` |

   The caught ones include a moved clock start, a deadline without the header time, the lines of
   an ended tool, the release of the reading slot, the cut-off check, deciding all frames in one
   go, the read to the end, the pace, `--skip-behind` counting, the warm-up before the header,
   `arrived_at`, the redirect of `--feed -`, the bounded queue, the paint bound, the early span and
   the exit code of `merge-feeds`. The check also found the defect L7 (not planted).

7. **Latency under a load made on purpose.** Round 6 reported that, before its merger fix, the 95th percentile from
   the arrival of frame `f + L` to the decision of `f` reached 3.4 and 4.4 s in 2 of 6 runs with
   the simulated slow calibration (20 ms detector, 439 ms calibration with `--skip-behind`) at a
   load of 6 to 29, and that this load was not repeated after the fix.
   - Hypothesis (fixed before the runs): after the fix this 95th percentile stays below 1 s in
     every run at a comparable load. Baseline arm "fixed": this branch. Control arm "control": the
     same tree with the merger deciding every due frame in one go (the merger before the round-6
     fix of item 10; the read to the end is kept). Same bench, seed 500 as in round 6
     (`python -m pitchvlm.evalx.livebench --runs 1 --delays 5,25 --profiles pnlcalib-paper --seed 500`
     in each tree), the arms alternating, 250 frames, n = 250 - L decided frames per row. Load made
     on purpose: 0 or 12 processes that spin (14 cores), on top of the other sessions of the
     machine (not controlled; the load average before each row is given). The fixed arm ran on the
     working tree, whose `merge-feeds` gained the end of item 2 (L7) between rows 1 and 3 of the
     second block; that change acts after the last frame only.
   - Milliseconds, median / 95th percentile / largest; exit codes of stream, merge-feeds and the two
     tools (merge-feeds 1: the skipped parts):

   | busy processes | run | arm | L | load | after f + L (ms) | line to decision (ms) | stream frame total (ms) | largest queue | exit codes |
   |---|---|---|---|---|---|---|---|---|---|
   | 0 | 1 | fixed | 5 | 27.1 | 450 / 505 / 896 | 158 / 338 / 385 | 30 / 34 / 48 | 10.0 | 0, 1, 0, 0 |
   | 0 | 1 | fixed | 25 | 21.7 | 461 / 509 / 914 | 170 / 364 / 414 | 32 / 35 / 48 | 10.0 | 0, 1, 0, 0 |
   | 0 | 1 | control | 5 | 18.0 | 445 / 504 / 880 | 156 / 336 / 390 | 30 / 32 / 46 | 10.0 | 0, 1, 0, 0 |
   | 0 | 1 | control | 25 | 14.3 | 474 / 513 / 888 | 178 / 388 / 429 | 34 / 37 / 53 | 10.0 | 0, 1, 0, 0 |
   | 0 | 2 | control | 5 | 10.8 | 458 / 507 / 893 | 164 / 356 / 413 | 32 / 35 / 47 | 10.0 | 0, 1, 0, 0 |
   | 0 | 2 | control | 25 | 9.5 | 470 / 512 / 901 | 176 / 377 / 420 | 32 / 36 / 51 | 10.0 | 0, 1, 0, 0 |
   | 0 | 2 | fixed | 5 | 8.8 | 450 / 507 / 902 | 161 / 347 / 393 | 31 / 33 / 50 | 10.0 | 0, 1, 0, 0 |
   | 0 | 2 | fixed | 25 | 8.2 | 470 / 512 / 896 | 175 / 379 / 424 | 33 / 36 / 50 | 10.0 | 0, 1, 0, 0 |
   | 0 | 3 | fixed | 5 | 7.7 | 1322 / 3994 / 4292 | 1053 / 3455 / 3489 | 52 / 82 / 128 | 49.0 | 0, 1, 0, 0 |
   | 0 | 3 | fixed | 25 | 21.5 | 1800 / 3904 / 3946 | 1533 / 3186 / 3321 | 52 / 82 / 113 | 49.0 | 0, 1, 0, 0 |
   | 0 | 3 | control | 5 | 25.2 | 452 / 508 / 920 | 161 / 346 / 395 | 31 / 33 / 50 | 10.0 | 0, 1, 0, 0 |
   | 0 | 3 | control | 25 | 20.3 | 461 / 507 / 883 | 171 / 369 / 412 | 32 / 35 / 51 | 10.0 | 0, 1, 0, 0 |
   | 0 | 4 | control | 5 | 16.0 | 450 / 508 / 890 | 160 / 346 / 383 | 31 / 33 / 47 | 10.0 | 0, 1, 0, 0 |
   | 0 | 4 | control | 25 | 15.6 | 464 / 511 / 920 | 173 / 375 / 424 | 32 / 36 / 52 | 10.0 | 0, 1, 0, 0 |
   | 0 | 4 | fixed | 5 | 12.3 | 449 / 506 / 898 | 158 / 344 / 389 | 30 / 33 / 47 | 10.0 | 0, 1, 0, 0 |
   | 0 | 4 | fixed | 25 | 10.7 | 476 / 515 / 896 | 179 / 389 / 442 | 34 / 37 / 62 | 10.0 | 0, 1, 0, 0 |
   | 12 | 1 | fixed | 5 | 36.0 | 5117 / 12687 / 12903 | 3880 / 7621 / 8007 | 75 / 185 / 292 | 49.0 | 0, 1, 0, 0 |
   | 12 | 1 | fixed | 25 | 43.9 | 7250 / 8618 / 8723 | 3005 / 5567 / 5703 | 57 / 159 / 234 | 49.0 | 0, 1, 0, 0 |
   | 12 | 1 | control | 5 | 46.1 | 3589 / 11423 / 11637 | 3250 / 6838 / 6959 | 77 / 162 / 277 | 49.0 | 0, 1, 0, 0 |
   | 12 | 1 | control | 25 | 49.6 | 6974 / 9762 / 9927 | 3396 / 5925 / 5994 | 68 / 150 / 216 | 49.0 | 0, 1, 0, 0 |
   | 12 | 2 | control | 5 | 62.6 | 2600 / 11087 / 11662 | 2270 / 6357 / 6411 | 81 / 166 / 232 | 49.0 | 0, 1, 0, 0 |
   | 12 | 2 | control | 25 | 62.0 | 7049 / 12218 / 12344 | 4203 / 6729 / 6859 | 74 / 159 / 226 | 49.0 | 0, 1, 0, 0 |
   | 12 | 2 | fixed | 5 | 68.0 | 1194 / 9661 / 10557 | 875 / 7252 / 7414 | 53 / 193 / 272 | 49.0 | 0, 1, 0, 0 |
   | 12 | 2 | fixed | 25 | 61.1 | 2764 / 11394 / 12048 | 2391 / 7094 / 7143 | 63 / 189 / 229 | 49.0 | 0, 1, 0, 0 |
   | 12 | 3 | fixed | 5 | 53.7 | 926 / 2583 / 2672 | 620 / 2292 / 2572 | 45 / 73 / 125 | 49.0 | 0, 1, 0, 0 |
   | 12 | 3 | fixed | 25 | 49.3 | 874 / 1948 / 2038 | 615 / 1735 / 1919 | 42 / 64 / 137 | 39.0 | 0, 1, 0, 0 |
   | 12 | 3 | control | 5 | 50.4 | 785 / 1631 / 1668 | 538 / 1337 / 1565 | 41 / 59 / 87 | 30.0 | 0, 1, 0, 0 |
   | 12 | 3 | control | 25 | 44.4 | 1144 / 3157 / 3322 | 827 / 2836 / 3179 | 44 / 84 / 149 | 45.0 | 0, 1, 0, 0 |
   | 12 | 4 | control | 5 | 41.9 | 722 / 1502 / 1570 | 409 / 1263 / 1461 | 40 / 59 / 124 | 30.0 | 0, 1, 0, 0 |
   | 12 | 4 | control | 25 | 50.5 | 817 / 1856 / 2024 | 557 / 1614 / 1910 | 43 / 62 / 104 | 36.0 | 0, 1, 0, 0 |
   | 12 | 4 | fixed | 5 | 63.6 | 593 / 1792 / 1896 | 372 / 1523 / 1786 | 42 / 63 / 89 | 33.0 | 0, 1, 0, 0 |
   | 12 | 4 | fixed | 25 | 66.8 | 860 / 2916 / 3119 | 655 / 2675 / 2858 | 45 / 76 / 187 | 49.0 | 0, 1, 0, 0 |
   | 12 | 5 | fixed | 5 | 63.3 | 591 / 1215 / 1270 | 363 / 1015 / 1177 | 39 / 56 / 96 | 25.0 | 0, 1, 0, 0 |
   | 12 | 5 | fixed | 25 | 61.8 | 608 / 1632 / 1757 | 386 / 1408 / 1632 | 40 / 64 / 107 | 33.0 | 0, 1, 0, 0 |
   | 12 | 5 | control | 5 | 54.2 | 794 / 1967 / 2024 | 533 / 1710 / 1949 | 42 / 65 / 91 | 38.0 | 0, 1, 0, 0 |
   | 12 | 5 | control | 25 | 53.7 | 5776 / 15484 / 16419 | 5611 / 7058 / 7213 | 93 / 208 / 295 | 49.0 | 0, 1, 0, 0 |
   | 12 | 6 | control | 5 | 51.0 | 613 / 1253 / 1364 | 383 / 951 / 1212 | 40 / 60 / 98 | 22.0 | 0, 1, 0, 0 |
   | 12 | 6 | control | 25 | 46.8 | 636 / 1543 / 1649 | 398 / 1281 / 1521 | 41 / 58 / 90 | 28.0 | 0, 1, 0, 0 |
   | 12 | 6 | fixed | 5 | 45.0 | 4488 / 5765 / 5900 | 2652 / 4329 / 4635 | 53 / 120 / 167 | 49.0 | 0, 1, 0, 0 |
   | 12 | 6 | fixed | 25 | 44.3 | 2508 / 8204 / 8347 | 2214 / 5564 / 5646 | 61 / 141 / 238 | 49.0 | 0, 1, 0, 0 |

   - **The hypothesis fails.** With no extra process (load 7.7 to 27.1) the fixed arm stayed below
     1 s in 6 of 8 rows (505 to 515 ms) and reached 3.9 s in 2 (one run, load 7.7 then 21.5); the
     control stayed below 1 s in 8 of 8. With 12 extra processes (load 36 to 68) every row of both
     arms was above 1 s: fixed 1.2 to 12.7 s, control 1.3 to 15.5 s.
   - Cause, from the same rows: the stream itself fell behind. In every row above 1 s the median
     frame time of the stream was 39 to 93 ms, at or above the 40 ms of a frame at 25 fps, the line
     to decision alone was 1.0 to 7.6 s at its 95th percentile, and the queue of the stream held up
     to 49 of its 50 lines; in the rows below 1 s the median frame time was 30 to 34 ms and the queue
     held at most 10 lines. The merger is not the cause: the control is no better. With the
     calibration skipping, the frames come in bursts of about 11 every 439 ms, so a stream that is
     slower than the pace collects a backlog that only the end of the clip empties.
   - Limit, written: on the build machine under load the decision of the stream does not keep a
     25 fps pace (round 6 found the same for the frame time, item 5 of its entry), and the delay
     then grows to many seconds. The fix that would help is in the stream (the decision runs the
     window of 123 + L frames again for every frame; round 5, item 7), not in the merger; it is not
     built here. On the run machine the 95th percentile of the frame time must be checked first
     (RUNBOOK section 14 already asks for it).

8. **The cache key of a video request** (V3 of round 6; the owner approved it).
   - What "the same" means. A video file is written from frames of a clip. A lossy codec gives
     other bytes, and other decoded pixels, from one codec or OpenCV build to the next, so neither
     the hash of the bytes nor a hash of the decoded video is stable. What stays the same is the
     request: the frames of the clip (their numbers), the rate and the size at which they are
     written, the range, the sampling rate and the media resolution, with the prompt, the schema,
     the model and the context of the request.
   - Built: `VideoPart.source` (`VideoFrames`: the frames, `write_fps`, `width`, `height` and
     `pixels_sha256`, the hash of the decoded pixels of the frames of the clip before the codec,
     `vlm.video.frames_digest`). For such a part `identity()` (in the key) holds the frames, the
     rate, the size, the range, the sampling rate, the resolution, the file type and the length,
     and neither hash. A part without `source` (a file the package did not write) keeps the hash
     of its bytes in the key. The item of a people video request is `video-` and 16 digits of a
     hash of that identity (it was the hash of the bytes). A cassette keeps `stored()`: the
     identity, `sha256` (the bytes that were sent) and `pixels_sha256`.
   - A hit compares. Same pixels, other bytes (the frames written again): the stored answer is
     used, a warning is logged, the store counts it (`CassetteStore.other_video_bytes`), and the
     command prints the count and writes `stored_answers_on_rewritten_videos` into its cost record:
     neither silent nor an error. Other pixels of the same frames (new footage under an old clip
     name, or another decoder of the source images): `OtherImagesError`, as for images, or the
     answer with a warning under `--allow-other-images`. A duration that does not fit the frames
     and the rate is refused when the part is made.
   - Rule 2: no path and no bytes of a video in a cassette (test: the cassette names no folder, no
     `.mp4`, no bytes of the file; its keys are listed).
   - No paid run has been made, so no stored answer needed a migration; the keys of video requests
     made before this change (only fake runs of the tests) are not valid any more.
   - Tests (`tests/test_vlm_video.py`): the same frames written again with other bytes hit and are
     counted once; the codecs `avc1` and `mp4v` give other bytes and the same key (skipped where
     OpenCV cannot write one of them); one other frame, the write rate, the frame size, the
     resolution, the prompt, the model, a range and a sampling rate each change the key; other
     pixels under the same frames are refused, and allowed with a warning; the cassette holds both
     hashes and no path.

9. **Proposed owner decisions.**
   - The wait of the merger (round 6): replace "it must be above the time of the slower tool per
     frame" by "it bounds the lag of a tool; give `--skip-behind` to a tool whose time per frame
     is at or above the frame interval" (L4).
   - Keep `FEED_REOPEN_WAIT_S` at 10 s: a tool whose reader stopped for good now ends 10 s later
     than before (the chain-ends test takes 12 s instead of about 2 s).
   - The early dissolve test stays on; for dissolves of 25 frames its span can end up to 9 frames
     before the final one (item 5), so a delay of at least 35 frames stays the advice.
   - No text of `CLAUDE.md` is proposed.

10. **Not verified.** Named pipes on Linux (built and run on macOS only; the open without
    blocking of `NamedPipeFeed` follows POSIX: it succeeds when a process has the pipe open for
    reading, also one that waits in its open); which program opens the pipes on the build
    machine (only what it does was measured); a real tool, the run machine, real footage, a real
    dissolve, flash or replay; a paid video request and the bytes of a video written on another
    machine (the test makes other bytes by another codec or by a changed writer).

11. **Second read of this work.** One subagent that had not seen this reasoning got the diff of
    the fixes and of the video key, and a copy to run. It found no major defect, 2 minor and 9
    small ones; each was run by it unless said:
    - Minor, fixed: opened first, `--feed` on a plain file emptied it before any check, the job
      file included (`--feed JOB`: exit 2 and an empty job). Now only a named pipe and `-` are
      opened first, and `--feed` on `--out` or `--job` is refused (`os.path.samefile`). Test
      `test_a_feed_that_names_the_job_file_is_refused_and_the_job_is_kept` (fails without it).
    - Minor (reasoning), fixed in RUNBOOK section 14: the tools load their models only after
      `merge-feeds` opens their pipes, so `START` 60 s ahead of terminal 1 could pass before the
      models are loaded; now 120 s, and the three terminals within 30 s.
    - Small, fixed: `--feed -` in a process that calls `main` left its descriptor 1 on the error
      stream after `main` (`StandardOutputFeed.close` gives it back). The first test of it ran in
      the test process, where `sys.stdout` is the capture file of pytest, and passed without the
      fix; the test now runs a process and fails without it.
    - Small, fixed: with `--allow-other-images`, a hit with other image bytes and a rewritten video
      did not count the video (test `test_a_rewritten_video_is_counted_also_when_other_image_bytes_are_allowed`).
    - Small, fixed: after a resend that broke, the reopen loop of `NamedPipeFeed` did not look at
      its deadline (bounded in practice: 0.33 s with a wait of 0.3 s); it does now.
    - Small, written as limits in the docstring of `NamedPipeFeed`: a stray reader that takes some
      bytes and closes before the next line, with `merge-feeds` open by then, takes them from the
      feed (run with fast stand-ins; real tools write the header after the tool is built); on
      Linux a reader that opens in the microseconds between the broken write and the close sees
      the end at once (reasoning); a child forked without exec keeps the old end open (reasoning).
    - Small, fixed: the docstring of `VideoPart` named `identity` as what a stored answer keeps;
      ARCHITECTURE now says that a cassette names all written frames of a video (frame numbers,
      no label data), also those that `stored_context` leaves out.
    - Its design note: the review of round 6 recommended stopping a hit on other bytes; the brief of
      this session asked for a counted note, not an error, and that is what is built (item 8).
    - Checked and found right by it: the key of a written video changes with frames, write rate,
      size, resolution, prompt, model, range and sampling rate and stays for other bytes; a
      written and a foreign video cannot share a key; the pixels check refuses new footage and a
      cassette without the hash; the counter is under a lock and counts once per hit; rule 2 holds
      for the cassette; the open that does not block gives `ENXIO` without a reader on Linux and
      macOS, and a reader that waits in its open counts as a reader on Linux.

12. **Test runs.** Build machine, `tools/run_tests.py`, 14 workers, after the last fix: default run
    3,516 passed, 1 skipped, 10 deselected, exit 0, 75.6 s (load 20.6); full run (`-m "slow or not
    slow"`) 3,526 passed, 1 skipped, exit 0, 51.7 s (load 14.5); on 72fad94 the full run gave 3,501
    passed. `python3 tools/guard.py --tracked`: exit 0. `stream --synthetic`: exit 0. The `demo`
    predictions of both engines are byte-equal to 72fad94 (SHA-256 `bb50efd43536dd71...` for the
    model engine, `1f97af18358d652b...` for the detector engine).

13. **Files outside the area of this session.** `src/pitchvlm/cli.py`: `SessionBackend.other_video_bytes`
    and the note in `close_session` (`print_rewritten_videos`), no option and no command;
    `docs/HANDOVER.md` (step 21 and decision 42).


## 2026-11-23 — A live output and a radar viewer (session "live output" of round 7)

Base `main` 72fad94. The command `stream` decided frame `t` once frame `t + L` was in, but wrote
its files only at the end of the clip. Built: a per-frame result feed (`liveout.py`, option
`stream --live-out FILE|-`), a radar viewer of that feed on the loopback address (`radar.py`,
command `radar`), one hook in `stream.py` (`DetectorStream.on_given`, 5 lines), a bench
(`evalx/liveoutbench.py`), RUNBOOK section 15, and the contracts in `docs/ARCHITECTURE.md`.
Everything is measured on the build machine with synthetic clips and stand-ins of the tools;
nothing ran with a real tool, real footage or a real camera.

1. **Format** `pitchvlm-live-1` (ARCHITECTURE, `liveout.py`): a header, one compact JSON line per
   decided frame in frame order (frame, time, `decided_at`, delay, wall clock of the decision,
   camera source, flags `transition` and `cut`, lines left out before it, and per person id,
   role, team key, team side, shirt number or null, pitch metres, box), an end line with the
   counts. No pixel and no label value (rule 2). The name of the option cannot be mixed up with
   `--feed` of `tools/run_tool.py` (a tool feed) or `--lines` of `stream` (its input).
2. **Final at delay L, no revision line.** The stream gives each state once and never changes
   it, so the line of frame `f` equals frame `f` of `result.json` (state, camera, delay). Measured
   on held-out seeds 600 to 609 (`liveoutbench revisions`; kinds of `evalx/streambench.py`; 160
   clips of 250 frames, 40,000 frame lines): **0 lines differ** from their frame of the result.
   What the result says more, and how often:

   | kind | condition | delay | clips | frames | lines unequal | dissolve frames | flag known late | player states | role/team/side of the track differs |
   |---|---|---|---|---|---|---|---|---|---|
   | plain | exact | 25 | 10 | 2500 | 0 | 0 | 0 | 39286 | 717 (1.83%) |
   | plain | exact | 5 | 10 | 2500 | 0 | 0 | 0 | 39275 | 1468 (3.74%) |
   | plain | noisy | 25 | 10 | 2500 | 0 | 0 | 0 | 42510 | 375 (0.88%) |
   | plain | noisy | 5 | 10 | 2500 | 0 | 0 | 0 | 41718 | 853 (2.04%) |
   | cut | exact | 25 | 10 | 2500 | 0 | 0 | 0 | 41526 | 40 (0.10%) |
   | cut | exact | 5 | 10 | 2500 | 0 | 0 | 0 | 41526 | 912 (2.20%) |
   | cut | noisy | 25 | 10 | 2500 | 0 | 0 | 0 | 44259 | 24 (0.05%) |
   | cut | noisy | 5 | 10 | 2500 | 0 | 0 | 0 | 43751 | 515 (1.18%) |
   | dissolve | exact | 25 | 10 | 2500 | 0 | 93 | 0 | 40450 | 40 (0.10%) |
   | dissolve | exact | 5 | 10 | 2500 | 0 | 93 | 85 | 41530 | 1020 (2.46%) |
   | dissolve | noisy | 25 | 10 | 2500 | 0 | 94 | 0 | 43119 | 24 (0.06%) |
   | dissolve | noisy | 5 | 10 | 2500 | 0 | 94 | 86 | 43766 | 699 (1.60%) |
   | dissolve-20 | exact | 25 | 10 | 2500 | 0 | 169 | 0 | 39618 | 41 (0.10%) |
   | dissolve-20 | exact | 5 | 10 | 2500 | 0 | 169 | 169 | 41710 | 1495 (3.58%) |
   | dissolve-20 | noisy | 25 | 10 | 2500 | 0 | 169 | 0 | 42820 | 59 (0.14%) |
   | dissolve-20 | noisy | 5 | 10 | 2500 | 0 | 169 | 169 | 44303 | 1063 (2.40%) |

   Load average 4.5 before, 10.2 after (14 cores); 4 processes; 3.6 min.

   - "flag known late": frames of a final dissolve span whose line went out without the flag
     `transition`. At L = 25 none (the early dissolve test of round 6 knows every dissolve in
     time); at L = 5 most dissolve frames went out before the dissolve was known (85 of 93,
     86 of 94, 169 of 169). Their states are those of the result: the stream then drops nothing
     on those frames, as before this change.
   - "role/team/side of the track differs": `result.json` lists tracks whose role, team key and
     side are those of the last state of the id (`stream.tracks_of_states`); a state given earlier
     can have another value. 0.05 % to 3.74 % of the player states, more at L = 5.
   - Decision: the format says "final at delay L"; no revision line. A consumer that needs the
     final role of a track reads `result.json` at the end. A revision line would send, for a
     dissolve, states that the stream does not change, and for a track a value that only the
     last state has.
3. **Policy for a slow consumer.**
   - A regular file: every line is written with `os.write` in the thread of the stream as the
     frame is decided (no buffer, so a reader sees it at once). Complete; a write error stops
     the command (exit 2). Its cost per frame is in item 5.
   - A pipe, a named pipe, a terminal or `-`: the line is made and checked in the thread of the
     stream, then handed to a writer thread; the stream never waits for the reader. While the
     reader holds the writer (one write has taken longer than `PIPE_BLOCKED_S` = 50 ms: the pipe
     is full), one line waits (`PIPE_PENDING_LINES = 1`) and a newer line makes the waiting one
     go, counted in `dropped` (each line carries the count so far, the end line the total). The
     header and the end line never go. Why one line: for a viewer the newest state matters; the
     pipe of the system already holds 16 to 64 kB (3 to 12 lines of about 5 kB), a second queue
     that no count sees, so a reader that falls behind reads lines up to that many frames old;
     `decided_unix` lets it see how old. A reader that goes away ends the output, not the stream.
   - A first version left a line out whenever one was waiting. The first latency run (seeds 610
     to 612, fast reader) lost 18 to 24 of 250 frames per run, all of them in the burst of the
     last 25 frames that `finish` decides together: there the writer only waited for its turn
     (the switch interval of Python is 5 ms), it was not held by the reader. Hence the rule
     "held for 50 ms"; a writer that is not held takes up to `PIPE_MAX_LINES` = 250 waiting
     lines. A test with a fast reader and a burst of 60 decisions fails without the rule.
   - At the end the stream waits at most 5 s (`CLOSE_WAIT_S`) for a slow reader, then reports the
     lines not delivered. Tested with a reader of one line per second (one byte per read, so the
     pipe is not emptied in chunks): 60 frames at 25 fps, every hand-over below 0.2 s (the bound
     of the test), frames in order with gaps, counts that add up.
   - `--live-out -`: the output of `stream` goes to the error stream (`contextlib.redirect_stdout`),
     the lines to a copy of the standard output.
   - Refused (`cli.refuse_live_out`): a path inside a work tree of the repository, also through a
     link; inside `--out`; a folder; an input file; the standard output named as a path
     (`/dev/stdout`, `/dev/fd/1`: give `-`).
4. **The viewer** (`radar`, standard library only). Binds to a loopback address written as numbers
   (a name, `0.0.0.0`, `::`, an IPv6 address with a zone or a mapped IPv4 address are refused
   before a socket is made); port 0 by default, printed. Serves `GET /` (one page: inline script
   and style, the pitch drawn on a canvas, no external resource) and `GET /events` (server-sent
   events of the lines, parsed and written again as compact JSON); 404 for every other path, 405
   for every other method, 403 for a `Host` that does not name the server (DNS rebinding) and for
   a request of a page of another site (`Origin`, `Sec-Fetch-Site`); `Cache-Control: no-store`,
   a content security policy with `default-src 'none'` and the SHA-256 of the script and the
   style, `nosniff`, `X-Frame-Options: DENY`. At most 8 event streams and 32 connections; a line
   of more than 1 MiB, a line that is not UTF-8 or JSON, a nesting too deep, and a frame line
   without a finite time are refused. A client more than 25 frames behind jumps to the newest
   frame. It ends when the feed ended and the pages got the end. The page shows the players by
   team colour, the shirt number when known, the ball, the time, the delay, how far the page is
   behind the decision, and the frames it did not show. **Looked at** in the built-in browser of
   the desktop app (Chromium), with `radar --replay` on a 120-frame synthetic feed: pitch, players,
   referee, goalkeepers, ball, header values and a planted `transition` flag shown; shirt numbers
   drawn on a copy of the feed with planted numbers (the stream itself has no number step); no
   console message (no CSP violation). Not looked at: Firefox and Safari, a live (not replayed)
   run in a browser, a phone.
5. **Measurements.**

   **Time per frame** (`liveoutbench timing`): `stream --synthetic --frames 250 --delay 25` as a
   process per run, seeds 600 to 602 (held out), 4 rounds (forward, reverse, forward, reverse),
   the median and the 95th percentile of the frame samples of `result.json` (the hook runs inside
   the frame sample, so the writing of a line is in it). Arms: `base` (a copy of `72fad94`),
   `none` (this code, no option), `file` (`--live-out` a file), `slow` (`--live-out -` into a
   reader of one line per second, one byte per read). The frames that `finish` decides at the end
   are in no frame sample.

   Run 1, load average 3.1 to 11.3 (14 cores), 12 runs per arm (n = 12 pairs):

   | arm | runs | median ms, mean (range) | p95 ms, mean (range) | load (range) |
   |---|---|---|---|---|
   | base | 12 | 19.53 (19.03 to 20.19) | 22.38 (21.41 to 23.60) | 3.4 to 8.5 |
   | none | 12 | 19.96 (19.00 to 25.44) | 24.08 (21.51 to 42.04) | 3.7 to 8.1 |
   | file | 12 | 19.98 (19.08 to 26.96) | 24.08 (21.23 to 43.67) | 3.3 to 9.1 |
   | slow | 12 | 19.60 (19.11 to 21.57) | 22.47 (21.62 to 25.26) | 3.1 to 11.3 |

   | difference (paired by seed and round) | median ms, mean [95% t interval] | p95 ms, mean [95% t interval] |
   |---|---|---|
   | file - base (n = 12) | +0.46 [-0.83, +1.74] | +1.70 [-2.01, +5.42] |
   | slow - base (n = 12) | +0.07 [-0.28, +0.43] | +0.10 [-0.39, +0.59] |
   | none - base (n = 12) | +0.43 [-0.57, +1.43] | +1.71 [-1.71, +5.13] |
   | file - none (n = 12) | +0.03 [-0.31, +0.36] | -0.00 [-0.62, +0.61] |
   | slow - none (n = 12) | -0.35 [-1.09, +0.38] | -1.61 [-4.67, +1.45] |

   The two largest values (median 25.4 and 27.0 ms, 95th percentile 42.0 and 43.7 ms) are both in
   one round (seed 600, round 3, arms `none` and `file`, load up to 9.1): the arm without the
   option jumped as much as the arm with it. The medians of the paired differences: `file - base`
   -0.17 ms (median) and +0.20 ms (95th percentile), `file - none` -0.01 and -0.05 ms.
   In the `slow` arm the writer left 223 to 226 of 250 lines out per run (24 to 27 went into the
   pipe; the reader read about one per second).

   Run 2 (after the change of the pipe rule of item 3), load average 4.8 to 60.8 (other sessions
   started their tests during it), same design:

   | arm | runs | median ms, mean (range) | p95 ms, mean (range) | load (range) |
   |---|---|---|---|---|
   | base | 12 | 28.83 (19.49 to 67.11) | 54.84 (21.60 to 153.06) | 5.0 to 57.4 |
   | none | 12 | 32.52 (19.79 to 72.20) | 58.69 (22.24 to 152.37) | 4.8 to 57.0 |
   | file | 12 | 31.58 (20.09 to 64.79) | 55.74 (22.67 to 150.45) | 5.1 to 56.7 |
   | slow | 12 | 31.32 (19.53 to 59.95) | 61.07 (22.28 to 137.75) | 5.3 to 60.8 |

   At that load the intervals are 10 to 30 ms wide: run 2 says nothing about 1 or 2 ms.

   Run 3 (the final code; started when the load of the last minute was below 10), load average
   5.3 to 28.3, arms `base`, `slow`, `file`, 12 runs per arm:

   | arm | runs | median ms, mean (range) | p95 ms, mean (range) | load (range) |
   |---|---|---|---|---|
   | base | 12 | 22.64 (19.65 to 35.77) | 28.98 (21.91 to 61.19) | 5.6 to 20.6 |
   | slow | 12 | 19.94 (19.07 to 20.65) | 22.37 (21.54 to 23.30) | 5.7 to 28.3 |
   | file | 12 | 19.94 (19.03 to 20.47) | 22.52 (21.60 to 24.45) | 5.3 to 24.3 |

   | difference (paired by seed and round) | median ms, mean [95% t interval] | p95 ms, mean [95% t interval] |
   |---|---|---|
   | file - base (n = 12) | -2.70 [-6.51, +1.11] | -6.45 [-15.61, +2.70] |
   | slow - base (n = 12) | -2.70 [-6.45, +1.04] | -6.60 [-15.68, +2.47] |

   Here `base` had the disturbed runs (largest median 35.8 ms, 95th percentile 61.2 ms); the
   medians of the paired differences are `file - base` -0.38 / -0.35 ms and `slow - base`
   -0.31 / -0.69 ms. `slow - file`: -0.00 ms [-0.20, +0.20] on the median, -0.15 ms
   [-0.46, +0.16] on the 95th percentile. The writer left 213 to 220 of 250 lines out per run.

   **Verdicts.** Hypothesis 1 (file, below 1 ms on the median and 2 ms on the 95th percentile
   against `72fad94`): the point estimates of run 1 are below both limits (+0.46 and +1.70 ms;
   medians of the differences -0.17 and +0.20 ms), but the upper ends of the 95% intervals are
   above them (+1.74 and +5.42 ms), because of one disturbed round that hit the arm without the
   option as much. Against the same code without the option, which is byte-equal in its result
   and its path to `72fad94` (item 6), the feed costs +0.03 ms [-0.31, +0.36] on the median and
   -0.00 ms [-0.62, +0.61] on the 95th percentile: inside the limits. So: **holds against the
   control of the same code; against `72fad94` itself not shown at 95%** with n = 12 at this
   load; run 3 says the same (no interval against `72fad94` inside the limits, every median of the
   differences below 0.4 ms). Hypothesis 2 (a reader of one line per second changes nothing):
   **holds**: run 1 `slow - base` +0.07 ms [-0.28, +0.43] and +0.10 ms [-0.39, +0.59] (the pipe
   rule of run 1 differs from the final one only when the writer is not held; here it is held),
   run 3 `slow - file` -0.00 ms [-0.20, +0.20] and -0.15 ms [-0.46, +0.16]. A repeat on a quiet
   machine with more rounds would settle hypothesis 1 against `72fad94` itself.

   **Latency** from the decision of a frame (`decided_unix`, the wall clock in the stream) to the
   event at a client of the event stream (`liveoutbench latency`): three processes, the stream
   of the synthetic clip paced at 25 frames per second with `--live-out` into a pipe, `radar -`
   on that pipe, and a client with `urllib` in the bench process; seeds 610 to 612, 250 frames.

   Run 1 (the first pipe rule; it left the burst of the last frames out):

   | run | load | frames received / decided | dropped by the writer | median ms | p95 ms | largest ms |
   |---|---|---|---|---|---|---|
   | seed 610 | 3.4 | 227 / 250 | 23 | 0.40 | 0.49 | 6.80 |
   | seed 611 | 3.1 | 232 / 250 | 18 | 0.42 | 0.54 | 5.95 |
   | seed 612 | 2.7 | 226 / 250 | 24 | 0.41 | 0.50 | 0.61 |
   | all (685 frames) | - | - | - | 0.41 | 0.51 | 6.80 |

   Run 2 (the rule of item 3), load average 47 to 49:

   | run | load | frames received / decided | dropped by the writer | median ms | p95 ms | largest ms |
   |---|---|---|---|---|---|---|
   | seed 610 | 48.6 | 227 / 250 | 2 | 0.86 | 57.56 | 176.14 |
   | seed 611 | 47.5 | 250 / 250 | 0 | 0.85 | 14.06 | 63.99 |
   | seed 612 | 48.5 | 250 / 250 | 0 | 1.18 | 23.99 | 64.64 |
   | all (727 frames) | - | - | - | 0.99 | 23.99 | 176.14 |

   In run 2 the client of seed 610 got 227 frames while the writer left out only 2: at that load
   the client fell more than 25 frames behind once and jumped to the newest frame (the rule of the
   viewer).

   Run 3 (the final code), load average 6.4 to 7.0 (last minute; 13 to 27 over 5 minutes):

   | run | load | frames received / decided | dropped by the writer | median ms | p95 ms | largest ms |
   |---|---|---|---|---|---|---|
   | seed 610 | 6.5 | 250 / 250 | 0 | 0.46 | 104.57 | 248.78 |
   | seed 611 | 7.0 | 249 / 250 | 1 | 0.43 | 32.89 | 93.77 |
   | seed 612 | 6.4 | 250 / 250 | 0 | 0.44 | 90.84 | 190.76 |
   | all (749 frames) | - | - | - | 0.44 | 74.27 | 248.78 |

   Split by where the frame was decided (the two runs with all 250 frames, n = 450 and 50):
   frames decided as the clip came in (0 to 224): median **0.44 ms**, 95th percentile **0.56 ms**,
   largest **2.86 ms**; the last 25 frames, which `finish` decides together when the clip ends:
   median 100.8 ms, 95th percentile 221.7 ms, largest 248.8 ms. In that burst the writer gets the
   interpreter lock once per switch interval while the stream decides the next frame, so the 25
   lines go out over about the time of the 25 decisions. Run 1 had no such tail because its rule
   left those lines out. A limit at the end of a clip only: in a live run one frame is decided per
   frame that comes in.

6. **Controls** (byte comparison against a copy of `72fad94` from `git archive`, seed 615, 250
   frames, delay 25): `stream --synthetic` without the option gives the same prediction file,
   the same `result` of `result.json`, the same identity and diagnostics, and the same video;
   with `--live-out` the same again, and `result.json` has the key `live_out` in addition. The
   prediction files of `demo` and `demo --engine detector` are byte-equal.
7. **Second readers.** Code (liveout, hook, CLI): 2 medium findings, both fixed with a test that
   fails without the fix: a writer thread that failed with an error other than its reader's
   ended without a word (now raised by `close`), and a link outside the repository that pointed
   into it passed the refusal. Low findings fixed: the report read outside the lock and a writer
   that went on after `close` (now copied under the lock and given up); a close without `finish`
   that left the writer waiting (now stops it); the header of a pipe written in the thread of
   the stream (now queued); `/dev/stdout` as a path and a path inside `--out`; the wording of
   "not delivered". Security (server, page): no high finding. Fixed with tests: no limit on the
   length of a line and a reading that took quadratic time (a 64 MB line took 93.5 s; now a cap
   of 1 MiB, linear); a page of another site could hold the event streams and start a replay
   (now 403); no cap on connections (now 32); a hostile feed gave a traceback (now a clean
   error); error pages without the security headers; mapped and zoned IPv6 addresses. Not
   changed: a local process can read the feed (no token, by design: the server is for this
   machine), and the read timeout is per read, not per request.
8. **Tests** (14 workers): default run 3,535 passed, 1 skipped, 11 deselected, exit 0 (load 5.1);
   full run 3,546 passed, 1 skipped, exit 0 (`72fad94`: 3,501 passed, 1 skipped). 45 new tests
   (`tests/test_liveout.py` 21, `tests/test_radar.py` 24); every fix of the second reads has a
   test that failed with the fix undone (8 checks). `tests/test_documented_commands.py` now reads
   the options of RUNBOOK section 15 too. An earlier default run at a load of 7 to 10 failed in the
   3 tests of `tests/test_live_feed.py` that are known to race when the named pipes are opened
   (another session has them), and in the guard test for two addresses of this change (fixed).
   After the merge of `main` d0c5589 (the second read of the live feed): default run 3,560
   passed, 1 skipped, 11 deselected, exit 0; full run 3,571 passed, 1 skipped, exit 0. The live
   run of RUNBOOK sections 14 and 15 as processes (two stand-in tools paced at 25 fps into named
   pipes, `merge-feeds`, `stream --lines - --live-out -`, `radar -`, a client; 125 frames, seed
   617): every exit code 0, 125 of 125 frames and the end event at the client. The same with a
   calibration that keeps its pipe open for 30 s after its last frame: `stream` exit 0, its live
   output complete (125 written, 0 left out, end line) and the end event at the client after
   13.2 s, while the hung tool ran to 43 s; `merge-feeds` exit 1 (`left open: calibration 1`).
9. **Not verified.** A real detector, calibration and camera; Linux (named pipes, the pipe size);
   a browser other than Chromium; a live (not replayed) run in a browser; the 95th percentile at
   25 fps with the tools on the run machine; hypothesis 1 against `72fad94` at 95% (item 5).
10. **Proposed text** (no change to `CLAUDE.md`): none.


## 2026-11-23 — The offline tests on Linux in CI

Session "eng - linux ci" of round 7. Base `main` 72fad94. The owner approved a workflow in the
public repository with no secret. Until now nothing of this project had run on Linux; the run
machine is Linux, the build machine is a Mac (arm64).

1. **The workflow** `.github/workflows/offline-tests.yml`, on every pull request and every push
   to `main`. Never `pull_request_target`; `permissions: contents: read`; the checkout keeps no
   credentials (`persist-credentials: false`); no secret and no token beyond the checkout; a
   `concurrency` group per ref that cancels older runs; `timeout-minutes: 30` per job; the runner
   image `ubuntu-24.04`. Jobs: "full run, Python 3.12" and "full run, Python 3.13" (the guard on
   the tracked files, `tools/run_tests.py -m "slow or not slow"`, both demos, `stream
   --synthetic`, each later step also after a failed one) and "full run in one process, Python
   3.13" (`python -m pytest -m "slow or not slow" -rs`). `tests/test_workflows.py` loads every
   workflow with the YAML parser of the project and checks these rules, actions pinned by full
   commit, a check sum for every download and hashes for every install.
2. **Actions**, pinned by commit; the commit of each release tag was read on 2026-11-23 through
   the API of github.com (published, not measured):

   | Action | Version | Commit | Release page, date |
   |---|---|---|---|
   | `actions/checkout` | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` | github.com/actions/checkout/releases/tag/v7.0.1, 2026-07-20 |
   | `actions/setup-python` | v7.1.0 | `6d4fd22b20465ccc0f7354476588e8df7c9ca0e0` | github.com/actions/setup-python/releases/tag/v7.1.0, 2026-10-22 |

3. **Packages.** `locks/ci-linux.txt` holds the versions of the build machine (read from its
   environment) for the offline tests; `locks/ci-linux-hooks.txt` those of its `pre-commit`.
   Exact versions with `--require-hashes --only-binary :all:`; each hash is the SHA-256 of a wheel
   on the JSON page of the release on pypi.org (every wheel for Linux x86_64 with CPython 3.12 or
   3.13, and the wheel for any platform), read on 2026-11-23 without a download. The date is the
   first upload of the release; the youngest is 25 days old (published, not measured):

   | Package | Version | Uploaded | | Package | Version | Uploaded |
   |---|---|---|---|---|---|---|
   | numpy | 2.5.4 | 2026-10-10 | | pre-commit | 4.7.0 | 2026-10-12 |
   | opencv-python-headless | 5.0.0.93 | 2026-07-02 | | cfgv | 3.5.0 | 2025-11-19 |
   | scipy | 1.18.1 | 2026-08-21 | | distlib | 0.4.4 | 2026-10-16 |
   | pydantic | 2.14.2 | 2026-10-29 | | filelock | 4.2.4 | 2026-10-19 |
   | pydantic-core | 2.50.2 | 2026-10-29 | | identify | 2.6.20 | 2026-09-26 |
   | pyyaml | 6.0.3 | 2025-09-25 | | nodeenv | 1.11.0 | 2026-09-26 |
   | pytest | 9.2.0 | 2026-10-27 | | platformdirs | 4.13.5 | 2026-10-19 |
   | pluggy | 1.7.0 | 2026-10-15 | | python-discovery | 1.6.4 | 2026-10-19 |
   | iniconfig | 2.3.1 | 2026-10-06 | | virtualenv | 21.14.5 | 2026-10-02 |
   | packaging | 26.3 | 2026-08-04 | | | | |
   | pygments | 2.21.0 | 2026-08-17 | | | | |
   | annotated-types | 0.8.0 | 2026-07-23 | | | | |
   | typing-extensions | 4.16.0 | 2026-07-02 | | | | |
   | typing-inspection | 0.4.4 | 2026-08-12 | | | | |

   `gitleaks` 8.30.1 (the version of the build machine; release of 2026-03-21) comes from the
   file `gitleaks_8.30.1_linux_x64.tar.gz` of its release page on github.com and is checked
   against the line of that file in `gitleaks_8.30.1_checksums.txt` (SHA-256 `551f6fc8...70eb`,
   in the workflow). The model SDK and the external tools are not installed. No lock file of
   another form is committed (rule 3).
4. **OpenCV: the headless wheel.** The full wheel `opencv-python` needs the system graphics
   library (`libGL.so.1`) on Linux; installing a system package needs the package index of the
   runner image, which is neither pinned nor hashed. The headless wheel of the same version gives
   the same `cv2` module without the window functions, and the package calls none of them
   (`imshow`, `namedWindow`, `waitKey`: no hit). `import cv2` is unchanged. The video writer of
   the Linux wheel has `mp4v` and `MJPG`; it has no H.264 encoder, so `vlm/video.py` falls back
   from `avc1` to `mp4v` there.
5. **The leak check in CI has no deny-list** (`.git/guard-deny.txt` stays on each machine): the
   guard prints its notice and the rule `deny` is off; every other rule of `--tracked` runs, and
   it passed. `doctor --no-api` is left out: it needs the hooks installed in the clone and the
   deny-list. The workflow names one host, github.com, which `tools/allowed_hosts.txt` has.
6. **One process, a second job: yes.** It runs beside the other jobs (no longer wall time than
   the slowest job), and it is the only run where all tests share one interpreter, as with
   `python -m pytest`. With `-rs` it also prints the reason of every skipped test.
7. **Prediction against result.** Hypothesis: the full offline run passes on Linux without a code
   change. The ranked list was written before the first result (notes of this group of sessions).

   | Predicted (rank) | Result on Linux |
   |---|---|
   | 1 live chain with a pace ends with code 2 before the tools end | passed |
   | 2-5 thread and pipe timing of `test_feed.py`, `test_live_feed.py` | passed |
   | 6 frame rate of a rewritten mp4 (FFmpeg of the wheel) | passed |
   | 7 dissolve found on decoded `mp4v` frames | passed |
   | 8 git process counts and messages of the guard | passed |
   | 9-10 `avc1` fallback, truncated Motion JPEG | passed |
   | process start method, `sysctl` (no risk: none used) | no failure |
   | skips: 9 to 10 | 26 (below) |
   | not predicted: YAML of the workflow | round 1 refused the file |
   | not predicted: a stop signal during `Thread.join` on Python 3.12 | round 2 failed (item 8) |

   So the hypothesis failed twice for reasons that the static read did not see: once in the
   workflow, once in the code (a real defect, item 8). None of the ten predicted risks failed.
8. **A defect of Python 3.12 in the wait for paid requests in flight.** Round 2 failed on Python
   3.12 only: `test_second_stop_signal_does_not_end_the_wait_for_the_requests_in_flight` logged
   "1 model requests were in flight" instead of 2. Cause (measured on the build machine with its
   Python 3.12.16 and 3.13.18, stdlib only): on 3.12 a `KeyboardInterrupt` during `Thread.join`
   of a running thread marks that thread as stopped (`is_alive()` false) and every later join of
   it returns at once; on 3.13 the thread stays alive and the next join waits. Upstream: CPython
   issue 90882 on github.com (python/cpython, "Thread spuriously marked dead after interrupting a
   join call", first filed in 2022 as bpo-46726), closed on 2025-03-05 with the words that it is
   fixed in Python 3.13 and that 3.12 still has it (read on 2026-11-23; published, not measured).
   The wait of
   `pipeline._finish_requests_in_flight` joined the threads of the pool, so on 3.12 a second
   Ctrl-C ended the wait while a paid request was still in flight: its answer could be lost
   before it was stored and recorded. Fix: `_InFlight` counts a request when its thread begins
   it, under one lock with a stop flag; at the stop the set is closed, so a request that has not
   begun is not sent (also one whose thread took it already, or whose submission an
   interruption cut short), and the wait sleeps in steps of 10 ms until no request is sent. It
   joins no thread and waits on no condition or future. The pool is shut down with `wait=False`;
   its threads end after their last request. A model of the test (the code of `pipeline.py`, a
   fake model, the first Ctrl-C while two requests are at the model, a real second signal 50 ms
   after the wait began; 100 runs per arm): Python 3.12.16, old code, the warning named 1 request
   in 100 of 100 runs and an answer was lost in 43 of 100; new code, 2 requests and both answers
   in 100 of 100; Python 3.13.18, new code, 100 of 100. Each new test fails with a code that has
   its defect: joins of the pool that return at once (as on 3.12); a request whose thread took it
   but had not begun at the stop; a request queued by an interrupted submission; a second Ctrl-C
   while the pool cancels a queued request (a wait on futures hung there). Two first versions of
   this fix were wrong and are replaced: commit 2776c9d (only the test: a slow start of the
   second thread) and commit e5daf53 (a wait on futures: a request queued by an interrupted
   submission was sent after the return, and a cancel cut short by a second Ctrl-C made the wait
   hang). Two second readers found both. In the command line a stop also sets the stop flag of
   the model clients, which refuse a new request, so the risk of these windows was low there.
   The run machine uses Python 3.13, where the joins of the old code were right; `pyproject.toml`
   allows 3.12.
   Planted bugs in the new wait, one at a time, against `tests/test_pipeline.py` and
   `tests/test_cli.py` (413 tests): no close at the stop, the stop flag ignored, no retry after a
   second signal, and P01 of the entry "A planted-bug check of the money guards" (no stop after a
   failure) each end with exit code 1 (2 to 4 tests fail); a wait that returns at once and a count
   that is never lowered each end with exit code 2 (a worker of the runner did not end normally).
   6 of 6 found.
   A third reader (a fuzzer that raised a Ctrl-C at each of 471 points of the calling thread,
   Python 3.13, and real signals on 3.12 and 3.13) found no fault for one Ctrl-C and one gap that
   is not new: two different stop signals pending at the same time (for example SIGINT and
   SIGTERM, or SIGHUP and SIGTERM when a terminal closes). Python runs one handler per check;
   the second raises at the next check, which can be the jump back of the retry loop or the entry
   of `_finish_requests_in_flight`, outside its `try`. The function then returns with requests in
   flight (12 of 12 runs of its script). The code of 72fad94 does the same (3 of 3 runs of the
   same script on 3.13), so it is not a regression; in the command line the stop flag of the
   clients still refuses new requests, but the measured cost can be printed before the ledger
   lines of the answers in flight. A fix needs the signal handler of `cli.py` to hold a second
   signal back while the pool waits, as it does for a call in the main thread
   (`_CALL_IN_FLIGHT`): proposed as its own task, not done here. It also named stdlib limits that
   are not regressions: on 3.13 a Ctrl-C at one jump inside `concurrent.futures.wait` can keep
   the locks of the futures and hang the exit (a window of microseconds, not hit in 292 runs with
   a real Ctrl-C); a Ctrl-C inside `Thread.start` of the pool can leave a thread that never
   starts; a Ctrl-C in a few places of the stdlib becomes `RuntimeError: release unlocked lock`.
9. **Skipped tests.** The build machine skips 1. Round 2 on Linux skipped 26: 21 tests of the git
   hooks and the secret scan (`pre-commit` and `gitleaks` were not installed; the run machine has
   both, so these skips were a hole: CI installs both now), 4 tests of a disk that does not tell
   letter cases apart (`tests/test_cli.py`: a Linux disk tells them apart, so another spelling is
   another path; right, and the tests of that side run), and the image reader of
   `tools/run_tool.py`. Round 3 skipped 5: the 4 tests of letter case and the image reader. The
   image reader skips on the build machine too (its environment has no Pillow); the script runs
   in the environment of the tool, which brings its own Pillow, and no version of Pillow is
   known from the build machine to pin. So Pillow stays out of the pins of CI, and the skip
   stays: CI tests what the build machine tests.
10. **Times of round 2** (run 45302990921, one run, 4 CPUs, a shared machine; measured): install
    13 s, guard below 1 s, full run in 4 processes 132 s (3.12 and 3.13), each demo 6 s, `stream
    --synthetic` 6 s, a job about 3 min; the job of one process 5 min 46 s (pytest 327 s).
    `stream --synthetic` on the runner: total per frame median 25.84 ms, 95th percentile 31.99
    ms, largest 39.07 ms. The runner had Python 3.12.16 and 3.13.18 and pip 26.3.
11. **What CI does not cover:** no model call, no real tool, no footage, no dataset, no GPU, and
    a GitHub runner is not the run machine. The full wheel `opencv-python` that `pyproject.toml`
    installs on the run machine is not tested in CI.
12. **The build machine** at the time of this work: the three tests of the live chain in
    `tests/test_live_feed.py` hang there at a low load (a stand-in tool ends with a broken pipe
    and `merge-feeds` waits in the open of the other named pipe). Session "eng - review round 6"
    owns that code and its fix; until the fix is on `main` the full run on the build machine can
    fail in these three tests only. They passed in CI.
13. **The secret scan of the tests read compiled files.** Round 3 failed in all three jobs in
    `test_text_of_the_repository_passes_the_secret_scan[tests]`: `gitleaks dir` read the folder
    on the disk, with the `__pycache__` files that the run had written, where the fake keys of
    the tests stand without their allow marks (rules `openrouter-key`, `google-auth-key`). The
    build machine never failed so: its shell sets `PYTHONDONTWRITEBYTECODE=1`, so Python writes no
    compiled file there; with that variable unset the old test failed on the build machine too.
    The run machine writes compiled files by default, so its first full run would have failed
    the same way. The test now scans a copy of the files that git tracks (`git ls-files`), which
    is what a commit holds; the scan stays on for every folder, and `.github` and `locks` are
    scanned too. A new test checks that the copy holds no compiled file.
14. **CI rounds.** Round 1 (6528870): the workflow file was refused (item 7). Round 2 (9f51927,
    run 45302990921): 3.13 and one process green, 3.12 one failure (item 8), 26 skipped. Round 3
    (f3731a7, run 45308492172): the stop-signal test passed on 3.12 and 3.13; 3504 passed, 1
    failed (item 13), 5 skipped in each job; runner 144.8 s (3.13) and 145.6 s (3.12), one
    process 249.1 s; the checksum of gitleaks passed and the 21 tests of the hooks ran. Round 4
    (611861c, run 45311659695): green in all three jobs, the first full pass on Linux: 3508
    passed, 5 skipped in each; runner 206.4 s (3.12) and 207.6 s (3.13), one process 249.9 s.
15. **Why the run in 4 processes took 132, 145 and 207 s in rounds 2 to 4.** A new clone has no
    stored test times, so `tools/run_tests.py` balances the workers by test count, and the wall
    time is that of the slowest worker. Its lines in the log: round 3, workers 38.6 to 138.5 s;
    round 4, 111.7 to 198.7 s (3 more tests moved the split, and a worker got the heavy files).
    The run in one process, the same tests in one interpreter, took 249.1 s in round 3 and 249.9
    s in round 4; round 2 took 326.7 s without the 21 tests of the hooks. So the hooks do not
    explain the longer runs; the split by count and a shared runner do (one run per round, so the
    spread of the runner is not known). CI keeps no times file between runs.
16. **After the merges of round 7** (`main` d0c5589, then aebbf1e: the second read of the live
    feed, the live output and the radar viewer). Build machine, 14 workers: default run 3,571
    passed, 1 skipped, 11 deselected, exit 0, 44.1 s (load 6.9); full run 3,582 passed, 1 skipped,
    exit 0, 49.3 s (load 21.5); the three tests of the live chain passed (also 3 runs of the live
    feed tests alone at a load of 12 to 17). `guard.py --tracked`: exit 0. The `demo` predictions of
    both engines are byte-equal to 72fad94 (SHA-256 `bb50efd43536dd71...` and
    `1f97af18358d652b...`); `stream --synthetic` exit 0. The RUNBOOK now says 11 slow tests (it
    said 9) and the HANDOVER the counts of this full run.
17. **The video writer on Linux has no H.264.** Round 5 (4fad128 with `main` d0c5589, run
    45313141791, green: 3532 passed, 6 skipped in each job; runner 115.0 s on 3.12 and 130.4 s on
    3.13, one process 272.7 s) skipped one more test: `tests/test_vlm_video.py:931`, "OpenCV
    cannot write the codec avc1 here". Measured: the headless wheel 5.0.0.93 on the Linux runner
    cannot open `avc1`, so `vlm/video.py` writes the videos of the model steps with `mp4v`
    (MPEG-4 Part 2) in an MP4 file; the annotated video (`render.py`) is `mp4v` on every machine.
    On the build machine (the full wheel 5.0.0.93, macOS) `avc1` opens through the FFmpeg backend
    of the wheel and through AVFoundation, and `mp4v` through FFmpeg only (a check of
    `cv2.VideoWriter`, 64 x 48 pixels). Published (read on 2026-11-23): the README of
    opencv-python (github.com/opencv/opencv-python and its page on pypi.org) says that all wheels
    ship with FFmpeg under the LGPLv2.1 and that the headless wheels have no GUI; it names no
    H.264 encoder and does not say that the video input and output of the four packages are the
    same. So whether the full wheel on Linux (the run machine) opens `avc1` is not known; the same
    FFmpeg in both is likely (inferred, not measured). The providers: the video page of Gemini
    (ai.google.dev, now aistudio.google.com/docs/video-understanding) lists MIME types such as
    `video/mp4` and names no codec; the video page of OpenRouter (openrouter.ai/docs) lists
    `video/mp4`, `video/mpeg`, `video/mov` and `video/webm` and names no codec. So an `mp4v` file
    is a `video/mp4` request of an accepted type, and whether the model decodes MPEG-4 Part 2 as
    well as H.264 is not published. The video cache key of pull request 25 holds the frames of the
    clip and the settings, not the bytes: an answer stored for an H.264 video of the build machine
    is found again for the `mp4v` video of the run machine and counted as other bytes (the test
    above checks it where both codecs open; on Linux it skips). The model would see other pixels
    than the stored answer saw only on a paid run after a replay that used another machine. What
    the owner checks on the run machine: the codec (RUNBOOK, section 1, the CI part: one free
    command), then `doctor --video` (gate G0), whose video checks send a video written by the same
    code. Open owner decision 44 of `docs/HANDOVER.md`, section 11.1.
18. **Readers of the change in `pipeline.py`.** Four readers that had not seen my reasoning read
    the code. The first found the cause on 3.12 (item 8), the second the two faults of e5daf53,
    the third and the fourth (the fourth from the diff against 72fad94 alone) found no fault for
    one stop signal, the same normal and failure paths as at 72fad94 (the fourth: 299 random
    cases, the same answers and the same errors on 3.12 and 3.13), and the gap of two different
    signals at the same time (item 8; 40 of 40 bursts of 2,000 signals from another process ended
    the wait early, on 3.12 and 3.13, and 20 of 20 on the code of 72fad94; 0 of 60 at the rate of a
    person). Both named limits that are not new: a single Ctrl-C logs no warning (the warning is
    for a second signal), an interruption at two instructions inside `executor.submit` can leave
    a thread of the pool that never ends, and on 3.12 a third signal inside the lock of a logging
    handler can block the wait (reasoning). The docstrings now name the gap of two signals.


## 2026-11-23 — Second read of the live output, and stop signals that cannot end the wait for paid requests

Session "eng - review round 7" of round 8. Base `main` 41c5552. An outside read of the code that
round 7 merged (pull requests 24, 25 and 26), with fixes, and the fix of the gap of stop signals
that the CI session of round 7 reported. No model call, no spend: every request went to a fake
model. Build machine, Python 3.13.18; the load of the machine (`vm.loadavg`, 1 min) was high
during the whole session (other sessions ran tests): it is in each table.

**S. Stop signals during the wait for paid requests (major, fixed).**

1. *What happened on 41c5552.* The handler of the command line (`cli._interrupt`) set
   `STOP_REQUESTED` and raised `KeyboardInterrupt` in the main thread, at whatever point that
   thread was. The wait of `pipeline._generate_all` caught one interruption and waited again,
   but a second one could come inside its `except` clause, at its entry or at the jump back of
   its loop. Worse, the next signal could come inside the handler itself, while
   `threading.Event.set` held the lock of its condition: the inner handler raised there, the
   lock stayed taken, and the outer handler (and every later one) waited for it for ever. The
   command then never ended: the threads of the pool stored their answers, but the command
   printed no measured cost and needed `kill -9`. Seen with `faulthandler`: the main thread
   stopped in `threading.py`, `Condition.__enter__`, called by `Event.set` inside `_interrupt`,
   nested many times.
2. *Fix.* New module `stopsignals.py`. While paid requests are in flight in the main thread
   (`finish_in_flight`, and now the whole of `_generate_all`), the handler only records the stop
   (`held_stops`, `hold_back`) and returns, however many signals come. The pool sends no request
   after a recorded stop (a request that has not begun is refused), the main thread looks at the
   record every 0.1 s and logs the warning "a stop signal came while N model requests were in
   flight" (now also after the first signal), and the first stop is raised with its cause when
   the requests in flight have ended. A signal that comes while the handler runs is only
   counted (`enter_handler`): nothing is raised inside the lock of the event. `STOP_REQUESTED` is
   a `StopEvent`: a `threading.Event` whose lock the same thread can take again, so the handler
   can set it also while the main thread is inside `wait` of the event (a model client that
   waits before a new attempt, in the main thread). Behaviour for one signal and for two: as
   documented (the requests in flight end, their answers are stored and recorded, exit code 2;
   a second signal does not shorten the wait). Changed: no number of signals ends the wait. The
   way to force the end: Ctrl-\ (SIGQUIT) or `kill -KILL`; the answers in flight are then lost,
   can still be billed, and have no ledger line (RUNBOOK, section 4; owner decision 45 of
   `docs/HANDOVER.md`, section 11.1). A defect of the first version of this fix (commit
   cf05905), found by its own test before any report: a thread pool run by another thread
   refused its requests while the main thread held a stop, and raised the internal refusal to
   its caller; fixed in 5cc5b8e (only a call in the main thread refuses). Found by the second
   reader of the fix (b2b0e72): (a) after the stop was raised, a further signal raised again,
   inside the cleanup of the command or inside the error handling of `main`, so the process ended
   without its error line and exit code 2 (reproduced in 2 of 12 runs of its script; no answer
   was lost): a command is now interrupted once, and later signals are only counted
   (`mark_raised`); (b) `test_measured_cost_of_a_run_that_is_stopped_two_times_holds_every_answer_in_flight`
   waited for the shutdown of the pool, which now comes after the requests in flight, so it
   failed after 60 s: it now waits for the recorded stop.
3. *Hypothesis S1 and its measurement.* S1: under a burst of 2,000 stop signals from another
   process (SIGINT and SIGTERM in turn, sent with `os.kill` as fast as the parent can: in 0.021 s, median, range 0.001 to
   0.039 s), the wait ends early in 0 of 40 runs, where it was 40 of 40 on 41c5552 (the CI
   session's number). Script (`round8-review-scripts/burst_measure.py` and `burst_child.py` in
   the notes folder of the round): a child process installs `cli._interrupt` as `main` does and
   runs `_generate_all` on 12 requests with 4 in parallel against a fake model; when 4 requests
   are at the model the parent sends the burst, then lets the model answer 200 ms later. Early
   end: an answer came after `_generate_all` returned, or the child hung (stopped by
   `faulthandler` after 15 s; the place is classified from its traceback). Arms: the source of
   HEAD (5cc5b8e) and of 41c5552, run by run in turn, the order flipped every run.

   | arm | runs | ended early | returned before the answers in flight | hung inside the handler | hung elsewhere | the 4 answers before the return | load, median [range] |
   |---|---|---|---|---|---|---|---|
   | HEAD 5cc5b8e | 40 | **0** (95 % interval 0 to 8.8 %) | 0 | 0 | 0 | 40 | 8.7 [5.7, 12.9] |
   | 41c5552 | 40 | **40** (91.2 to 100 %) | 17 | 14 | 9 | 0 | 8.6 [5.7, 13.0] |
   | control, 1 signal, HEAD | 10 | 0 | 0 | 0 | 0 | 10 | 6.4 to 7.0 |
   | control, 1 signal, 41c5552 | 10 | 0 | 0 | 0 | 0 | 10 | 6.4 to 7.0 |

   S1 holds (measured; intervals: Clopper-Pearson). Again on the final code (232c9fd, with the
   rule "interrupted once" of item 2), the same script, 40 runs, load 38 to 53 (the tests ran at
   the same time): 0 of 40 early, the 4 answers before the return in 40 of 40. On 41c5552, in 4 of the 17 early returns the
   pool then started all 8 requests that were still queued: the fake model does not read the stop
   event; the real clients refuse a new request once `STOP_REQUESTED` is set, so there they would
   not be sent (reasoning, not measured). "Hung elsewhere": the child was stopped by
   `faulthandler` with no handler in its main stack; not looked at further. The pilot before the
   fix (4 runs, load 20 to 25) gave 4 of 4 early, 3 of them hangs in the handler.

4. *Test.* `tests/test_stopsignals.py`: three bursts of 2,000 signals from the test process to a
   child (about 5 s; a child that hangs is stopped after 30 s), and five tests without real
   signals (a signal inside the handler; a hold that records 2,000 signals; nested holds and
   other threads; the event set by the thread that holds its lock; no request after a recorded
   stop) and the pool of another thread. On 41c5552 the burst test fails (the child hung inside
   the handler) and the others cannot import `stopsignals`.

**Findings of the read, by severity.** Fixed means: a test that fails on 41c5552 passes now.

| # | Severity | Where (41c5552) | What | State |
|---|---|---|---|---|
| S | major | `cli.py:5002-5016`, `pipeline.py:461-533` | item S above: a burst of signals ended the wait for paid requests, or blocked the command for ever | fixed, `tests/test_stopsignals.py` |
| R1 | minor | `radar.py:251`, `radar.py:176`, `radar.py:269` | a number too large for a float (`1e400`) anywhere but `t_s` passed the check, then the events (JSON without infinity) stopped `radar` with a `ValueError` and a traceback, as a fault of the code; a whole number of 401 digits in `t_s` raised `OverflowError` (found by the second reader in the first fix) | fixed: both refused as wrong input (whole numbers up to 2**53, what the page can hold); `test_a_number_too_large_for_a_float_is_refused_as_wrong_input_not_as_a_fault` |
| R2 | minor | `radar.py:278-291` | a second `stream` to the same `--live-out` file empties it (`O_TRUNC`); the viewer read on from its old place and waited for ever; a file replaced under its name was never seen either | fixed: `RadarError`; `test_the_viewer_stops_when_its_file_is_emptied_or_replaced` |
| R3 | minor | `radar.py:580` | the queue of connections that wait for their accept held 16, the server takes 32: while the loop of the server was late, connections 17 to 32 were reset (macOS) or hung (Linux); `test_connections_above_the_limit_are_closed_at_once` failed once in the one-process run at a load of 58 (a check without the server loop: the 17th connection is reset at 16, none at 32) | fixed: the queue holds `MAX_CONNECTIONS`; `test_connections_up_to_the_limit_that_come_while_the_server_is_late_wait_for_their_accept` |
| B1 | minor (a claim) | `evalx/liveoutbench.py:117-128` | the bench `revisions` rebuilds each line from the result with the line's own `decided_at`, `transition` and `cut`, and both sides take `fps=25.0`: its "0 of 40,000 lines differ" covers the people, the ball and the camera of each frame, not those three flags, `t_s` or `delay_frames` | written here; the flags are covered by `tests/test_liveout.py` (planted bugs below) |
| N1 | nit | `liveout.py:395-413` | `close` can count a line that its writer has just written, but whose lock it has not taken again, as not delivered (`stuck`), and then leaves the file descriptor open | not fixed: a window of a few instructions, only at the end, only the count |
| N2 | minor (was nit) | `liveout.py:346`, `398` | the counts of lines not delivered: a header that its reader never took counted as a line (the report says "frame lines and the end line"), one that went out counted nowhere, so `written + dropped + undelivered` was `given + 1` or `given + 2` by the order of the writer and the close of the reader; Linux CI failed on it once (pull request 27, 1 of 3 jobs, `undelivered=22` for 20 frames); a writer that fails with an error that is not one of its reader does not count the line it was making | the first fixed in the commit after 86d7b2f: the header is in no count, also when its write is still held at the close; both orders are forced in `test_a_reader_that_goes_away_ends_the_output_and_not_the_stream` and `test_a_header_whose_write_is_still_held_at_the_close_is_in_no_count` (both fail on 86d7b2f). The second is not observable (`close` raises then) and stays. Two tests left to timing were made deterministic: a close without `finish` now reads the lines first, and the first radar test sends its frames after its client has the header |
| N3 | nit | `liveout.py:195` | the docstring says that each line is made and checked in the thread of the stream; for a pipe, the JSON is made in the writer | not fixed (text) |
| N4 | nit | `cli.py` `_stream_command` | `live_out.close()` runs before `result.json` is written: a writer that failed (not its reader) would stop the command without its result files | not fixed: same unreachable failure as N2 |
| N5 | nit | `radar.py` `_check_same_feed` | a new `stream` that writes more bytes than the old file had before the next look (50 ms) is not seen; a file system whose inode numbers change between looks (some network mounts) would stop the viewer | not changed: the local disk of the run machine keeps the inode under `O_TRUNC` (second reader, reasoning) |
| N6 | design | `radar.py:797` | a live viewer that no page ever opened ends as soon as the feed ended | not changed: as documented (`--replay` for a finished file) |

The security read of the radar server found no fault: it binds only addresses that
`check_loopback` accepts (numbers, loopback, no zone, no mapped IPv4: `localhost`, `0.0.0.0`,
`::ffff:127.0.0.1`, `::1%lo0`, `127.1` and `0127.0.0.1` are refused before a socket is made); the
`Host` header must be exactly one of `<address>:<port>` and `localhost:<port>`; `Origin` and
`Sec-Fetch-Site` refuse the requests of other sites; only `/` and `/events` are served, the query
and the absolute form of a path are 404; the request line and headers are bounded by
`http.client` (65,536 bytes per line, 100 headers) and a request body is never read; 32
connections and 8 event streams at most, each read and write at most 10 s; every value of the
feed reaches the page through `textContent` or the canvas, and the events are JSON written again,
whose `\n` and `\r` are escaped. Limit (accepted): a local program can hold the 32 connections.
The workflow of CI (read-only token, actions pinned by hash, no value of a pull request in a
command) and the video cache key of pull request 25 (the frames, size, rates and range; the bytes
and the pixels in the cassette and compared on a hit) gave no finding. The resend of the feed
from its header (`NamedPipeFeed`) gives one reader each line once, from the header; a second
header reaches `merge-feeds` only in the documented case of a reader that took bytes and left,
and is then counted as a broken line.

**Planted bugs.** 64 single-line bugs, one at a time, each against the tests of its file (`liveout.py`
and `radar.py`: `tests/test_liveout.py` and `tests/test_radar.py`; the wait, the hold and the
handler: `tests/test_pipeline.py`, `tests/test_stopsignals.py` and the tests of stop signals of
`tests/test_cli.py`; the resend and the merger: `tests/test_live_feed.py`,
`tests/test_live_feed_planted.py`, `tests/test_run_tool.py`, `tests/test_feed.py`), full run with
`-x`, a time limit of 900 s; script `round8-review-scripts/planted.py`. The bugs of `liveout.py`,
`radar.py` and the resend ran on 5cc5b8e; those of the wait, the hold and the handler ran again on
b2b0e72, because the first tree held the stale test of item 2 (b), which failed with every bug.
First run: **47 caught** (one of them a hang that the time limit ended), **17 missed**. Twelve of
the missed ones matter and are now caught by a new test (each run again against it); three are
equivalent (L07 in practice, L10, R03); L12 plants into an option that no caller uses
(`LiveOut(close_fd=False)`; proposed: remove it); P07 opens a window of a few instructions and has
no test. Each new test passes on the code and fails with its bug.



Not verified: Python 3.12 and Linux (CI runs both after the push); the burst on a machine with a
low load; a real model client (the second reader checked the main thread inside
`STOP_REQUESTED.wait(10)` in a hold with a script: it returned after 0.2 s, for 1 signal and for
a burst of 3,000).

## 2026-11-23 — A faster stream and an overload policy (session "stream speed" of round 8)

Base `41c5552`. Build machine (14 cores), synthetic clips with the stand-ins of the tools, held-out
seeds 2000-2039 (2040 for the profile and the smoke runs only). Nothing here is a result on real
footage, a real tool or the run machine. GS-HOTA was not computed: its evaluator is not installed
on the build machine (`evalx/scorer.py`), and nothing is installed here; the price is given with
the box recall and the identity values of `evalx.metrics` instead.

1. **Where the time of a frame goes** (measured, one run per row, median ms per frame of the frame
   samples, code `41c5552`):

   | case | read | motion | decide | total |
   |---|---|---|---|---|
   | `stream --synthetic`, 100 frames (default), idle | 3.29 | 7.35 | 3.80 | 15.69 |
   | `stream --synthetic`, 250 frames, idle | 3.43 | 7.31 | 8.40 | 20.16 |
   | `stream --synthetic`, 500 frames, idle | 3.21 | 7.21 | 7.88 | 19.47 |
   | live input (PNG folder, merged lines file, 300 frames), idle | 5.30 | 7.56 | 9.01 | 22.60 |
   | the same, 12 busy processes (6 runs, OpenCV threads 14, 1 or 2) | 6.1 to 7.7 | 9.8 to 13.5 | 10.0 to 12.4 | 29.7 to 37.3 |

   Inside `motion`: `estimate_motion` 5.3 ms (corners 1.0, two optical flows about 2.0 each), the
   block correlations 1.2 ms. Inside `decide` (300 frames): the ball tracker 1.5, the mean pitch
   positions 1.5, `finalize_tracks` 1.0, the joins 0.7, the kit roles 0.7 ms. Parsing a line
   (JSON, the adapters) is 0.2 ms, outside the frame sample. The gap from about 15 ms to 39-93 ms
   in the slow rows of round 7 has three parts: the default synthetic clip of 100 frames never
   fills the decision window of 148 frames (decide 3.8 against 9 ms), a PNG takes 5.3 ms to read
   against 3.3 ms for the synthetic picture, and under load every stage takes 1.5 to 3 times
   longer (the round-7 rows ran at a load of 36 to 68 with the other sessions). The number of
   OpenCV threads made no clear difference under load (6 single runs, noisy: inferred, weak).

2. **What was built for speed** (no change of output): a frame feeder (`stream.FrameFeeder`)
   reads the pictures on one thread and measures the camera motion (`motion.PairMeter`: the small
   images and `estimate_motion`) on a second, at most 2 frames ahead each, while the frame thread
   does the pair test (`ShotMeter.take`), the camera, the kit colours, the matching and the
   decision. A first version ran the pair test on the measuring thread too: 11.1 ms of wall time
   for 7.5 ms of work per frame there (it waited for the interpreter lock after each OpenCV
   call), so the pair test moved to the frame thread. Exact savings in the window: the chains of
   the camera motion are kept until a cut changes or a step is forgotten, the mean pitch positions
   are taken in one loop, the window copies its tracks without validating them again (decide
   3.7 to 3.25 ms at 100 frames, measured 3 times each), and the target side of a block
   correlation is computed once per pair.

3. **Hypothesis A1** (fixed before the runs): the time per frame of `stream --synthetic` falls by at
   least 30% against `41c5552`. The frame total of `result.json` no longer holds the reading and
   the motion (they run on the feeder), so it would show a drop that is not a speed-up; the
   metric fixed before the runs is the interval between the ends of two pushes of the frame path
   (all inputs are ready on the synthetic clip, so it is the time per frame the stream can keep),
   median per run, one harness for both trees. 10 rounds, seed 2000 + round for both arms, the
   order turned every round, load 7.3 to 11.3:

   | arm | runs | median interval, mean (range) | CPU of the process per frame, mean |
   |---|---|---|---|
   | `41c5552` | 10 | 16.04 ms (15.03 to 16.71) | 23.2 ms |
   | this branch | 10 | 10.88 ms (10.06 to 12.08) | 25.7 ms |

   Ratio new / base per round: mean 0.678, 95% interval 0.649 to 0.707, range 0.602 to 0.737.
   **A1 holds, narrowly**: the gate (mean ratio at most 0.70) passes, the upper end of the interval
   lies just above it. The gain is parallel work, not less work: the process uses 11% more CPU per
   frame. These timings ran on the code before the second read; the timing after it is item 10.
   Output: `result.json` (without the measured times and the stamps) and the prediction file
   byte-equal to `41c5552` on all 20 seeds 2000-2019; the live output lines (without
   `decided_unix`) equal on all 20 (102 lines each); both demos keep their prediction hashes
   (`bb50efd43536...`, `1f97af18358d...`).

4. **The overload policy, version 1** (`stream --max-behind S`, design in `docs/ARCHITECTURE.md`
   before the code): drop a frame whose line waited more than `S` in the stream, judged once, by
   the feeder, before the picture is read. Hypothesis B1 (fixed before the runs): with 12 busy
   processes and `S` = 0.25 s, the 95th percentile of the time from the arrival of frame `f + L` at
   the tools to the decision of `f` stays below 1.0 s in every run. Slow simulated calibration (20
   and 439 ms, `--skip-behind`), seeds 2020-2024, 5 runs per delay, the policy off and on back to
   back, the order turned every round. On: 8 of 10 runs below 1 s (557 to 955 ms), 2 above (1034
   and 1679 ms); off: 10 of 10 above 1 s (1.05 to 10.1 s). **B1 failed for version 1.** The second
   reader found the causes: the wait was judged up to 6 frames before the frame path took the
   frame, and a line that waits in a full queue (the reader of the pipe stops) is not seen (a run
   of the reader: 0 frames dropped, a real lateness of 2.95 s).

5. **The overload policy, version 2** (fixed before its runs): the queue of lines has no bound under
   the policy, so its reader never waits and every wait is seen; the policy looks at each frame
   twice, at the feeder and when the frame path takes it; a frame whose line waited more than
   `S / 2` is not used; when it waited more than `S` at the frame path, the decision due with it
   (frame `t - L`) is dropped too. So every state given while the stream runs is decided at most
   `S` plus one push after the line that let it be decided was read. A single bar was tried on the
   price model first: keeping and dropping then alternate, and every late line cost a second
   frame (factor 2: 85% of the frames without a state instead of 47%); the half bar for frames
   keeps the waits below `S`, so few measured frames lose their decision. Setting fixed before the
   runs: `S` = 0.3 s. Milliseconds, the 95th percentile of each run (n = 250 - L decided frames per
   run, fewer with the policy on):

   | load | L | arm | after frame f + L, p95 of each run | line to decision, p95 | frames dropped | load average |
   |---|---|---|---|---|---|---|
   | 12 busy | 5 | off | 886, 693, 623, 899, 9111 | 467 to 6026 | - | 7.4 to 52.0 |
   | 12 busy | 5 | on | 527, 527, 537, 535, 884 | 261 to 273 | 83 to 90 of 250 | 23.5 to 55.6 |
   | 12 busy | 25 | off | 1306, 1478, 1483, 805, 8935 | 657 to 6115 | - | 35.6 to 60.6 |
   | 12 busy | 25 | on | 497, 485, 500, 497, 619 | 224 to 308 | 102 to 134 of 250 | 37.5 to 52.5 |
   | none | 5 | off | 502, 501, 502, 509, 511 | 225 to 232 | - | 2.4 to 4.8 |
   | none | 5 | on | 502, 500, 504, 507, 501 | 185 to 198 | 31 to 33 of 250 | 2.3 to 4.4 |
   | none | 25 | off | 504, 506, 507, 503, 506 | 245 to 254 | - | 2.3 to 3.9 |
   | none | 25 | on | 482, 488, 482, 488, 479 | 194 to 201 | 37 to 38 of 250 | 2.2 to 3.8 |

   **B1 holds for version 2**: 10 of 10 runs below 1.0 s (485 to 884 ms) with 12 busy processes,
   where the policy off gave 623 ms to 9.1 s (6 of 10 above 1 s). The line to decision stays at or
   below `S` plus one push (224 to 308 ms). The price is large on this feed: the calibration skips
   behind, so the merged lines come in groups of about 11 every 439 ms, and the last lines of a
   group wait more than 0.15 s even without load. With 12 busy processes 33 to 54% of the frames
   were dropped; without load 12 to 15%, for no gain in latency (the 0.5 s is the share of the
   tools and the merger). `S` must lie above twice the wait of a group (item 9).

6. **Hypothesis B2** (fixed before the runs): without extra load and with the policy on, 0 frames
   are dropped and the result equals the policy off, seeds 2020-2039. The slow profile is not
   deterministic from run to run (the calibration skips by timing: a smoke run of seed 2040 gave
   another result with 0 drops), so B2 used the steady profile (4.4 ms per tool, no skip), `S` =
   0.3 s, load 2.6 to 50 (falling after the busy runs). **B2 holds**: 0 frames dropped in 20 of 20
   runs, the result equal to the policy off in 20 of 20; the largest wait of a line at the frame
   path was 24 ms (the bar is 150 ms); the 95th percentile after frame f + L was 39 to 49 ms off and
   40 to 47 ms on.

7. **The price on synthetic clips** (a model, not a measurement of an overload:
   `python -m pitchvlm.evalx.livebench --price 1,1.25,1.5,2,3,4 --seeds 2000-2004 --kinds plain,cut
   --max-behind 0.3`): a stream that needs `k` frame intervals per used frame gives the wait of each
   line, the real policy and the real stream do the rest, scored against the synthetic truth.
   Means over 5 seeds (every other range lies within 0.012 of its mean; id switches 3 to 5 and 4 to 5):

   | kind | k | frames dropped | box recall | box precision | coverage | id switches | tracks per true track |
   |---|---|---|---|---|---|---|---|
   | plain | 1 | 0.000 | 1.000 | 0.999 | 0.967 (0.962 to 0.978) | 4.6 (3 to 5) | 1.20 (1.13 to 1.22) |
   | plain | 1.25 | 0.144 | 0.858 | 0.999 | 0.842 | 4.6 | 1.20 |
   | plain | 1.5 | 0.324 | 0.674 | 0.999 | 0.655 | 4.6 | 1.20 |
   | plain | 2 | 0.492 | 0.507 | 0.999 | 0.492 | 4.6 | 1.20 |
   | plain | 3 | 0.660 | 0.339 | 1.000 | 0.329 | 4.6 | 1.20 |
   | plain | 4 | 0.744 | 0.255 | 1.000 | 0.247 | 4.4 | 1.19 |
   | cut | 1 | 0.000 | 1.000 | 1.000 | 0.578 | 22.0 | 1.96 |
   | cut | 2 | 0.492 | 0.506 | 1.000 | 0.290 | 22.0 | 1.96 |
   | cut | 4 | 0.744 | 0.255 | 1.000 | 0.144 | 22.0 | 1.96 |

   A dropped frame has no state, so the box recall falls with the share of dropped frames, one for
   one; the precision, the id switches and the tracks per true track stay: the frames that are kept
   keep their ids across the gaps. On real footage, where the matching across a gap is harder, this
   is not known.

8. **Second read.** One subagent with fresh context got the diff, the requirements and the old code.
   It found 2 major, 6 minor and 9 small points; each was run by it unless said:
   - Major, fixed: `--max-behind` with `--live-out` always stopped (the live output was told of the
     policy after its header). Now before; a command-line test runs both
     (`test_stream_with_a_wait_no_line_can_keep_drops_every_frame_and_says_so_in_every_output`).
   - Major, fixed: the wait was not seen while the queue of lines was full (item 4); version 2.
   - Minor, fixed in `docs/ARCHITECTURE.md`: the bound counted 3 frames ahead, the code holds 6;
     the feeder has two threads; the wait for the feeder lies outside the frame samples; the
     signatures; "byte-equal with the policy on" now says what the policy adds.
   - Minor, fixed: a dropped frame looked like an empty measured frame in the video (it is darkened
     now and says `dropped: not measured`); `result.json` names the frames; the prediction file
     has no flag of its own (a format limit, written).
   - Small, fixed: `--lines -` from a regular file was taken (now refused: no live clock); two
     weak tests (one always true, one without its control) replaced; the count on a live line is
     now written as what it is (the dropped frames before the line). Written as limits: the
     missing parts count every line, dropped or not; `latency` counts given states only; with a
     video file a frame that is not read is still decoded at the next read; a reading thread that
     waits on a silent pipe stays until the process ends.
   - Checked and found right by it: no output change on 3 kinds x 3 seeds x 6 settings (72 hashes
     equal to `41c5552`, 54 through the feeder); the caches; the order of `take`; the errors at their
     frame; the refusals; the radar reader with the new keys.
   - The fixes of version 2 were read again by the same reader: item 10.

9. **Proposed owner decisions** (`docs/HANDOVER.md`, section 11.1, numbers 46 to 48): the policy in
   a live run and its setting (here 0.3 s; on the run machine set `S` above twice the wait of a
   group of lines, measured there); a dropped frame shows nothing (built) or the gap fill of its
   window; the live output keeps the version `pitchvlm-live-1` with two more keys. Proposed text
   of `CLAUDE.md`: none.

10. **After the second read.** The same reader checked the fixes of version 2: no major point; 2
    minor ones fixed (with no delay a late frame was counted twice, as not used and as decided
    late: now once, test `test_without_delay_a_late_frame_is_not_used_and_counted_once`, which
    fails without the fix; the RUNBOOK gave 0.3 s as the example where its own rule asks for
    more: now the rule and the measured values), and small ones fixed (`/dev/null` as the
    standard input passed the check of a live pipe: now only a pipe or a socket; the feeder times
    left out frames that the second look did not use; tests for both refusals and for a frame
    that only the second look drops). B1, B2 and B0 ran on the code before these fixes; they do
    not touch a delay above 0 with a pipe on the standard input. Final measurements on the last
    commit, back to back, the order turned every round, load 2.9 to 6.0:

    | measurement | `41c5552` | this branch | n |
    |---|---|---|---|
    | A1: median interval of `stream --synthetic`, mean (range) | 15.41 ms (14.95 to 15.84) | 10.21 ms (9.79 to 10.63) | 10 runs per arm, seeds 2000-2009 |
    | A1: ratio per round | | mean 0.663, 95% interval 0.652 to 0.673 | 10 |
    | CPU of the process per frame | 22.5 ms | 25.1 ms | 10 |
    | live input (PNG folder, merged lines file, 300 frames, seed 2040): median interval | 21.6 to 22.0 ms | 13.0 to 13.2 ms | 3 runs per arm |

    **A1 holds** on the last commit (the whole interval below 0.70). Output on the last commit:
    `result.json` and the prediction file byte-equal to `41c5552` on 20 of 20 seeds (2000-2019),
    the live lines on 20 of 20, both demo hashes kept. Tests on the last commit: default run 3,602
    passed, 1 skipped, 11 deselected, exit 0; full run 3,613 passed, 1 skipped, exit 0; one process
    3,613 passed, 1 skipped, exit 0 (283 s); `tools/guard.py --tracked` exit 0; `stream
    --synthetic` exit 0.

11. **Not verified.** A real tool, real footage, the run machine, Linux (the threads of the feeder
    and the reading of a named pipe were run on macOS only), a video file as the clip under the
    policy, GS-HOTA (its evaluator is not installed here), a setting other than 0.3 s, and the
    loss of a run with drops on footage, where the matching across a gap is harder than on the
    synthetic clip. The load of the other sessions was not controlled (the load average of each
    run is given).

12. **Files outside the area of this session**: `src/pitchvlm/liveout.py` (additive: `overload`,
    `skipped`, two keys only with the policy), `src/pitchvlm/render.py` (a keyword `dropped` of
    `render_clip`), `src/pitchvlm/cli.py` (the options and the input loop of `stream`; the signal
    handling is not touched), `tests/test_live_feed_planted.py` (one argument of a namespace).

## 2026-11-23 — Shirt numbers in the live path (session "numbers in the stream" of round 8)

Context: `stream` refused every number engine but `none`, so the live feed and the radar showed
no shirt number. The clip path asks the model for the number of every track after the whole clip
(`configs/engines/detector_calibration_numbers.yaml`). This entry builds the live form of that
step: a worker beside the stream (`src/pitchvlm/livenumbers.py`). Build machine only: the model
is the fake backend of the synthetic scene, no call was made, nothing here is a measurement of a
real model, a real clip or a real cost.

1. **Engines.** `stream` takes the number engine `model` besides `none`
   (`stream.STREAM_NUMBERS`). The frame path is the same code for both: the stream decides its
   states with no number. With `none` there is no worker and no hook, and every output is the one
   of `41c5552` (checked byte for byte below).
2. **Off the frame path.** `DetectorStream.on_arrived(frame, image, people)` (the one hook added
   to the stream, stage `numbers` of the frame sample; since item 12 at the start of `_arrive`) hands a reference to the image and the boxes of
   every stride frame (the keyframe step, 12 frames at 25 fps) and of the frames of the kit legend
   (0, 1 and 2 s) to one worker thread, through a queue of at most 8 images: a full queue drops the
   new image and counts it (`images_dropped`); no image is copied on the frame path. Each given
   state goes into the same queue as a reference (never dropped: the stream keeps every state).
   The worker cuts the crops (the padding of the clip path), and frees them when their frame is
   decided. Nothing assumes that every frame of a track is there: a frame without image has no
   crop, a frame without decision has no state (the overload policy of the speed session drops
   frames; item 12).
3. **When a track is asked.** Per stream id (the id of the live feed, which is final): the crops
   of its observed states of a player or a goalkeeper (most observed states) go to a pool (at most
   16, `crop_choice`). The id is asked once: `ready` when `crop_choice` gives 4 crops at least
   1 s apart (seen for at least 3 s); `closed` when it has had no observed state for the history of
   the stream (123 frames: no window can give the id again), with a crop; `end` at the end of the
   stream, with a crop. The request is that of the clip path (`pipeline._number_request` with up
   to `number_crops` = 8 crops, the team key `none`), after one kit legend request; the answer is
   decided with `tracking.decide_number(reading, ())`, the rule of the clip path.
4. **How an answer reaches the outputs.** The thread of the stream takes the answers that came
   after each decision and writes each as a number line of the live output: a new format
   `pitchvlm-live-2` (the lines of `pitchvlm-live-1` plus `{"kind": "number", "id", "number",
   "export", "display", "frames": [first, last], "crops", "reason", "decided_at",
   "decided_unix"}`). A number line comes after the frame lines it was read from and changes none;
   the pipe policy never leaves one out. Without the number step the output stays
   `pitchvlm-live-1`. The radar reads both: it keeps the newest number of every id, gives all of
   them to a page that opens late, and draws the number of an id when its frame line has none.
   At the end `apply_numbers` gives every state and track of an id its number line, so
   `result.json`, the prediction file and the video hold what the feed said
   (`tests/test_livenumbers.py`). Why not revise frame lines: a frame line is final by the
   contract of round 7, and a reader that drew a frame must not have to draw it again.
5. **End of the stream and stops.** `finish` asks the open ids, waits for the kit legend, sends
   the waiting requests and waits for every request in flight; `abort` (an early stop, also
   Ctrl-C through the exit stack of the command) sends nothing more and waits for the requests in
   flight. Both reuse `pipeline._InFlight` and `pipeline._finish_requests_in_flight`, and (after
   the merge of the second read of round 7) hold stop signals back with `stopsignals.held_stops`,
   as `pipeline._generate_all` does: a stop during the wait is recorded, no id is asked and no
   request is sent after it, a warning names the requests in flight, and the stop is raised when
   they have ended; no number of signals ends the wait (`Ctrl-\` or `kill -KILL` does, owner
   decision 45; tested with 50 signals). After the
   first failure (a budget stop, an error of the model, an answer that cannot be parsed) no
   request is sent any more; the stream goes on, prints the failure, and the command exits with 1.
6. **Money: the cap.** A live stream has no known length, so the command takes a cap of number
   requests per stream, `--numbers-max` (default 50, `cli.EXPECTED_NUMBER_REQUESTS` of one clip).
   The estimate is 1 kit legend + the cap, printed before the first frame by `open_session`; the
   budget guard of the cost ledger of the cache folder stops the run above 1.3 times it, as for
   `run`. When the cap is reached a ready id gets no request (fail closed): a warning says it once,
   the end prints the count (`refused_by_cap`). Estimate of one pilot clip with the default cap
   (estimated, not measured; token estimates of `vlm.cost`, `gemini-4-argon`, standard tier):
   0.463 USD until 30 Nov, 0.926 USD from 1 Dec; the run stops above 0.602 / 1.204 USD. With
   `--synthetic` the model is the fake one: no ledger, no cost.
7. **Measurements** (build machine, synthetic clips, the fake model of the scene behind a set
   delay per answer; `src/pitchvlm/evalx/numberbench.py`; thresholds and method fixed in the
   Parallax record before the first run; held-out seeds 2200 to 2239, used before only by the git
   states of a guard test, never by a synthetic clip; the tests use seeds 2290 to 2296). The load of
   the machine came from two other sessions of this round and was not under my control.
   - **N1, cost to the frame path** (hypothesis: the worker adds at most 1.0 ms to the median and
     2.0 ms to the 95th percentile of the time per frame, stage `total`, against the same code with
     the number engine `none`). 750-frame clips, seeds 2200 to 2205, frames as fast as the stream
     goes (the worker competes with the frame path all the time), a live output to a file in both
     arms, pairs back to back on the same seed with the order flipped every pair, 6 pairs per answer
     delay (6 runs per arm and delay, 36 runs). Load average 9.7 before, 4.9 after (1 min).
     Code of `9c4e3a9` (the fixes of item 11 came after; they add one flag test to the hand-over).

     | Answer delay | Pairs | Median per frame, none / model (median of runs) | Paired difference of the median: mean [95% t] (median) | 95th percentile, none / model | Paired difference of the 95th percentile: mean [95% t] (median) | Verdict (mean against the threshold) |
     |---|---|---|---|---|---|---|
     | 0.5 s | 6 | 20.97 / 21.71 ms | +0.55 [+0.05, +1.05] (+0.48) ms | 23.84 / 24.91 ms | +1.05 [+0.33, +1.78] (+1.18) ms | pass |
     | 2 s | 6 | 21.22 / 20.90 ms | −0.14 [−0.77, +0.49] (−0.33) ms | 24.36 / 23.96 ms | −0.25 [−1.13, +0.64] (−0.49) ms | pass |
     | 5 s | 6 | 19.65 / 21.11 ms | +2.23 [−2.10, +6.55] (+0.62) ms | 22.23 / 24.38 ms | +4.33 [−7.81, +16.48] (+1.13) ms | **fail** |
     | all | 18 | | +0.88 [−0.36, +2.12] (+0.37) ms | | +1.71 [−1.57, +5.00] (+0.65) ms | pass on the means, not shown at 95% |

     **N1 fails as written** (the 5 s arm). Its mean comes from two pairs: pair 3, where the model
     run had a median of 29.9 ms against 19.4 ms (+10.4 ms), and pair 2, where both arms were slow
     (the run with `none` had a 95th percentile of 38.0 ms). The paired medians of all three arms
     are below the thresholds, and with 5 s per answer the threads of the pool only wait, so I infer
     a load spike of the other sessions, not a cost of the worker; this is an inference, not a
     measurement. What the 0.5 s arm shows is a real cost of about +0.5 ms on the median and +1 ms
     on the 95th percentile (the upper end of the interval of the median is 1.05 ms): the worker
     takes the lock of the interpreter to cut crops, to read the states and to build requests, and
     the fake model reads the truth in Python in the threads of the pool, which a real model (a
     network call) does not. A replication of the 5 s arm on `9763ee1` (6 more pairs, the same
     seeds) ran at a load average of 52 falling to 7.7: paired difference of the median −4.16
     [−14.58, +6.27] ms, of the 95th percentile −0.02 [−17.06, +17.01] ms; in pair 2 the run with
     `none` had a median of 68.8 ms. It decides nothing: on this machine, under the load of the
     other sessions, 6 pairs cannot show a cost of 1 ms. After the merge of the second read of
     round 7, the 5 s arm again on `165b668` (6 pairs, both orders, the same seeds, the load average
     of 1 min noted before and after each run: from 7.0 to 36.6, rising during pairs 2 to 4): paired
     difference of the median +4.66 [−3.67, +12.99] ms (median of the pairs +2.20), of the 95th
     percentile +14.14 [−20.25, +48.53] ms (−0.77). **Fail as measured.** The two largest
     differences came with the load rising during the run with numbers (pair 0: 39.7 against
     20.7 ms at a load of 17.7 to 18.7; pair 2: 25.3 against 20.7 ms while the load rose from 7.0 to
     20.9); in pair 4 at a load of 35 both arms took 33.6 to 33.9 ms. N1 stays failed as written; a
     run on a quiet machine (or the run machine) is the open check.
   - **N2, the numbers against the clip path** (hypothesis: at the end of the stream the set of
     (track, number) equals that of the clip path with the same engine file and the same fake
     answers, on all 40 seeds). 500-frame clips (20 s), frames paced at 25 fps, the cap set to 1000 so
     that it takes no number; a stream id is mapped to the clip-path track that holds most of its
     observed boxes. **N2 fails: 37 of 40 seeds equal**, the same 37 at each answer delay (0.5, 2
     and 5 s, 120 runs). The three other seeds, each looked at by hand:
     - 2227 (10 tracks) and 2223 (1 track): the stream itself votes these players `referee` (seed
       2227: ids 10 and 12 have 434 and 452 observed states as referee, 22 as player; seed 2223: id
       19 has 145 as referee, 78 as player), where the clip path votes them players. The worker asks
       only ids that most states call a player or a goalkeeper, as the clip path does, so they get no
       number. The cause is the role of the stream (its kit clusters and kit roles in a window),
       which the number engine `none` shows the same; it is not a fault of the number step, and it
       is a lead for the stream: a whole team called referee on 1 of 40 seeds.
     - 2235 (1 track): the clip path gives one person two tracks (5 and 10, 304 and 145 boxes), both
       with the number 11; the stream gives one id over the boxes of both, with the number 11. The
       numbers agree box for box; the set of my comparison counts the second track of the clip path
       as missing. An identity difference, not a number difference.
     Requests per clip: median 38 (21 to 40) in the stream, median 38 (38 to 40) in the clip path;
     none above the default cap of 50. No image was dropped by the queue in the 120 runs, and no
     failure.
   - **N3, the time from the first frame of an id to its number line** (description; arrival of the
     first frame of the id to the moment the stream writes the line; 40 seeds per delay, 1,517
     lines each; the load average rose from 12.7 to 56 during the runs):

     | Answer delay | Median | 95th percentile | Largest | `ready` lines: median / 95th (n = 1,010) | `closed` (n = 219) | `end` (n = 288) |
     |---|---|---|---|---|---|---|
     | 0.5 s | 6.22 s | 9.94 s | 11.66 s | 6.26 / 6.90 s | 9.54 / 11.10 s | 3.56 / 4.91 s |
     | 2 s | 12.15 s | 18.36 s | 20.16 s | 12.42 / 18.06 s | 17.38 / 19.47 s | 9.65 / 11.09 s |
     | 5 s | 49.25 s | 60.67 s | 64.57 s | 50.06 / 60.09 s | 60.50 / 64.39 s | 41.54 / 43.89 s |

     The floor is the rule: 4 crops 1 s apart on a stride of 12 frames are frames 0, 36, 72 and 108
     of the id (4.3 s), plus the delay of the stream (1 s), plus one answer. Above it the pool of
     `max_parallel` = 4 threads queues the requests: about 20 ids are ready in the same second at the
     start of a clip, and 4 threads answer 0.8 requests per second at 5 s per answer, so at 5 s the
     line of an id comes about 45 s after the floor. `end` lines are short because their ids were
     first seen late. On a real model the time per answer is not known (owner decision 53).
8. **Checks** (on `9763ee1` with these documents). The runner: 3,598 passed, 1 skipped (default),
   3,609 passed, 1 skipped (`-m "slow or not slow"`), exit 0; one process: 3,608 passed, 1 skipped,
   1 failed at a load average of 43 (`tests/test_run_tests.py::test_ctrl_c_stops_every_worker_and_their_children`:
   its runner did not end within 60 s; the file is not touched here, and the test passed 3 times of 3
   alone). `tools/guard.py --tracked`: no finding. With the number engine `none`, against `41c5552`:
   the predictions of `demo` (SHA-256 `bb50efd43536…` and `1f97af18358d…`) and both videos byte for
   byte, the result and the prediction file of `stream --synthetic` equal, and its live feed equal
   line for line but for `decided_unix` (the wall clock). 27 new tests in `tests/test_livenumbers.py`:
   with the module copied into the tree of `41c5552`, 12 of the first 23 fail there (the 11 others
   test the module alone, which does not exist there), and the changed test of `tests/test_stream.py`
   fails there too.
9. **Owner decisions** (HANDOVER 11.1, items 49 to 53): the cap per stream (and whether a cap per
   minute is needed for long live runs); the readiness rule of 4 crops; that two ids of one person
   both show the number and `stream` has no roster names; that one answer that cannot be parsed
   stops the asking for the rest of the stream; the threads of the step (N3).
10. **Second read.** A reader with fresh context read the change (code only) and found: a history
    of 0 frames failed after the model session was opened, without printing the measured cost; a
    fault of the code that reads an answer was swallowed (no failure, the step went on); after a
    fault of the worker the frame path still kept every state and arrival time (memory growth); the
    test of `abort` depended on 0.3 s of scheduling; the stream tests could drop images on a slow
    machine. All fixed in `9763ee1`; the first three and the closing of the session each have a test
    that fails on the code before it (4 tests).
    Kept as they are: the count of undelivered lines of the pipe report now holds number lines too
    (its words say "frame lines and the end line"); `stream --help` lists the model options also
    for the number engine `none` (they are refused there).
11. **Not verified.** A real model (its time per answer, how it reads crops of a live clip), the
    real number of ids of a clip and so the real cost, a real camera, Linux timing (CI runs the
    tests only).
12. **The merges of round 8** (added 2026-11-24). (a) The second read of round 7 (`b0889a6`): the
    end-of-stream wait and `abort` hold stop signals (item 5). (b) The count rule of `main`
    (`3764d2a`): the header is in no count of `LiveReport`; number lines have counts of their own
    outside it, frozen by `close` (`LiveOut.number_counts`: `given`, `written`, `undelivered`,
    the last two adding up to the first), written to `result.json` as `numbers.live_out` (before
    the merge the key was `numbers.number_lines_given_to_live_out`). The rule as `main` states it,
    `written + dropped + undelivered == given + 1`, holds when the end line did not go out; when
    it went out, `written + dropped == given` (`written` counts frame lines only). (c) The frame
    feeder and the overload policy (`b4f19a4`): the hook moves to the start of `_arrive`, where the
    thread of the stream has the picture (`push`, or `push_fed` with the picture the feeder read),
    after the motion stage; a dropped frame calls nothing; a dropped decision writes the number
    lines that came (`wire`, `on_dropped`), so a run of dropped frames does not hold them back.
    The live output is one of four (`liveout.py`, table): `pitchvlm-live-1` without the number
    step and `pitchvlm-live-2` with it, each with the two overload keys only when the policy is
    on; a reader of version 1 reads the first two and refuses the name `pitchvlm-live-2`. Owner
    decision 48 (whether the overload keys get a name of their own) is described there, not
    decided: the other choice would add a third and a fourth name, or one version with both
    parts optional. With the number engine `none` and the policy off every output stays that of
    `41c5552` (checked below); `result.json` of `stream` has the timing key `feeder_timing` of
    the speed branch. Tests: frames dropped in the middle of tracks give the same numbers (an
    injected still clock); the number step and the policy in one stream, whose end holds 20 stop
    signals; a dropped decision writes the numbers that came. The bench feeds its frames through
    the frame feeder since `9ab0d29`, as `stream` does.

## Open questions (to settle with measurements)

- Does the streaming path keep up with 25 fps on the run machine with the tools on the same
  machine (the 95th percentile of the package plus that of the slower tool below 40 ms), and how
  long is the lag of the dissolve test on real dissolves (entry "A streaming path for the detector
  engine", items 6 and 7)?
- Can Gemini point at pitch landmarks well enough for a homography (gate G3)?
- Recall on far-side players about 46 x 99 px (gate G2)?
- Shirt-number reading at real player size (gates G5a, G5b)?
- Identity over 30 s at 2 keyframes per second (gate G4)?
- Real tokens, cost and latency per request (cost pilot)?
- Type of the `jersey` field in the current label files (string or number)?
- Length of a Google API key in the `AQ.` format (the guard expects 20 or more characters after
  the prefix)?
- Does the body field `service_tier` of an answer name the tier that served the call?
- Share of real clips where the team-side rule is right, and the right value for its 3 m limit?
- Largest output of a real answer, and the limit on output tokens that follows from it?
- Does a broadcast graphic disturb the measured camera motion on real footage?
- Does the paint-alignment score separate good from bad pitch mappings on real frames (AUROC,
  per engine), and how many real frames cannot judge a mapping?
- How many hard cuts does the cut test find in the demo clip, and how many does it miss?
- Does the window test find the dissolves of real broadcast clips, and does it raise false
  transitions in fast real pans (entry "A dissolve is a transition, not a shot")?
- How often does the optical flow fail for several pairs in a row inside a real dissolve (the
  window test then misses it), and how often does OpenCV give a cut broadcast file a last frame
  of odd length, an estimated frame count or a field rate (entry "Second read of the dissolve
  test, the video frames, the overlays and the guard", items 2 and 10)?
- Right value of the two-people margin (6 m) and of the limit for a repeated box (0.6) on real
  answers (RUNBOOK section 8, facts 13 and 15)?
- Do two crops that are less than one second apart agree on a wrong number, and does the model
  read a number of two digits in parts (fact 14)?
- How many landmark fits does the share rule refuse on real answers (fact 12)?
- How often is the flex tier busy, and does a real answer ever reach the token cap of one
  request (facts 7 and 17)?
- Which coordinate convention does each OpenRouter model answer in, and does any provider of it
  meet the routing settings (zero data retention, no data collection, all parameters)?
- Is `usage.cost` of OpenRouter in USD (one credit = one USD), and is a call refused with an
  error status never billed?
- Do the providers of the OpenRouter models take the strict answer schema with `minItems` and
  `maxItems`?
- How often do swaps, wrong team keys for seconds and returns into the image happen in real
  clips, and how pure are the links of the team slots there (entry "Track repairs and tracking
  values without labels" of 2026-11-22, item 9)?
- Recall, precision and small-box recall of RF-DETR Small and Large with their pretrained
  weights on the frames of the people probe, and their time per frame on the run machine?
- Position error and share of frames with a camera of PnLCalib on the frames of the pitch
  probe? Do the GPU and the CPU of the run machine give the same values for both tools?
- Detector engine (2026-11-22): G2 values of RF-DETR Medium; time per frame of each tool and of
  each stage of the package on a full clip of the run machine; role and team-side accuracy of
  the kit colours on real kits (stripes, similar colours, a goalkeeper kit near a team kit);
  share of frames with a ball and its error; GS-HOTA of the detector engine against the model
  engine on the dev clips; do the GSR label files hold ball boxes?
- Detector engine and the track repairs (entry "A real-time detector engine next to the model
  engine" of 2026-11-22, item 5): does the split of team swaps cut wrongly where two boxes
  overlap and a torso shows the shirt of the other player? Should the tracking values without
  labels of a detector run be read at the keyframes, so that both engines are compared on the
  same frames and with steps of the same length?

Decisions for the owner that are open, each with its recommendation:

The complete list of 23 Nov 2026, with the decisions of the later entries, the options, the
default when nothing is decided and what each one blocks, is in `docs/HANDOVER.md`, section 11.1.

- **Label penalty.** Keep the growth with missed keyframes for the role and the team key, or
  let it grow for the team key only. Recommendation: keep it now. Take the growth for the team
  key only when the people probe (G2) shows about 10% wrong roles per keyframe.
- **Team side.** (a) When the goalkeepers of the two goal zones name different answers, let
  the players decide and not the heavier zone. (b) Give a goalkeeper track that shares its
  time with a heavier goalkeeper track of the same goal zone no vote and no zone side. (c) Let
  a goalkeeper deep in its goal decide also against a player gap of 3 m or more.
  Recommendation: do (a) and (b) in `tracking.py` and `evalx/oracle.py` together, after the
  label stage has printed the facts of every clip; leave (c) until the labels show how many
  clips have a player gap of 3 m or more with the wrong sign.
- **Requests in flight at a budget stop.** A stop comes after the answer that crosses a limit,
  and up to `max_parallel - 1` more answers are paid. The stronger rule counts the estimate of
  every call in flight against the run limit before a new call is sent
  (`cli.SessionBackend`). Recommendation: decide after the cost pilot, when the real cost of
  one answer is known; with 4 requests in flight the overshoot is 3 answers.
- **Install rule of the build machine.** `CLAUDE.md` and `docs/PLAN.md` say that the build
  machine installs nothing. The sentences name no owner, no host and no reason.
  Recommendation: keep them: a clone on the build machine needs the rule to act right.
- **Two texts that the guard refuses although they are no leak**: a route `/home/<word>` and a
  bare list of 32 or more check sums. Recommendation: keep both refusals; each has the form of
  a leak that the first review closed (RUNBOOK, section 9, says how to write such text).
- **Settings of the tool measurements** (section "Both specialist tools are measured on the
  probe frames", item 3). Three experiment files use the settings of the tools as they come;
  two more measure the antialiased resize of the detector. Recommendation: measure these first.
  Add one setting at a time after the first values, where a gate is missed: for RF-DETR a frame
  shape that keeps the side ratio, for PnLCalib the thresholds of its benchmark script and,
  when weak cameras miss the limit of the 90th percentile, its limit for the reprojection error.
- **Run files of the tools on the results branch** (item 8 of the same section). The owner
  decision of 13 Nov lets three kinds of text return: metrics, stored model answers and the
  cost ledger. Recommendation: let the run files return too; they are the stored answers of the
  two tools, without pixels and without label data. After a yes, add them to rule 2 of
  `CLAUDE.md`, to section 6 of the RUNBOOK, to section 9 of the HANDOVER and to the list of
  `docs/PLAN.md` (section "Two machines"). Until then the folder `results/tools` is not
  committed (RUNBOOK, section 3).
- **A tool that passes its gate cannot be used by the pipeline yet** (item 1 of the same
  section). Needed in M4, before the first clip that uses a tool: a setting that names the
  engine of the people step and of the pitch mapping, a path of `run` that reads the answers of
  a tool, and for RF-DETR one more step that gives each box its role, team key and number (a
  paid model step on crops, or plain code with the kit legend). None of it is designed, and no
  budget line holds it. Recommendation: decide on 24 Nov with the measured values; build the
  switch only for a step whose Gemini gate failed and whose tool passed (rule 8). When G3 fails
  and PnLCalib passes, the switch is small, because a camera matrix needs no other step. When G2
  fails and RF-DETR passes, plan the step for role and team first, with its cost.
  Built on 2026-11-22 after the owner decision of that day (entry "A real-time detector engine
  next to the model engine"); what stays open is listed at the end of this list.
- **Frames of gate G3.** `docs/PLAN.md` and `configs/gates.yaml` name the frames of the people
  probe plus 3 of the dev clips. Both pitch experiment files (the model step and PnLCalib)
  measure the group `probe` only, and the RUNBOOK has no command for the 3 dev clips.
  Recommendation: keep the gate text, and add the run on the group `dev` with `--max-clips 3`
  for both engines when the values of the group `probe` are near a gate line; the frames of a
  gate are a part of the gate (rule 7).
- **Branch `results` on a public remote** (section "The code repository is public").
  Recommendation: do not push the branch `results` to the public remote before this is
  decided. Keep it on the run machine, or give it a second remote that is private. The build
  machine can read the files only from a remote: with the first way it reads no results until
  this is decided. For the second way the second remote must hold the commits of `main` before
  the first push of `results`, for example as a copy of the public repository that the git
  host makes (not tried), and `git fetch <remote>` must have run: the guard judges only the
  commits that the remote of the push does not have (measured in a scratch repository). A
  push to a remote that does not have the merge commit of pull request 1 judges the identity
  of that commit and is refused, and that commit cannot be changed (RUNBOOK, section 9). The
  other way is a change of the guard: the identity rule leaves out the commits that a remote
  of the repository already has.
- **Licence of this repository.** It has no licence file, and one script imports GPL code of
  another project when it runs. Recommendation: decide both before the two tools are installed
  on the run machine (M2), and add a licence file.
- **Overlay images of the kill switch of G3.** Built on 2026-11-22 (entry "A video file into
  the detector engine, and the check images of gate G3"): `overlay probe|tool|run` (RUNBOOK,
  section 3, "Check images of the pitch mappings"). The gate file is unchanged. Owner to do: after
  the first 4 clips of the pitch probe, run the overlay with `--count 5`, check the images, write
  `right` or `wrong` into each `by_eye` of the list file and the counts into this file before the
  kill switch is judged; for M3 the same with `--count 10` for the model step and PnLCalib.
- **OpenRouter: rule text** (entry "Image models through OpenRouter" of 2026-11-22, item 9).
  Accept, change or refuse the proposed text of the first paragraph and of rules 4 and 5 of
  `CLAUDE.md`. Recommendation: accept; the code already keeps it, and the doctor line `data policy`
  checks it.
- **OpenRouter: price table on the estimate side.** The table holds the highest published price
  of any provider of a model. Recommendation: keep it so until the first doctor names the
  provider that the routing settings choose, then keep the highest price anyway: a later request
  can be routed to another provider.
- **OpenRouter: token estimates per model.** `DEFAULT_TOKEN_ESTIMATES` is one table for all
  models, measured for none yet. Image tokens of other models differ. Recommendation: after the
  first doctor of a model, compare its measured tokens with the table; when they differ by more
  than 1.5 times, add estimates per model (a small change of `vlm/cost.py`) before its first probe.
- **OpenRouter: which models to try.** Candidates with a published price: Qwen3-VL 235B (first
  guess `xy_1000`), Qwen2.5-VL 72B and Gemini 2.5 Flash through OpenRouter (`yx_1000`).
  Recommendation: one doctor of Qwen3-VL first (0.015 USD, estimated), then the people
  probe of the model that passes, on the frames of the Gemini people probe.
- **Paint alignment: the rule** (entry "Paint alignment of pitch mappings, and hard cuts" of
  2026-11-22, item 4). Recommendation: run the free measurement of the RUNBOOK (section 3) on
  the cached answers of the pitch probe and on the PnLCalib run file, and make the score a rule
  of `check_homography` only for an engine that meets every condition of item 4.
- **Shot type in the landmark answer** (same entry, item 8). Recommendation: decide after G3,
  with the cut counts of the demo clip; the field changes a paid prompt and every cache key of
  the landmarks step.
- **Tracks and states at a cut** (same entry, item 9, points c and d). A shown shirt number of
  one team joins two tracks across a cut, and the frames between the last keyframe before a cut
  and the cut get no state. Recommendation: keep both: a number is identity, and a position
  carried into another shot is a guess.
- **Measurement script in `tools/`** (same entry, item 9, point e). `tools/synthetic_diagnostics.py`
  sits beside the scripts of the specialist tools. Recommendation: keep it there; it imports
  only the package and runs offline.
- **Track repairs: G4 limit per clip or over the clips** (entry "Track repairs and tracking
  values without labels" of 2026-11-22, item 11 (a)). Is `track_ratio_max` of G4 read per clip
  or over the 6 dev clips? Recommendation: per clip, as the RUNBOOK script reads it; a clip
  with many fragments is the case that the limit is for.
- **Track repairs: default after G4** (same entry, item 11 (b)). Does a repair that passes G4
  but lowers AssA of `eval` become the default? Recommendation: no; a repair becomes the
  default when it passes G4 and does not lower AssA on the same clips.
- **Track repairs: a limit for `purity`** (same entry, item 11 (c)). Recommendation: decide it
  before the dev runs of the arms (rule 7); without a limit a repair can pass G4 and mix people.
- **Track repairs: cost of the split arm** (same entry, item 11 (d)). It pays new number
  requests for the tracks that it changes (about 0.009 USD each, estimated); the team slots are
  free on stored answers. Recommendation: run the team slots arm first, and the split arm after
  its estimate is printed and accepted.
- **Rule 8 for two first-class engines** (entry "A real-time detector engine next to the model
  engine" of 2026-11-22, item 7). Recommendation: apply the proposed text after the first G2 and
  dev values of both engines exist. It assumes the proposed rules 4 and 5 of the OpenRouter entry (item
  "OpenRouter: rule text" above); the first paragraph of `CLAUDE.md` then also names the tools.
- **Size of the detector.** Recommendation: take the smallest RF-DETR size that passes G2 on
  the probe frames, and RF-DETR Medium until then.
- **What the AGPL or the GPL means for `tools/run_tool.py`.** Recommendation: decide it with the
  licence of the repository, before an AGPL tool is added; until then add only tools whose code
  and weights are Apache-2.0, MIT or BSD, besides PnLCalib.
- **Soccer-trained detector** (RUNBOOK, section 12). Recommendation: read the terms of the
  SoccerNet-GSR train data first; build the converter and train only with a budget line for the
  GPU time and a decision that research-only weights may drive the demo.
- **Video files for the detector engine.** Built on 2026-11-22 (entry "A video file into the
  detector engine, and the check images of gate G3"): `video-frames`
  writes the frames of a video as PNG below the data folder, the clip job names the video by
  its SHA-256, and `run --clip <video>` reads the clip run only with that video (RUNBOOK,
  section 11, "A video file as the clip"). Owner to do: put the demo clip below
  `~/pitchvlm-data`, run `video-frames` on it, compare one frame read by Pillow in a tool
  environment with the frame of `run` (item 9 (a) of that entry), and keep about 1 GB of disk per
  clip of 30 s (estimated).
- **Clip runs on the results branch.** They are tool runs (item "Run files of the tools on the
  results branch" above). Recommendation: decide both together; each part passes the guard.
- **New dates of the schedule** (entry "Rehearsal of the handover and the runbook" of
  2026-11-22). On 22 Nov no step of M2 has run, so every date of `docs/PLAN.md` from M2 on has
  slipped by at least 5 days, and the standard prices end on 30 Nov. Recommendation: set the new
  dates after the first session (HANDOVER, section 6) with the measured cost of the pilot; keep
  the half-price tier for every clip run from 1 Dec, as the plan says.
