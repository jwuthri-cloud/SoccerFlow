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

## 2026-11-22 — The offline tests in several processes

1. **Goal and result.** The default run of the offline tests took 110 to 155 s in one process.
   Hypothesis: the test files can run in N processes with the same collected tests and the same
   outcomes as one process, and the default run then takes at most 40 s on the build machine.
   Baseline: `python -m pytest` in one process on the same tree. **Result: met on the build
   machine.** The new runner `tools/run_tests.py` ran the default run in 27.4 to 37.2 s (14 runs,
   load average 5.7 to 27 on 14 cores), against 142.5 to 144.5 s in one process (4 runs). The
   full run took 28.3 to 36.4 s against 164.3 s. Same test ids and the same outcome of every
   test. Every time here is measured on the build machine while other sessions ran; the load
   average (1 min) is given with each run. Nothing ran on the run machine.

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
    - No run on the run machine, on Linux, or on a quiet machine. The target of 40 s holds here
      under a load of 6 to 27; on a machine with fewer cores or slower process starts it can be
      missed.
    - The reading of macOS's `PermissionError` in item 4 (b) is inferred from the code of the
      system, not proved by a test that forces it.
    - The mapping of JUnit XML names to test ids follows pytest 9.2; another pytest version may
      fail the id check of every run (it cannot pass a wrong one).
    - The gap of fault P05 (item 9) is not closed here.
    - An earlier version of the Ctrl-C test, before fix (b), left two idle test processes on the
      build machine; they ended by themselves after one hour.

## Open questions (to settle with measurements)

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
