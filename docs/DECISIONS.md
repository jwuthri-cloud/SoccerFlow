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
     frame on an NVIDIA RTX 2080 Ti for the same voting with line refinement. That time was
     measured with the reading of the image, at 960 x 540 pixels, with other weights and other
     thresholds: it is not for the settings above.
5. **What guards against a wrong script.** The script was written without the tools, and its
   tests use stand-ins. An error of a tool on a frame stops the run and names the frame: no
   frame counts as a frame without an answer because of an error. And: (a) the script refuses
   another version of `rfdetr` and another commit of PnLCalib than those that were read, and a
   PnLCalib clone with a changed tracked file; (b) the pitch result holds `axis_check`, the
   median position error for the matrices as they are and with the sign of x, of y or of both
   changed. The pitch convention is right when the matrices as they are give the smallest value
   and that value is small (some metres at most). When another sign gives a smaller value,
   `tool-score` says so, writes the metrics file and ends with the exit code 1: a large value
   as given next to one small other value means that the script has the wrong pitch convention.
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
   runs. This repository is private. Before it is published, the owner checks what the licence
   of PnLCalib means for that script (`docs/PLAN.md`, risks).
10. **Review of this change (2026-11-14).** Five reviews, each with one view (the package code,
    the script, the command line, the measurement, the documents), and a second reading of each
    finding. 50 findings, 48 confirmed: 6 major, 19 minor and 23 small ones. Repaired, each
    code change with a test:
    - The script: a box or a score of the detector that is not a finite number stops the run
      at its frame; an error inside an installed tool keeps its traceback; a PnLCalib clone
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
      the run file and whether the script is the one of the work tree; an output file that is
      an input file is refused; `sync` names a run file in every folder below `tools`.
    - The documents: what the two commands do not do (item 1), which pitch values of two
      engines are not on the same frames, the size rule of the axis check, the stricter
      plausibility rule of a tool record, and four new open owner decisions.
    Kept as it is: one run of the detector serves one score limit. A run file with a lower
    limit could serve a higher one, but then the metrics file would not prove that the tool ran
    with the settings of the experiment file. Named by the review and not built: the share of
    all frames within an error limit, by-view values per clip, and the limit for the
    reprojection error of PnLCalib (item 3).

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
- Right value of the two-people margin (6 m) and of the limit for a repeated box (0.6) on real
  answers (RUNBOOK section 8, facts 13 and 15)?
- Do two crops that are less than one second apart agree on a wrong number, and does the model
  read a number of two digits in parts (fact 14)?
- How many landmark fits does the share rule refuse on real answers (fact 12)?
- How often is the flex tier busy, and does a real answer ever reach the token cap of one
  request (facts 7 and 17)?
- Recall, precision and small-box recall of RF-DETR Small and Large with their pretrained
  weights on the frames of the people probe, and their time per frame on the run machine?
- Position error and share of frames with a camera of PnLCalib on the frames of the pitch
  probe? Do the GPU and the CPU of the run machine give the same values for both tools?

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
  `CLAUDE.md`, to section 6 of the RUNBOOK and to section 9 of the HANDOVER.
- **A tool that passes its gate cannot be used by the pipeline yet** (item 1 of the same
  section). Needed before M4: a setting that names the engine of the people step and of the
  pitch mapping, a path of `run` that reads the answers of a tool, and for RF-DETR one more
  step that gives each box its role, team key and number (a paid model step on crops, or
  plain code with the kit legend). None of it is designed, and no budget line holds it.
  Recommendation: decide on 24 Nov with the measured values; build the switch only for a step
  whose Gemini gate failed and whose tool passed (rule 8). When G3 fails and PnLCalib passes,
  the switch is small, because a camera matrix needs no other step. When G2 fails and RF-DETR
  passes, plan the step for role and team first, with its cost.
- **Frames of gate G3.** `docs/PLAN.md` and `configs/gates.yaml` name the frames of the people
  probe plus 3 of the dev clips. Both pitch experiment files (the model step and PnLCalib)
  measure the group `probe` only, and the RUNBOOK has no command for the 3 dev clips.
  Recommendation: keep the gate text, and add the run on the group `dev` with `--max-clips 3`
  for both engines when the values of the group `probe` are near a gate line; the frames of a
  gate are a part of the gate (rule 7).
