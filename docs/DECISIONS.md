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
   - A 200 answer without usage, without a cost or with `is_byok: true` raises
     `UnconfirmedCallError`: the ledger cannot price it. A 200 answer with an error, a finish
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
   Estimated, not measured: the first doctor of `qwen/qwen3-vl-235b-a22b-instruct` costs about
   0.02 USD (6 calls at the token estimates of Gemini; image tokens of Qwen differ).
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
     run without the SDK and without a network. Only `tools/run_tool.py` imports an external tool
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
   the plausibility check), every valid mapping of the pitch probe and every PnLCalib mapping of
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
- **Overlay images of the kill switch of G3.** The gate file wants five overlay images that
  are checked by eye before the kill switch counts (`overlays_checked_by_eye: 5`), and M3 asks
  the owner to check 10. No command writes an overlay of a pitch mapping on its frame, for the
  model step or for PnLCalib. Recommendation: build a small command that draws the pitch lines
  of a mapping on its frame into `~/pitchvlm-data` (images stay on the run machine), before
  the pitch probe of M3. The other way, to take the line out of the kill switch, is a change
  of the gate and needs its own entry here (rule 7).
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
  Recommendation: one doctor of Qwen3-VL first (about 0.02 USD, estimated), then the people
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
