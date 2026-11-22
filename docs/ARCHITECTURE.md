# Architecture and module contracts

This file is the contract between modules. A module may change its inside freely. It must not
change a signature or a convention listed here without updating this file and the callers.

## 1. What the system does

A short soccer broadcast clip goes in. An annotated video, a 2D pitch radar and a SoccerNet-GSR
prediction file come out. A vision language model (Gemini) answers four kinds of stateless
requests. Plain code does everything else.

| # | Step | Module | Engine |
|---|---|---|---|
| 1 | Keyframes | `frames.keyframe_indices` | code |
| 2 | Kit legend (once per clip) | `vlm.steps` | model |
| 3 | People per keyframe | `vlm.steps` | model |
| 4 | Pitch landmarks per keyframe | `vlm.steps` | model |
| 5 | Camera motion between frames | `motion` | code |
| 6 | Pitch mapping per frame | `geometry`, `motion` | code |
| 7 | Tracks | `tracking` | code |
| 8 | Shirt number per track | `vlm.steps` | model |
| 9 | Votes, team side, names | `tracking`, `roster` | code |
| 10 | Fill, export, draw | `fill`, `export_gsr`, `render` | code |

## 2. Conventions (do not deviate)

- **Image**: pixels, origin top-left, x right, y down. Images are BGR `uint8` arrays `(H, W, 3)`.
- **Model coordinates**: integers 0-1000, normalised to the image. Boxes are
  `[ymin, xmin, ymax, xmax]`, points are `[y, x]`. Only `vlm/coords.py` converts.
- **Pitch**: metres, origin at the centre mark, x to the right goal, y towards the main camera.
  Far touchline y = -34, near touchline y = +34, pitch 105 x 68. A far-side player has negative y.
- **Homography**: 3x3 `float64`, maps image to pitch: `pitch ~ H @ (x_px, y_px, 1)`. A
  `HomographyRecord` stores the matrix as nested tuples, not as an array: convert with
  `geometry.matrix_to_rows` and `geometry.matrix_from_record`.
- **Frames**: 0-based index. SoccerNet file `000001.jpg` is frame 0. `t = frame / fps`. A frame
  index outside `0 .. n_frames - 1` is an error; a negative index never counts from the end.
- **Arrays of points** have shape `(n, 2)`, `float64`.
- **Records** are the frozen pydantic models of `schemas.py` (base class `Record`); the transport
  types of `vlm/types.py` are records too. Update with `model_copy(update=...)`. `Record.model_copy`
  validates the copy like a new record (plain pydantic does not): a misspelled field name, a wrong
  value and a broken record rule raise `ValidationError`, and a list becomes a tuple.
- **Shirt numbers** are whole numbers from 0 to 99 in every record (`schemas.ShirtNumber`).
- **Finite numbers**: a `Box`, a `LandmarkObs`, a `HomographyRecord` and the pitch position of a
  `PlayerState` (`pitch_x`, `pitch_y`) refuse NaN and infinity. A person without a pitch position
  has no state.
- **Team**: `team_key` (`A`, `B`, `none`) is the model's label from the kit legend. `team_side`
  (`left`, `right`) is the side whose goal the team defends as seen by the camera; it is what the
  scorer compares.
- **Errors**: raise a clear exception for wrong input. Return `None` only where the contract says
  so (for example "no mapping for this frame"). No silent fallbacks.
- **Only `vlm/client.py` may import the model SDK.** Everything else must import and run without it.
- **No machine-specific text** in tracked files: no absolute home paths, no user names, no private
  host names, no keys. Use `~` and environment variables.
- Python 3.12+, type hints on every function, docstrings in plain full sentences, no dead code.

## 3. Existing foundation (already written; read these first)

- `schemas.py` — `Record` (base class), `ShirtNumber`, `Box`, `PersonDet`, `FramePeople`,
  `LandmarkObs`, `FrameLandmarks`, `KitLegend`, `NumberReading`, `HomographyRecord`, `TrackObs`,
  `Track`, `PlayerState`, `FrameState`, `ClipInfo`, `ClipResult`.
  - `KitLegend.prompt_text()` leaves the `uncertain` notes out, so that they do not change the
    prompt hash of the later requests.
- `pitch.py` — `LANDMARKS` (35 real marks, ids 1..35), `landmark_xy`, `landmark_legend_text`,
  `inside_pitch`, `pitch_polylines`, `PitchCanvas`, `draw_landmark_diagram`.
  - `inside_pitch` and `PitchCanvas.to_px` take points of shape `(n, 2)`, one point of shape
    `(2,)`, or an input without values. Any other shape raises `ValueError`.
  - `pitch_polylines(arc_samples)` needs at least 8 samples. A `Polyline` holds an array, so two
    polylines are equal only when they are the same object.
- `vlm/types.py` — `VLMRequest`, `VLMResponse`, `Usage`, `RequestMeta`, `TextPart`, `ImagePart`,
  `Backend` (protocol), `ServiceTier`, `BilledTier`, `BackendError`, `BilledAnswerError`,
  `UnconfirmedCallError`, `FlexBusyError`, `CacheMiss`, `OtherImagesError`.
  - `VLMRequest.cache_key()` is a method. The key is the first 32 hex characters of the SHA-256
    of these values as JSON with sorted keys: clip id, step, frame, item, repeat, model, thinking
    level, the resolutions of the images, the prompt hash (all text parts), the hash of the answer
    schema, and the context. The key has no pixel bytes and no service tier.
  - The key has no pixel bytes because another version of the image library can give the same
    picture other bytes: a miss would be paid again without a word, and a hit with other image
    bytes is an error that costs nothing (`vlm/cache.py`). A replay builds its requests from the
    frames, so it runs on the machine that has the footage. A cassette can be read as text on
    any machine.
  - `VLMRequest.context` is plain JSON data that says what the images show (for example the
    frames and boxes of crops). It is never sent to the model, and it is part of the cache key. So
    two requests that show other images must differ in frame, item or context.
  - `VLMRequest.stored_context()` is the part of the context that a cassette keeps: the entry
    `frames` (`STORED_CONTEXT_KEYS`). Every other entry stays on the machine that made the
    request. The boxes of a probe request are boxes of the label files, and label data must not
    be part of a result file.
  - `VLMResponse.service_tier` is the billed tier (`BilledTier`: `standard`, `flex` or
    `priority`): the tier whose price the answer costs. The real backend takes the tier that the
    answer reports, else the tier of the request. A reported tier without a price counts as
    `priority`, the most expensive tier. A request can ask for `standard` or `flex` only
    (`ServiceTier`). The cost ledger prices the billed tier.
  - `BilledAnswerError(message, billed)` is a `BackendError`: the model answered and billed the
    call, but the answer cannot be used. `billed` is the cost record of the call, a `VLMResponse`
    without text. It is never stored as an answer.
  - `UnconfirmedCallError(message)` is a `BackendError`: the call ended by a client timeout, by
    a lost connection, or by an answer with a success status whose body the SDK could not read.
    No token usage came back. The provider can still bill such a call, and the cost ledger has
    no line for it.
  - `FlexBusyError(message)` is a `BackendError`: the flex tier answered a request with the
    status 503 or 429 (no capacity) for the whole waiting time of the backend. The provider
    refused these attempts, so none of them is billed.
  - `OtherImagesError(message)` is a `BackendError`: a stored answer was recorded with other
    image bytes than its request shows now.
- `config.py` — `StepSettings`, `TrackingConfig`, `PipelineConfig`, `load_config`,
  `THINKING_LEVELS`, `MAX_PARALLEL_LIMIT`.
  - `load_config` lays the file over the defaults. A section that names one key keeps the
    defaults of its other keys: `number: {model: ...}` keeps the low resolution of the number step.
    An empty file gives the defaults. Any other content that is not a mapping raises `ValueError`.
  - A `StepSettings` record refuses a thinking level that its model does not have.
    `THINKING_LEVELS` lists `low`, `medium`, `high` for `gemini-4-argon`, `gemini-4-carbon` and
    `gemini-3.8-flash`, these and `xhigh` for `gemini-4-hydrogen`, and `minimal`, `low`, `medium`,
    `high` for `gemini-robotics-er-2-preview`. A model that is not in the table is not checked.
  - `TrackingConfig` fields and defaults: `max_speed_mps` 9.0, `reach_margin_m` 2.0,
    `max_image_jump` 2.5, `mismatch_penalty` 0.5, `max_missed_keyframes` 4, `max_stitch_gap_s`
    3.0, `stitch_speed_mps` 4.0, `min_observations` 2. Section `tracking.py` says what each one
    does. `configs/default.yaml` writes out every setting of `PipelineConfig` with one comment line.
  - `PipelineConfig.max_parallel` is a whole number from 1 to 8 (`MAX_PARALLEL_LIMIT`), 4 by
    default. A budget stop comes after the answer that crosses a limit, and the requests in
    flight at that time are paid too. So the upper limit bounds that overshoot at the cost of
    7 answers.
- `frames.py` — `FrameSource` (protocol), `ImageFolderSource`, `VideoFileSource`, `keyframe_indices`.
  - `read(frame)` of both sources raises `IndexError` for an index outside the clip and returns a
    new array on every call, so the caller may draw on it.
  - `ImageFolderSource`: the name of every frame file is a whole number, and the files are read
    in the order of their numbers (`2.jpg` before `10.jpg`). Hidden files are skipped. An image
    file whose name is not a number, and two files with the same number, raise `ValueError`.
    `path(frame)` gives the image file of a frame: the job file of an external tool names it.
  - `VideoFileSource` decodes a frame when it is read and keeps only a few frames, so the memory
    use does not grow with the clip. The decoder runs forward: a read of an earlier frame starts
    it again at the first frame. Read frames in increasing order where you can.
  - `keyframe_indices` raises `ValueError` unless the frame count, the frame rate and the
    interval are above zero.

## 4. Module contracts

Type aliases used below:

```python
MotionWarp = Callable[[int, int, np.ndarray], np.ndarray | None]  # (src_frame, dst_frame, xy) -> xy in dst frame; None = the points cannot be moved
PitchMap = Callable[[int, np.ndarray], np.ndarray | None]          # (frame, image xy) -> pitch metres; None = frame has no mapping
```

### `geometry.py` — homography from landmarks

```python
@dataclass(frozen=True)
class HomographyFit:
    H: np.ndarray            # 3x3, image -> pitch
    inliers: np.ndarray      # bool, one per input pair
    reproj_px: float         # median reprojection error of the inliers, image pixels

def apply_homography(H: np.ndarray, xy: np.ndarray) -> np.ndarray
def invert_homography(H: np.ndarray) -> np.ndarray | None
def are_collinear(xy: np.ndarray, tolerance: float) -> bool
def fit_homography(image_xy, pitch_xy, *, ransac_px: float = 8.0, min_inliers: int = 4,
                   min_near_share: float = 0.4) -> HomographyFit | None
def landmark_pairs(obs: FrameLandmarks) -> tuple[np.ndarray, np.ndarray]   # (image_xy, pitch_xy) that a fit can use
def count_near(H: np.ndarray, image_xy: np.ndarray, pitch_xy: np.ndarray, *, ransac_px: float) -> int
def fit_from_landmarks(obs: FrameLandmarks, *, ransac_px: float = 8.0,
                       min_near_share: float = 0.4) -> HomographyRecord
def check_homography(H, image_size: tuple[int, int], feet_xy: np.ndarray | None = None, *,
                     pitch_margin_m: float = 3.0, min_inside_fraction: float = 0.9) -> list[str]
```

- `fit_homography` fits pitch -> image with `cv2.findHomography(..., cv2.RANSAC, ransac_px)` so that
  the threshold is in pixels, then inverts.
- An inlier of `fit_homography` is a pair whose pitch point lands at most `ransac_px` from its
  image point under the returned matrix. The marks of the solver are not used, because with
  exactly 4 pairs it marks every pair.
- `fit_homography` returns `None` when there are fewer than `min_inliers` inliers, when more than
  4 pairs are given and fewer than 5 of them are inliers or fewer than `min_near_share` of them
  (0.4, `geometry.MIN_NEAR_SHARE`) are near the fit, when the matrix has no usable inverse, or
  when the inlier points cannot fix a homography: all of them, or all but one, lie close to one
  line (a midfield view with marks on the halfway line only). The line rule holds for the pitch
  points (0.5 m) and for the image points (1 px), so two landmarks at one image point, or three
  image points on one line, give no fit from four pairs.
- A pair is near a fit when its error is at most 3 times `ransac_px`
  (`geometry.NEAR_LIMIT_FACTOR`): a point with a wrong landmark id is not, a point with some
  noise is. The share is not counted with `ransac_px` itself, because landmark noise of the
  size of that threshold would refuse fits that are as exact as the others. With
  `min_near_share` 0 the share rule is off.
- `landmark_pairs` returns the point pairs of a landmarks answer that a fit can use: it drops
  unknown landmark ids, and every observation of an id that the answer names more than once.
  `count_near` counts the pairs that are near a given image-to-pitch mapping, with the limit of
  the share rule; a mapping without a usable inverse has none.
- `fit_from_landmarks` fits the pairs of `landmark_pairs`. It never raises for a well-formed
  answer: it returns a record with `source="none"` when no fit exists.
- `invert_homography` returns `None` for a matrix whose smallest singular value is at most 1e-12
  of its largest one, so a matrix that is singular only up to rounding errors has no inverse.
- `apply_homography` takes points of shape `(n, 2)` or one point of shape `(2,)`; any other shape
  raises `ValueError`. `are_collinear` needs a finite tolerance that is not negative; with 0 the
  points must lie exactly on one line.
- `check_homography` returns a list of problems (empty = plausible): mirrored or flipped mapping
  (the image centre must map into the pitch plus 30 m; walking right in the image must increase
  pitch x; walking down must increase pitch y), and fewer than `min_inside_fraction` of the foot
  points inside the pitch plus margin. A negative or NaN `pitch_margin_m` and a
  `min_inside_fraction` outside 0 to 1 raise `ValueError`.

### `motion.py` — camera motion between frames

```python
def estimate_motion(prev: np.ndarray, nxt: np.ndarray, *, mask: np.ndarray | None = None,
                    ransac_px: float = 3.0, min_matches: int = 30) -> np.ndarray | None   # 3x3: prev px -> nxt px

class MotionField:
    def __init__(self, steps: Sequence[np.ndarray | None])    # steps[i]: frame i px -> frame i+1 px
    n_frames: int                                              # len(steps) + 1
    def between(self, src: int, dst: int) -> np.ndarray | None # 3x3 src px -> dst px, both directions
    def warp(self, src: int, dst: int, xy: np.ndarray) -> np.ndarray | None
    @classmethod
    def identity(cls, n_frames: int) -> "MotionField"
    def to_json(self) -> dict ; @classmethod def from_json(cls, data: dict) -> "MotionField"

def bridged_warp(motion: MotionField, homographies: Sequence[HomographyRecord]) -> MotionWarp
def compute_motion_field(source: FrameSource, *, people: Mapping[int, Sequence[Box]] | None = None,
                         downscale: float = 0.5) -> MotionField
def propagate_homographies(key_fits: Mapping[int, HomographyRecord], motion: MotionField, n_frames: int,
                           *, max_gap_frames: int = 75) -> list[HomographyRecord]
```

- `estimate_motion` follows corners with sparse optical flow and fits one homography with RANSAC.
  `mask` is a `uint8` image, 0 where corners must not be taken (people, graphics). It needs a
  finite `ransac_px` above 0. `compute_motion_field` masks the people of the keyframes; it masks
  no broadcast graphics.
- `MotionField.between` and `MotionField.warp` raise `TypeError` for a frame index that is not a
  whole number, also for `True`.
- `bridged_warp` moves image points like `motion.warp` wherever the motion chain between the two
  frames exists. Across a missing step it goes through the pitch mappings of the two frames:
  image to pitch with the mapping of the source frame, then pitch to image with the inverse
  mapping of the destination frame. For a camera that only turns and zooms this is the image
  motion of every point. The answer is `None` when the chain is broken and one of the two frames
  has no mapping. `pipeline.run_clip` gives this warp to the tracking and to the dense output.
- `propagate_homographies` returns one record per frame. A frame with its own valid landmark fit
  keeps it (`source="landmarks"`). Any other frame takes the nearest keyframe fit within
  `max_gap_frames` whose motion chain exists: `H_t = H_k @ M(t -> k)`, `source="propagated"`.
  Otherwise `source="none"`.

### `tracking.py` — tracks, votes, team side, numbers

```python
def associate(keyframes: Sequence[FramePeople], *, fps: float, warp: MotionWarp, to_pitch: PitchMap,
              config: TrackingConfig = TrackingConfig()) -> list[Track]
def stitch_fragments(tracks: Sequence[Track], *, fps: float, to_pitch: PitchMap,
                     config: TrackingConfig = TrackingConfig()) -> list[Track]
def vote_role_team(track: Track) -> tuple[Role, TeamKey]
def assign_team_sides(tracks: Sequence[Track], mean_pitch_x: Mapping[int, float]) -> dict[TeamKey, TeamSide]
def goal_side(mean_x: float | None) -> TeamSide | None   # the goal zone that a mean pitch x lies in, or None
def decide_number(reading: NumberReading | None, frame_numbers: Sequence[int | None]) -> tuple[int | None, int | None]
def finalize_tracks(tracks: Sequence[Track], *, readings: Mapping[int, NumberReading],
                    mean_pitch_x: Mapping[int, float], fps: float, to_pitch: PitchMap,
                    config: TrackingConfig = TrackingConfig(), roster: Roster | None = None) -> list[Track]
```

- `associate`: Hungarian matching (`scipy.optimize.linear_sum_assignment`) between consecutive
  keyframes. The foot point of the newest observation of a track is moved on with 0.85 of the
  velocity of the track and then into the later frame with `warp`. The velocity is the motion of
  the foot point between the last two observations of the track with the camera motion taken
  out; it is used for at most twice the gap between these two observations. Cost = distance
  between this expected foot point and the foot point of the detection in units of box height
  (mean of the two boxes), plus `mismatch_penalty` when the role or the team key of the detection
  differ from the vote of the track so far. The team key `none` on either side is no difference.
  The penalty counts once more for every keyframe that the track missed. A pair is forbidden
  when the cost is above `max_image_jump`, or when both pitch positions exist and their distance
  is above `max_speed_mps * dt + reach_margin_m`. A forbidden pair costs just above
  `max_image_jump` in the solver, so a track that lost its person and a new detection do not
  shift the other matches. When `warp` returns `None` or points that are not finite, the track
  gets no detection in that frame: a raw image position is never used. `pipeline.run_clip`
  passes `motion.bridged_warp(field, homographies)`, which crosses a missing motion step through
  the pitch mappings. Unmatched tracks stay alive for `max_missed_keyframes` keyframes. Track ids
  are 0, 1, 2, ...
- `stitch_fragments` first drops tracks shorter than `min_observations`; they take no part in
  the joins. It joins two of the other tracks when all of this is true: they have the same voted
  team key and role; they do not show different shirt numbers (the most frequent per-frame
  number of each, when both have one); they do not overlap in time and the time gap between them
  is at most `max_stitch_gap_s` (3 seconds by default); both pitch positions are known and their
  distance is at most `stitch_speed_mps * dt + reach_margin_m` (`stitch_speed_mps` is 4 m/s by
  default, lower than `max_speed_mps`); and the join has no rival. A track is joined to the first
  candidate that follows it only when no other candidate is in the image at the same time as
  that one, and the same rule holds backwards. A candidate that starts after the first one ended
  is no rival, so chains of fragments become one track.
- `vote_role_team`: majority over the observations, weighted by box height. The team key is
  voted among `A` and `B` only: an observation with `none` gives no vote. The result is `none`
  when no observation has a team, and for a referee or an "other" person.
- `assign_team_sides` uses two kinds of evidence. Players: among tracks voted `player`, the team
  key with the lower mean pitch x is `left`; with one team key only, the sign of its mean x
  decides. Goalkeepers: a track voted `goalkeeper` with the team key `A` or `B` whose mean pitch
  x lies in the depth of a penalty area (36 m or more from the halfway line,
  `tracking.GOAL_ZONE_MIN_X_M`) says that its team defends that goal; it counts with its
  observations times its depth in that zone. The players decide alone when both team keys are
  seen and their means are at least 3 m apart (`tracking.CLEAR_TEAM_GAP_M`, an estimate). Else
  the goalkeepers decide when they name one answer with more weight. Else the players decide.
  `mean_pitch_x` maps track id to the mean pitch x of that track (tracks without a position are
  absent).
- `goal_side` returns the side of the goal zone that a mean pitch x lies in (`left` or `right`),
  or `None` for a value outside both zones and for `None`. `tracking` and the team-side rule of
  the labels (`evalx/oracle.py`) both use it, so the goal zone has one definition.
- `decide_number` returns `(export, display)`. Export = the number of the reading when its
  confidence is `high` or at least 2 crops show it; else the most frequent number over the crop
  numbers and the per-frame numbers together (crops first in a tie); else the number of the
  reading, also with a low confidence; else `None`. In that vote, when the reading names a
  number of two digits, a crop or a frame that shows one of these digits alone counts for the
  number of the reading: a number of two digits is often seen in parts. Display = the number of
  the reading only when confidence is `high` and at least 2 crops agree with it; else `None`.
- `finalize_tracks` fills `role`, `team_key`, `team_side`, `number_export`, `number_display`,
  `name`. Referees and "other" get no team side and no number. A goalkeeper whose mean pitch x
  lies in a goal zone has the side of that goal, whatever its team key is. Names come from the
  roster by (team side, displayed number). `fps` must be a finite number above zero.
- `finalize_tracks` lets a displayed number name one person per team side. The tracks with the
  same side and displayed number are one group. Two tracks of a group are two people when the
  first observation of the later one is further from the observation of the other track that
  is nearest in time than `max_speed_mps * dt + 6 m` (`tracking.TWO_PEOPLE_MARGIN_M`, an
  estimate); `to_pitch` gives the two positions, and without both of them this cannot be told.
  Tracks of a group that overlap in time are never linked: the track with the most observations
  keeps the number, and a track that overlaps it loses the displayed number and keeps the export
  number. A group that holds two people keeps the number on its longest track only, and no
  track of it is linked. In every other group the kept tracks that follow each other become
  one track when they have the same role and both pitch positions are known: one person who
  was out of the image. The track has the id and the place of the earliest fragment. So the
  result can have fewer tracks than the input.

Track repairs (2026-11-22). Both are arms of gate G4 and are off by default
(`TrackingConfig.split_team_swaps`, `TrackingConfig.team_slots`, both `False`); the default
behaviour above does not change until G4 has measured them (`docs/DECISIONS.md`, 2026-11-22).

```python
SWAP_MIN_VOTES = 4                 # votes for its own key that a run of one team key needs (estimate)
SWAP_CONTACT_BOX_HEIGHTS = 1.0     # contact at a team-key change, foot points in box heights (estimate)
TEAM_SLOTS = {"player": 10, "goalkeeper": 1}   # people of one team side and role in a clip
@dataclass(frozen=True)
class TeamRun: team_key: TeamKey; start: int; stop: int    # observations[start:stop]
def team_key_runs(observations: Sequence[TrackObs]) -> list[TeamRun]
def split_team_swaps(tracks: Sequence[Track]) -> list[Track]
def link_team_slots(tracks: Sequence[Track], *, fps: float, to_pitch: PitchMap,
                    config: TrackingConfig = TrackingConfig()) -> list[Track]
```

- `team_key_runs`: the observations with the key `A` or `B` vote; votes in a row with one key
  are a run, and the strength of a run is its number of votes for its own key. While a run is
  weaker than `SWAP_MIN_VOTES`, the weakest run (the first in a tie) takes the key of its
  neighbours and joins them. So each change between two runs left is a change of the team key
  that held for 4 votes on both sides. One run left gets the key of most votes (`A` in a tie).
  A run starts at the first vote of its key; the first run starts at observation 0 and the last
  ends at the last observation; `none` belongs to the run it lies in. No vote gives `[]`.
- `split_team_swaps` (before `stitch_fragments`, in `pipeline._tracks` when
  `split_team_swaps` is on): a track of a player or a goalkeeper is cut at the first vote of a
  run when, at that change, a person of the other team was in contact with it. The change spans
  the observations from the last vote of the run before to the first vote of the run after.
  Contact: in the frame of one of them, an observation of another track at most
  `SWAP_CONTACT_BOX_HEIGHTS` box heights away (foot points, mean height of the two boxes), whose
  run has the other key than the run of this track there. The pieces are fragments with ids
  0..n-1 in the order of the first frame; the joins of `stitch_fragments` link each with the
  tracks of its own team. Limits: a swap of two teammates shows no change; a change with fewer
  than 4 votes on a side (a swap just before a track ends) is not seen; a model that names the
  wrong team for seconds next to an opponent gives a wrong cut, and the two pieces are not joined
  again.
- `link_team_slots` (at the end of `finalize_tracks` when `team_slots` is on; the tracks need
  their votes, side and numbers): the tracks of a player or a goalkeeper with a team side form
  one group per (side, role). The rule holds over the whole clip, not per frame, because the
  camera shows a part of the pitch: a group with `n` tracks and `s` slots holds at least `n - s`
  later parts of other tracks of the group. Two tracks can be linked when the earlier one ends
  before the later one starts, they show no different displayed numbers and no different most
  frequent per-frame numbers, both pitch positions (end of the earlier, start of the later) are
  known, and their distance is at most `max_speed_mps * dt + reach_margin_m`. The links are
  taken by distance (then time gap, then ids), each track with at most one earlier and one
  later link, until the group holds `s` tracks or no link is left; a group with `s` tracks or
  fewer does not change. A chain becomes one track with the id and place of its earliest track,
  all observations, the team key voted over them, and the numbers of its longest track with a
  displayed number (else of its longest track); the name is set later. Limits: the distance is
  the only evidence, so two players who come back where the other one left are linked the wrong
  way; a substitution or a red card adds people to a side. The slots are not used in
  `stitch_fragments`: the joins run before the shirt numbers are read, and a slot link there
  could weld two players whose numbers `finalize_tracks` would tell apart.
- `stitch_fragments` was reviewed against the failure "a wide gate welds two players" with two
  cases (`tests/test_tracking.py`, section "Stitching gates"): two teammates who cross while both
  are hidden are not welded (the rival rule refuses both joins, also when one comes back a
  keyframe later); a teammate who enters where a player left is joined when the player does
  not come back within the stitch time and nothing else tells them apart (a limit of the data,
  no gate change), and is not joined when the player comes back while the teammate is in view
  or when the numbers differ. No gate was tightened.

### `roster.py`

```python
class Roster(Record):
    left: dict[int, str] ; right: dict[int, str] ; left_name: str = "" ; right_name: str = ""
    def name_for(self, side: TeamSide, number: int) -> str | None
def load_roster(path: Path) -> Roster    # YAML or JSON: {left: {name, players: {number: name}}, right: {...}}
```

`load_roster` raises `ValueError` with the file name for a key that is written twice in one
mapping (a shirt number, a side, a team name), for a file that is not UTF-8 text and for a file
that is not valid YAML or JSON. The same shirt number on both sides is allowed.

### `fill.py` — dense output

```python
def dense_frame_states(tracks: Sequence[Track], homographies: Sequence[HomographyRecord], n_frames: int,
                       *, warp: MotionWarp, image_size: tuple[int, int], max_fill_gap_frames: int,
                       pitch_margin_m: float = 3.0) -> list[FrameState]
```

One `FrameState` per frame. For each track: pitch position at every observation whose frame has
a valid mapping (foot point through `H`); linear interpolation in pitch space for the frames
between two such observations that follow each other; no extrapolation. Boxes between two
keyframes: both keyframe boxes are warped into the frame and blended linearly (raw linear
interpolation when `warp` returns `None`). `observed=True` only at keyframe observations.
Positions further than `pitch_margin_m` outside the pitch are dropped. Tracks without a decided
role are skipped. A negative or NaN `pitch_margin_m` raises `ValueError`.

Image border: `image_size` is `(width, height)` and has no default. A frame without an
observation gets no state when the blended box of the person lies fully outside the image: the
person left the image and came back, and the labels have nobody there. A state at an
observation is always kept.

Gap limit: two observations with a pitch position that are more than `max_fill_gap_frames` frames
apart are not joined by interpolation. Inside such a gap the track has no state, also at an
observation without a pitch position. The argument is a number of frames and has no default.
`pipeline.run_clip` passes `round(config.tracking.max_stitch_gap_s * fps)`, so the dense output
uses the time limit of `stitch_fragments`. Reason: in GS-HOTA a prediction at a wrong place costs
more than a missing one.

### `vlm/coords.py`

```python
def box_from_model(box_2d: Sequence[float], width: int, height: int) -> Box | None   # None when degenerate
def point_from_model(point: Sequence[float], width: int, height: int) -> tuple[float, float]  # -> (x_px, y_px)
def box_to_model(box: Box, width: int, height: int) -> list[int]
def point_to_model(x: float, y: float, width: int, height: int) -> list[int]         # -> [y, x]
```

### `vlm/steps.py` and `vlm/prompts/` — the four requests

Prompt texts live in `vlm/prompts/<name>.txt` (versioned text, loaded with `importlib.resources`).
Every builder returns a `VLMRequest`; every parser turns answer text into a record and raises
`ParseError` (subclass of `ValueError`) on invalid JSON. Parsers drop single bad items (degenerate
box, unknown landmark id, value out of range) instead of failing the whole frame, and clamp
coordinates to 0-1000.

```python
def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes
def build_kit_legend_request(images: Sequence[np.ndarray], *, clip_id: str, settings: StepSettings, repeat: int = 0) -> VLMRequest
def parse_kit_legend(text: str) -> KitLegend
def build_people_request(image: np.ndarray, legend: KitLegend, *, clip_id: str, frame: int,
                         settings: StepSettings, repeat: int = 0, item: str | None = None) -> VLMRequest
def parse_people(text: str, *, frame: int, width: int, height: int) -> FramePeople
def build_landmarks_request(image: np.ndarray, *, clip_id: str, frame: int, settings: StepSettings, repeat: int = 0) -> VLMRequest
def parse_landmarks(text: str, *, frame: int, width: int, height: int) -> FrameLandmarks
CROP_PADDING = 0.15   # a number crop is larger than the box by this share of the box size, on every side
def crop_rect(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None   # (left, top, right, bottom); None = outside the image
def build_number_request(crops: Sequence[np.ndarray], legend: KitLegend, team_key: TeamKey, *, clip_id: str,
                         track_id: int | None, crop_frames: Sequence[int], crop_boxes: Sequence[Box],
                         image_size: tuple[int, int], crop_upscale: float,
                         settings: StepSettings, repeat: int = 0) -> VLMRequest
def parse_number(text: str) -> NumberReading
```

`meta.step` values: `kit_legend`, `people`, `landmarks`, `number`. The number request sets
`context = {"frames": [...], "boxes": [[x1, y1, x2, y2], ...]}`. Its `meta.item` is
`crops-<16 hex digits>` without a track id and `track-<id>-<8 hex digits>` with one. The landmarks
request sends two images: the numbered diagram (`pitch.draw_landmark_diagram()`, resolution
`medium`) and the frame; its text includes `pitch.landmark_legend_text()`.

The cache key has no pixels, so the stored answer of a number request must be found by its
crops. Two things make sure of that. The context (crop frames and boxes) is part of the cache
key. And the item holds a hash of the crop frames, the crop rectangles and the enlargement,
because the enlargement changes the pixels but not the context. `build_number_request` is the
only place that makes this item. It takes the crops as they are sent (cut and enlarged by the
caller), the size of the frames (`image_size`, width and height) and the enlargement
(`crop_upscale`). The crop rectangle of a box is `crop_rect`: the box plus `CROP_PADDING` on
every side, in whole pixels, cut at the image border. `pipeline` and `probes` cut their crops
with `crop_rect`, so the hashed rectangles are the pixel areas that are sent.

A track id of the pipeline is not a lasting name: it is a place in the track list, and after a
change in the tracking code or settings `track 12` can be another person. So `pipeline` gives
`track_id=None`, and the same crops find their stored answer again whatever id their track
has. `probes` gives the track id of the label file, which is a lasting name.

The cache key of a kit legend request has no frame. A kit legend that is asked from other
frames of the same clip finds the stored answer with other image bytes, so it needs
`allow_other_images` (section `vlm/cache.py`).

Answer formats (also the `response_schema` of each request):

```json
{"team_a": {"shirt": "", "shorts": "", "socks": ""}, "team_b": {"shirt": "", "shorts": "", "socks": ""},
 "goalkeepers": [{"kit": "", "team": "A|B|none"}], "referee_kit": "", "uncertain": [""]}
{"people": [{"box_2d": [0, 0, 0, 0], "role": "player|goalkeeper|referee|other", "team": "A|B|none", "number": null}]}
{"view": "left|centre|right", "landmarks": [{"id": 1, "point": [0, 0]}]}
{"per_crop": [null], "number": null, "confidence": "high|medium|low"}
```

Prompt rules: describe kits by colour only; never ask about faces or names; ask for a tight box
that includes the feet; a number only when clearly readable, else null; landmarks only when visible
inside the frame, never extrapolated.

### `vlm/cache.py`

```python
class CassetteStore:
    def __init__(self, root: Path, *, allow_other_images: bool = False)
    def path_for(self, request: VLMRequest) -> Path      # root/<clip_id>/<step>/<cache_key>.json
    def get(self, request: VLMRequest) -> VLMResponse | None
    def check_writable(self, request: VLMRequest) -> None   # OSError when the cassette could not be written; creates nothing
    def put(self, request: VLMRequest, response: VLMResponse) -> Path

class CachedBackend:    # implements Backend
    def __init__(self, store: CassetteStore, inner: Backend | None = None,
                 on_new_answer: Callable[[VLMRequest, VLMResponse], None] | None = None)
    def generate(self, request: VLMRequest) -> VLMResponse
```

A cassette is a JSON text file: key, meta, context, model, thinking level, prompt name and hash,
image resolutions, image hashes, and the response. The context of a cassette is
`VLMRequest.stored_context()`: for a number request the frames of the crops, and never their
boxes. The key of the cassette still stands for the whole context. A cassette never contains
image bytes and never a box of a label file.

A hit returns the stored answer. A hit whose request shows other image bytes than the cassette
was recorded with raises `OtherImagesError`, and the model is not asked: the stored answers of a
clip belong to its pixels. A store with `allow_other_images=True` returns the stored answer and
logs a warning. A miss first checks that the cassette can be written (`OSError` before the model
is asked), then calls `inner`, stores the answer and calls `on_new_answer` (used for the cost
ledger). A paid answer reaches `on_new_answer` also when its cassette cannot be written; the
error of the store is raised after that, and an error of the callback replaces it. A miss with
`inner=None` raises `CacheMiss` (replay mode). It is safe to call from several threads.

No lock keeps two commands apart that work on one clip at the same time: both can pay for the
same request. The second answer then logs a warning, both answers are in the cost ledger (two
lines with the same `request`), and the newer one stays in the store.

Cassettes go to the `results` branch, so they must pass the guard and the secret scan of the git
hooks. The default rules of the secret scan take the line `"key": "<32 hex digits>"` for a key.
`.gitleaks.toml` allows this one line in a file `<clip>/<step>/<32 hex digits>.json`. Keep the
name and the place of that field, or change the rule with it.

Rule for an answer that was billed and then rejected (`BilledAnswerError`): its cost reaches the
cost ledger and the budget guard in exactly one place per call path, the wrapper that reports the
new answers of the real backend to the cost meter. That wrapper passes `error.billed` to the same
callback as a new answer, stores nothing and raises the error again. For `run` and `probe` the
wrapper is `CachedBackend`. For the contract stage of `doctor`, which has no cache, it is
`cli.MeteredBackend`. A real backend gets one of the two wrappers, never both, and no other code
records `billed`. An error of the callback (a budget stop) replaces the `BilledAnswerError`.

### `vlm/cost.py`

```python
@dataclass(frozen=True)
class Price: model: str; valid_from: date; input_usd_per_million: float; output_usd_per_million: float
PRICES: tuple[Price, ...]            # dated table, see below
FLEX_TIER_MODELS: frozenset[str]     # the models that have a flex tier
FLEX_FACTOR = 0.5
PRIORITY_FACTOR = 1.8
HIGHEST_PRICE_TIER = "priority"
MAX_LEDGER_FILE_BYTES = 1_000_000
def price_for(model: str, on: date) -> Price                      # KeyError for an unknown model
def has_flex_tier(model: str) -> bool
def cost_usd(model: str, usage: Usage, tier: BilledTier, on: date) -> float
def billed_usd(response: VLMResponse) -> float                    # cost of an answer at the price of its UTC day
class TokenEstimate(Record): input_tokens: int; output_tokens: int   # output includes thinking
def estimate_usd(requests: Iterable[VLMRequest], estimates: Mapping[str, TokenEstimate], on: date) -> float  # keyed by meta.step
def ledger_file_number(first_name: str, name: str) -> int | None  # 1 for ledger.jsonl, 2 for ledger-0002.jsonl, None for any other name
class Ledger:
    def __init__(self, path: Path)                                # JSON lines, append-only, thread-safe
    def files(self) -> list[Path]                                 # ledger.jsonl, ledger-0002.jsonl, ... by number
    def record(self, run_id: str, request: VLMRequest, response: VLMResponse) -> float   # returns USD
    def check_writable(self) -> None                              # OSError when a line could not be added; creates nothing
    def total_usd(self) -> float ; def run_usd(self, run_id: str) -> float
class BudgetExceeded(RuntimeError)
class BudgetGuard:
    def __init__(self, ledger: Ledger, *, hard_stop_usd: float = 135.0, run_factor: float = 1.3)
    def start_run(self, run_id: str, estimate_usd: float, *, limit_usd: float | None = None) -> None
    def check(self, run_id: str) -> None                          # raises BudgetExceeded
```

`cost = (input * in_price + (output + thought) * out_price) / 1e6`, times `FLEX_FACTOR` (0.5) for
tier `flex` and `PRIORITY_FACTOR` (1.8) for tier `priority`. Prices in USD per million tokens
(input / output): `gemini-4-argon` and `gemini-4-carbon` 2 / 10 from 2026-10-13 and 4 / 20 from
2026-12-01; `gemini-4-hydrogen` 1.5 / 7.5; `gemini-3.8-flash` 0.75 / 3.75 and 1.5 / 7.5 from
2027-01-01; `gemini-robotics-er-2-preview` 1 / 5 and 2 / 10 from 2027-01-01. Default token
estimates until measured (`DEFAULT_TOKEN_ESTIMATES`): people 1500 / 1400, landmarks 2200 / 800,
number 2500 / 400, kit_legend 3600 / 600. No test pins these values: after the cost pilot, change
the table and this line together.

Flex tier: `FLEX_TIER_MODELS` holds `gemini-4-argon`, `gemini-4-carbon`, `gemini-4-hydrogen` and
`gemini-3.8-flash`. The provider documents no flex tier for `gemini-robotics-er-2-preview`. The
flex tier of a model without one has no price: `cost_usd` and `estimate_usd` raise `ValueError`
for it, `cli.estimated_usd` raises `CliError` before the estimate is printed, the real backend
does not send such a request, and the contract checks of `doctor` plan none.

Ledger: one or more JSON lines files in the cache folder: `ledger.jsonl`, then
`ledger-0002.jsonl`, `ledger-0003.jsonl` and so on. A file takes lines until it would pass
1,000,000 bytes (`MAX_LEDGER_FILE_BYTES`, half of the size limit of the guard), so every ledger
file passes the guard. Totals are read from all files. The files must be a complete series
with the numbers 1 to n, each one time: `total_usd` and `run_usd` raise `ValueError` when the
first file is missing, when a number is missing or is there two times, for a damaged line and
for a line with a negative cost. `record` still adds its line to the last file, so a paid
answer never loses its line. `ledger_file_number` is the one place that knows the names of the
ledger files; the listing of the `sync` command uses it too.

A ledger line holds `time`, `run_id`, `clip_id`, `step`, `frame`, `item`, `repeat`, `request`
(the cache key of the request), `model`, `service_tier` (the billed tier), `attempts`,
`input_tokens`, `output_tokens`, `thought_tokens` and `usd`. Two lines with the same `request`
are two payments for the same question. A ledger file whose last line has no line end gets one
before the next line. `BudgetGuard.start_run` raises `OSError` when the ledger cannot be
written, so a run whose answers cannot be recorded does not start.

### `vlm/client.py` — the real backend (the only SDK import)

```python
class GeminiBackend:    # implements Backend
    def __init__(self, *, api_key_env: str = "GEMINI_API_KEY", timeout_s: float = 300.0,
                 flex_timeout_s: float = 900.0, max_attempts: int = 4, flex_busy_wait_s: float = 900.0,
                 max_usd_per_10_min: float = 8.0, stop: threading.Event | None = None)
    def generate(self, request: VLMRequest) -> VLMResponse
```

Three more keywords are for tests and wrappers: `client` (anything with `.interactions.create`
in the place of the SDK client), `clock` and `sleep`.

Interactions API, `store=False`, no sampling parameters, the request's thinking level, JSON answer
with the request's schema, images inline with their resolution, the request's service tier. No
request limits its output tokens. `usage.output_tokens` excludes thinking tokens;
`usage.thought_tokens` holds them. The key is read from the environment at call time and never
logged. The client needs SDK 3.x (`google-genai>=3.0,<4`): the SDK gets one attempt per call,
and the backend does the retries and reports the number of attempts.

One call of `client.interactions.create` is one HTTP request: the backend passes
`automatic_continuation=False`. By default the SDK continues an answer that stops at the token
cap of one request (status `requires_continuation`; 32,768 tokens of thinking and output on
`gemini-4-argon` and `gemini-4-carbon`) with further billed requests. Such an answer is not
continued: it raises `BilledAnswerError` with the usage of its one request.

Timeouts and retries: a call with the standard tier gets the client timeout `timeout_s` (300 s)
and a call with the flex tier `flex_timeout_s` (900 s); the SDK client is built with the longer
one. A failed call is tried again, with backoff, after the status codes 408, 409, 429 and 5xx
and after a lost connection (the SDK error `APIConnectionError`, known by its class name). A
client timeout (`APITimeoutError`) is not tried again. An SDK error with a success status (an
answer whose body the SDK could not read, `APIResponseValidationError`) is not tried again
either. A call that ends by a timeout, by a lost connection or by such an answer raises
`UnconfirmedCallError`, and a warning is logged: the call may still be billed, and the cost
ledger has no line for it. The status code of an SDK error is read from `status_code` only.

Flex tier: a request for the flex tier that gets the status 503 or 429
(`FLEX_BUSY_STATUS_CODES`, the documented answers of a busy flex tier) is tried again with the
same backoff until `flex_busy_wait_s` (900 s, `FLEX_BUSY_WAIT_S`) have passed since its first
attempt. These attempts are not billed and are not counted in `max_attempts`. Then the request
raises `FlexBusyError`. A standard request and every other failure of a flex request keep
`max_attempts` (4).

Stop request: `stop` is an event. While it is set, the backend sends no request and no new
attempt: a wait for a retry or for room in the spend window ends at once
(`SpendRateLimiter.wait_for_room(stop)`), and the request raises a `BackendError` that says so;
a warning is logged. A call in flight is not cut off.

Money: a spend-rate limiter keeps the cost of one backend under `max_usd_per_10_min`. It lives
in one process: two commands at the same time do not see the spend of each other. A missing SDK
or key, a model without a price and a flex request for a model without a flex tier raise
`BackendError` before any call. An answer whose token usage is known but that cannot be used (it
is not completed, it has no text, or its usage reports no output tokens) raises
`BilledAnswerError` with the cost record of the call. Its cost is already in the spend-rate
limiter. The backend itself writes no cost ledger.

### `synth.py` and `vlm/fake.py` — test world

```python
class SyntheticScene:
    def __init__(self, *, n_frames: int = 100, fps: float = 25.0, width: int = 1280, height: int = 720, seed: int = 0)
    info: ClipInfo
    def image(self, frame: int) -> np.ndarray
    def homography(self, frame: int) -> np.ndarray           # true image -> pitch
    def people(self, frame: int) -> FramePeople               # true visible people
    def landmarks(self, frame: int) -> FrameLandmarks         # true visible landmarks
    def frame_state(self, frame: int) -> FrameState           # true dense state
    def kit_legend(self) -> KitLegend
    def true_number(self, frame: int, box: Box) -> int | None # by best IoU >= 0.3
    def gsr_labels(self) -> dict                               # content of a Labels-GameState.json
class SyntheticSource:   # FrameSource over a scene
    def __init__(self, scene: SyntheticScene)

class FakeBackend:       # implements Backend, answers from the scene's ground truth
    def __init__(self, scene: SyntheticScene, *, miss_rate: float = 0.0, box_noise_px: float = 0.0,
                 point_noise_px: float = 0.0, seed: int = 0)
```

The scene has two teams of 11 (10 players and a goalkeeper each), one referee, a fixed-position
camera on the near side that pans and zooms, pitch markings drawn through the camera, and people
drawn as kit-coloured boxes. Team A defends the left goal. Ground truth is exact.

### `labels.py` — SoccerNet-GSR label reader

```python
class GsrAthlete(Record): track_id: int; box: Box | None; pitch_x: float | None; pitch_y: float | None; role: str; team: str | None; jersey: int | None
class GsrFrame(Record): frame: int; image_id: str; file_name: str; scored: bool; athletes: tuple[GsrAthlete, ...]; lines: dict[str, tuple[tuple[float, float], ...]]
class GsrLabels(Record): clip_id: str; fps: float; n_frames: int; width: int; height: int; frames: tuple[GsrFrame, ...]; jersey_is_str: bool; version: str | None
def load_gsr_labels(path: Path) -> GsrLabels
def gsr_frame_states(labels: GsrLabels) -> list[FrameState]     # ground truth in the internal schema
def gsr_frame_people(labels: GsrLabels, frame: int) -> FramePeople
```

Label facts: file `<clip>/Labels-GameState.json`; keys `info`, `images`, `annotations`,
`categories`. `images[i]`: `image_id` (string), `file_name` (`000001.jpg`), `is_labeled`,
`has_labeled_person`, `has_labeled_pitch`, `has_labeled_camera`. A frame is scored only when all
three `has_labeled_*` flags are true. Athlete annotations have `supercategory == "object"`,
`track_id`, `attributes` `{role, jersey, team}`, `bbox_image` `{x, y, x_center, y_center, w, h}`
(pixels, x and y are the top-left corner) and `bbox_pitch` with `x_bottom_middle`,
`y_bottom_middle` (metres) and four more keys. Roles: `player`, `goalkeeper`, `referee`, `other`;
a role `ball` is skipped. `jersey` is a string of digits in the paper sample (the type must be
detected and stored in `jersey_is_str`) or null. `team` is `left`, `right` or null. Pitch
annotations have `supercategory == "pitch"` and `lines`: `{class name: [{x, y}, ...]}` normalised
to 0-1.

`load_gsr_labels` puts the images in the order of the numbers in their file names (`2.jpg`
before `10.jpg`), the rule of `ImageFolderSource` for the frame files, so a label and its
picture get the same frame index. It raises `ValueError` with the file and the place when the
name of an image file is not a whole number, when two images have the same number or the same
image id, when `info.seq_length` differs from the number of images, when the frame rate is not a
finite number above zero, when a track id is not a whole number, when an image holds two
athletes with the same track id (the ball does not count), when a shirt number is neither a
string of the digits 0 to 9 nor a whole number, and when a pitch position is not finite.
`n_frames` is the number of images. The reader is strict on purpose: it has not read a real
label file yet (RUNBOOK, section 8).

### `export_gsr.py` — prediction writer and checker

```python
def prediction_items(frames: Sequence[FrameState], image_ids: Mapping[int, str], *, jersey_as_str: bool) -> list[dict]
def predictions_path(root: Path, split: str, tracker: str, clip_id: str) -> Path   # root/SoccerNetGS-<split>/<tracker>/<clip_id>.json
def write_predictions(frames, image_ids, root: Path, *, split: str, tracker: str, clip_id: str, jersey_as_str: bool) -> Path
def check_predictions(payload: dict, *, expected_image_ids: Collection[str], jersey_as_str: bool) -> list[str]
def image_ids_without_items(payload: dict, *, expected_image_ids: Collection[str]) -> list[str]
```

File content: `{"predictions": [item, ...]}`. Item: `id` (running integer as string), `image_id`
(the label's string), `track_id` (int, unique per frame), `supercategory: "object"`,
`category_id: 1`, `attributes: {role, team, jersey}`, `bbox_pitch` with all six keys
(`x_bottom_left`, `y_bottom_left`, `x_bottom_middle`, `y_bottom_middle`, `x_bottom_right`,
`y_bottom_right`; left and right repeat the middle), and `bbox_image` when a box exists. `team` is
the team side for players and goalkeepers, else null. `jersey` is the export number for players
only (string or int as `jersey_as_str` says), else null. Frames that are not in `image_ids` are
skipped. `check_predictions` returns a list of problems (empty = valid) and tests every rule above
plus: no unknown image ids. A scored frame without an item is not a problem: the scorer reads
such a file and counts the labelled people of that frame as missed. `image_ids_without_items`
returns the expected image ids without an item, sorted, so that a command can report their
number. It raises `ValueError` when the content holds no list of predictions.

### `evalx/metrics.py`, `evalx/oracle.py`, `evalx/scorer.py`

```python
# metrics.py
def match_boxes(pred: Sequence[Box], truth: Sequence[Box], iou_threshold: float = 0.5) -> list[tuple[int, int]]
@dataclass(frozen=True)
class DetectionScores: tp: int; fp: int; fn: int   # properties recall, precision
def detection_scores(pred: Sequence[FramePeople], truth: Sequence[FramePeople], *, iou_threshold: float = 0.5,
                     max_height_px: float | None = None) -> DetectionScores
def foot_errors_m(pred, truth, to_pitch: PitchMap, *, iou_threshold: float = 0.5) -> np.ndarray
def position_errors_m(pred: Sequence[FrameState], truth: Sequence[FrameState], *, max_match_m: float = 10.0) -> np.ndarray
@dataclass(frozen=True)
class IdentityScores: coverage: float; switches: int; track_ratio: float; purity: float
def identity_scores(pred: Sequence[FrameState], truth: Sequence[FrameState], *, max_match_m: float = 5.0) -> IdentityScores
def number_exact_match(pred: Mapping[int, int | None], truth: Mapping[int, int | None]) -> float
def summarize(values: np.ndarray) -> dict[str, float]       # n, median, p90, mean
def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]   # mean, low, high

# oracle.py (label-only checks, no model call)
def shift_states(frames: Sequence[FrameState], dx_m: float, dy_m: float = 0.0) -> list[FrameState]
def keep_frames(frames: Sequence[FrameState], keep: Collection[int]) -> list[FrameState]
def resample_states(frames: Sequence[FrameState], keyframes: Sequence[int]) -> list[FrameState]
def team_side_by_rule(frames: Sequence[FrameState]) -> dict[int, TeamSide]   # player track id -> side from the team-side rule
@dataclass(frozen=True)
class TeamSideFacts: agrees: bool; decided_by: Literal["players", "goalkeepers"]; players_alone_agree: bool
                     player_mean_gap_m: float | None; goalkeeper_tracks: int
                     goalkeepers_in_own_goal_zone: int; goalkeepers_in_other_goal_zone: int
def team_side_facts(frames: Sequence[FrameState]) -> TeamSideFacts
def label_homography_residuals_m(labels: GsrLabels) -> np.ndarray

# scorer.py (runs the stand-alone evaluator in its own interpreter)
def trackeval_command(python_exe: Path, script: Path, gt_root: Path, pred_root: Path, *, split: str,
                      tracker: str, clips: Sequence[str]) -> list[str]
def parse_detailed_csv(path: Path) -> dict[str, dict[str, float]]   # clip or "COMBINED" -> {"HOTA", "DetA", "AssA"}
def run_trackeval(...same arguments..., output_dir: Path,
                  drop_env: Collection[str] = ()) -> dict[str, dict[str, float]]
```

- The matchers gate before the assignment. In `match_boxes` an IoU below `iou_threshold` counts as
  0 for the Hungarian method. In the position matcher of `position_errors_m` and `identity_scores`
  a pair further apart than `max_match_m` costs just above `max_match_m`. So a pair that does not
  count never pushes a pair that counts out of the pairing. The pairs that do not count are then
  removed from the result. `max_match_m` must be a finite number of 0 or more, else `ValueError`.
- `detection_scores` with `max_height_px`: only truth boxes up to that height count; a prediction
  matched to a taller truth box is ignored, not a false positive.
- `foot_errors_m` maps both matched foot points with the same `to_pitch`, so it measures box
  placement only.
- `identity_scores`: `coverage` = mean over truth tracks of the largest share of the track covered
  by one predicted id; `switches` = id changes along truth tracks; `track_ratio` = predicted
  tracks / truth tracks.
- `identity_scores` also gives `purity` (2026-11-22): the mean over the predicted ids that are
  matched in a frame of the largest share of their matched frames that one truth track covers.
  The three values of gate G4 do not see a predicted id that follows two people one after the
  other (a wrong join); `purity` does. No gate uses it.
- `t_interval` raises `ValueError` for an empty sequence, for a value that is not finite and for
  a confidence outside 0 to 1.
- `resample_states` keeps the truth at `keyframes` and interpolates linearly in pitch space
  between them with the true ids (the ceiling of the keyframe design). A track is filled only
  between two keyframes that both hold it.
- Team-side rule on the labels: `team_side_by_rule` and `team_side_facts` apply the rule of
  `tracking.assign_team_sides` to the labelled teams, with the same two limits
  (`tracking.CLEAR_TEAM_GAP_M`, `tracking.GOAL_ZONE_MIN_X_M`) and the same goal zone
  (`tracking.goal_side`). The players decide alone when both teams have players and their
  means are at least 3 m apart; else the goalkeepers with a labelled side who stand in a goal
  zone; else the players. An exact tie of the player means that no goalkeeper decides counts
  as wrong. The rule is judged on the player tracks.
  `TeamSideFacts` also says what decided, whether the players alone are right, the gap between
  the player means and how many goalkeepers stand in the goal zone of their own and of the other
  side. `team_side_facts(frames).agrees` says whether the rule gives every player track its
  labelled side; the function raises `ValueError` when no player has a labelled side. A test
  gives random clips to this rule and to `assign_team_sides` and compares the answers.
- Scorer flags: `--GT_FOLDER <gt_root>/SoccerNetGS-<split>`,
  `--TRACKERS_FOLDER <pred_root>/SoccerNetGS-<split>`, `--SKIP_SPLIT_FOL True`,
  `--TRACKER_SUB_FOLDER ""`, `--TRACKERS_TO_EVAL <tracker>`, `--SPLIT_TO_EVAL <split>`,
  `--SEQ_INFO <clip> ...`. With its default settings the evaluator reads the labels from
  `<GT_FOLDER>/<split>/<clip>/Labels-GameState.json` and gives only the prediction folder the
  prefix `SoccerNetGS-`. This package keeps the clips and the predictions of a split below
  `SoccerNetGS-<split>`, so the command names both split folders itself, and
  `--SKIP_SPLIT_FOL True` tells the evaluator to add none.
- The command names no output folder, so the evaluator writes its result files into the folder
  of the prediction files of the tracker. Every run writes three detailed CSV files there:
  `person_detailed.csv` with one row per clip and the row `COMBINED`, and
  `cls_comb_cls_av_detailed.csv` and `cls_comb_det_av_detailed.csv` with the row `COMBINED` only.
  `run_trackeval` moves these files into `output_dir` and reads the scores from
  `person_detailed.csv` alone. It raises `FileExistsError` before the run when the tracker folder
  already holds a detailed CSV file, `RuntimeError` when the evaluator stops with an error, and
  `FileNotFoundError` when the run wrote no `person_detailed.csv`.
- The evaluator process gets the environment of the command without the variables that
  `drop_env` names. The evaluator is the code of another project and needs no key, so the
  commands `oracle` and `eval` pass the name of the model key variable (`cli.KEY_VARIABLE`).
- The scorer is `sn-trackeval` in a separate environment (Python 3.12, numpy 1.26). The script
  that the wrapper starts belongs to a clone of the scorer repository, and it puts that clone
  first on its Python path. So the code that scores is the code of the clone (the tag `v0.4.1`,
  see the RUNBOOK), and not the package 0.4.0 of the environment, which only brings the
  dependencies. The files `oracle.json` and `scores.json` record the commit of that clone under
  `provenance.scorer_commit` (`cli.scorer_commit`; `null` when the script lies in no clone).
  The clone folder and the top folder that git names are compared as places on the disk, so a
  path that is typed in another letter case gives the commit too.
- The wrapper was never run with the real evaluator. The points above were read in the source
  of `sn-trackeval`; the first `oracle` run on the run machine confirms them (RUNBOOK, section 8).

`evalx/diagnostics.py` and `evalx/trackbench.py` (2026-11-22):

```python
# diagnostics.py: tracking values without labels; no gate uses them
BORDER_BOX_HEIGHTS = 1.0
@dataclass(frozen=True)
class TeamCounts: ids: int; max_alive: int; max_seen: int; keyframes_over_slots: int
@dataclass(frozen=True)
class TrackDiagnostics: tracks: int; births_inside: int; ends_inside: int; steps: int
                        impossible_steps: int; max_speed_mps: float; top_speed_mps: dict[str, float]
                        team_changes: int; tracks_with_team_change: int; teams: dict[str, TeamCounts]
                        alive_per_keyframe: tuple[tuple[int, int, int], ...]
    def as_dict(self) -> dict[str, object]
    def lines(self) -> list[str]
def track_diagnostics(result: ClipResult, *, max_speed_mps: float) -> TrackDiagnostics

# trackbench.py: synthetic bench of the track repairs (python -m pitchvlm.evalx.trackbench)
FAULTS = ("none", "swap", "team_noise", "occlusion", "mixed")
ARMS = {"current": ..., "swaps": ..., "slots": ..., "both": ...}   # TrackingConfig per arm
@dataclass(frozen=True)
class CaseResult: fault: str; seed: int; arm: str; scores: IdentityScores; tracks: int; swaps: int
def run_case(fault: str, seed: int, *, n_frames: int = 750) -> list[CaseResult]
def table_lines(results: Sequence[CaseResult]) -> list[str]
```

- `track_diagnostics` reads a `ClipResult` only. A birth (an end) inside is a track that starts
  after the first (ends before the last) observation of the clip with its box more than one box
  height from every image border. A step is the way between two `observed` states of one track
  (keyframes, not filled frames), in metres per second; steps above `max_speed_mps` (the
  tracking setting) are counted, and `top_speed_mps` summarises the top speed of each track
  (`summarize`). Team-key changes are counted with `tracking.team_key_runs` on the tracks of
  players and goalkeepers. Per team key `A` and `B`: the ids, the most tracks alive (between
  their first and last observation) and seen at one keyframe, and the keyframes with more alive
  than 11; `alive_per_keyframe` is the series. `max_speed_mps` must be a finite number above 0.
- `run` writes the values into `result.json` under `diagnostics` and prints `lines()`; `demo`
  writes them into `summary.json` under `diagnostics` and prints them.
- `run_case` runs every arm on one synthetic case: exact answers without per-frame numbers,
  true camera motion and pitch mappings, a confident number reading for the people of a random
  half of the scene, and one fault: `none`, `swap` (the tracks of two players of different teams
  at most one box height apart exchange their later observations, up to two per clip, after
  `associate`), `team_noise` (10 % of the answers of players and goalkeepers name the other
  team, 10 % no team), `occlusion` (4 players missing for 2.4 to 4.8 s each), `mixed` (all).
  The seed fixes the scene and the faults. `table_lines` gives the means per fault and arm and
  the paired differences against `current` with 95 % t intervals.

### `render.py`

```python
@dataclass(frozen=True)
class RenderStyle: ...   # colours per team side / role, trail seconds (4.0), radar scale, font sizes
def draw_overlay(image: np.ndarray, players: Sequence[PlayerState], style: RenderStyle = RenderStyle()) -> np.ndarray
class RadarRenderer:
    def __init__(self, fps: float, style: RenderStyle = RenderStyle())
    def draw(self, state: FrameState) -> np.ndarray          # keeps trails per track id
def compose(overlay: np.ndarray, radar: np.ndarray) -> np.ndarray   # video on top, radar below, same width
class VideoSink:                                              # context manager, OpenCV writer, fourcc mp4v
    def __init__(self, path: Path, fps: float, size: tuple[int, int])
    def write(self, image: np.ndarray) -> None
def render_clip(source: FrameSource, result: ClipResult, out_path: Path, style: RenderStyle = RenderStyle()) -> Path
```

Labels show `#<number_display> <name>` when present. Colours come from the team side; referees
and others are grey. A sink that cannot open raises `OSError`. `render_clip` writes the video as
`<name>.partial<ending>` in the same folder and renames it when it is complete, so a file with
the final name is always a complete video. A failed write removes the unfinished file and leaves
an older video as it was. The radar joins two remembered positions of a track only when they
come from two frames that were drawn one after the other. Where a track has no state in a drawn
frame, its trail has a gap.

### `pipeline.py`, `cli.py`, `doctor.py`, `fallbacks/` (integration layer)

```python
def run_clip(source: FrameSource, backend: Backend, config: PipelineConfig = PipelineConfig(), *,
             roster: Roster | None = None, motion: MotionField | None = None) -> ClipResult
def fit_problems(record: HomographyRecord, feet_xy: np.ndarray, image_size: tuple[int, int], *,
                 pitch_margin_m: float) -> list[str]
def crop_choice(boxes: Sequence[tuple[int, Box]], fps: float, max_crops: int) -> tuple[tuple[int, Box], ...]
```

CLI commands: `demo` (synthetic clip, fake backend, no network), `run`, `doctor`, `data`,
`oracle`, `probe`, `tool-job`, `tool-score`, `eval`, `sync`. Exit code 0 means done, 1 means
that a check or a gate failed, 2 means that the command could not run. Fallback adapters convert
the JSON output of external tools (RF-DETR boxes, PnLCalib mappings) into `FramePeople` and
`HomographyRecord`.

Stop signals: SIGINT (Ctrl-C), SIGTERM and SIGHUP stop a command with exit code 2. The handler
sets the event `cli.STOP_REQUESTED`, which the real backend gets as its stop event, so no
request and no new attempt is sent after the signal. A model call that runs in the main thread
(the kit legend request of `run`, every request of `probe` and of the contract checks of
`doctor`) is not cut off: the signal is held back until the call has ended and its answer is
stored and recorded (`cli.finish_in_flight`). A second signal during that call is held back too.

Output and errors: the standard output of every command is line buffered. So the cost estimate
and the budget line of a paid command can be read in a pipe or a file before the first model
call, and the line `error:` is the last line also when both streams go to one place. Wrong
content of a settings file, a roster file, a label file, an experiment file, a split list or a
run file of a tool ends with one line `error: the <kind> <path> cannot be used: <fault>` and no
traceback (`cli.wrong_file`). A clip folder that does not fit its label file, or whose labels
have no scored frame, ends with `error: the clip folder <path> cannot be used: <fault>`
(`cli.dataset_clip`, `cli.matched_clip`), and a group of the split list that names a clip twice
with one line too (`cli.group_clips`).
`cli._configure_logging` sets the loggers `httpx2`, `httpcore2`, `httpx` and `httpcore` to
WARNING (`cli.HTTP_LOGGERS`), so that no request line goes to the log; from version 3.0 the SDK
sends its requests with `httpx2`.

`pipeline.run_clip`:

- A person whose box has an IoU above 0.9 (`REPEATED_BOX_MIN_IOU`) with a person of the same
  people answer that is kept before it is dropped before the camera motion and the tracking
  step, and a warning names the number of repeated boxes. For two people with the same role and
  the same team key the limit is 0.6 (`REPEATED_BOX_SAME_LABELS_MIN_IOU`, an estimate until the
  people probe). The person that the answer lists first is kept.
- `fit_problems` is the plausibility check of a landmark fit, used by `run_clip` and by the
  pitch probe. Every fit must have the pitch the right way round (`check_homography` without
  foot points). Every fit may put 1 person (`MAX_PEOPLE_OUTSIDE`) of the players, goalkeepers
  and referees outside the pitch plus the margin. With more people outside, a fit with fewer
  than 7 agreeing landmarks (`STRONG_FIT_MIN_INLIERS`) must put 90 percent of them on the pitch
  (the rule of `check_homography`), and a fit with 7 or more must put two thirds of them on the
  pitch (`STRONG_FIT_MIN_SHARE_ON_PITCH`).
- A keyframe keeps its own landmark fit when 7 or more landmarks agree with it. A keyframe
  without a fit takes the mapping that the camera motion carries over from the nearest keyframe
  with 7 or more agreeing landmarks within `max_gap_frames`; its record then has the source
  `propagated`. A keyframe with a weaker fit takes that mapping only when more than half of its
  own usable landmarks are within 3 times `ransac_px` of it (`geometry.count_near`); else it
  keeps its own fit. Without such a keyframe in reach it keeps its own weaker fit. A warning is
  logged when no keyframe of a clip has a mapping.
- The tracking and the dense output get `motion.bridged_warp(field, homographies)`, so people
  are followed across a missing motion step through the pitch mappings of the two frames.
- `crop_choice` chooses the crops of a number request from the frames and boxes of one person.
  The tallest boxes are taken first, each at least `MIN_CROP_GAP_S` (1 s) away from the others,
  at most `max_crops`. When this leaves one box and the person has more sightings, the sighting
  that is furthest in time from it is taken too, so a track with two observations or more gets
  two crops. The pipeline and the number probe both choose their crops with it; `probes` has no
  own copy of the rule or of the constant.
- No shirt-number request is sent when no frame of the clip has a pitch mapping: no track gets
  a state then, so the answers would be paid for nothing.
- After an interruption (Ctrl-C or a stop signal) the thread pool of the model requests returns
  only when every request in flight has finished, so that their answers are stored and recorded.
  This holds also when further stop signals come during the wait; a warning then names the
  number of requests in flight.

Money rules of the command line (`cli.py`):

- A paid command prints its cost estimate before the first model call and the measured cost
  after. The estimate of `run` assumes 50 number requests (`cli.EXPECTED_NUMBER_REQUESTS`, a
  guess from the synthetic clip). The cost record of a run and of a replay holds `requests`
  (the real number of requests per step), `stored_answers` and `stored_usd`; in a replay every
  answer is a stored one.
- The limit of a run, 1.3 times its estimate, holds for the new answers of the run together
  with the stored answers that it uses, each at the cost that it had (`cli.CostMeter`,
  `cli.SessionBackend`). A paid run checks this limit before every model call. So a run that is
  started again after a budget stop pays for no further answer until the estimate is corrected.
- The contract checks of `doctor` are the one run with its own limit: the spend limit of gate G0
  (`g0_contract.spend_usd_max`) in the place of 1.3 times the estimate
  (`BudgetGuard.start_run(..., limit_usd=...)`, `cli.start_meter(..., spend_limit_usd=...)`).
  The limit holds for one doctor run with all its models together. The checks do not start
  when their estimate is above that limit, also with `--start-over-budget`. Their answers are
  not stored, so the rule above does not hold for them: a doctor that is started again pays
  for every answer again, up to its spend limit at each start. The hard stop holds for the
  doctor like for every run.
- A paid command refuses to start when the ledger total plus its estimate is above the hard
  stop, unless `--start-over-budget` is given. The hard stop itself always holds.
- The ledger of the cache folder of a command is the memory of the budget guard. A command
  that uses the default cache folder needs the work tree of the branch `results`
  (`../pitchvlm-results`); no command makes it.
- An answer is counted in the measured cost before its ledger line is written, so the printed
  cost also holds an answer whose line could not be written.
- `run` and `probe` have the option `--allow-other-images` (`CassetteStore(allow_other_images=True)`).
  A repeat index above 0 prints a line: the stored answers of other repeat indexes are not used.
- `run` and `demo` refuse a clip with fewer keyframes than `tracking.min_observations`. The clip
  id of a path is spelled as the disk stores it (`cli.disk_spelling`).

Output files of the command line:

- `run` writes `result.json` with `provenance` (commit, settings, cost record), `prediction_file`
  and `result`. With a label file, `prediction_file` holds the problems of the format checker,
  the number of scored frames and the number of scored frames without a prediction; without
  one it is `null`. The exit code is 1 for a problem of the format checker and for a clip
  without any result: no scored frame has a prediction, or, without a label file or without a
  scored frame, no frame holds a person (`cli.empty_result_reason`). The files are written in
  both cases, so the scorer reads an empty prediction file and counts the clip as missed.
- `demo` writes `summary.json` with the position error, the identity values (coverage, switches,
  track ratio), the verdict of the format checker and the same two frame counts. It follows the
  same rule for the exit code 1.
- `run` and `demo` also write and print the tracking values without labels
  (`evalx.diagnostics`): `result.json` and `summary.json` hold them under `diagnostics`
  (2026-11-22). The identity values of `summary.json` hold `purity` too.
- `oracle` writes `oracle.json`. Its part `team_side_rule` holds the share of clips where the
  rule of the pipeline is right (the value of gate G1), the share where the players alone are
  right, the clips that the goalkeepers decided, and the `TeamSideFacts` of every clip.
- `sync` lists the text result files (every ledger file is named as a cost ledger), checks them
  with `tools/guard.py --paths` and then with the secret scan of the git hooks
  (`gitleaks dir` with the file `.gitleaks.toml` of the results work tree), and prints the git
  commands. It runs none of them. Without `gitleaks` it says that the secret scan did not run.

An experiment file of `probe` (`cli.Experiment`) names the settings of its model step under
`step`. That section is laid over the defaults of the pipeline step that the probe measures
(`cli.default_step_settings`: people probe `people`, pitch probe `landmarks`, number probe
`number`). So a key that is left out keeps the default of that step: a number probe keeps the low
resolution.

`probes.py` measures one model step on labelled frames (`probe_people`, `probe_pitch`,
`probe_number`). The result of the pitch probe holds `n_frames_valid_without_share_rule` and
`valid_share_without_share_rule`, for all frames and by view: the count and the share of the
frames whose landmark fit is valid with `min_near_share` 0. The difference to `valid_share` is
what the share rule of the landmark fit refuses on real answers. The number probe chooses its
crops with `pipeline.crop_choice`, so it measures the crops that the pipeline sends.

An external tool is measured on the frames of a probe, with the values of that probe and
without a model call:

```python
def probe_frames(clips: Sequence[LabelledClip], *, frames_per_clip: int, seed: int) -> dict[str, list[int]]
def score_people_answers(clips, people_of: Callable[[str, int], FramePeople], tool: Mapping[str, Any], *,
                         frames_per_clip: int, seed: int, small_box_max_height_px: float) -> dict[str, Any]
def score_pitch_mappings(clips, mapping_of: Callable[[str, int], HomographyRecord], tool: Mapping[str, Any], *,
                         frames_per_clip: int, seed: int) -> dict[str, Any]
```

- `probe_frames` gives the frames that `probe_people` and `probe_pitch` sample for the same
  clips, frames per clip and seed. The two score functions ask `people_of(clip_id, frame)` or
  `mapping_of(clip_id, frame)` for exactly these frames. An answer that names another frame
  raises `ValueError`.
- `score_people_answers` returns the keys of `probe_people` without `prompts` and
  `n_answers_not_parsed`. Both functions score the answers with the same code
  (`probes._score_people`), so the same boxes give the same values. `counts.fp_off_pitch` of
  both results is the number of wrong boxes whose foot point the label mapping of the frame
  puts more than `PITCH_MARGIN_M` outside the pitch: the pipeline drops such people, and a
  detector cannot leave the spectators out as the people prompt tells the model. A frame
  without a label mapping adds nothing to this count.
- `score_pitch_mappings` returns the keys of `probe_pitch` that do not belong to a landmark fit.
  A frame is valid when the tool gave a matrix. The plausibility check is
  `pipeline.fit_problems`; a tool record has no count of agreeing landmarks, so the check wants
  9 of 10 people on the pitch; a landmark fit with 7 or more agreeing landmarks needs 2 of 3.
  The error values are counted on the valid frames of that engine, so two engines that refuse a
  different number of frames are not measured on the same frames there. `axis_check` holds the
  median position error of the valid frames for the mapped points as they are and with the sign
  of x, of y or of both changed (`probes.AXIS_SIGNS`). It tests the pitch convention of the tool
  output. The convention is right when `as_given` is the smallest value and small (some metres
  at most). A large `as_given` next to one small other value means that the script that ran the
  tool has a fault (then `plausible_share` is 0), or that the tool mirrors the pitch on more
  than half of the frames (then it is above 0). When all four values are large, the check
  decides nothing. The command `tool-score` prints the check and ends with the exit code 1 when
  another value is smaller than `as_given` (`cli.print_axis_check`).
- The settings block of a tool result holds the tool section of the experiment file under
  `tool`, and the seed and the measurement values of the probe. It has no `step_settings` and
  no `repeat`.

`fallbacks/runs.py` holds the two files that connect the package with the script
`tools/run_tool.py`, which runs a tool in the environment of the tool and does not import the
package:

- The job file (`ToolJob`, format `pitchvlm-tool-job-1`): the name of the experiment, the tool
  section, and one entry per frame with the clip, the frame index and the path of the image
  below the data folder. The command `tool-job` writes it from an experiment file
  (`cli.ToolExperiment`: `name`, `group`, `frames_per_clip`, `seed`, `tool`).
- The run file (`ToolRun`, format `pitchvlm-tool-run-1`): the name and the tool section of the
  job, the facts of the run (`runner`), and one entry per frame with `seconds` (the time of the
  tool for the frame, without the reading of the image file) and `payload` (the answer in the
  format of the adapter module of the tool). No frame may have two answers. The file is read
  strictly (`load_tool_run`): a key that comes twice is refused, a number must be written as a
  number and a truth value as true or false.
- The tool section is `RfdetrSettings` (`name`, `model`, `min_score`, `antialias`) or
  `PnlcalibSettings` (`name`, `kp_threshold`, `line_threshold`, `pnl_refine`); the key `name`
  says which one. `antialias` and `pnl_refine` take only true and false.
  `cli.TOOL_PROBES` names the probe of each tool: `rfdetr` people, `pnlcalib` pitch.
- The command `tool-score` refuses a run file whose name or tool section differs from the
  experiment file, and a run file without an answer for a frame of the probe
  (`cli.probe_answers`). So the settings of the experiment file are those that the tool ran
  with. Answers for other frames are counted (`provenance.n_answers_not_used`) and not
  measured. The metrics file holds the values of the probe, `seconds_per_frame` (the summary of
  the times of the measured frames) and, under `provenance`, the facts of the run (`runner`),
  the SHA-256 of the run file (`run_file_sha256`) and `script_is_current`: whether
  `runner.script_sha256` is the SHA-256 of the `tools/run_tool.py` of the work tree. The
  command refuses a work tree with changes like `probe`. Both tool commands refuse an `--out`
  file that is the experiment file or the split list, and `tool-score` also one that is the run
  file (`cli.refuse_input_as_output`).
- `fallbacks.rfdetr.people_from_detections` has no default score limit: the experiment file
  names it, and the script gives the same value to the tool as its threshold.

`tools/run_tool.py` (`rfdetr` and `pnlcalib` sub-commands) reads a job file and writes a run
file. It checks the version of `rfdetr` (`RFDETR_VERSIONS`) and the commit of the PnLCalib clone
(`PNLCALIB_COMMIT`) and refuses another one, and a clone with a changed tracked file, because it
was written from the source of these versions and never ran with the tools. It takes the
variable `GEMINI_API_KEY` out of its environment before it loads a tool, and it writes its own
SHA-256 into the facts of the run. Before the tool is built it checks the place of the run
file (`check_out_place`: no folder, write access, not the job file), so that the answers of a
long run are not lost at its end. An error of a tool on a frame is not caught: the script
names the frame and stops, so no frame counts as a frame without an answer because of an
error. For PnLCalib it binds the package names `model` and
`utils` to the two folders of the clone before it imports from them, loads the weight files as
plain weights, and takes the mapping of the grass from the camera of the tool
(`camera_homography`: the projection matrix without the column of the height). The world frame
of the tool is the pitch convention of this package, so no axis is changed. The script holds
the two format names and the keys of the tool sections again; a test compares them with the
package.

`doctor.py` holds the checks of the `doctor` command:

- Environment checks: Python, packages, the model SDK (it must be a version 3.x,
  `doctor.SDK_MAJOR_VERSION`), the key variable, the guard hooks, the deny-list and the free
  disk space. The check `guard hooks` asks git for the place of the hooks `pre-commit` and
  `pre-push` (`git rev-parse --git-path`). It needs an executable hook file that the pre-commit
  framework wrote for `.pre-commit-config.yaml`, and a hook list that names `tools/guard.py` and
  `gitleaks` for both stages. The check `deny-list` fails when the git folder of the clone has
  no file `guard-deny.txt`; it counts the entries and never shows one.
- Contract checks (gate G0), per model, on three synthetic frames of 1920 x 1080 pixels. The
  checks `boxes` and `landmarks` are contract verdicts with limits in `doctor.py`: at least half
  of the possible pairs of an answered box and a true box exist and their median IoU is at
  least 0.3; the median point error is at most 0.05 of the image side. The checks
  `box accuracy` and `landmark accuracy` have the status `info`: they report the mean IoU and
  the mean point error next to `g0_contract.box_mean_iou_min` and
  `g0_contract.point_error_max_fraction` of the gate file and are no verdict.
  `doctor.check_boxes` and `doctor.check_landmarks` return a list of checks.
- The check `flex tier` sends one people request through the flex tier and names the time of
  the answer. A model that is not in `cost.FLEX_TIER_MODELS` gets no flex request, and the
  check is a fact without a verdict. A request for which the flex tier had no capacity
  (`FlexBusyError`) is not counted. When no request found capacity, the check has the status
  `info` ("not checked"), gate G0 passes without a verdict on the flex tier, and the detail
  says to run the doctor again before the first run through the flex tier.
- The checks `store flag`, `token sum` and `served tier` read the raw calls of a recording SDK
  client. `served tier` fails when the answer of any call reports another tier than the call
  asked for. It reads the field `service_tier` of the answer body; the first real run must
  show that this field names the serving tier (RUNBOOK, section 8).
- The answers of the contract checks are not stored, so `cli.MeteredBackend` reports their
  cost. A doctor run that the budget guard stops still writes its report with
  `stopped_by_budget`. The report then holds the checks of the models that were finished and
  the cost record of every paid answer, and no check of the model that was in progress.
- The check `spend` compares the measured cost of one doctor run with
  `g0_contract.spend_usd_max`. The budget guard stops the run above the same value. So the
  check fails only together with `stopped_by_budget` and the exit code 2. Its value and its
  limit are the record of the cost in the report.

### `tools/guard.py` — repository guard

A script outside the package, with the Python standard library only (Python 3.9 or later). It
has no import interface: other code starts it as a process.

```
python3 tools/guard.py (--staged | --push | --tracked | --paths FILE ...) [--repo DIR]
```

- Exit code 0 means clean, 1 means violations, 2 means a usage error or a git error. Each
  violation is one line on the standard output: `<path>:<line or ->: <rule>: <message>`. A notice
  about a missing deny-list goes to the error stream.
- `--staged` (hook before a commit) reads the git index: every file that the next commit adds
  or changes, also a renamed file and a file that was a link. It checks the author and the
  committer of the next commit.
- `--push` (hook before a push) reads commits, not the working tree: every commit of a local
  branch, of a tag or of HEAD that the remote-tracking branches of the remote do not have. For
  each commit it checks the files that the commit adds or changes (for a merge commit: the files
  that differ from every parent), the message, the author and the committer. Git does not
  record which tags a remote has, so the mode reads every local tag: its name, what it points
  at and, for an annotated tag, the tagger and the message. It also reads the names of the
  local branches that the remote does not have, and the name that the hook runner gives for the
  first branch of the push on the remote (`PRE_COMMIT_REMOTE_BRANCH`).
- `--tracked` reads every tracked file from the working tree. `--paths` reads the given files;
  `cli sync` uses it for the result files.
- Rules: `ext`, `binary`, `size` (2 MB for every file; the package splits its cost ledger into
  files of at most 1 MB, section `vlm/cost.py`), `base64`, `pixels`, `dataset` (the label file
  and the result file of a clip run by name; label data, prediction data, the result of a clip
  run and records of the label format at any depth of a `.json` or `.jsonl` file), `secret`,
  `lock`, `ignored`, `host`, `home`, `deny`, `identity` (the clone needs a local `user.name` and
  `user.email`, and every author, committer and tagger must have both), `tag` (a tag must point
  at a commit). The rules `secret`, `host`, `home` and `deny` also read the path of every
  checked file and the names of branches and tags. The RUNBOOK, section 9, describes each rule
  and the deny-list `.git/guard-deny.txt`.
- The guard starts a fixed number of git processes, whatever the number of files, branches and
  tags is.
- A file of the repository must pass every rule. A test checks this for the files of the
  repository that holds the test. A value that only looks like a secret must be built at run
  time in a test, as `tests/test_guard.py` does.

## 5. Tests

- `pytest` runs offline and without the model SDK. No test may open a network connection.
- A test that needs an outside tool (`git`, `gitleaks`, `pre-commit`) is skipped on a machine
  without that tool. Every test of the guard in `tests/test_guard.py` builds a git repository,
  so these tests are skipped without `git`. The tests of the secret scan run the hooks of
  `.pre-commit-config.yaml` on result files, and `gitleaks` on the text of the repository: a
  value that only looks like a key needs the mark `gitleaks:allow` on its line. The tests of the
  guard run the hooks through the hook runner, as an installed hook does, for a push whose
  files are not in the working tree.
- Unit tests live in `tests/test_<module>.py`. Use small hand-made fixtures; integration tests use
  `SyntheticScene`.
- Test the sign conventions explicitly: a far-side point has negative y; model boxes are
  `[ymin, xmin, ymax, xmax]`.
- No external tool is installed where the tests run. `tests/test_run_tool.py` replaces each
  tool by a stand-in with the call signatures of the real one, and one test of the image reader
  is skipped without Pillow. So these tests say what the script calls, and not that the real
  tools answer as the script expects.
- No media files in the repository: build images in the test.
