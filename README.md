# gym-technique-analyzer

Lifting technique from a phone video: upload a set of squats, RDLs or bench
press, get every repetition broken down into the numbers that describe it —
depth, range of motion, tempo, spinal flexion, knee cave, left/right symmetry —
and, if you ask for it, a DTW comparison of each rep against reference lifters.

> [!NOTE]
> The default mode needs no reference lifter at all: knee cave, spine flexion,
> hip/knee balance and hip-knee coordination are measured from your own video,
> against metrics chosen with a strength coach. That half stands on its own, and
> it is the half that tells you whether *this* set was better than your last one.
>
> The optional comparison mode is the one to read carefully: it measures
> **similarity to a specific, small set of reference repetitions** (24 reps from
> two lifters, see [Reference set](#reference-set)). A high score means "this rep
> moved like those reps did"; a low one means it didn't, which may be body
> proportions or tempo rather than a fault. It gets better the more lifters go
> into the reference pool — trAIner's `src/scripts/extract_reference_keypoints.py`
> takes a video and writes one reference JSON per repetition, ready to drop into
> `references/`.

## Index

- [Motivation](#motivation)
- [Quick start](#quick-start)
- [The two modes](#the-two-modes)
- [How it works](#how-it-works)
- [What you get per rep](#what-you-get-per-rep)
- [Reference set](#reference-set)
- [Design choices](#design-choices)
- [Architecture](#architecture)
- [HTTP API](#http-api)
- [Tests](#tests)
- [Future improvements](#future-improvements)
- [License](#license)

## Motivation

Filming your sets is free and everybody already does it; getting anything out of
the footage is the hard part. Watching your own squat back tells you very little
beyond "looked fine" — you cannot eyeball how much your spine rounded, whether
rep 6 was 200 ms slower out of the hole than rep 1, or whether your left elbow
is consistently lagging your right by 20°. A pose model can measure all of that
frame by frame, so the video you were filming anyway becomes a set of numbers
you can track across sessions.

The analysis engine here started life inside **trAIner** (a full training app);
this repository is the standalone version of it — the CV pipeline plus a small
FastAPI/HTMX frontend built to the same shape as **vbt-tracker**, so all three
projects run the same way.

## Quick start

This app brings no database or broker of its own — it plugs into the ones
already running on the host:

| Needs | Uses | Reached over |
|---|---|---|
| Postgres | trAIner's `trainer-db`, in a **`technique` schema of its own** | network `trainer_default` |
| Redis (broker, result backend, SSE pub/sub) | the shared `redis`, on databases **3 / 4 / 5** | network `infra_network` |

Nothing outside the `technique` schema is read or written, and the Redis
databases are picked to stay clear of the ones trAIner uses. Both networks are
declared `external`, so their own projects have to be up first.

```bash
git clone <this-repo> gym-technique-analyzer
cd gym-technique-analyzer
cp .env.example .env               # then set POSTGRES_PASSWORD (the same one trAIner uses)

docker compose up --build
```

The `technique` schema and its `analyses` table are created on first start.
To point somewhere else entirely — a standalone Postgres, another Redis —
override `POSTGRES_HOST` / `POSTGRES_DB` / the three Redis URLs in `.env`; see
`.env.example`.

Then open <http://localhost:8000>, pick an exercise, upload a video, and watch
the pipeline stages tick past live while it processes.

The pose model (SynthPose, a ViT-Huge) is downloaded on the first analysis into
`data/hf-cache/`, so the first run is slower than the rest. By default the
worker auto-detects its device and will fall back to **CPU**, where pose
estimation takes minutes per clip; if you have an NVIDIA GPU with the
`nvidia-container-toolkit` installed, uncomment the GPU block in
`docker-compose.yml` (the torch wheels bring their own CUDA runtime, so no CUDA
base image is needed).

**Filming**: one lifter in frame, whole body visible, camera static.

- **Squat / RDL** — **30° off the sagittal plane**, front or back. Not side-on:
  that angle is what makes both knees visible (so knee cave can be measured at
  all) and keeps the plates from hiding the back.
- **Bench press** — **45° between the frontal and transverse planes**, i.e.
  looking down at the lifter from in front and above, where the bar path and
  both arms are unobstructed.

The angles are not incidental; see [Design choices](#design-choices) for why
they are what they are.

## The two modes

Every rep gets a 0–100 score, made of two independent halves:

```
score = 0.5 · movement pattern (DTW)  +  0.5 · relational metrics
```

The checkbox next to the upload form decides whether the first half runs.

**Technique only** (default) — the relational half alone. Knee valgus, spine
flexion range, hip/knee ROM ratio and hip-knee coordination for the sagittal
lifts; left/right symmetry for bench. Each is scored 0–100 and averaged. Two of
them are judged against an absolute cutoff that needs no reference at all
(valgus, bench asymmetry); the others against the pooled range of the reference
reps (see [Reference set](#reference-set)). No DTW is involved, which is the
whole point of the split — an analysis you run without the comparison still
comes back with a score.

**Technique + DTW comparison** — adds the movement-pattern half: each rep is
dynamic-time-warped against every reference rep on file for that exercise, and
you get the closest match, its score, and a per-joint breakdown of where the
divergence is. DTW absorbs tempo differences, so a slow rep and a fast rep with
the same shape score the same; and matching against *individual* reps rather
than an averaged curve conditions on body type by similarity, since the mean of
differently-proportioned lifters is a trajectory none of them actually performs.

When only one half is available the score is that half, rather than a combined
number that quietly means something different from rep to rep. The UI always
says which halves went into it.

The split exists because the two halves answer genuinely different questions.
"Is my back rounding more than it should?" is a measurement against a range.
"Does my hinge look like theirs?" is a shape comparison, and inherits every
limitation of the reference pool.

## How it works

1. **Preprocess** — resample to 15 FPS and cap the long side at 1920 px
   (`video_services.py`). Consecutive frames of a barbell lift are highly
   redundant; 15 FPS still resolves a 0.6 s concentric into ~9 samples.
2. **Pose** — YOLO11m detects people, an IoU tracker keeps the principal one
   across frames, and **SynthPose** (ViTPose-Huge, 52 keypoints) estimates the
   skeleton (`pose_inference_service.py`). SynthPose over stock COCO-17 for the
   spine: it predicts anatomical markers (C7, T6, T11, L2, ASIS) that COCO
   simply doesn't have, and the spine metrics below are built on them. Outputs
   an annotated `pose.mp4` alongside the keypoint JSON.
3. **Normalize** — per-exercise origin and scale (`normalization_service.py`):
   hip-centred with femur/torso-length scaling for the sagittal lifts,
   shoulder-centred with shoulder-width scaling for bench. Both use a
   *median over the clip* rather than per-frame values, so a single bad frame
   can't rescale the skeleton. This is what makes the metrics comparable
   between people of different sizes and between clips shot at different
   distances.
4. **Count reps** — a 4-state hysteresis machine per exercise on the primary
   joint angle, cross-confirmed by a second signal (`rep_counting_service.py`):
   knee + hip for the squat, hip + torso inclination for the RDL, elbow
   (both arms) for bench. Hysteresis rather than peak-picking because a lifter
   pausing mid-rep must not read as two reps. Each rep carries its own
   eccentric/concentric split, taken at the frame the primary angle bottoms out.
5. **Measure** — per-rep metrics from the normalized keypoints
   (`technique_service.py`, `relational_metrics.py`,
   `coordination_service.py`, `bench_symmetry_service.py`). All of them are
   *relations between joints* (ratios, ranges, phase) rather than absolute
   positions, so they survive differences in body proportions and ROM.
6. **Compare** (optional) — DTW per joint angle against each reference rep
   (`dtw_service.py`). Distance is `dtw-python`'s normalized distance in
   degrees; a score of `100·(1 − distance/15°)` per angle, floored at 0 and
   averaged across the exercise's angles, gives the per-reference score, and the
   rep keeps its best match. The 15° denominator is the "totally different
   movement" anchor: reps averaging 15° of warped deviation per joint score 0.

7. **Score** — each metric becomes a 0–100 sub-score (against the reference band
   or an absolute cutoff), those are averaged into the relational half, and the
   DTW best match is the pattern half (`scoring_service.py`). This one runs when
   the report is requested rather than at analysis time, so rebuilding a band or
   changing a weight re-scores every past analysis without touching a video.

Stages 1–6 run sequentially inside one Celery task, publishing each transition
to Redis so the browser can show them live; the annotated video is transcoded to
H.264 on the way out, because OpenCV writes MPEG-4 Part 2 and no browser will
decode it.

## What you get per rep

The panel opens with the set's headline score and how it was composed, then one
card per rep: its own score, tempo chips (total / eccentric / concentric), the
metric grid, any cues raised, and — in comparison mode — the closest reference
with its per-joint bars.

The grid is deliberately short. It is exactly the set that feeds the relational
half of the score, rather than everything the engine can measure — and each badge
is the colour of that metric's own sub-score, so what you see and what counts
cannot disagree:

| Squat & RDL | Bench press |
|---|---|
| Knee cave (verdict) | Elbow L/R asymmetry (verdict) |
| Spine flexion range | Shoulder L/R asymmetry (verdict) |
| Hip/knee ROM ratio | |
| Hip-knee out of phase | |
| Hip-knee MARP | |

The movement pattern is the other half of that judgement, and it is the DTW
comparison: its per-joint bars are where the shin angle on a squat, or the
shoulder angle on a bench, actually gets assessed.

Descriptive quantities — per-joint ROM, joint angles at the bottom, peak torso
lean, whether the counter confirmed a full bottom position — are still computed
and written to `technique.json` for every rep, so they are one line in
`app/exercises.py` away if you want them on screen. They are just not a verdict
about anything on their own.

A few of the metrics are worth spelling out:

- **Spine flexion range** — the (mid-hip → L2 → C7) angle is ~180° with a
  neutral spine and closes as the back rounds. Reported as the *range within the
  rep*, not the absolute angle, so it measures how much you rounded rather than
  each person's individual neutral posture.
- **Hip/knee ROM ratio** — hip range divided by knee range. A high ratio is a
  hip-dominant hinge; a low one on an RDL means you are squatting it.
- **Hip-knee MARP** (mean absolute relative phase, via a Hilbert transform) and
  **out-of-phase %** — whether hip and knee move together or one leads. 0° MARP
  is perfect synchrony.
- **Knee cave** — ankle width divided by knee width over the deepest quarter of
  the rep. Differential (both knees vs both ankles) because that cancels the
  common-mode perspective offset of an angled camera, and evaluated at the
  bottom because valgus appears under load at depth. It is reported as **n/a on
  a side-on camera** — see below.

## Reference set

`references/` holds 26 individual reference repetitions as normalized per-rep
keypoint JSONs: 9 bench press, 12 squat (two lifters), 5 RDL (one), all filmed
to the protocol. Which metrics are worth measuring at all, and which executions
are good enough to serve as references, were decided in consultation with a
professional strength coach rather than picked for being easy to compute. The DTW comparison scores each of your
reps against every file in `references/<exercise_id>/` (1 = bench, 6 = squat,
7 = RDL).

### Adding your own references

`scripts/extract_reference_keypoints.py` turns a video of a good set into the
per-rep files this directory holds. It runs the clip through the *same*
preprocessing, pose estimation, normalization and rep counting a user submission
goes through — references measured any other way are not comparable to the
submissions they will be scored against.

```bash
# 1. put the clip somewhere the container can see (./data is bind-mounted)
cp squat_alex.mp4 data/reference-videos/

# 2. one JSON per rep, plus an annotated video to review
docker compose exec worker python /srv/scripts/extract_reference_keypoints.py \
    /srv/data/reference-videos/squat_alex.mp4 6 \
    --out /srv/data/reference-out/6 --name alex

# 3. watch reference-out/6/alex_pose.mp4, delete any rep you would not want a
#    stranger measured against, then keep the rest
mv data/reference-out/6/alex_rep_*.json references/6/

# 4. rebuild the band — NOT optional, see below
docker compose exec worker python -m services.celery.reference_band_service 6 /srv/references
```

Before any of that, check the framing: **squat/RDL at 30° off the sagittal
plane, bench press at 45° between the frontal and transverse planes**, exactly as
a submission would be filmed. A reference shot at the wrong angle does not
announce itself — it just quietly makes everyone scored against it look worse.
Step 4 is not optional either: the band is a set of numbers derived from these
exact reps, so leaving a stale one in place scores every future submission
against a range that no longer describes its own references.

Three things to keep in mind about it:

- **It is small and it is two people** — and RDL is one person, five reps. Since
  best-of-references matching works by finding the reference rep most like you,
  a pool that small mostly cannot contain anyone built like you. Everything in
  [the note at the top](#gym-technique-analyzer) follows from this.
- **The framing is checked, and it matters.** Every rep here measures 0.36–0.61
  on the ankle-separation test (≈21–38° off sagittal). An earlier set, shot
  side-on, measured 0.005–0.067 — and with it the knee-cave metric returned
  nothing at all for every single reference rep, silently, because the geometry
  it needs was not in the picture. If you add references, check this before
  trusting them.
- **A clip that is itself in the reference set scores 100.** Reps match
  themselves at zero distance. Useful as a sanity check that the pipeline is
  wired up, misleading as a result.

`references/bands/<exercise_id>.json` holds the pooled acceptable range of each
relational metric, computed from those same reps. It is generated, not authored
— rebuild it whenever you add reference reps:

```bash
docker compose exec worker python -m services.celery.reference_band_service 6 /srv/references
```

The band has to be *built*, never copied in from elsewhere: it is a set of
numbers produced by one particular implementation of the metrics, and it means
nothing apart from it. Change how a metric is computed and the band must be
regenerated with it, or every rep gets scored against a range that no longer
describes the same quantity.

## Design choices

- **One camera, at a deliberately intermediate angle.** The complete answer is
  3D: lift the 2D keypoints into a skeleton and every metric becomes measurable
  in every plane at once. That was tried and abandoned — the 2D→3D lift models
  did not preserve limb length through the movement, with the quadriceps
  stretching and shrinking by as much as **70%** between phases of the same rep.
  Metrics built on segment lengths and joint angles cannot survive that. The
  other complete answer is several cameras, one per plane, which is a real
  imposition on someone filming themselves between sets. So the design settles
  on a single camera at an angle chosen to see as much as possible of both
  planes: **30° off sagittal** for squat and RDL, **45° between the frontal and
  transverse planes** for bench press.
- **The filming angle is what makes knee cave measurable.** Knee cave is read
  from how far apart the two knees track relative to the two ankles, evaluated
  over the deepest quarter of the rep — a differential, because that cancels the
  common-mode perspective offset of an angled camera, and at the bottom, because
  valgus appears under load at depth. All of that needs both knees visible,
  which is exactly what the 30° angle buys and a pure side view does not. The
  metric therefore checks the geometry before trusting itself: it measures how
  far the ankles actually project apart relative to the shank length in the same
  frame, and returns nothing below a 0.15 cutoff. Footage shot to the protocol
  measures around 0.58; footage shot dead side-on sits at 0.005–0.07, and before
  the check existed one such clip reported a *capped, severe* knee cave built
  entirely out of about a pixel of keypoint jitter, because dividing two
  near-zero widths amplifies noise without bound.
- **Per-side angles, since both sides are visible.** Knee and hip angles are
  computed separately for left and right (`angles2d_service.py`) rather than
  taken from one dominant side and mirrored, which is what the angled camera
  makes possible — and what lets bench press report L/R symmetry as a real
  measurement.
- **The visible leg is picked by how long it projects, not by confidence.** The
  DTW comparison reads one leg, and it has to be the leg the camera can see. The
  near leg is closer to the lens, so perspective renders it larger while the far
  one is smaller and partly hidden behind it: the median projected **shank**
  length separates the two by 12–20% on real footage. The shank specifically,
  because it holds its orientation through the whole movement in all three lifts,
  so its projected length reflects distance to the camera and little else — the
  femur does not (in an RDL the thighs stay near vertical and the two femurs came
  out 0.6% apart, a coin flip), and the trunk is worse still at 0.6–6.4%, sitting
  near the midline where perspective barely separates the sides.

  Keypoint *confidence*, the obvious first choice, is not a proxy for visibility:
  on one squat the model scored the occluded leg 10.6% higher than the visible
  one. Worse, confidence was averaged over whatever frames it was handed, so the
  same clip answered "left" over the whole video and "right" over five of its six
  individual reps — meaning a submission was compared against reference reps cut
  from *the same footage* leg-against-leg. Matching sides matters more than which
  side wins: cross-matched, a rep scored 68 against itself; on the same leg, 100.
- **Landmark choice is a fallback chain, not a contest.** SynthPose returns 52
  keypoints: the first 17 are COCO, the remaining 35 anatomical markers. `L_Hip`
  (COCO, id 11) is the hip joint centre and `l_ASIS` (id 29) the iliac spine,
  several centimetres forward and above it — different points, not competing
  estimates of one. Preferring whichever scored higher on a given frame swapped
  between them mid-rep and put 47° steps into the hip-angle series. The rule is
  now COCO first, marker only when the COCO point is occluded. This stayed
  invisible for as long as clips were shot side-on, where the two landmarks
  project almost on top of each other; the 30° angle separates them laterally and
  the swap became large.
- **Knee cave is judged per exercise.** The neutral ankle/knee width ratio is not
  the same lift to lift: a squat drives the knees out, so knees wider than the
  feet is the expectation and parity is already a fault (flagged above 0.90); an
  RDL is a narrow stance with the knees soft and over the feet, which lands
  around 1.0–1.15 as plain geometry (flagged above 1.20). One shared cutoff told
  all five reference RDL reps their knees were caving in. The RDL figure rests on
  those five reps from one athlete, so it is provisional.
- **Metrics that are relations, not absolutes.** Every scored metric is a ratio,
  a range or a phase relationship between joints. That is what makes a band
  pooled across differently-proportioned lifters meaningful at all: a tall
  lifter's absolute knee angle at depth says little, the relationship between
  their hip and knee travel says a lot.
- **Best-of-references, not a mean reference.** See
  [The two modes](#the-two-modes).
- **Joint angles rather than keypoint coordinates for DTW.** Angles are
  invariant to mirroring, camera distance and limb length; raw coordinates are
  not, and would score a tall lifter as "different" for being tall.
- **Reps are counted once, and everything reuses those windows.** The rep
  counter's boundaries drive the metrics, the DTW segmentation and the numbering
  in the UI, so rep 3 means the same thing everywhere.

## Architecture

FastAPI + HTMX frontend, a Celery worker for the CV pipeline, Postgres for the
analysis index (schema `technique`, table `analyses`, inside trAIner's
database), and per-analysis
artifacts on disk under `data/analyses/<id>/`. Pipeline progress is pushed to
the browser over Server-Sent Events, sourced from a Redis pub/sub channel per
analysis — no polling on either the worker or the browser side.

```
video upload -> FastAPI (POST /analyses) -> Celery task
                                                 |
             preprocess -> pose -> normalize -> count reps -> measure -> [compare]
                                                 |
                                                 v
                                      Redis pub/sub (stage) --> SSE --> browser
                                                 |
                                                 v
                    Postgres (status) + JSON artifacts / pose.mp4 on disk
```

```
app/                     # web layer
├── main.py              # FastAPI app, index page
├── routers/             # analyses (HTML partials + JSON API), events (SSE)
├── tasks.py             # the Celery task: the pipeline, stage by stage
├── exercises.py         # exercise catalog + which metrics each one shows
├── report.py            # engine JSON -> the shape the frontend renders
├── models.py            # SQLAlchemy: technique.analyses
├── pubsub.py            # Redis pub/sub behind the SSE stream
└── templates/, static/  # Jinja + htmx, per-rep cards rendered by static/js/reps.js
src/services/celery/     # analysis engine (pure functions, no web/DB imports)
├── video_services.py            # resample + resize
├── pose_inference_service.py    # YOLO11m + SynthPose + tracking
├── normalization_service.py     # per-exercise origin/scale
├── rep_counting_service.py      # 4-state hysteresis counters
├── angle_extraction_service.py  # joint angles from keypoints
├── angles2d_service.py          # per-frame angle channels
├── technique_service.py         # reference-free per-rep metrics
├── scoring_service.py           # 0.5 pattern + 0.5 relational metrics
├── reference_band_service.py    # pooled metric range from the reference reps
├── relational_metrics.py        # valgus, spine flexion, hip/knee ratio
├── coordination_service.py      # hip-knee phase, MARP
├── bench_symmetry_service.py    # bench L/R symmetry
├── feedback_service.py          # bilingual cue text
└── dtw_service.py               # DTW vs the reference reps
references/              # reference repetitions, per exercise id
```

The engine has no FastAPI, SQLAlchemy or Celery imports anywhere in it — every
service takes data or paths in and hands data or paths back, which is what makes
it runnable from a script or a notebook as easily as from the worker.

Neither Postgres nor Redis is bundled: `docker-compose.yml` is the only compose
file, and it attaches the two services to the existing `trainer_default` and
`infra_network` networks (see [Quick start](#quick-start)).

## HTTP API

The browser talks to the same endpoints you can script against. The HTML ones
return htmx partials; the JSON ones are typed with Pydantic and documented at
`/docs`.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/exercises` | the exercise catalog (slug, engine id, name) |
| `POST` | `/analyses` | creates an analysis and queues it (multipart: `video`, `name`, `exercise`, `weight`, `compare`) |
| `GET` | `/api/analyses` | every analysis with its status |
| `GET` | `/analyses/{id}/results` | the full per-rep report (409 until it's done) |
| `GET` | `/analyses/{id}/events` | SSE stream of pipeline stages |
| `GET` | `/analyses/{id}/video` | the annotated `pose.mp4` |

```bash
curl -F video=@squat.mp4 -F name="Squat 5x5" -F exercise=squat \
     -F weight=100 -F compare=true http://localhost:8000/analyses
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the parts with real logic in them — the per-rep measurements
(including that knee cave is reported on a frontal view and withheld on a
sagittal one) and the engine-JSON→UI mapping. It runs on synthetic skeletons, so
it needs neither a GPU nor the pose model.

## Future improvements

- **A reference set worth the name.** The comparison mode is only as good as
  what it compares against, and right now that is 26 reps from two people, with
  RDL resting on a single athlete. More lifters beats more reps per lifter:
  best-of-references matching conditions on body type by similarity, so its
  whole value comes from having *someone* built like you in the pool.
- **Bar path.** Nothing here tracks the bar itself, only the body. Bar drift
  forward out of the hole, or an uneven path on a bench press, are faults a
  lifter can act on immediately, and the 45° bench angle already sees them.
- **Per-rep angle traces in the UI.** The angle series are computed and thrown
  away after the metrics are derived; plotting knee/hip angle against time per
  rep, with the eccentric/concentric split shaded, would show *where* in the rep
  a fault happens rather than just that it did.

## License

MIT — see [LICENSE](LICENSE).
