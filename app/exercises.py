"""Exercise catalog and per-rep metric display specs.

The engine (`src/services/celery`) identifies exercises by the numeric ids it
inherited from trAIner (1 = bench, 6 = squat, 7 = RDL); the web layer addresses
them by slug. This module is the single place those two worlds meet, and it also
declares which metrics each exercise shows, in what order, and how a value maps
to an ok / warn / bad badge.

The list per exercise is deliberately short: it is exactly the set of metrics
that feed the relational half of the technique score (knee valgus, spine flexion,
hip/knee ROM ratio and the two coordination measures for the sagittal lifts; L/R
symmetry for bench), not every quantity the engine can produce. Descriptive
numbers like per-joint ROM and depth are computed and stored in `technique.json`,
they are simply not what the grid is for. The other half of the score is the
movement pattern — including the shin angle on a squat — which is the DTW
comparison, rendered as its own per-joint block.

Each metric's ok / warn / bad badge comes from the 0-100 sub-score the scorer
gives it (`scoring_service`), so the colour on screen and the number feeding the
rep's score can never disagree. A metric with no sub-score — no band for this
exercise, or a value the camera angle could not measure — renders plain.
"""

from dataclasses import dataclass, field

# Imported from the engine so a hint can never quote a threshold the scorer
# no longer uses.
from services.celery.bench_symmetry_service import ASYM_THRESH_DEG

OK, WARN, BAD, INFO, NA = "ok", "warn", "bad", "info", "na"

# Sub-score cutoffs for the badge colour, shared with the DTW scores in the UI.
GOOD_SCORE, FAIR_SCORE = 80.0, 60.0


def status_for(value, score: float | None) -> str:
    if value is None:
        return NA
    if score is None:
        return INFO
    return OK if score >= GOOD_SCORE else (WARN if score >= FAIR_SCORE else BAD)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str = "°"
    digits: int = 1
    hint: str = ""


@dataclass(frozen=True)
class Exercise:
    slug: str
    id: int
    name: str
    metrics: list[MetricSpec] = field(default_factory=list)



# The relational half of the technique score, shared by squat and RDL — the same
# set the scorer averaged into its metric sub-score.
_SCORED_SAGITTAL = [
    MetricSpec("valgus_index", "Knee cave", unit="", digits=2,
               hint="Ankle/knee width at the bottom; >1 means the knees track inside the feet. "
                    "Needs the camera ~30° off the sagittal plane — reported as n/a on a pure side view."),
    MetricSpec("spine_flex_delta_deg", "Spine flexion range",
               hint="How much the thoraco-lumbar angle closed within the rep. Higher = more rounding."),
    MetricSpec("hip_knee_rom_ratio", "Hip/knee ROM ratio", unit="", digits=2,
               hint="Higher = more hip-dominant. A low ratio on an RDL means you are squatting it."),
    MetricSpec("hip_knee_lag_pct", "Hip-knee lag", unit="%",
               hint="Signed lag between the hip and knee angular-velocity signals, as a "
                    "share of the rep: 0 means they turn around together, positive means "
                    "the hip leads (hips shooting up), negative means the knee leads."),
    MetricSpec("marp_deg", "Hip-knee MARP",
               hint="Mean absolute relative phase over the rep: 0° = the two joints stay "
                    "perfectly coupled through the whole movement."),
]

SQUAT = Exercise(slug="squat", id=6, name="Squat", metrics=list(_SCORED_SAGITTAL))

RDL = Exercise(slug="rdl", id=7, name="Romanian deadlift", metrics=list(_SCORED_SAGITTAL))

BENCH = Exercise(
    slug="bench", id=1, name="Bench press",
    # Bench has no reference band: the reference reps give it a movement pattern
    # to match, and its relational half is the L/R symmetry check, scored against
    # an absolute cutoff.
    metrics=[
        MetricSpec("elbow_avg_asymmetry_deg", "Elbow L/R asymmetry",
                   hint=f"Mean left-right elbow-angle difference across the rep; flagged above {ASYM_THRESH_DEG:.0f}°."),
        MetricSpec("shoulder_avg_asymmetry_deg", "Shoulder L/R asymmetry",
                   hint=f"Mean left-right shoulder-angle difference across the rep; flagged above {ASYM_THRESH_DEG:.0f}°."),
    ],
)

EXERCISES: dict[str, Exercise] = {e.slug: e for e in (BENCH, SQUAT, RDL)}
BY_ID: dict[int, Exercise] = {e.id: e for e in EXERCISES.values()}

# Human labels for the angle channels the DTW comparison reports per joint.
JOINT_LABELS: dict[str, str] = {
    "knee_angle": "Knee",
    "hip_angle": "Hip",
    "tibia_angle": "Shin",
    "lumbar_angle": "Lumbar",
    "elbow_angle": "Elbow",
    "shoulder_angle": "Shoulder",
}


def get(slug: str) -> Exercise:
    try:
        return EXERCISES[slug]
    except KeyError:
        raise ValueError(f"unknown exercise {slug!r}; available: {sorted(EXERCISES)}") from None
