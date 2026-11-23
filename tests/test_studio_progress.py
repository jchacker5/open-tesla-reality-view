import json

import pytest

from tsplat.studio import progress
from tsplat.studio.progress import Tracker


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize(
    ("line", "hit"),
    [
        ("I20261122 10:01:02.123 4242 feature_extraction.cc:301] Processed file [12/240]", ("features", 12, 240)),
        ("Processed file [240/240]", ("features", 240, 240)),
        ("I20261122 10:05:00.000 1 pairing.cc:88] Processing block [3/18]", ("match", 3, 18)),
        (
            "I20261122 10:09:00.000 1 incremental_pipeline.cc:1] Registering image #57 (num_reg_frames=19)",
            ("map", 19, 0),
        ),
        ("Undistorting image [7/240]", ("undistort", 7, 240)),
        ("1234/15000 Steps (12.5/s, 18m 22s remaining)", ("train", 1234, 15000)),
        ("  ⠙ [00:01:39] ███░░░ 2000/30000 Steps (20.1/s, 23m 13s remaining)", ("train", 2000, 30000)),
        ("$ colmap mapper ...", None),
        ("Processed file", None),
        ("pairs: 1234", None),
    ],
)
def test_match_progress(line, hit):
    got = progress.match_progress(line)
    assert (got[:3] if got else None) == hit


def test_brush_rate_and_remaining():
    _, cur, total, extra = progress.match_progress("500/1000 Steps (25.0/s, 20s remaining)")
    assert extra == {"rate": 25.0, "remaining": 20.0}


@pytest.mark.parametrize(
    ("text", "secs"),
    [("18m 22s", 1102), ("1h 2m", 3720), ("12s", 12), ("3.5s", 3.5), ("00:18:40", 1120), ("18:40", 1120), ("?", None)],
)
def test_parse_duration(text, secs):
    assert progress.parse_duration(text) == secs


def test_ansi_and_carriage_returns():
    raw = (
        "\x1b[2K\r\x1b[32m100/200 Steps (10.0/s, 10s remaining)\x1b[0m"
        "\r\x1b[2K200/200 Steps (10.0/s, 0s remaining)\npartial"
    )
    lines, rest = progress.split_lines(raw)
    assert rest == "partial"
    clean = [progress.strip_ansi(x) for x in lines if progress.strip_ansi(x).strip()]
    assert clean == ["100/200 Steps (10.0/s, 10s remaining)", "200/200 Steps (10.0/s, 0s remaining)"]


@pytest.mark.parametrize(
    ("line", "stage"),
    [
        ("[drive1] 4 cams, 10.0s +20.0s @ 3.0 fps", "frames"),
        ("ego mask: prebuilt for front", "masks"),
        ("$ colmap feature_extractor ...", "features"),
        ("$ colmap matches_importer ...", "match"),
        ("$ colmap exhaustive_matcher ...", "match"),
        ("$ colmap mapper ...", "map"),
        ("$ colmap global_mapper ...", "map"),
        ("$ colmap image_undistorter ...", "undistort"),
        ("$ /Users/me/.local/opt/brush/brush_app /work/x/dense ...", "train"),
        ("splat -> work/x/viewer/scene.ply (80 MB)", "export"),
        ("$ colmap rig_configurator ...", None),
        ("$ colmap model_converter ...", None),
    ],
)
def test_stage_markers(line, stage):
    assert progress.match_stage(line) == stage


def test_tracker_runs_through_a_pipeline_log():
    clk = Clock()
    t = Tracker(n_images=240, map_total=60, clock=clk)
    log = [
        "[drive1] 4 cams, 0s +20s @ 3 fps",
        "ego mask: guessed",
        "$ colmap feature_extractor ...",
        "Processed file [120/240]",
        "$ colmap matches_importer ...",
        "Processing block [1/2]",
        "$ colmap mapper ...",
        "Registering image #1 (num_reg_frames=30)",
        "Registering image #2 (num_reg_frames=12)",  # a second model: stays monotonic
        "$ colmap image_undistorter ...",
        "Undistorting image [240/240]",
        "Undistorting image [60/240]",  # second pass (masks)
        "$ /x/brush_app dense",
        "7500/15000 Steps (10.0/s, 12m 30s remaining)",
        "splat -> scene.ply",
    ]
    events = []
    for line in log:
        clk.t += 10
        events += t.feed(line)
    starts = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "start"]
    assert starts == list(progress.STAGES)
    done = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "done"]
    assert done == list(progress.STAGES[:-1])
    prog = {e["stage"]: e for e in events if e["type"] == "progress"}
    assert prog["features"]["frac"] == 0.5
    assert prog["map"]["cur"] == 30 and prog["map"]["frac"] == 0.5  # 30 of 60 rig frames, not regressed to 12
    assert prog["undistort"]["cur"] == 300 and prog["undistort"]["total"] == 480  # pass 2 of 2
    assert prog["train"]["stage_eta"] == 750.0  # Brush's own estimate wins
    assert t.seconds["frames"] == 10.0


def test_tracker_never_goes_backwards():
    t = Tracker(n_images=10)
    t.feed("$ colmap mapper ...")
    assert t.feed("$ colmap feature_extractor ...") == []
    assert t.stage == "map"
    assert t.feed("Processed file [1/10]") == []  # progress for an earlier stage is ignored


def test_tracker_mapper_without_rig_counts_images():
    t = Tracker(n_images=40, map_total=10)
    t.feed("$ colmap mapper ...")
    ev = t.feed("Registering image #3 (num_reg_frames=25)")
    assert ev[-1]["total"] == 40 and ev[-1]["cur"] == 25


def test_finish_and_resume():
    t = Tracker(n_images=8)
    ev = t.finish("masks")  # prep exited 0 without printing markers
    assert [(e["stage"], e["status"]) for e in ev] == [("frames", "done"), ("masks", "start"), ("masks", "done")]
    assert t.finish("masks") == []
    r = Tracker(n_images=8, start_stage="train")
    ev = r.feed("$ /x/brush_app dense")
    assert [(e["stage"], e["status"]) for e in ev] == [("train", "start")]  # earlier stages already done


def test_eta_total_uses_stage_eta_and_priors():
    clk = Clock()
    t = Tracker(n_images=100, clock=clk)
    rates = {s: 1.0 for s in progress.STAGES}
    t.begin("train")
    t.progress("train", 50, 100, rate=10.0)  # 5 s left in train
    assert progress.eta_total(t, rates, 100, 100) == 5.0 + 1.0  # + export (1 unit)
    t2 = Tracker(n_images=100, clock=clk)
    t2.begin("features")  # no progress yet: prior (100 s) minus elapsed
    clk.t += 30
    assert progress.eta_total(t2, rates, 100, 100) == 70 + 100 * 3 + 100 + 1


def test_history_rates(tmp_path):
    a = tmp_path / "a"
    (a / "stages").mkdir(parents=True)
    (a / "stages" / "train.done").write_text(json.dumps({"seconds": 300, "units": 1000}))
    (a / "stages" / "features.done").write_text(json.dumps({"seconds": 120, "units": 240}))
    b = tmp_path / "b"
    b.mkdir()
    (b / "timings.json").write_text(json.dumps({"sfm": 1000, "undistort": 60, "train": 4380}))
    (b / "meta.json").write_text(
        json.dumps({"frames": {"front": 60, "back": 60, "left_repeater": 60, "right_repeater": 60}})
    )
    (b / "recon_args.json").write_text(json.dumps({"steps": 30000}))
    rates = progress.history_rates(tmp_path)
    assert rates["train"] == pytest.approx((0.3 + 0.146) / 2)
    assert rates["features"] == pytest.approx((0.5 + 1000 * 0.3 / 240) / 2)
    assert rates["undistort"] == pytest.approx(0.25)
    assert "frames" not in rates
    est = progress.estimate_build(rates, 240, 15000)
    assert est["frames"] == pytest.approx(progress.DEFAULT_RATES["frames"] * 240)


# Lines pipeline v2 (feat/pipeline-v2) prints when its optional steps finish.
V2_LOG = [
    ("[x] 6 cams, 0s +20s @ 6 fps", None),
    ("ego mask: guessed", None),
    ("sky (segformer): front 31%, back 28%", ("sky", "masks")),
    ("$ colmap feature_extractor ...", None),
    ("$ colmap image_undistorter ...", None),
    ("color norm (LAB, to front): back dL +2.1, left_repeater dL -0.4", ("color", "undistort")),
    (
        "dense init: +120000 depth points (from 3000000), +8000 road-ribbon points (SfM had 40000)",
        ("depth", "undistort"),
    ),
    ("$ /x/brush_app dense", None),
    ("clean: 900000 -> 850000 splats (prune 40000, haze 10000)", ("clean", "export")),
    ("splat -> work/x/viewer/scene.ply (120 MB)", None),
    (
        "collision: ground 80x200 cells of 0.5 m (90% from splats, the rest from the path), 12 wall cells",
        ("collision", "export"),
    ),
]


def test_pipeline_v2_substages():
    t = Tracker(n_images=8)
    subs = []
    for line, want in V2_LOG:
        evs = t.feed(line)
        got = [(e["substage"], e["stage"]) for e in evs if e["type"] == "substage"]
        assert got == ([want] if want else []), line
        subs += [e for e in evs if e["type"] == "substage"]
    assert [e["label"] for e in subs] == ["Sky masks", "Colour match", "Depth init", "Prune & clean", "Collision"]
    assert t.stage == "export"  # substages never move the core stages
    assert set(progress.STAGES) == {"frames", "masks", "features", "match", "map", "undistort", "train", "export"}


def test_unknown_stage_names_are_tolerated():
    t = Tracker(n_images=8)
    assert t.begin("relight") == [] and t.progress("relight", 1, 2) == []
    assert progress.label("relight") == "relight" and progress.label("depth") == "Depth init"
