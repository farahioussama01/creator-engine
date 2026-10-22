"""The whole loop, small: two rounds on simulated pods, then the HTML report."""

from creator_engine.demo import run_demo


async def test_demo_loop_and_report(cfg, tmp_path):
    logs: list[str] = []
    out = await run_demo(
        cfg, pieces=6, rounds=2, time_scale=0.002, workdir=tmp_path / "demo", log=logs.append
    )
    assert len(out["rounds"]) == 2 and all(r["keepers"] > 0 for r in out["rounds"])
    assert out["rounds"][0]["quotas"] == {"cafe-pov": 2, "golden-street": 2, "mirror-walk": 2}
    html = open(out["report"], encoding="utf-8").read()
    for section in (
        "Cost per keeper",
        "Where the money went",
        "How to halve it",
        "Same clip, hosted",
        "Identity holds",
        "Formats: what travels",
        "Publishing queue",
        "SIMULATED GPU",
    ):
        assert section in html, section
    assert "<script src" not in html and "http://" not in html.split("<style>")[0]  # self-contained
    assert any("enrolled mila" in line for line in logs)
