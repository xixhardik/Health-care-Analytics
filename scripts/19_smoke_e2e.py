"""Sprint 6 - local end-to-end smoke test.

Proves the full chain against a **real running server** over HTTP, not a test
client: upload -> create -> process -> result -> slice -> report.

The frontend leg is verified by checking that the production bundle exists and
that the same endpoints the browser calls return what it expects, since a
headless browser is not part of this project's toolchain.

Usage
-----
    # terminal 1
    uvicorn backend.app.main:app --port 8000
    # terminal 2
    python scripts/19_smoke_e2e.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "extracted" / "images" / "33_t2.mha"

OK, BAD = "PASS", "FAIL"
checks: list[tuple[str, str, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    checks.append((OK if passed else BAD, name, detail))
    print(f"  [{OK if passed else BAD}] {name}" + (f": {detail}" if detail else ""))
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:3000")
    parser.add_argument("--volume", default=str(SAMPLE))
    parser.add_argument("--keep", action="store_true",
                        help="Keep the analysis instead of deleting it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    volume = Path(args.volume)
    if not volume.exists():
        print(f"Sample volume not found: {volume}")
        return 2

    client = httpx.Client(base_url=args.base, timeout=60.0)

    print("=" * 74)
    print("SPRINT 6 END-TO-END SMOKE TEST")
    print("=" * 74)
    print(f"  backend : {args.base}")
    print(f"  volume  : {volume.name}")

    # ---------------------------------------------------------------- health
    print("\n[1] Backend health")
    try:
        response = client.get("/api/health")
    except httpx.ConnectError:
        print(f"  Cannot reach the backend at {args.base}.")
        print("  Start it with: uvicorn backend.app.main:app --port 8000")
        return 2
    health = response.json()
    check("health returns 200", response.status_code == 200)
    check("segmentation model loaded once at startup",
          health["segmentation_model"]["loaded"] is True,
          f"{health['segmentation_model']['architecture']} on "
          f"{health['segmentation_model']['device']}")
    check("serving the Sprint 3 model (not Sprint 4)",
          health["segmentation_model"]["source_sprint"] == "Sprint 3 Coverage")
    check("Sprint 5 post-processing active",
          health["postprocessing_method"] == "Sprint 5 5A+5D")
    check("finding estimators loaded",
          len(health["finding_models"]["supported_targets"]) >= 7,
          ", ".join(health["finding_models"]["supported_targets"]))
    check("unsupported targets declared with reasons",
          set(health["finding_models"]["unsupported_targets"]) ==
          {"spondylolisthesis", "modic_type"})

    # ---------------------------------------------------------------- upload
    print("\n[2] Upload and validation")
    bad = client.post("/api/analysis/upload",
                      files={"file": ("notes.txt", b"x" * 4000, "text/plain")})
    check("bad extension rejected with a useful code",
          bad.status_code == 400
          and bad.json()["error"]["code"] == "UNSUPPORTED_FORMAT",
          bad.json()["error"]["message"][:70])
    check("no traceback leaks to the client",
          "Traceback" not in bad.text and 'File "' not in bad.text)

    with volume.open("rb") as handle:
        response = client.post(
            "/api/analysis/upload",
            files={"file": (volume.name, handle, "application/octet-stream")},
        )
    check("real volume accepted", response.status_code == 201, response.text[:100])
    created = response.json()
    analysis_id = created["analysis_id"]
    study = created["study"]
    print(f"      analysis_id={analysis_id}")
    check("study described from the file itself",
          study["slice_count"] > 0 and len(study["dimensions"]) == 2,
          f"{study['slice_count']} slices, {study['dimensions']} px, "
          f"modality {study['modality']}")
    check("analysis id is not derived from the filename",
          analysis_id not in volume.name and len(analysis_id) == 16)

    # ---------------------------------------------------------------- run
    print("\n[3] Non-blocking job start")
    started = time.perf_counter()
    response = client.post(f"/api/analysis/{analysis_id}/run")
    elapsed = time.perf_counter() - started
    check("run returns 202 immediately", response.status_code == 202,
          f"{elapsed:.2f}s")
    check("run does not block on the pipeline", elapsed < 2.0, f"{elapsed:.2f}s")

    # ---------------------------------------------------------------- poll
    print("\n[4] Real progress tracking")
    trace: list[tuple[int, str]] = []
    deadline = time.time() + 600
    state: dict = {}
    while time.time() < deadline:
        state = client.get(f"/api/analysis/{analysis_id}/status").json()
        point = (state["progress"], state["stage"])
        if not trace or trace[-1] != point:
            trace.append(point)
            print(f"      {state['status']:<11}{state['progress']:>4}%  "
                  f"{state['stage']}")
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    check("analysis completed", state.get("status") == "completed",
          state.get("status", "unknown"))
    values = [p for p, _ in trace]
    check("progress is monotonic (no fake or rewinding bar)",
          values == sorted(values), str(values))
    check("progress reached 100", values and values[-1] == 100)
    check("multiple real stages reported", len(trace) >= 3,
          f"{len(trace)} distinct stage/progress points")

    # ---------------------------------------------------------------- result
    print("\n[5] Result from the real pipeline")
    result = client.get(f"/api/analysis/{analysis_id}/result").json()
    discs = result["discs"]
    check("result returns 200 with discs", len(discs) > 0,
          f"{len(discs)} discs identified")
    check("measurements are real numbers in a plausible range",
          all(d["measurements"]["height_mm_central"] is not None
              and 1.0 < d["measurements"]["height_mm_central"] < 25.0
              for d in discs),
          ", ".join(f"{d['measurements']['height_mm_central']:.2f}mm" for d in discs))
    check("disc areas present",
          all(d["measurements"]["area_mm2"] for d in discs),
          ", ".join(f"{d['measurements']['area_mm2']:.0f}mm2" for d in discs))
    check("unsupported findings are null with a reason, not fabricated",
          all(d["modic_type"] is None and d["spondylolisthesis"] is None
              for d in discs))
    graded = [d["pfirrmann_grade"] for d in discs if d["pfirrmann_grade"] is not None]
    check("Pfirrmann reported for this T2 study",
          len(graded) == len(discs) and all(1 <= g <= 5 for g in graded),
          f"grades {graded}")
    check("template summary present with scope statement",
          any(f["category"] == "Scope" for f in result["summary"]["findings"]))
    check("validated metrics are the published figures",
          result["validated_metrics"]["disc_indexing_percent"] == 93.84
          and result["validated_metrics"]["segmentation_macro_foreground_dice"] == 0.90001)
    check("processing provenance recorded",
          result["processing"]["segmentation_source"].startswith("Sprint 3"),
          f"{result['processing']['postprocessing_method']}, "
          f"{result['processing']['duration_seconds']}s on "
          f"{result['processing']['device']}")

    # ---------------------------------------------------------------- slices
    print("\n[6] Slice retrieval")
    middle = result["study"]["slice_count"] // 2
    sizes = {}
    for mode in ("original", "mask", "overlay"):
        response = client.get(f"/api/analysis/{analysis_id}/slice/{middle}",
                              params={"mode": mode})
        sizes[mode] = len(response.content)
        ok = (response.status_code == 200
              and response.headers["content-type"] == "image/png"
              and response.content[:8] == b"\x89PNG\r\n\x1a\n")
        check(f"slice renders as PNG in {mode} mode", ok,
              f"{len(response.content):,} bytes")
    check("slice count exposed for the viewer",
          response.headers.get("X-Slice-Count") == str(result["study"]["slice_count"]))
    check("completed slices are cacheable",
          "immutable" in response.headers.get("cache-control", ""))

    filtered = client.get(f"/api/analysis/{analysis_id}/slice/{middle}",
                          params={"mode": "mask", "classes": "2"})
    check("class filtering happens server-side",
          filtered.content != client.get(
              f"/api/analysis/{analysis_id}/slice/{middle}",
              params={"mode": "mask", "classes": "1,2,3"}).content)

    oob = client.get(f"/api/analysis/{analysis_id}/slice/99999")
    check("out-of-range slice rejected cleanly",
          oob.status_code == 404
          and oob.json()["error"]["code"] == "SLICE_OUT_OF_RANGE")

    # ---------------------------------------------------------------- report
    print("\n[7] Report and download")
    report = client.get(f"/api/analysis/{analysis_id}/report")
    check("report endpoint returns the result payload",
          report.status_code == 200 and report.json()["analysis_id"] == analysis_id)

    markdown = client.get(f"/api/analysis/{analysis_id}/download",
                          params={"fmt": "md"})
    body = markdown.text
    check("markdown report downloads as an attachment",
          markdown.status_code == 200
          and "attachment" in markdown.headers.get("content-disposition", ""),
          f"{len(body):,} chars")
    for heading in ("## 1. Study Information", "## 4. Disc-Level Analysis",
                    "## 5. Quantitative Measurements",
                    "## 6. Model-Derived Findings", "## 7. Limitations",
                    "## 8. Research Disclaimer"):
        check(f"report contains '{heading}'", heading in body)
    check("report carries the radiologist-review disclaimer",
          "require review by a qualified radiologist" in body)
    check("report keeps postoperative analysis out of scope",
          "postoperative" in body.lower())

    json_download = client.get(f"/api/analysis/{analysis_id}/download",
                               params={"fmt": "json"})
    check("json report downloads", json_download.status_code == 200,
          f"{len(json_download.content):,} bytes")

    # ---------------------------------------------------------------- history
    print("\n[8] History")
    history = client.get("/api/analysis").json()
    entry = next((i for i in history["items"] if i["analysis_id"] == analysis_id), None)
    check("analysis appears in history", entry is not None)
    if entry:
        check("history carries real counts",
              entry["disc_count"] == len(discs)
              and entry["slice_count"] == result["study"]["slice_count"],
              f"{entry['disc_count']} discs, {entry['slice_count']} slices")

    # ---------------------------------------------------------------- frontend
    print("\n[9] Frontend")
    bundle = ROOT / "frontend" / ".next"
    check("frontend production build present", bundle.exists(),
          str(bundle.relative_to(ROOT)))
    try:
        page = httpx.get(args.frontend, timeout=10.0)
        check("frontend dev/prod server responds", page.status_code == 200,
              f"{args.frontend} -> {page.status_code}")
        check("frontend serves the workstation shell",
              "Lumbar MRI Analysis" in page.text)
    except httpx.HTTPError:
        check("frontend server reachable", False,
              f"not running at {args.frontend} (start with: npm run dev)")

    # ---------------------------------------------------------------- cleanup
    if not args.keep:
        client.delete(f"/api/analysis/{analysis_id}")
        print(f"\n  cleaned up analysis {analysis_id}")
    else:
        print(f"\n  kept analysis {analysis_id}")

    failures = sum(1 for status, _, _ in checks if status == BAD)
    print("\n" + "=" * 74)
    for status, name, detail in checks:
        if status == BAD:
            print(f"  [{BAD}] {name}" + (f": {detail}" if detail else ""))
    print(f"  {len(checks) - failures} passed, {failures} failed")
    print("=" * 74)
    print("FRONTEND -> API -> ML PIPELINE -> RESULT -> SLICE -> FRONTEND")
    print(f"  upload         {volume.name} ({study['slice_count']} slices)")
    print(f"  segmentation   {health['segmentation_model']['architecture']}")
    print(f"  indexing       {result['processing']['postprocessing_method']}")
    print(f"  discs found    {len(discs)}")
    print(f"  slice images   {sizes}")
    print(f"  duration       {result['processing']['duration_seconds']}s "
          f"on {result['processing']['device']}")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
