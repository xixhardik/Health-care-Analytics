"""Sprint 7 - live end-to-end verification of the demo click-path.

Sprint 6's ``19_smoke_e2e.py`` proves the core chain. This script proves the
surface Sprint 7 added, against a **real running server** over HTTP with no
mocked responses:

* health serves the pipeline identity, the metric table and the research record
* CORS actually permits the frontend origin
* ``POST /api/analysis/sample`` creates a real analysis from the bundled volume
* progress comes from real stage boundaries
* every disc value carries a provenance tag, and unavailable values carry a
  reason instead of a fabricated zero
* ``highlight_disc`` changes the rendered slice, which is what keeps the viewer
  in step with the selected disc
* the report and download carry the pipeline version

It walks the same order as ``outputs/reports/sprint7_final/demo_checklist.md``,
so a failure here maps to a step in the demo.

Usage
-----
    # terminal 1
    uvicorn backend.app.main:app --port 8000
    # terminal 2
    cd frontend && npm run dev
    # terminal 3
    python scripts/20_smoke_e2e_sprint7.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "SPIDER-Lumbar-v1"

OK, BAD = "PASS", "FAIL"
checks: list[tuple[str, str, str]] = []
timings: dict[str, float] = {}


def check(name: str, passed: bool, detail: str = "") -> bool:
    checks.append((OK if passed else BAD, name, detail))
    print(f"  [{OK if passed else BAD}] {name}" + (f": {detail}" if detail else ""))
    return passed


def timed(label: str):
    """Context-manager-free timer: returns a stop() that records and returns ms."""
    start = time.perf_counter()

    def stop() -> float:
        elapsed = (time.perf_counter() - start) * 1000.0
        timings[label] = elapsed
        return elapsed

    return stop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend", default="http://127.0.0.1:3000")
    parser.add_argument("--origin", default="http://localhost:3000",
                        help="Origin header used for the CORS checks.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the analysis instead of deleting it.")
    return parser.parse_args()


def main() -> int:  # noqa: PLR0915 - a linear smoke test reads better flat
    args = parse_args()
    client = httpx.Client(base_url=args.base, timeout=120.0)

    print("=" * 78)
    print("SPRINT 7 LIVE END-TO-END VERIFICATION")
    print("=" * 78)
    print(f"  backend  : {args.base}")
    print(f"  frontend : {args.frontend}")
    print(f"  origin   : {args.origin}")

    # ------------------------------------------------------ 1. connectivity
    print("\n[1] Backend connectivity and health")
    stop = timed("health_ms")
    try:
        response = client.get("/api/health")
    except httpx.ConnectError:
        print(f"  Cannot reach the backend at {args.base}.")
        print("  Start it with: uvicorn backend.app.main:app --port 8000")
        return 2
    latency = stop()
    health = response.json()
    check("health returns 200", response.status_code == 200, f"{latency:.0f} ms")
    check("service reports ok (model loaded)", health["status"] == "ok",
          f"{health['segmentation_model']['architecture']} on "
          f"{health['segmentation_model']['device']}")

    # ------------------------------------------------- 2. pipeline identity
    print("\n[2] Pipeline identity and served research figures")
    pipeline = health["pipeline"]
    check("health serves the pipeline version",
          pipeline["version"] == PIPELINE_VERSION, pipeline["version"])
    check("pipeline names all three stages",
          pipeline["preprocessing"] == "Sprint 1"
          and pipeline["segmentation"] == "Sprint 3"
          and pipeline["postprocessing"] == "Sprint 5",
          f"{pipeline['preprocessing']} / {pipeline['segmentation']} / "
          f"{pipeline['postprocessing']}")
    check("the rejected Sprint 4 experiment is declared, not hidden",
          "Sprint 4" in pipeline["excluded"],
          pipeline["excluded"].get("Sprint 4", "")[:60] + "...")
    check("dataset named", "SPIDER" in pipeline["dataset"], pipeline["dataset"])

    metrics = health["validated_metrics"]
    check("published segmentation Dice served",
          metrics["segmentation_macro_foreground_dice"] == 0.90001
          and metrics["segmentation_vertebra_dice"] == 0.91663
          and metrics["segmentation_disc_dice"] == 0.87827
          and metrics["segmentation_canal_dice"] == 0.90514)
    check("published Sprint 5 indexing figures served",
          metrics["disc_indexing_percent"] == 93.84
          and metrics["disc_indexing_baseline_percent"] == 84.08,
          f"{metrics['disc_indexing_baseline_percent']}% -> "
          f"{metrics['disc_indexing_percent']}%")
    check("published measurement errors served",
          metrics["disc_height_mae_mm"] == 0.6492
          and metrics["disc_area_mae_mm2"] == 21.4831
          and metrics["intensity_ratio_pearson_r"] == 0.9726)
    check("end-to-end Pfirrmann QWK served",
          metrics["pfirrmann_qwk_end_to_end"] == 0.6543)
    check("test split size stated", metrics["test_patients"] == 33)
    check("metrics are labelled as describing the pipeline, not the study",
          "not this study" in metrics["note"].lower())

    table = health["metric_table"]
    kinds = {row["kind"] for row in table}
    check("metric table served for the UI (no hard-coded frontend metrics)",
          len(table) >= 10, f"{len(table)} rows")
    check("segmentation and post-processing metrics kept separate",
          kinds == {"segmentation", "postprocessing"}, str(sorted(kinds)))

    progress_record = health["research_progress"]
    rejected = [s for s in progress_record if s["outcome"] == "rejected"]
    check("research progress served including a negative result",
          len(progress_record) >= 4 and len(rejected) >= 1,
          f"{len(progress_record)} stages, rejected: "
          f"{', '.join(s['sprint'] for s in rejected)}")
    check("Sprint 4 is the rejected one",
          any(s["sprint"] == "Sprint 4" for s in rejected))

    check("provenance labels served",
          set(health["provenance_labels"]) ==
          {"segmentation_derived", "model_prediction", "unsupported"},
          " / ".join(health["provenance_labels"].values()))
    check("evidence labels served and qualitative",
          set(health["evidence_labels"]) == {"strong", "moderate", "modest", "weak"}
          and not any("%" in v for v in health["evidence_labels"].values()),
          " / ".join(health["evidence_labels"].values()))
    check("disclaimer refuses the medical-device claim",
          "not a medical device" in health["disclaimer"].lower())
    check("research notice served", "prototype" in health["research_notice"].lower())
    check("sample study advertised as available",
          health["sample_study_available"] is True)

    # ------------------------------------------------------------- 3. CORS
    print("\n[3] CORS for the frontend origin")
    preflight = client.request(
        "OPTIONS", "/api/health",
        headers={"Origin": args.origin,
                 "Access-Control-Request-Method": "GET"},
    )
    allow_origin = preflight.headers.get("access-control-allow-origin")
    check("preflight accepted for the frontend origin",
          preflight.status_code in (200, 204) and allow_origin == args.origin,
          f"{preflight.status_code}, allow-origin={allow_origin}")
    allow_methods = (preflight.headers.get("access-control-allow-methods") or "")
    check("preflight allows the methods the UI uses",
          all(m in allow_methods for m in ("GET", "POST", "DELETE")),
          allow_methods)
    with_origin = client.get("/api/health", headers={"Origin": args.origin})
    check("actual request carries the allow-origin header",
          with_origin.headers.get("access-control-allow-origin") == args.origin)
    hostile = client.get("/api/health",
                         headers={"Origin": "http://evil.example.com"})
    check("an unlisted origin is not granted access",
          hostile.headers.get("access-control-allow-origin") is None)

    # --------------------------------------------------- 4. sample endpoint
    print("\n[4] Load Sample Study")
    stop = timed("sample_ms")
    response = client.post("/api/analysis/sample")
    latency = stop()
    check("sample study created", response.status_code == 201,
          f"{response.status_code}, {latency:.0f} ms")
    if response.status_code != 201:
        print("  Cannot continue without a sample analysis.")
        return 1
    created = response.json()
    analysis_id = created["analysis_id"]
    study = created["study"]
    print(f"      analysis_id={analysis_id}")
    check("sample is flagged as a sample, not passed off as an upload",
          created["is_sample"] is True)
    check("sample study described from the real volume",
          study["slice_count"] > 0 and len(study["dimensions"]) == 2,
          f"{study['filename']}, {study['slice_count']} slices, "
          f"{study['dimensions']} px, modality {study['modality']}")
    check("analysis id is a random non-identifying token",
          len(analysis_id) == 16 and analysis_id not in study["filename"])

    # ------------------------------------------------------------- 5. run
    print("\n[5] Non-blocking start")
    stop = timed("run_ms")
    response = client.post(f"/api/analysis/{analysis_id}/run")
    latency = stop()
    check("run returns 202 immediately", response.status_code == 202,
          f"{latency:.0f} ms")
    check("run does not block on the pipeline", latency < 2000, f"{latency:.0f} ms")

    # ------------------------------------------------------- 6. real progress
    print("\n[6] Real progress from pipeline stage boundaries")
    trace: list[tuple[float, int, str]] = []
    began = time.perf_counter()
    deadline = began + 900
    state: dict = {}
    while time.perf_counter() < deadline:
        state = client.get(f"/api/analysis/{analysis_id}/status").json()
        point = (state["progress"], state["stage"])
        if not trace or (trace[-1][1], trace[-1][2]) != point:
            at = time.perf_counter() - began
            trace.append((at, state["progress"], state["stage"]))
            print(f"      {at:6.2f}s  {state['status']:<11}"
                  f"{state['progress']:>4}%  {state['stage']}")
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    timings["processing_s"] = time.perf_counter() - began

    check("analysis completed", state.get("status") == "completed",
          f"{state.get('status', 'unknown')} in {timings['processing_s']:.2f}s")
    if state.get("status") != "completed":
        print(f"  error: {json.dumps(state.get('error'))}")
        return 1
    values = [p for _, p, _ in trace]
    check("progress is monotonic (no fake or rewinding bar)",
          values == sorted(values), str(values))
    check("progress reached 100", values[-1] == 100)
    check("several real stages reported", len(trace) >= 3,
          " -> ".join(stage for _, _, stage in trace))

    # ---------------------------------------------------------- 7. result
    print("\n[7] Result from the real pipeline")
    stop = timed("result_ms")
    result = client.get(f"/api/analysis/{analysis_id}/result").json()
    latency = stop()
    discs = result["discs"]
    check("result returns discs", len(discs) > 0,
          f"{len(discs)} discs, {latency:.0f} ms")
    check("result carries the pipeline version",
          result["pipeline"]["version"] == PIPELINE_VERSION)
    check("result keeps the sample flag", result["is_sample"] is True)
    check("timepoint is a single baseline, no longitudinal claim",
          result["timepoint"]["id"] == "baseline"
          and result["timepoint"]["is_baseline"] is True
          and result["timepoint"]["acquired_at"] is None)
    check("disclaimer travels with the result",
          "qualified radiologist" in result["disclaimer"])

    # ------------------------------------------- 8. provenance on every value
    print("\n[8] Provenance and evidence on every disc value")
    allowed_provenance = {"segmentation_derived", "model_prediction", "unsupported"}
    allowed_strength = {"strong", "moderate", "modest", "weak"}
    lines = [line for disc in discs for line in disc["structured"]]
    check("every disc carries structured label/value/provenance lines",
          all(disc["structured"] for disc in discs),
          f"{len(lines)} lines across {len(discs)} discs")
    check("every provenance value is from the declared closed set",
          all(line["provenance"] in allowed_provenance for line in lines),
          str(sorted({line["provenance"] for line in lines})))
    check("segmentation-derived measurements are present and labelled",
          any(line["provenance"] == "segmentation_derived" for line in lines))
    check("model estimates are labelled as estimates",
          any(line["provenance"] == "model_prediction" for line in lines))
    unavailable = [line for line in lines if not line["available"]]
    check("unavailable values exist and every one carries a reason",
          len(unavailable) > 0
          and all(line["unavailable_reason"] for line in unavailable),
          f"{len(unavailable)} unavailable lines")
    check("an unavailable value is never rendered as a negative finding",
          all(str(line["value"]).lower() not in ("0", "no", "none", "normal", "false")
              for line in unavailable),
          str(sorted({line["value"] for line in unavailable})))
    check("available model estimates carry a qualitative strength",
          all(line["strength"] in allowed_strength
              for line in lines
              if line["available"] and line["provenance"] == "model_prediction"))
    check("no numeric confidence percentage is served as a value",
          not any("confidence" in line["label"].lower() for line in lines))

    # ------------------------------------- 9. measurements and findings
    print("\n[9] Measurements and model-estimated findings")
    heights = [d["measurements"]["height_mm_central"] for d in discs]
    check("central heights are real numbers in a plausible range",
          all(h is not None and 1.0 < h < 25.0 for h in heights),
          ", ".join(f"{h:.2f}mm" for h in heights))
    areas = [d["measurements"]["area_mm2"] for d in discs]
    check("disc areas present", all(a for a in areas),
          ", ".join(f"{a:.0f}mm2" for a in areas))
    check("measurements declare their source",
          all(d["measurements"]["source"] == "segmentation_derived_measurement"
              for d in discs))

    estimates = [f for d in discs for f in d["findings"]
                 if not f.get("unavailable_reason")]
    check("model estimates carry their validated metric and value",
          all(f.get("validated_metric") and f.get("validated_value") is not None
              for f in estimates),
          f"{len(estimates)} estimates")
    check("binary estimates carry the prevalence baseline for context",
          all(f.get("prevalence_baseline") is not None
              for f in estimates if f["name"] != "pfirrmann_grade"))
    narrowing = next((f for d in discs for f in d["findings"]
                      if f["name"] == "narrowing"), None)
    check("narrowing served with its published PR-AUC",
          narrowing is not None and narrowing["validated_value"] == 0.8779
          and narrowing["prevalence_baseline"] == 0.323,
          f"PR-AUC {narrowing['validated_value']} vs baseline "
          f"{narrowing['prevalence_baseline']}" if narrowing else "missing")
    graded = [d["pfirrmann_grade"] for d in discs if d["pfirrmann_grade"] is not None]
    check("Pfirrmann reported for this T2 study",
          len(graded) == len(discs) and all(1 <= g <= 5 for g in graded),
          f"grades {graded}")
    pf = next((f for d in discs for f in d["findings"]
               if f["name"] == "pfirrmann_grade"), None)
    check("Pfirrmann served with the END-TO-END QWK, not the ground-truth one",
          pf is not None and pf["validated_value"] == 0.6543,
          f"{pf['validated_metric']} {pf['validated_value']}" if pf else "missing")

    # ------------------------------------- 10. unsupported targets
    print("\n[10] Unsupported targets stated, never guessed")
    check("modic_type is null on every disc",
          all(d["modic_type"] is None for d in discs))
    check("spondylolisthesis is null on every disc",
          all(d["spondylolisthesis"] is None for d in discs))
    unsupported = result["summary"]["unsupported_targets"]
    check("both unsupported targets declared with a reason",
          set(unsupported) == {"spondylolisthesis", "modic_type"}
          and all(v for v in unsupported.values()))
    check("the Modic reason explains it was never modelled",
          "never modelled" in unsupported["modic_type"].lower())
    check("the spondylolisthesis reason cites too few positives",
          "positive" in unsupported["spondylolisthesis"].lower())
    check("summary states the single-timepoint scope",
          any(f["category"] == "Scope" for f in result["summary"]["findings"]))
    # The wording carries a parenthetical example, so match on the claim rather
    # than on one contiguous phrase.
    detection = next((f["text"] for f in result["summary"]["findings"]
                      if f["category"] == "Detection"), "")
    check("summary refuses to assert an anatomical level name",
          "no anatomical level name" in detection.lower()
          and "asserted" in detection.lower(),
          detection[:110] + "...")
    check("summary explains why no level name is given",
          "which vertebra is l5" in detection.lower())

    # ---------------------------------------- 11. slices and overlays
    print("\n[11] Slice images and segmentation overlays")
    middle = result["study"]["slice_count"] // 2
    payloads: dict[str, bytes] = {}
    for mode in ("original", "mask", "overlay"):
        stop = timed(f"slice_{mode}_ms")
        response = client.get(f"/api/analysis/{analysis_id}/slice/{middle}",
                              params={"mode": mode})
        latency = stop()
        payloads[mode] = response.content
        ok = (response.status_code == 200
              and response.headers["content-type"] == "image/png"
              and response.content[:8] == b"\x89PNG\r\n\x1a\n")
        check(f"slice renders as PNG in {mode} mode", ok,
              f"{len(response.content):,} bytes, {latency:.0f} ms")
    check("overlay differs from the plain image (the mask is really drawn)",
          payloads["overlay"] != payloads["original"])
    check("slice count exposed for the viewer",
          response.headers.get("X-Slice-Count")
          == str(result["study"]["slice_count"]))
    check("completed slices are cacheable",
          "immutable" in response.headers.get("cache-control", ""))

    only_disc = client.get(f"/api/analysis/{analysis_id}/slice/{middle}",
                           params={"mode": "mask", "classes": "2"}).content
    all_classes = client.get(f"/api/analysis/{analysis_id}/slice/{middle}",
                             params={"mode": "mask", "classes": "1,2,3"}).content
    check("class filtering happens server-side (hidden class not transmitted)",
          only_disc != all_classes,
          f"{len(only_disc):,} vs {len(all_classes):,} bytes")

    # ------------------------------ 12. disc selection -> viewer highlight
    print("\n[12] Disc selection drives the viewer highlight")
    target = discs[0]
    focus = target.get("representative_slice_index")
    check("selected disc reports a representative slice for the viewer",
          isinstance(focus, int) and 0 <= focus < result["study"]["slice_count"],
          f"disc {target['index']} -> slice {focus}")
    plain = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                       params={"mode": "overlay"})
    stop = timed("slice_highlight_ms")
    highlighted = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                             params={"mode": "overlay",
                                     "highlight_disc": target["index"]})
    latency = stop()
    check("highlight_disc accepted on the slice endpoint",
          highlighted.status_code == 200
          and highlighted.headers["content-type"] == "image/png",
          f"{latency:.0f} ms")
    check("highlighting actually changes the rendered slice",
          highlighted.content != plain.content,
          f"{len(plain.content):,} -> {len(highlighted.content):,} bytes")
    if len(discs) > 1:
        other = discs[1]
        second = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                            params={"mode": "overlay",
                                    "highlight_disc": other["index"]})
        check("highlighting a different disc renders differently",
              second.content != highlighted.content,
              f"disc {target['index']} vs disc {other['index']}")
    bad_highlight = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                               params={"mode": "overlay", "highlight_disc": 0})
    check("out-of-range highlight rejected cleanly with an error envelope",
          bad_highlight.status_code in (400, 422)
          and "error" in bad_highlight.json(),
          f"{bad_highlight.status_code} "
          f"{bad_highlight.json().get('error', {}).get('code')}")
    oob = client.get(f"/api/analysis/{analysis_id}/slice/99999")
    check("out-of-range slice rejected cleanly",
          oob.status_code == 404
          and oob.json()["error"]["code"] == "SLICE_OUT_OF_RANGE")

    # ------------------------------ 12b. finding-aware red overlay (Sprint 7.1)
    print("\n[12b] Findings overlay")
    overlay = result["finding_overlay"]
    marked = overlay["disc_indices"]
    check("the result serves the overlay decision from the server",
          isinstance(marked, list) and isinstance(overlay["discs"], list),
          f"discs marked: {marked or 'none'}")
    check("the ordinal Pfirrmann grade is not an overlay trigger",
          "pfirrmann_grade" not in overlay["eligible_findings"],
          ", ".join(overlay["eligible_findings"]))
    check("no unsupported target is an overlay trigger",
          not {"spondylolisthesis", "modic_type"}
          & set(overlay["eligible_findings"]))
    note = overlay["note"].lower()
    check("the overlay states it is not a pixel-level pathology claim",
          "damaged tissue" in note and "pixel-level pathology model" in note
          and "disc regions associated with" in note)

    # Cross-check the served decision against the discs' own findings.
    eligible = set(overlay["eligible_findings"])
    expected = {
        disc["index"] for disc in discs
        if any(f.get("name") in eligible and f.get("value") is True
               and not f.get("unavailable_reason")
               and f.get("source") == "model_prediction"
               for f in disc["findings"])
    }
    check("marked discs are exactly the discs with a positive supported finding",
          set(marked) == expected,
          f"served {sorted(marked)} vs derived {sorted(expected)}")

    off = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                     params={"mode": "overlay"})
    explicit_off = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                              params={"mode": "overlay", "highlight": "disc"})
    check("overlay OFF is byte-identical to the pre-existing rendering",
          off.content == explicit_off.content,
          f"{len(off.content):,} bytes")
    check("overlay OFF reports no marked discs",
          off.headers.get("X-Finding-Discs") == "")

    stop = timed("slice_findings_ms")
    on = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                    params={"mode": "overlay", "highlight": "findings"})
    latency = stop()
    check("overlay ON renders a PNG",
          on.status_code == 200 and on.content[:8] == b"\x89PNG\r\n\x1a\n",
          f"{len(on.content):,} bytes, {latency:.0f} ms")
    header = [int(v) for v in (on.headers.get("X-Finding-Discs") or "").split(",")
              if v]
    check("the response reports which discs it marked",
          header == sorted(marked), str(header))
    if marked:
        check("overlay ON changes the image", on.content != off.content)
        emphasised = client.get(
            f"/api/analysis/{analysis_id}/slice/{focus}",
            params={"mode": "overlay", "highlight": "findings",
                    "highlight_disc": marked[0]})
        check("selecting a marked disc renders it more strongly",
              emphasised.content != on.content,
              f"disc {marked[0]} emphasised")

    for mode in ("original", "mask", "overlay"):
        response = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                              params={"mode": mode, "highlight": "findings"})
        check(f"overlay renders in {mode} mode", response.status_code == 200)

    again = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                       params={"mode": "overlay"})
    check("the MRI underneath is unchanged once the overlay is turned off",
          again.content == off.content)

    bad_mode = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                          params={"mode": "overlay", "highlight": "tumour"})
    check("an invalid highlight mode returns INVALID_REQUEST, not an image",
          bad_mode.status_code == 422
          and bad_mode.json()["error"]["code"] == "INVALID_REQUEST",
          f"{bad_mode.status_code} "
          f"{bad_mode.json().get('error', {}).get('code')}")
    bad_alpha = client.get(f"/api/analysis/{analysis_id}/slice/{focus}",
                           params={"mode": "overlay", "highlight": "findings",
                                   "finding_opacity": 1.8})
    check("an out-of-range finding opacity returns INVALID_REQUEST",
          bad_alpha.status_code == 422
          and bad_alpha.json()["error"]["code"] == "INVALID_REQUEST")

    # ------------------------------------------- 13. report and download
    print("\n[13] Report and download")
    stop = timed("report_ms")
    report = client.get(f"/api/analysis/{analysis_id}/report")
    latency = stop()
    check("report endpoint returns the payload",
          report.status_code == 200
          and report.json()["analysis_id"] == analysis_id,
          f"{latency:.0f} ms")
    check("report payload carries the pipeline version",
          report.json()["pipeline"]["version"] == PIPELINE_VERSION)

    stop = timed("download_md_ms")
    markdown = client.get(f"/api/analysis/{analysis_id}/download",
                          params={"fmt": "md"})
    latency = stop()
    body = markdown.text
    check("markdown report downloads as an attachment",
          markdown.status_code == 200
          and "attachment" in markdown.headers.get("content-disposition", ""),
          f"{len(body):,} chars, {latency:.0f} ms")
    check("report states the pipeline version",
          PIPELINE_VERSION in body)
    for heading in ("## 1. Study Information", "## 2. Processing Information",
                    "## 3. Segmentation Overview", "## 4. Disc-Level Analysis",
                    "## 5. Quantitative Measurements",
                    "## 6. Model-Derived Findings", "## 7. Limitations",
                    "## 8. Research Disclaimer"):
        check(f"report contains '{heading}'", heading in body)
    check("report carries the radiologist-review disclaimer",
          "require review by a qualified radiologist" in body)
    check("report keeps postoperative assessment out of scope",
          "postoperative" in body.lower())
    check("report labels validated performance as held-out test, not per-patient",
          "held-out test" in body.lower())
    check("report claims no composite damage score",
          "damage score" not in body.lower()
          and "damage percentage" not in body.lower())

    stop = timed("download_json_ms")
    json_download = client.get(f"/api/analysis/{analysis_id}/download",
                               params={"fmt": "json"})
    latency = stop()
    check("json report downloads", json_download.status_code == 200,
          f"{len(json_download.content):,} bytes, {latency:.0f} ms")

    # ---------------------------------------------------------- 14. history
    print("\n[14] History")
    history = client.get("/api/analysis").json()
    entry = next((i for i in history["items"]
                  if i["analysis_id"] == analysis_id), None)
    check("analysis appears in history", entry is not None)
    if entry:
        check("history carries real counts",
              entry["disc_count"] == len(discs)
              and entry["slice_count"] == result["study"]["slice_count"],
              f"{entry['disc_count']} discs, {entry['slice_count']} slices")
        check("history marks the sample study", entry.get("is_sample") is True)

    # --------------------------------------------------------- 15. frontend
    print("\n[15] Frontend and its API configuration")
    bundle = ROOT / "frontend" / ".next"
    check("frontend build output present", bundle.exists())
    env_local = ROOT / "frontend" / ".env.local"
    env_example = ROOT / "frontend" / ".env.example"
    configured = ""
    for candidate in (env_local, env_example):
        if candidate.exists():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                if raw.strip().startswith("NEXT_PUBLIC_API_URL"):
                    configured = raw.split("=", 1)[1].strip()
            if configured:
                break
    probe = client.get("/api/health", headers={"Origin": args.origin})
    check("frontend is configured to the backend that is running",
          bool(configured) and probe.status_code == 200,
          f"NEXT_PUBLIC_API_URL={configured or 'unset'}")
    try:
        page = httpx.get(args.frontend, timeout=15.0)
        check("frontend server responds", page.status_code == 200,
              f"{args.frontend} -> {page.status_code}")
        check("frontend serves the application shell",
              "Lumbar MRI Analysis" in page.text)
        check("research notice is in the server-rendered shell",
              "prototype" in page.text.lower())
    except httpx.HTTPError:
        check("frontend server reachable", False,
              f"not running at {args.frontend} (start with: npm run dev)")

    # ---------------------------------------------------------- cleanup
    if not args.keep:
        client.delete(f"/api/analysis/{analysis_id}")
        print(f"\n  cleaned up analysis {analysis_id}")
    else:
        print(f"\n  kept analysis {analysis_id}")

    # ---------------------------------------------------------- summary
    failures = sum(1 for status, _, _ in checks if status == BAD)
    print("\n" + "=" * 78)
    if failures:
        print("FAILURES")
        for status, name, detail in checks:
            if status == BAD:
                print(f"  [{BAD}] {name}" + (f": {detail}" if detail else ""))
        print()
    print(f"  {len(checks) - failures} passed, {failures} failed")

    print("\nTIMINGS")
    print(f"  health                 {timings.get('health_ms', 0):8.0f} ms")
    print(f"  create sample study    {timings.get('sample_ms', 0):8.0f} ms")
    print(f"  run (submit)           {timings.get('run_ms', 0):8.0f} ms")
    print(f"  analysis wall time     {timings.get('processing_s', 0):8.2f} s")
    print(f"  result                 {timings.get('result_ms', 0):8.0f} ms")
    for mode in ("original", "mask", "overlay"):
        print(f"  slice ({mode:<8})       "
              f"{timings.get(f'slice_{mode}_ms', 0):8.0f} ms")
    print(f"  slice (highlighted)    {timings.get('slice_highlight_ms', 0):8.0f} ms")
    print(f"  slice (findings red)   {timings.get('slice_findings_ms', 0):8.0f} ms")
    print(f"  report                 {timings.get('report_ms', 0):8.0f} ms")
    print(f"  download md            {timings.get('download_md_ms', 0):8.0f} ms")
    print(f"  download json          {timings.get('download_json_ms', 0):8.0f} ms")

    print("\nDEMO PATH")
    print(f"  sample study    {study['filename']} "
          f"({study['slice_count']} slices, {study['dimensions']} px)")
    print(f"  pipeline        {pipeline['version']}")
    print(f"  segmentation    {health['segmentation_model']['architecture']}")
    print(f"  indexing        {result['processing']['postprocessing_method']}")
    print(f"  discs found     {len(discs)}")
    print(f"  grades          {graded}")
    print(f"  provenance      {sorted({l['provenance'] for l in lines})}")
    print(f"  unavailable     {len(unavailable)} lines, all with reasons")
    print(f"  findings red    disc {sorted(marked) or 'none'} "
          f"(from the validated disc segmentation, not a pathology mask)")
    print(f"  duration        {result['processing']['duration_seconds']}s on "
          f"{result['processing']['device']}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
