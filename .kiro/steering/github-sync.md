---
inclusion: always
---

# GitHub synchronisation policy

Applies to every session in this repository. It exists so that work reaches
GitHub only after it has been shown to work, and so that the dataset, model
weights and patient-derived artefacts this project handles never leave the
machine.

**Repository:** `https://github.com/xixhardik/Health-care-Analytics.git` (private)
**Tracked branch:** `main` → `origin/main`

---

## 1. Validate before committing

After every completed implementation task, run the gates relevant to what
changed. Run them from the repository root unless noted.

| Scope of change | Gate |
| --- | --- |
| Backend (`backend/`, `ml/`, `src/`) | `python -m pytest backend/tests -q` |
| Frontend behaviour | `cd frontend; npx vitest run` |
| Frontend types | `cd frontend; npm run typecheck` |
| Frontend or build configuration | `cd frontend; npm run build` |
| Live API / end-to-end surface | `python scripts/19_smoke_e2e.py` and `python scripts/20_smoke_e2e_sprint7.py` against running servers |

Also run any test that specifically covers the subsystem touched — for example
`backend/tests/test_findings_overlay.py` for overlay work, or
`backend/tests/test_storage_atomicity.py` for storage work. A change confined to
documentation needs no code gate; verify instead that the diff contains only the
intended files.

Known-good baselines to compare against, so a regression is visible:

- backend `pytest backend/tests -q` → 106 passed
- frontend `npx vitest run` → 30 passed
- frontend `npm run typecheck` → clean
- frontend `npm run build` → success, 9 routes
- `scripts/19_smoke_e2e.py` → 48/48
- `scripts/20_smoke_e2e_sprint7.py` → 119/119

Note for Windows: `npm run build` fails with `EPERM ... .next\trace` while a dev
server holds `.next`. Stop the dev server first rather than retrying.

## 2. If a required gate fails

Do **not** commit. Do **not** push. Report which gate failed, the failing output,
and the affected files. Fix the cause and re-run, or stop and report if the fix
is outside the task's scope. Never weaken a test or an ignore rule to get a
commit through.

## 3. If the gates pass

1. Inspect `git status --short --branch` and `git diff` (and `git diff --cached`
   once staged) before staging anything.
2. Stage **only** the files belonging to the completed task, by explicit path:
   `git add -- path/one path/two`.
3. Never use `git add -A`, `git add .`, or `git add -u`.
4. Never stage unrelated changes the user has in progress. If the working tree
   contains edits that are not part of this task, leave them alone and say so in
   the report.

## 4. Commit message

Conventional Commits, imperative mood, subject under ~70 characters:

`feat:` new capability · `fix:` bug fix · `docs:` documentation ·
`refactor:` no behaviour change · `test:` tests only · `chore:` tooling,
config, dependencies · `perf:` performance

Use the body for why, not what. Example: `feat: add findings overlay to MRI viewer`.

## 5. Push

Push to the current tracked branch: `git push -u origin main`. Confirm the
remote and branch are the expected ones first — `git remote -v` and
`git branch --show-current` — before anything leaves the machine.

## 6. Verify after pushing

Confirm these three agree, when a network call is practical:

```
git rev-parse HEAD                      # local
git rev-parse origin/main               # remote-tracking ref
git ls-remote origin refs/heads/main    # the server itself
```

The third matters: the second is only a local cache of what the server said last.

## 7. Report

- commit hash (full and short)
- commit message
- branch and upstream
- push status
- changed files
- validation results, with pass counts

## 8. Never do these

- force push (`--force`, `--force-with-lease`)
- rewrite history (`rebase`, `filter-branch`, `filter-repo`)
- `git reset --hard`
- delete branches
- amend or rewrite an already-pushed commit without explicit approval
- `git clean -fd` over unexamined files
- overwrite or discard unrelated user changes
- modify `git config`

If one of these genuinely looks necessary, stop and ask first.

## 9. Never commit or push these

This project handles licensed medical research data. The following stay local,
permanently:

- `data/raw/`, `data/processed/`, `data/extracted/` — the SPIDER dataset and
  everything derived from it, including `images.zip` (3.5 GB) and per-patient
  radiological gradings
- MRI volumes: `.mha`, `.mhd`, `.nii`, `.nii.gz`, `.nrrd`, `.dcm`
- rendered MRI slices, overlays and figures: any `.png`/`.jpg`/`.svg` of patient
  anatomy, and `outputs/visualizations/`
- model checkpoints and weights: `outputs/checkpoints/`, `outputs/models/`,
  `.pt`, `.pth`, `.ckpt`, `.h5`, `.onnx`, `.joblib`, `.pkl`, `.npy`, `.npz`
- generated patient-specific analysis data:
  `outputs/reports/finding_assessments/` and `outputs/app_data/`
- CSV files excluded by `.gitignore` (`outputs/**/*.csv`) — per-patient,
  per-series and per-slice measurement tables
- `.env`, `.env.local`, or any real secret. Only `*.env.example` is committed,
  and only because it contains paths and `localhost` URLs
- credentials, API keys, tokens, private keys: `.pem`, `.key`, `.pfx`, `.npmrc`,
  `.netrc`, `id_rsa`
- `node_modules/`, `.next/`, `__pycache__/`, `.pytest_cache/`, `.venv/`,
  `*.tsbuildinfo`

Embedded images in notebooks count as raster MRI. Before staging any `.ipynb`,
check that its outputs carry no `image/png`, `image/jpeg` or `data:image/`
payloads; strip them if they do, preserving source, markdown, metadata and
non-image outputs.

## 10. Respect `.gitignore` and `.gitattributes`

Both are deliberate and were audited file by file. Do not add exceptions,
negations or `--force` flags to make a commit succeed. If something genuinely
needs to be tracked, raise it and get approval rather than editing the ignore
rules in passing. `.gitattributes` is `* text=auto`, which keeps line endings
repo-controlled rather than dependent on each contributor's `core.autocrlf`.

## 11. Check the staged set before every push

After staging and before committing, confirm:

- no file exceeds 10 MB (GitHub hard-rejects over 100 MB, and nothing here should
  approach it — the largest tracked file is ~366 KB)
- no path matches any category in §9
- the staged file count matches what the task actually touched

`git diff --cached --name-only` is the list to check. Checking the working tree
is not sufficient — verify what is actually staged.

## 12. Stay inside the task

Do not fold unrelated cleanup, reformatting, dependency bumps or refactors into a
task's commit. If something unrelated needs fixing, mention it and let the user
decide whether it becomes its own task and its own commit.

---

## Frozen artefacts

These are validated research outputs. Do not modify them to make anything pass:

- `src/` — the frozen ML layer (Sprint 1 preprocessing, Sprint 3 U-Net,
  Sprint 5 post-processing). `ml/` is a serving adapter that delegates into it.
- `outputs/checkpoints/sprint3_coverage/best_val_dice.pt` — the served model
- `outputs/reports/sprint5_indexing/test_results.json` — the served
  post-processing configuration, read at startup
- `outputs/models/findings/` — the persisted finding estimators
- published metrics in `backend/app/pipeline_info.py`

No retraining, and no changes to preprocessing, segmentation or post-processing
as a side effect of another task.
