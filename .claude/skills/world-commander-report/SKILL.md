---
name: world-commander-report
description: >-
  Build, update, and publish a World Commander experiment report as the self-contained web page on
  the shared PRIVATE GitLab Pages hub (the world-commander-bench convention). Use this whenever the
  user wants to build/regenerate/refresh an experiment report, add results to it, publish it, "send
  the report link", or asks for the report URL — in any World Commander repo (world-commander-motion,
  world-commander-bench). It covers the build (experiments/build_report.py), the publish-to-hub flow,
  and private-only delivery. The always-on guardrails live in the report-convention memory; this
  skill is the procedure.
---

# World Commander experiment report — build & publish

The deliverable for a World Commander experiment is **one self-contained `report.html`** on a
**shared private GitLab Pages hub**, where all experiments cross-link via a switcher bar
(`Grid Arena (E1) · StarCraft II (E2) · Embodiment (E3) · Crowd Motion (E4)`). The reference
implementation lives in the sibling repo `world-commander-bench` (`scripts/build_report.py` +
`arena/viz.py:build_html_report`); this repo's `experiments/build_report.py` follows it.

## Guardrails (also in the `report-convention` memory — restated because they matter)

- **Web page is the deliverable.** Hand back the hub URL, not the markdown. `REPORT.md` is the
  source-of-record / working notes; the web page is what you give people.
- **Private only — never public.** Pages is members-only (`pages_access_level=private`). Public
  GitHub Pages publishing **caused a GitHub account suspension on 2026-06-20**; the old
  `publish_report.sh` is retired. Local preview binds `127.0.0.1` and is reached over an SSH
  tunnel — never `0.0.0.0`, never a public host.
- **Omit the collaborator's name.** Don't name the partner professor or describe the work by that
  person's research "direction" in any report/output. (Their own `world-commander/plan/
  research-plan.md` lists roles — don't edit that; only your own outputs.)

## 1 — Build

```bash
python experiments/build_report.py            # -> experiments/report.html (self-contained)
```

`build_report.py` reads the numeric results from `experiments/results/*.json`, embeds figures and
the interactive-replay frames as base64 data-URIs, and emits one HTML file in the bench style:
`noindex`/"members-only" masthead → relative-link **switcher bar** → interactive replay viewer →
sections of **intro (what/why) → figure → table → caption (the takeaway, in bold)** → a
metric-definitions `<dl>` → footer. Keep section prose in that voice. If you added a new experiment,
add a section in `build_report.py` (and a results JSON) — don't hand-edit `report.html`, it's
generated.

## 2 — Publish to the shared hub

```bash
python experiments/build_report.py --publish   # also copies report.html -> ../world-commander-bench/motion.html
cd ../world-commander-bench
git add motion.html && git commit -m "update motion report" && git push gitlab main
```

Push to the bench repo's **`gitlab`** remote, **not** its `origin` (that's the suspended GitHub).
The `pages` CI job (`.gitlab-ci.yml`) copies `motion.html` into the site. Confirm it deployed:

```bash
glab ci status        # expect: (success) deploy  pages
```

If this report is a *new* experiment (not the existing E4 motion report), also add it to the
switcher bar in every report (`report.html`, `sc2.html`, `embodiment.html`, and the `arena/viz.py`
template) and to the bench `.gitlab-ci.yml`, mirroring how `motion.html`/E4 was wired in.

## 3 — Deliver

Hand back the **private hub URL** (members-only; the user views it signed in as a project member):

```
https://world-commander-bench-087cae.gitlab.io/motion.html
```

A non-member (or anyone not signed in) gets a 302 to GitLab auth — that redirect confirms it's
live AND private, which is what you want.

## Local preview (optional, for yourself)

```bash
bash experiments/serve_report.sh               # serves 127.0.0.1:8899
# from a laptop:  ssh -L 8899:localhost:8899 <amax41>   then open http://localhost:8899/report.html
```

Use this only to eyeball the page before publishing. It is not a way to share — the hub link is.

## Verify before you call it done

- `report.html` is self-contained (no `src="figures/..."` left — all `data:` URIs).
- The switcher bar lists this report and links the others with relative paths.
- The CI pipeline is green and the hub URL returns 302 (live + private).
