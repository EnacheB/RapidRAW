# RapidRAW fork

Upstream RapidRAW (Tauri: Rust `src-tauri/`, React `src/`) with features on top,
one commit per feature. Keep entropy low: the fewest moving parts, reusing existing
code paths.

**Machine setup and the owner's workflow** (checkouts, test builds on their
desktop, the package, hidden GUI tests, what to ask first): load the `rapidraw-dev`
skill before working here.

**Public repo.** Every tracked file and commit message is public. Write them for any
reader: technical facts, numbers, anonymous labels (img01). The owner's details (home
paths, usernames, host names, their photos and what they show, desktop setup) live
in the private `rapidraw-dev` skill; experiment outputs live outside the repo.

## Goal and pipeline principles

**Goal:** a Sony A7 III RAW opens looking at least as good as the camera's JPEG,
every slider acts on the RAW's full range, and the pipeline is technically correct
and explainable. Later goals: film sims that make every photo look great, faster
culling. Library management is out of scope.

Check every pipeline change against these principles, and say so explicitly when
a design bends one. The stage map is the Pipeline panel (`src/utils/pipeline.ts`);
known open violations are under "Open pipeline issues" below.

1. **Scene light until one rendering step.** From decode to the tone mapper,
   pixels are linear light in float. 1.0 is *not* white there; nothing clips or
   assumes 0–1. Exactly one stage, the rendering (the "tone mapper": look + tone
   curve), turns scene light into a picture. After it come only display
   finishing steps (brightness, display LUT, grain) and output encoding. Curves
   are a scene edit on purpose: they run on linear light just before the
   rendering, on a graph whose axis is that light encoded as sRGB.
2. **Never throw information away before the rendering.** No clamps, no 8-bit
   intermediates, no changing colours that aren't actually clipped. A shortcut
   (e.g. a fast preview) may cost speed or resolution, never different output.
3. **Order follows physics.** Sensor fixes (black/white level, clipped-highlight
   reconstruction, demosaic, denoise) → camera colour (white balance, camera RGB
   → working space) → lens and geometry → scene edits (exposure, local contrast,
   colour) → rendering → display finishing → output. A stage that looks at other
   data (luma, a blurred neighbourhood) must read it from the same point in the
   pipeline it acts on.
4. **Fixed per camera goes in the rendering; varies per shot goes in sliders.**
   The camera's colour and tone are the rendering and move no sliders, like
   Lightroom's Profile menu. Per-shot choices (exposure, white balance, shadow
   lift, faces) are slider values that Auto sets and the user can see. Never hide
   a per-shot correction inside the rendering.
5. **Looks and film sims replace or follow the rendering, never sit mid-pipeline.**
   Either a scene-referred look that *is* the rendering, or a display LUT after it.
   One rendering at a time.
6. **Local edits change slider values per pixel,** so each stage still runs once
   (masks already work this way). Face-aware exposure is a slider value, not a
   separate pass.
7. **Every stage is listed in the Pipeline panel** (`src/utils/pipeline.ts`) in
   the order it runs. Keep the panel and the shader in step.
8. **Store recipes, not pixels.** Keep parameters, offsets and masks, and re-derive
   pixels from the original. Stored pixels freeze a lossy copy (as 8-bit AI patches once did).

**Before designing or changing a pixel-processing stage, load the `pipeline-design`
skill and work through it.**

## Git layout

- `upstream`: github.com/CyberTimon/RapidRAW. `origin`: this fork.
- **`fork-tooling`**: `upstream/main` plus one commit, "Fork tooling (never upstream)":
  this file, `CLAUDE.md`, `.cargo/config.toml`, `test-build.sh`, `.claude/`.
- **`local`**: the one line of development. `fork-tooling` plus a straight line of
  commits, one per feature (`git log --oneline fork-tooling..local`).

A new feature is a new commit on top of `local`, or a worktree branch off `local`
that fast-forwards back onto it. A fix to an existing feature folds into its commit:
```sh
git commit --fixup=<feature commit>
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash fork-tooling
```
Fork-only files (the Fork tooling list above) change in their own commits
("Docs: …"), never inside a feature commit, so a feature can go upstream without them.

### Pull upstream (or change the Fork tooling commit)
Move `fork-tooling`, then `local`, then every worktree in flight:
```sh
old=$(git rev-parse fork-tooling); oldlocal=$(git rev-parse local)
git fetch upstream && git rebase upstream/main fork-tooling
git rebase --onto fork-tooling $old local             # in the checkout that has local
git -C <worktree> rebase --onto local $oldlocal      # each worktree in flight
```
A worktree with uncommitted work needs a WIP commit first. A worktree left on the
old base goes stale unnoticed, so move every one.

### Offer a feature upstream
Related upstream issues: Heal #1362, #1456. `git switch -c pr/<feature> upstream/main`,
cherry-pick the feature's commit from `local` (never Fork tooling or "Docs:" commits;
resolve any dependency on earlier `local` commits), push to `origin` and open the PR.

## Builds

| Build | For | How | Cost |
|---|---|---|---|
| Iterate | checking a change | `npx tsc --noEmit`; in `src-tauri/`: `cargo check`, `cargo test`, harnesses with `cargo build --profile fast --example NAME` | seconds after an edit |
| Test build | trying a change in the app | `./test-build.sh` → `src-tauri/target/fast/RapidRAW`, runnable in place | ~10 s after an edit |
| Performance | speed numbers | `--release`: fat LTO, one codegen unit | ~8–10 min per build, mostly on one core |

- `--release` changes speed, not pixels, so harnesses and A/B comparisons use
  `--profile fast`.
- `.cargo/config.toml` gives every checkout one shared dependency cache and its own
  `src-tauri/target/` for finished binaries, so concurrent agents never run each
  other's builds. The first build of a new checkout compiles only the app crate (~2 min).
- Cargo takes turns per profile ("Blocking waiting for file lock"), so a build can
  wait on another agent's, and a busy machine skews timings: check `pgrep -a rustc`
  before measuring.

## Testing

- A running RapidRAW holds the single-instance D-Bus name and real settings. Run a
  test build isolated: `XDG_DATA_HOME`, `XDG_CONFIG_HOME` and `XDG_CACHE_HOME` pointed
  at a scratch dir, under `dbus-run-session`. For `tauri dev`, override the identifier:
  `npx tauri dev --config '{"identifier":"io.github.CyberTimon.RapidRAW.dev"}'`.
- Open a **copy** of a photo in a scratch dir: the app writes `.rrdata` sidecars next
  to the image.
- Stop test processes by PID.
- `WAYLAND_DEBUG=client` on the app logs every Wayland request (buffer formats, subsurfaces).

**Harnesses** (`src-tauri/examples/`) build with
`cargo build --profile fast --example NAME` into `target/fast/examples/`.
`render OUTDIR 'ADJUSTMENTS_JSON' imgs…` runs the full GPU pipeline to PNGs
(`'{"toneMapper":"profile"}'` for the camera profile's look; `LONG_EDGE=` sets the size).

**Face auto-exposure harness** (no GUI, ~70 ms/photo; writes before/after PNGs + JSON):
```sh
cd src-tauri && cargo build --profile fast --example face_auto
ORT_DYLIB_PATH=resources/libonnxruntime.so target/fast/examples/face_auto [--agx] OUTDIR imgs…
```

## Background

- Linux preview slowness: upstream hard-disables the direct wgpu renderer on Linux,
  so every preview is a JPEG sent over IPC into WebKitGTK (#1060, tauri#9220).
  The "direct wgpu preview display" commit draws through a Wayland subsurface instead.
  Next speed step: render only the visible region when zoomed in.
- Open face-exposure TODO: contrast, clarity, vibrance and dehaze in Auto still
  meter linear data as if it were display-referred. Face Auto predicts its result
  with its own copy of exposure → highlights → tone mapper (`face_exposure.rs`),
  which can drift from the shader.
- Camera profiles (DCP): render with the profile's look through the `render`
  harness; `face_auto` takes `--profile`.

## Open pipeline issues

- **Scene stages that assume 1.0 is white (principle 1).** Contrast skips values above 1.0,
  HSL clamps saturation, colour grading splits tones at fixed linear 0.1 and 0.5,
  positive vignette and dehaze blend toward 1.0. Harmless with Basic or Profile
  (they clip at 1.0), arbitrary with AgX. Upstream's design; fix one stage at a time,
  relative to a mid-grey anchor (0.18), when a stage is touched anyway.
- **Camera look (principle 4).** The camera JPEG's fixed part (tone curve, colour look,
  baseline brightness) belongs in the Profile rendering, fitted once from RAW+JPEG
  pairs; Match camera keeps only the per-shot part in Exposure and Shadows. A
  per-shot fitted tone curve was tried and dropped: on 30 shots it narrowed the
  median L* gap to the camera JPEG only from 0.75 to 0.56, and most of the curve
  was the same on every shot.
- **Thumbnails clip highlights at 1.0 (principle 2, minor).** The fast decode
  (`raw_processing.rs`) is used only for thumbnails and export-size estimates.
- **Decode clips colours outside Rec.709 (principle 2, negligible).** Measured at ≤0.03% of
  pixels on 25 A7 III photos; the real fix is a wide-gamut working space.
