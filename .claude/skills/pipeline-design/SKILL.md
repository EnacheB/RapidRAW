---
name: pipeline-design
description: Design checklist for RapidRAW pixel processing. MUST be loaded before
  designing or changing any stage - new adjustments, stage order, masks and local
  edits, Auto features, tone mappers, looks and film sims, camera profiles,
  heal/inpaint, decode - and when the user asks whether a design is right.
---

# Pipeline design checklist

The principles are in AGENTS.md ("Goal and pipeline principles"). This checklist
turns them into questions to answer before building. Read the code for where a
stage sits today; the Pipeline panel (`src/utils/pipeline.ts`) lists the stages in order.

Vocabulary: **scene-referred** data is linear light, float, before the rendering
(tone mapper); values can exceed 1.0 and nothing is white yet. **Display-referred**
data is the finished picture after the rendering: 0 is black, 1 is white.

1. **Scene or display?** If the stage models light (exposure, white balance, glow,
   lens blur, haze) or must preserve data (heal/clone), it acts on scene-referred
   data, before the rendering, and loses nothing. Roughly, a rendering maps light to
   screen as `display ≈ light^(1/2.2)`, plus a shoulder that rolls off values
   above 1.0 (Basic clips them instead). If the stage is a taste adjustment on the
   final image (grain, display LUT), it acts after the rendering, and it can't
   recover anything the rendering already clipped. Curves are the deliberate
   exception: they act on scene light, just before the rendering.

2. **Position.** What must run before it? For example, denoise runs before sharpening,
   or sharpening amplifies noise. What does it feed, and does it stay correct when
   any earlier slider moves?

3. **Scope.** Global, smooth local (varies with the image's own brightness or
   neighbourhood, like Shadows/Highlights), or masked (a region the user or AI
   picks)? Global keeps the relationships between areas natural but changes
   everything. Smooth local adapts without hard edges, but it can halo or flatten
   the image. Masks are precise, but they can look pasted-on and depend on mask
   quality. State the choice and why. If the stage can be local, its value must
   blend linearly per mask (principle 6).

4. **Auto: where it measures, where it acts.** Which pipeline point does it measure:
   scene-referred light, or the rendered picture? Why that one? Does the result go
   stale when the user changes other sliders? It acts by setting visible sliders
   (principle 4). Metering data as if it were in the other domain is a known bug
   of this class.

5. **Neighbourhood operations** (blur, clarity, dehaze, sharpening, heal). Which
   pipeline point does the neighbourhood come from? It must match the point the
   stage acts on, or edits made earlier skew the comparison (Dehaze once compared
   an edited pixel with a blur of the unedited image). Is the radius
   resolution-independent, so preview and export match? Does it halo at strong
   edges?

6. **Channels.** Applying a curve or look to R, G and B separately shifts hues in
   saturated colours (skies go cyan, skin goes orange); hue-preserving versions
   change brightness and saturation only. Which does this stage need, and why?

7. **Parity.** The preview must match the export apart from resolution. Fast paths
   (thumbnails, previews) may be slower or smaller, never different.

## Report before building

- The answers above, briefly, for anything not obvious.
- Which principles the design touches, any it bends and why.
- The main alternative rejected and why.
- If it violates a principle, say so plainly first, in words a non-expert can
  follow.
