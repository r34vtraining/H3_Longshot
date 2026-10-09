# ComfyUI — MiniMax H3 Long Shot

## What's new in 1.5.1

**Fix:** a run where every Shot is locked or reused no longer stops with "Reference audio is
connected — connect the H3 audio VAE to audio_vae" when reference audio is wired in. The
audio VAE is only needed (and only loaded) when a Shot actually renders.

## What's new in 1.5.0

**Clip Shots (1.5)**
- New nodes **Clip**, **Clip Shot** and **Clip Pixels**: a real video becomes a piece
  of the timeline. It is never sampled; the generated Shots beside it are pinned into
  and out of it (`join = bridge`) or meet it with a hard cut (`join = cut`). Adjacent
  clips always cut.
- **Clip Pixels** swaps the clip's original frames back in after decode / upscale.
- A clip's identity comes from its source, so a re-encoded clip after a restart is
  still the same clip and nothing around it re-renders.

**Timeline mode (1.4)**
- New **Timeline Shot** node (`shot_id`, seconds, text, `shot_seed`, lock, join,
  frames). A chain of Timeline Shots switches the Long Shot node into timeline mode.
- **Re-roll in place:** re-render a middle Shot pinned at both ends; the Shots after it
  keep their takes. The end pin is written back exactly, so the join is bit-exact.
- **Prepend and insert:** add Shots before Shot 1 or between Shots.
- **Takes store:** every render is kept as a take
  (`output/longshot/<cache>/takes/`), with fingerprints so unchanged Shots are reused
  from memory or disk; locked Shots load their take and never re-render.
- **Seams:** every join is checked and reported (`ok`, `cut`, `mismatch`); broken
  joins are left as a hard cut until the next Shot is re-rolled.
- Segments saved by earlier versions are adopted on the first timeline run.
- The model only loads when something actually renders (locked Shots and clips need
  none).

**Housekeeping**
- Tests use neutral sample names; README documents timeline mode, Clip Shots and all
  seven nodes.

---

Render one continuous shot longer than a single H3 generation. Each **Shot**
node in a chain becomes one generation; they're stitched in latent space and
decoded once, so the joins are seamless. Optional lip sync to a song.

A single Shot works too: it's one ordinary generation with nothing to stitch, so
the same workflow covers any length.

Requires the [H3 Prompt Compiler](https://github.com/r34vtraining/H3_Prompt_Compiler) pack (for the Shot and Ref Prompt
Builder r2v nodes) and a ComfyUI build with native MiniMax H3, arbitrary-frame
guides, and the V3 node API.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/r34vtraining/H3_Prompt_Compiler
git clone https://github.com/r34vtraining/H3_Longshot
```

Restart ComfyUI. No extra dependencies (`safetensors` ships with ComfyUI). Seven nodes appear under **MiniMax H3**: Long Shot, Timeline Shot, Clip, Clip Shot, Clip Pixels, Song Track, and RefMod Carrier.

**Updating from an earlier version.** The new widgets (`save_to_disk`, `cache_name`)
come after the existing ones, so saved workflows keep their values. If your saved
Long Shot shows `max` in `reuse_segments` (from the older widget shift), set it back
to on and `ref_image_size` to your choice.

---

## Wiring

```
Shot ─→ Shot ─→ Shot ─→ [shots] Ref Prompt Builder r2v ─→ long_shot ─→ [prompt] MiniMax H3 Long Shot ─→ latent ─→ VAE Decode
                                                                         ↑ model, clip, vae               └→ VAE Decode Audio
                                                                         ↑ noise, sampler, sigmas
```

- **model / clip / vae** — your usual H3 loaders. Apply `ModelSamplingMiniMaxH3`
  to the model as you normally would.
- **noise / sampler / sigmas** — `RandomNoise`, `KSamplerSelect`,
  `BasicScheduler`. The reference setting is euler / beta / 20 steps.
- **prompt** — the `long_shot` output of **MiniMax H3 Ref Prompt Builder r2v**,
  with your Shot chain wired into the builder's `shots` input. Each Shot is one
  segment: its `text` is that segment's prompt and its `seconds` is its length.
  `cut_verb` is ignored — this is one continuous shot.

Everything else on the builder applies to every segment: subject definitions,
task types, summary, retention analysis, style line, soundscape, and music.
Because each segment is its own one-shot generation, any `[Shot N]` mentioned
in those sections becomes `[Shot 1]` — so `(appears in [Shot 1], [Shot 3])`
becomes `(appears in [Shot 1])`. The builder's own `prompt` output still shows
the whole sequence as one prompt, for reference.

With no references connected at all, Long Shot writes each segment in the base
format instead, and only the builder's style line, soundscape, and music apply.

Decode the output **in a single pass** with the native decoders. That single
decode is what makes the joins invisible.

### The settings that matter

**overlap_frames** — hidden context shared across each join (5, 22, 39, 56…).
22 is the proven default. More carries motion better but re-samples more.

**style_line** (on the builder) — restated at the start of every segment. Put
your look here, not in the first Shot, or later segments drift.

**reuse_segments** — on by default. Reuses segments that would come out
identical, so a re-run only renders what changed. See *Re-rolling and building
Shot by Shot* below.

**save_to_disk** / **cache_name** — on by default. Finished segments are also
saved to disk, so a crash or restart doesn't cost finished Shots. See *Saved
segments* below.

**dry_run** — outputs the full plan and every segment's prompt in a second,
without sampling. Each segment costs minutes; check the plan first. Wire the
`plan` output to a text preview node. The `latent` output is blocked during a
dry run, so everything wired after it — decoders, upscalers, video saves — is
skipped instead of producing an empty clip.

---

## Lip sync to a song

```
Load Audio (trimmed) ─┬─→ MiniMax H3 Song Track ─→ song ─→ [song] Long Shot
                      │       ↑ audio_vae
                      └─→ (mux onto the final video)
```

**Use Song Track for lip sync, not `ref_audio`.** A `ref_audio` input is a
reference — every segment receives the whole clip as guidance for how voices
or music should sound, with no position on the timeline. That's right for a
voice timbre. Song Track instead gives each segment only its own slice of the
song, pinned to that segment's frames, which is what keeps the mouth in step
with the song across a long chain.

Trim the song in your audio loader, to start where the video starts. Song Track
encodes it **once**, as-is. The clip only needs to cover the Shots' total
length; Long Shot tells you if it falls short. Long Shot then gives each segment its exact slice of that encoding,
pinned to the segment's timeline — the mechanism H3 uses for lip sync.

What happens under the hood:

- With a song connected, the song is the only source of audio. The continuation
  guide carries video only, so two audio guides never compete over the same frames.
- The song is sliced as **latents**, not waveform. Cutting and encoding the
  waveform per segment would add edge artifacts at every join.
- The output latent's audio is replaced by the song itself, token for token.
  For the cleanest result, mux the loader's trimmed audio onto the decoded
  video instead of using the decoded audio.

### Writing the lyrics

Each Shot needs the lyrics sung **during its window** in `<d>` tags. The plan
output tells you exactly which part of the song each Shot covers:

```
Segment 2: asked 6.5s, got 6.38s · shows 00:07.3–00:13.7 · window 175f (22f hidden overlap)
  lyrics for this Shot: song 00:06.4–00:13.7 (includes 0.92s overlap)
```

The range includes the hidden overlap, because the pinned audio spans the
whole window — so the text should too. Times count from the start of the
clip Song Track receives, so add your loader's start time to find the same
spot in the full song.

---

## Re-rolling and building Shot by Shot

Long Shot remembers every segment it renders. On the next run, any segment that
would come out identical is reused instead of sampled, so only the Shots you
changed, and the ones after them, render again.

- **Build it up a Shot at a time.** Render Shots 1–2. Add Shot 3 and queue:
  Shots 1–2 are reused and only Shot 3 renders. A segment depends only on the
  segments before it, never on how many come after.
- **Re-roll one Shot.** Set that Shot's `shot_seed` (on the Shot node) to any number.
  `-1` follows Long Shot's seed: base + Shot number − 1 with `seed_mode`
  increment. Only that Shot and the ones after it render again.
- **Change a middle Shot** and it renders, along with every Shot after it,
  since they continue from its ending.
- **Interrupted?** Queue again. The segments that finished are kept.

Set **RandomNoise** to *fixed* (its control after generate). On *randomize*,
the base seed changes every queue and nothing can be reused.

Each segment line in the plan says what will happen, dry run included:

```
Segment 4: asked 5s, got 4.96s · … · seed 1683 · reused (disk)
Segment 5: asked 9s, got 9.04s · … · seed 1684 · reused (memory)
Segment 6: asked 7s, got 7.04s · … · seed 77 (Shot seed) · will render — seed changed
```

The reasons: *prompt changed*, *seed changed*, *length changed*, *references
changed*, *song changed*, *model, CLIP or VAE changed* (a new LoRA, a reloaded
model), *sampler, sigmas or size changed*, *follows a changed segment*, or
*first run*.

Segments are kept in RAM until ComfyUI restarts, up to the 64 most recent, and
on disk (below). Turn `reuse_segments` off to render everything fresh and store
nothing.

---

## Timeline mode: re-roll in place, add Shots before Shot 1 (1.4)

Chain **MiniMax H3 Timeline Shot** nodes instead of MiniMax H3 Shot and the
chain becomes a **timeline of pieces** instead of a one-way chain. Each
Timeline Shot carries an id, an optional locked take, a join mode and a length
to keep. H3 Long Shot Studio builds these for you.

- **Every finished Shot is kept as a take:**
  `output/longshot/<cache_name>/takes/<shot_id>__<seed>__<fp8>.safetensors`
  (the full window, video and audio, fp16).
- **A locked Shot** (`lock` = a take file name) loads its take instead of
  sampling. If the file is gone, the run stops with "Shot N's approved take is
  missing; re-render it or unlock it".
- **A Shot that renders is pinned on both sides:**
  - its start is guided to the tail of the Shot before it, as in round 1, and
    that shared zone is dropped afterwards;
  - when the Shot after it is locked, its last `overlap_frames` are also guided
    to a fixed slice at `window − overlap`. That slice is its own old tail when
    the next Shot was continued from it, otherwise the next Shot's head.
  - After sampling, the end slice is **written back exactly** (video and
    audio), so the locked Shot after it stays bit-identical.
- **Re-roll in place.** Lock every Shot but one and give it a new seed (keep
  its `frames`). It renders once, pinned to its neighbours, and the timeline
  and song timing are unchanged.
- **Add a Shot before Shot 1.** The new Shot is end-pinned to old Shot 1's
  first frames; old Shot 1 now drops those frames, which appear unchanged as
  the new Shot's tail.
- **Shared zones.** Every boundary has an `overlap_frames` zone owned by the
  left Shot; the right Shot drops its first `overlap_frames`. So any take fits
  anywhere: one rendered first drops its head when something goes before it,
  and the total is always 17n + 5 frames.
- **Fingerprints by pins.** A rendered Shot's fingerprint is its own inputs
  (prompt, seed, length, references, song slice, model recipe) plus what it is
  pinned to. Moving Shots around never re-renders an unchanged one; changing
  what it is pinned to does.
- **Seams.** The plan reports a *hard cut* where a locked Shot no longer
  follows the Shot before it (for example after removing the Shot in between).
  Re-rolling that Shot bridges the join. `join = cut` makes a join an
  intentional hard cut: no start pin.
- **Audio grid.** Audio runs at 40 Hz, so a take moved along the timeline
  can get a slot one token (25 ms) longer or shorter; Long Shot trims or
  repeats one edge token. With a song connected, the song's slice rides on the
  start guide and the end pin carries video only.
- **Round 1 saved segments** are adopted as takes on the first timeline run, as
  long as the timeline is laid out as it was.

### Clip Shots: real video in the timeline (1.5)

A piece of the timeline can be a real video clip instead of a generation:

```
VHS Load Video (force_rate 24) → MiniMax H3 Clip → MiniMax H3 Clip Shot → (next Shot…)
```

- **MiniMax H3 Clip** resizes and centre-crops the frames to the film's size,
  trims them to a valid length (17k + 5 frames, snapped down), and encodes them with
  the H3 video VAE, and their sound (or silence) with the H3 audio VAE.
- **MiniMax H3 Clip Shot** puts the clip in the Shot chain like a Shot. It is
  never sampled.
- **Joins.** With `join = bridge` (the default), the generated Shot before the clip is
  end-pinned to the clip's first frames, and the Shot after it is start-pinned to its
  last frames, so they flow into and out of it. With `cut`, they meet with a hard cut.
  Two clips next to each other always cut.
- **The shared zone applies to clips too.** A clip that isn't first drops its first
  `overlap_frames` (0.92 s at 22): on a bridge, that's where the Shot before it
  arrives; on a cut, those frames are hidden.
- **Original pixels.** By default the clip is decoded from its latent like everything
  else (seamless joins, slightly softer). **MiniMax H3 Clip Pixels**, wired between
  VAE Decode and the video save (with Long Shot's latent), puts each clip's original
  frames back for its span: sharper, with a possible faint seam at its edges.
- **Sound.** The clip's sound plays in its span. With a song connected (lip sync),
  the song replaces the whole soundtrack, clips included.
- A clip must plausibly match the film (same character, similar look and light): H3
  bridges motion and continuity, not unrelated footage.

**Quality notes**

- A Shot pinned at both ends has to travel from its start to a fixed end state.
  Give it at least 5 s, and write the arrival into its text ("…ends with her at
  the doorway, facing screen left"). If the two states are too different, the
  last second may visibly morph.
- End pins are 22-frame guide clips, a less-tested shape than a single last
  frame. Check them early on real renders; the fl2va model is expected to
  behave best.
- The first frames after a pinned join may decode very slightly differently
  (the decoder looks back a few frames). This is expected to be invisible.

---

## Saved segments (crash-proof resume)

Every finished segment is also written to

```
ComfyUI/output/longshot/<cache_name>/segments/<fingerprint>.safetensors
```

When you queue again, after a crash, a restart or days later, Long Shot looks in RAM,
then on disk, and only samples what's in neither. You never point at a file;
just queue again.

| reuse_segments | save_to_disk | Behaviour |
|---|---|---|
| on | on (default) | memory + disk |
| on | off | memory only |
| off | — | nothing is reused or stored |

- **cache_name** names the folder (letters, digits, `. _ -`; anything else becomes `_`).
  Use one per project. H3 Long Shot Studio sets it to the project's slug.
- **Files.** Each file holds the segment's video and audio latents in fp16 (cast back
  on load), plus metadata: segment, seconds, seed, shapes, Long Shot version and
  creation time. That's about 1 MB per second of video at 0.6 MP and about 1.5 MB/s at
  0.9 MP. Delete the folder whenever you like.
- **Safe writes.** Writes go to a `.tmp` file first and are renamed when complete, so a
  crash mid-write never leaves a file that looks valid. Leftover `.tmp` files are
  removed on the next run. A damaged or wrong-shape file is ignored, and that
  segment renders again.

### What makes a segment "the same" after a restart

The in-memory check identifies the model by object identity, which no restart
survives. For disk, Long Shot fingerprints the **recipe** instead:

- It follows its `model`, `clip`, `vae`, `audio_vae`, `sampler` and `sigmas` wires
  upstream through the workflow and hashes every node's class and settings. That
  includes model file names, LoRA strengths, shifts, Sage settings, scheduler and steps.
- Node ids aren't part of it, so a rebuilt or renumbered graph matches.
- Each model file's **size and modified time** go in with its name. Replacing a file
  under the same name counts as a change; moving your models to another drive costs
  one re-render.
- Seeds, prompts, lengths, references (by content), the song slice and first/last
  frames were already restart-proof.

If a run comes from outside ComfyUI's executor (no workflow to read), segments stay
in memory only, and the log says so.

### The model only loads when something renders

`model`, `clip`, `vae`, `audio_vae` and `sigmas` are *lazy* inputs. Long Shot works
out the plan first and asks ComfyUI to load them only if a segment actually needs
sampling. A dry run, or a re-run where every segment comes from memory or disk,
never loads the model. After a restart you can re-decode or check a finished
chain in seconds.

---

## Timing

H3 moves in 17-frame steps (~0.7s), so durations snap to the grid. Long Shot
snaps each segment's **end point** on the global timeline rather than its
length. Rounding therefore never accumulates: every boundary lands within
~0.35s of where you put it, however many segments you chain. This is what
keeps a long music video in step with its song.

H3 is trained on windows of roughly 124–362 frames (5–15s). The plan flags any
segment outside that range.

---

## References

Long Shot has the same growing reference inputs as the native **MiniMax H3
Reference to Video** node — connect one and the next slot appears:

| Inputs | Up to | Label |
|---|---|---|
| `ref_image_0`, `ref_image_1`… | 9 | `<Picture 1>`, `<Picture 2>`… |
| `ref_video_0`… | 3 | `<Video 1>`… |
| `ref_video_audio_0`… | 3 | soundtrack of the same-numbered video |
| `ref_audio_0`… | 3 | standalone audio |

They go to the native node exactly as connected, so labels are numbered exactly
as the native node numbers them. Audio labels count **video soundtracks first,
then standalone audio**: with `ref_video_audio_0` connected, it's `<Audio 1>`
and `ref_audio_0` becomes `<Audio 2>`. A soundtrack only counts when its
same-numbered video is connected too.

Connect the audio VAE to `audio_vae` whenever any reference audio is connected.

References ride through **every** segment, which is what keeps a character
consistent across the joins. The text encoder runs per segment, but the VAE
encodes of the references happen once per window length and are reused, so a
reference video isn't re-encoded on every segment.

`first_frame` / `last_frame` use the image-to-video path instead, so they can't
be combined with reference inputs.

---

## RefMods

Connect **Load H3 RefMods** straight to Long Shot's `refmods` input:

```
Load H3 RefMods ─→ mods ─→ [refmods] MiniMax H3 Long Shot
```

Long Shot does what **H3 RefMod Text Encode** does, on every segment. Each
RefMod is shown to the text encoder under its own label, and its reference
goes to the model in the same order. That pairing is what lets a line in your
subject definitions, like `<hero> … whose appearance comes from <Picture 2>`,
actually point at the RefMod.

RefMods take the next free labels after your reference inputs. With one
`ref_image` connected, the first image RefMod is `<Picture 2>`. Only inputs
that actually arrive are numbered: a muted or bypassed loader takes no label,
and slot names don't matter. The `plan` output lists every live reference at
the top, dry run included, so you can write your Subject boxes to match:

```
Reference labels — use these in your subject definitions:
  <Picture 1> = ref_image_0
  <Picture 2> = hero_face (RefMod)
  <Video 1> = hero_walk (RefMod)
```

A RefMod bundle with several members gets one label per member.

Each row's strength from the loader applies, and a row at 0 is left out
entirely, label included. Visual RefMods are decoded for the text encoder once
per run, not once per segment, so connect the H3 video VAE. RefMods count as
references, so they can't be combined with `first_frame` / `last_frame`.

**H3 RefMod Step Curve** patches the model rather than the conditioning, so it
goes on the `model` wire before Long Shot as usual. It still finds the RefMods.

### extra_refs: Apply H3 RefMod without labels

```
MiniMax H3 RefMod Carrier ─→ [conditioning] Apply H3 RefMod ─→ [extra_refs] Long Shot
```

`extra_refs` takes reference blocks only. Use it when you want Apply H3
RefMod's retention, curve, or scramble controls. Apply never shows a RefMod to
the text encoder, so these blocks have **no label**: nothing in your prompt can
point at them, and they act as unlabelled guidance. For a RefMod your prompt
refers to, use `refmods`.

The Carrier supplies the empty conditioning Apply needs, so only RefMod's own
blocks reach `extra_refs`. Don't feed Apply from a Reference to Video node that
has references connected, or those references go in twice. Long Shot warns
when that happens, and also when the same RefMods seem to be connected to both
`refmods` and `extra_refs`.

---

## Limits

- **The joins are structurally seamless, not semantically guaranteed.** The
  latent stitch is exact, but whether the action reads as continuous still
  depends on the model, prompts, and seed.
- A segment opening mid-action restates that segment's Shot text. If a Shot
  describes a one-time action, keep it inside one segment.
- The stitched latent grows with every segment and is decoded at the end, so
  long pieces need VAE memory for the full length.

---

## How the stitch works

H3 video tokens come in groups of five covering 17 frames, each group opening
with a 1-frame token — which is why frame counts must be 17k+5. With a 17k+5
overlap, each continuation's guide is a slice of the previous latent starting
exactly on a group boundary, and the new tokens complete the previous clip's
final partial group. One group straddles each join, so a single decode sees one
continuous causal sequence. Audio boundaries are computed from global frame
position, never accumulated, so they can't drift.

Approach after [ttulttul/ComfyUI-Minimax-H3-Continuation](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation),
which uses only ComfyUI's native guide API — no monkey-patching, so it
survives ComfyUI updates.

---

## Front-end hooks (H3 Long Shot Studio)

Long Shot also reports its work for front ends like H3 Long Shot Studio. None of this
changes what it renders.

- **Plan in `/history`.** The node returns its plan as a UI output: `text` holds the
  same text as the `plan` output, and `plan_json` holds one row per segment with
  `{index, seconds, frames, start, end, start_frame, window_frames, seed, own_seed,
  status: "reused" | "render", reason}`. Dry runs include it too. On ComfyUI builds that
  support `has_intermediate_output`, the plan is resent when a whole run is served from
  ComfyUI's cache.
- **Progress over the websocket.** For each segment, Long Shot sends `mmh3.longshot` with
  `{segment, of, status: "reused" | "rendering" | "done", seed, seconds}` (plus
  `source: "memory" | "disk"` on reused segments) to the client that queued the prompt.
- Plan rows carry `source: "memory" | "disk"` for reused segments too.
- **Timeline mode** rows add `{kind, id, locked, take, join, seam: "ok" | "mismatch" | "cut",
  pins: {start: {from}, end: {to, kind: "old_tail" | "head", take}}}`; locked Shots have
  `status: "locked"` and progress `source: "take"`.

---

## Tests

`tests/` runs the nodes against a real ComfyUI source tree (CPU mode, no model
weights). Sampling is simulated as a perfect continuation, so the suite proves
the bookkeeping is bit-exact — stitching, audio sync, and song slicing — and
that the model's own `PackedLayout` places every guide on the right frames.

```
set COMFYUI_ROOT=C:\path\to\ComfyUI
python -m pytest tests -q
```

`test_timeline.py` covers timeline mode:

- planner grid invariants for random mixes of fixed and requested pieces (totals
  17n + 5, every shared zone on a group boundary, song slices by global position);
- without locks, a timeline renders exactly like a round-1 chain;
- re-roll in place: one sampler call, every other Shot bit-identical, the end pin
  is the old tail and is written back exactly even when the sampler drifts;
- prepend keeps old Shot 1's visible frames; fully locked runs sample nothing;
- moving pieces re-renders nothing unchanged; changing a pin source does;
- missing take error, take switch with zero sampler calls, hard-cut reporting,
  cut joins, size/overlap checks, dry runs from metadata only, lazy model loading,
  song guides, round-1 segment adoption, and the ±1-token audio fit.

`test_frontend_hooks.py` covers the plan UI output, `plan_json` and the progress events,
including a cached re-run through the real executor. `test_disk_segments.py` covers
saved segments:

- recipe fingerprints are identical with renumbered node ids, and change with any
  upstream setting or a replaced model file;
- save → restart → zero sampler calls, with exactly the fp16-rounded latent;
- a resumed chain continues within fp16 rounding;
- `.tmp` and corrupt files are handled;
- memory-only and off modes;
- through the real executor: a dry run and a fully reused run never load the model.

Two optional suites run only when pointed at the other packs:
`MMH3_PROMPT_PACK` (the comfyui-minimax-h3 folder) for the builder hand-off,
and `REFMOD_PACK` (the ComfyUI-MiniMaxH3Mod folder) for the RefMod chain.
