# ComfyUI — MiniMax H3 Long Shot

Requires the [H3 Prompt Compiler](https://github.com/r34vtraining/H3_Prompt_Compiler) pack (for the Shot and Ref Prompt Builder r2v nodes) and a ComfyUI build with native MiniMax H3, arbitrary-frame guides, and the V3 node API.

Render one continuous shot longer than a single H3 generation. Each **Shot**
node in a chain becomes one generation; they're stitched in latent space and
decoded once, so the joins are seamless. Optional lip sync to a song.

A single Shot works too: it's one ordinary generation with nothing to stitch, so
the same workflow covers any length.

Requires the **comfyui-minimax-h3** prompt pack (for the Shot and Ref Prompt
Builder r2v nodes) and a ComfyUI build with native MiniMax H3, arbitrary-frame
guides, and the V3 node API.

---

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/r34vtraining/H3_Prompt_Compiler
git clone https://github.com/r34vtraining/H3_Longshot
```

Restart ComfyUI. No dependencies. Three nodes appear under **MiniMax H3**: Long Shot, Song Track, and RefMod Carrier.

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

## RefMods and other reference blocks

Long Shot encodes each segment itself, so there's no conditioning wire to
splice a node like **Apply H3 RefMod** into. Use the `extra_refs` input, fed
from **MiniMax H3 RefMod Carrier**:

```
MiniMax H3 RefMod Carrier ─→ [conditioning] Apply H3 RefMod ─→ [extra_refs] Long Shot
```

RefMod needs a conditioning to attach its references to. The Carrier supplies
an empty one, so the only thing reaching `extra_refs` is RefMod's own reference
blocks. Long Shot adds them to every segment, after the native references.

Don't feed RefMod from a Reference to Video node that has references connected.
Long Shot adds every reference block it receives, so those references would go
in twice: once through Long Shot's own inputs and again through `extra_refs`.
Long Shot logs a warning if `extra_refs` comes from anything other than the
Carrier.

**H3 RefMod Step Curve** patches the model rather than the conditioning, so it
goes on the `model` wire before Long Shot as usual.

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

## Tests

`tests/` runs the nodes against a real ComfyUI source tree (CPU mode, no model
weights). Sampling is simulated as a perfect continuation, so the suite proves
the bookkeeping is bit-exact — stitching, audio sync, and song slicing — and
that the model's own `PackedLayout` places every guide on the right frames.

```
set COMFYUI_ROOT=C:\path\to\ComfyUI
python -m pytest tests -q
```

Two optional suites run only when pointed at the other packs:
`MMH3_PROMPT_PACK` (the comfyui-minimax-h3 folder) for the builder hand-off,
and `REFMOD_PACK` (the ComfyUI-MiniMaxH3Mod folder) for the RefMod chain.
