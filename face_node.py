"""MiniMax H3 Face Refine (1.8.0): faces too small to come out well, redrawn by H3 as
steady close-ups and blended back in. The face finding / following / cropping is in
faces.py; this is the ComfyUI node and the H3 resampling."""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch

from . import faces
from . import nodes as _n
from . import store as disk

logger = logging.getLogger("MiniMaxH3LongShot")

FACE_FOLLOW = ("auto", "fixed", "follow")
FACE_WHICH = ("all", "largest")
AUDIO_PER_FRAME = 40.0 / 24.0          # H3's sound latent runs at 40 Hz, the picture at 24 fps
_FACE_PLANS = OrderedDict()            # (images identity, settings) -> plan, from the lazy check to the run


def face_prompt(with_ref=False, label="the person", describe=""):
    """The close-up's text: a face filling the frame, keeping its expression and mouth
    movement. with_ref: ref2va's six sections, with <Picture 1> as the face's identity."""
    who = (describe or "").strip().rstrip(".")
    look = f" {who[0].upper() + who[1:]}." if who else ""
    action = ("The face keeps exactly the same expression, head movement, eye movement and mouth "
              "movement as before, in time with the sound. Sharp natural detail in the eyes, eyelashes, "
              "skin texture, lips and teeth. The camera is static.")
    sound = "overall_soundscape: The person's own voice and the room around them, unchanged.\nnon_diegetic_music: N/A"
    if not with_ref:
        return ("integrated_multimodal_description: [Shot 1] Cinematic, live-action, realistic. "
                f"An extreme close-up of a person's face filling most of the frame.{look} {action}\n" + sound)
    name = (label or "").strip() or "the person"
    return (f"subject_definitions: <Subject 1> is {name}, whose face and appearance come from <Picture 1>.\n"
            "summary: [reference generation] An extreme close-up of <Subject 1>'s face.\n"
            "retention_analysis: <Subject 1> (appears in [Shot 1]): fully_preserved - face, identity, "
            "hairstyle and skin tone.\n"
            "detailed_description: Cinematic, live-action, realistic. [Shot 1] An extreme close-up of "
            f"<Subject 1>'s face filling most of the frame.{look} {action}\n" + sound)


def detector_folder():
    """models/facedetect/mtcnn (or a facedetect folder from extra_model_paths.yaml)."""
    import folder_paths
    if "facedetect" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["facedetect"] = (
            [os.path.join(folder_paths.models_dir, "facedetect")], set())
    roots = folder_paths.get_folder_paths("facedetect") or [os.path.join(folder_paths.models_dir, "facedetect")]
    for r in roots:
        d = os.path.join(r, "mtcnn")
        if faces.weights_present(d):
            return d
    return os.path.join(roots[0], "mtcnn")


def face_store(cache_name):
    """<output>/longshot/<project>/faces: redrawn close-ups (latents), reused while the
    picture under them, their sound and the settings stay the same."""
    if not cache_name:
        return None
    return os.path.join(os.path.dirname(disk.TakeStore(cache_name, _n.STORE_ROOT).folder), "faces")


def skip_ranges(text):
    """'[[start, end], ...]' (frames, end exclusive) the face pass leaves alone."""
    try:
        rows = json.loads(text) if str(text or "").strip() else []
    except ValueError:
        return []
    out = []
    for r in rows if isinstance(rows, list) else []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            out.append((a, b))
    return out


def audio_slice(latent, start, frames):
    """The sound latent under frames start .. start+frames-1 (zeros past its end), or None."""
    if latent is None:
        return None
    lat = latent.get("samples") if isinstance(latent, dict) else None
    if lat is None or not getattr(lat, "is_nested", False):
        return None
    a = lat.tensors[1]
    n_tok = round(frames / 24.0 * 40.0)
    a0 = int(round(start * AUDIO_PER_FRAME))
    piece = a[..., a0:a0 + n_tok]
    if piece.shape[-1] < n_tok:
        piece = torch.cat([piece, torch.zeros(*piece.shape[:-1], n_tok - piece.shape[-1],
                                              dtype=a.dtype, device=a.device)], -1)
    return piece.clone()


class MiniMaxH3FaceRefine:
    """1.8.0: faces too small to come out well are redrawn by H3 as close-ups.

    Each face is found and followed over the frames, cut out as one steady square
    (fixed, or slowly following a moving face), enlarged to work_size and resampled
    with H3 at low denoise (the sigmas). The sound of those frames is held while the
    picture is redrawn, so the mouth is drawn against the real sound. The close-up is
    shrunk back and blended in under a soft face mask, its broad colour matched to the
    original. Nothing else in the picture changes.

    Without a ref_image the close-up is conditioned like t2va / fl2va (text only); with
    one (the ref2va model) that picture is <Picture 1>, the face's identity. A long
    video is done in windows of up to max_window frames that crossfade."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "model": ("MODEL", {"lazy": True, "tooltip": "The redraw model with its model sampling "
                                "(a low video shift, e.g. 4) and LoRAs: the fl2va model by default."}),
            "clip": ("CLIP", {"lazy": True}),
            "vae": ("VAE", {"lazy": True}),
            "noise": ("NOISE", {"lazy": True}),
            "sampler": ("SAMPLER", {"lazy": True}),
            "sigmas": ("SIGMAS", {"lazy": True, "tooltip": "A BasicScheduler with denoise below 1 "
                                  "(about 0.3): lower keeps the face and mouth closer to what they were."}),
        }, "optional": {
            "latent": ("LATENT", {"tooltip": "The take or film these frames were decoded from: its "
                                  "sound is held while the faces are redrawn."}),
            "ref_image": ("IMAGE", {"tooltip": "The face's identity picture, for the ref2va model."}),
            "text": ("STRING", {"default": "", "multiline": True, "tooltip": "The close-up's prompt. "
                                "Empty: a standard one (a face filling the frame, same expression)."}),
            "max_face": ("INT", {"default": 150, "min": 8, "max": 4096, "tooltip": "Faces shorter than "
                                 "this (pixels) are redrawn; bigger ones are left alone."}),
            "which": (list(FACE_WHICH), {"default": "all"}),
            "work_size": ("INT", {"default": 640, "min": 256, "max": 1536, "step": 32,
                                  "tooltip": "The close-up's size while it's redrawn (square)."}),
            "padding": ("FLOAT", {"default": 0.8, "min": 0.2, "max": 3.0, "step": 0.05,
                                  "tooltip": "Room around the face on each side, as a share of its size."}),
            "follow": (list(FACE_FOLLOW), {"default": "auto", "tooltip": "auto: the crop follows a face "
                                           "that moves, else it stays put."}),
            "mask_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.5, "step": 0.05}),
            "feather": ("FLOAT", {"default": 0.3, "min": 0.02, "max": 1.0, "step": 0.01}),
            "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "hold_audio": ("BOOLEAN", {"default": True}),
            "max_window": ("INT", {"default": 362, "min": 22, "max": 1000, "tooltip": "Longest piece "
                                   "redrawn in one go (frames); longer videos go in crossfading windows."}),
            "skip": ("STRING", {"default": "", "tooltip": "[[start, end], ...] frames to leave alone."}),
            "cache_name": ("STRING", {"default": "", "tooltip": "The project (Long Shot's cache_name): "
                                      "redrawn close-ups are kept there and reused."}),
            "cache_key": ("STRING", {"default": "", "tooltip": "What identifies the redraw model and its "
                                     "settings (H3 Long Shot Studio sets it)."}),
            "detect_every": ("INT", {"default": 2, "min": 1, "max": 24}),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3"

    # --- the plan: faces, tracks, crops (no model needed) ---------------------------
    @staticmethod
    def _settings(kw):
        return (int(kw.get("max_face", 150)), str(kw.get("which", "all")),
                round(float(kw.get("padding", 0.8)), 4), str(kw.get("follow", "auto")),
                int(kw.get("max_window", 362)), str(kw.get("skip", "")), int(kw.get("detect_every", 2)))

    @classmethod
    def plan(cls, images, kw, device=None):
        key = (id(images), tuple(images.shape), cls._settings(kw))
        got = _FACE_PLANS.get(key)
        if got is not None:
            return got
        n, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        if device is None:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        det = faces.detector(detector_folder(), device)
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(100)
        except Exception:
            pass

        def tick(k, of):
            if pbar is not None:
                pbar.update_absolute(int(100 * k / max(of, 1)), 100)
        dets = faces.detect_frames(det, images, stride=int(kw.get("detect_every", 2)), min_face=12, progress=tick)
        tracks = faces.build_tracks(dets, n)
        wins = faces.windows(n, int(kw.get("max_window", 362)))
        skip = skip_ranges(kw.get("skip", ""))
        jobs = []
        for k, (s, f) in enumerate(wins):
            for c in faces.plan_crops(tracks, s, f, max_face=float(kw.get("max_face", 150)),
                                      which=str(kw.get("which", "all")), padding=float(kw.get("padding", 0.8)),
                                      follow=str(kw.get("follow", "auto"))):
                if skip:
                    fr = np.arange(s, s + f)
                    for a, b in skip:
                        c.presence = np.where((fr >= a) & (fr < b), 0.0, c.presence)
                    if not (c.presence > 0).any():
                        continue
                jobs.append((k, c))
        plan = {"n": n, "H": H, "W": W, "tracks": tracks, "windows": wins, "jobs": jobs}
        _FACE_PLANS[key] = plan
        while len(_FACE_PLANS) > 4:
            _FACE_PLANS.popitem(last=False)
        return plan

    @staticmethod
    def job_key(plan, i, images, kw, noise):
        """A redraw's file name: the picture under the crop, its sound, where the crop is,
        the prompt, the seed and the redraw model's settings (cache_key)."""
        k, c = plan["jobs"][i]
        s, f = plan["windows"][k]
        win = images[s:s + min(f, plan["n"] - s)]
        audio = audio_slice(kw.get("latent"), s, f)
        ref = kw.get("ref_image")
        return "f" + _n._digest([
            str(kw.get("cache_key", "")), str(kw.get("text", "")), s, f, round(float(c.side), 1),
            round(float(np.median(c.cx)), 1), round(float(np.median(c.cy)), 1), bool(c.follow),
            int(kw.get("work_size", 640)), bool(kw.get("hold_audio", True)), int(getattr(noise, "seed", 0) or 0),
            faces.region_hash(win, faces._slice_crop(c, 0, win.shape[0])),
            _n._digest(audio) if audio is not None else "", _n._digest(ref) if ref is not None else ""])[:24]

    def check_lazy_status(self, images, model=None, clip=None, vae=None, noise=None, sampler=None,
                          sigmas=None, **kw):
        have = {"model": model, "clip": clip, "vae": vae, "noise": noise, "sampler": sampler, "sigmas": sigmas}
        if not kw.get("enabled", True):
            return []
        try:
            plan = self.plan(images, kw)
        except FileNotFoundError:
            return []                                   # run() reports it
        if not plan["jobs"]:
            return []
        need = ["vae", "noise"]
        folder = face_store(kw.get("cache_name", ""))
        # a kept redraw counts only with its seed, so the noise comes first
        cached = (folder is not None and bool(kw.get("cache_key")) and noise is not None and all(
            os.path.isfile(os.path.join(folder, self.job_key(plan, i, images, kw, noise) + ".safetensors"))
            for i in range(len(plan["jobs"]))))
        if noise is not None and not cached:
            need += ["model", "clip", "sampler", "sigmas"]
        return [x for x in need if have[x] is None]

    # --- the run -------------------------------------------------------------------
    def run(self, images, model=None, clip=None, vae=None, noise=None, sampler=None, sigmas=None, **kw):
        report = {"frames": int(images.shape[0]), "width": int(images.shape[2]), "height": int(images.shape[1]),
                  "windows": [], "tracks": [], "crops": [], "treated": 0}
        if not kw.get("enabled", True):
            report["note"] = "off"
            return self._out(images, report)
        try:
            plan = self.plan(images, kw)
        except FileNotFoundError as err:
            raise ValueError(str(err)) from None
        report["windows"] = [list(w) for w in plan["windows"]]
        report["tracks"] = [t.as_dict() for t in plan["tracks"]]
        if not plan["jobs"]:
            report["note"] = "no small faces"
            return self._out(images, report)
        if vae is None:
            raise ValueError("MiniMax H3 Face Refine needs the video VAE.")
        ref = kw.get("ref_image")
        text = str(kw.get("text") or "").strip() or face_prompt(with_ref=ref is not None)
        work = max(256, int(kw.get("work_size", 640)) // 32 * 32)
        n = plan["n"]
        weights = faces.window_weights(plan["windows"], n)
        folder = face_store(kw.get("cache_name", ""))
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        out = images.clone()
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(len(plan["jobs"]))
        except Exception:
            pass
        for i, (k, c) in enumerate(plan["jobs"]):
            mm.throw_exception_if_processing_interrupted()
            s, f = plan["windows"][k]
            real = min(f, n - s)
            win = images[s:s + real]
            if real < f:                                    # padded to 17k+5: the last frame repeats
                win = torch.cat([win, win[-1:].expand(f - real, -1, -1, -1)], 0)
            crop = faces.extract(win, c, work, dev)
            name = self.job_key(plan, i, images, kw, noise)
            keep = folder is not None and bool(kw.get("cache_key"))
            v = self._load(folder, name) if keep else None
            cached = v is not None
            if v is None:
                if model is None or clip is None or sigmas is None or sampler is None or noise is None:
                    raise ValueError("MiniMax H3 Face Refine needs the model, CLIP, noise, sampler and sigmas.")
                v = self.redraw(model, clip, vae, noise, sampler, sigmas, crop, audio_slice(kw.get("latent"), s, f),
                                text, f, work, ref, bool(kw.get("hold_audio", True)), i)
                if keep:
                    self._save(folder, name, v)
            dec = vae.decode(v)
            if dec.dim() == 5:
                dec = dec.reshape(-1, *dec.shape[-3:])
            dec = dec[:f, ..., :3].float().clamp(0, 1).to(dev)
            if dec.shape[1] != work or dec.shape[2] != work:
                dec = torch.nn.functional.interpolate(dec.movedim(-1, 1), size=(work, work), mode="bicubic",
                                                      align_corners=False).movedim(1, -1).clamp(0, 1)
            dec = faces.match_color(dec, crop, float(kw.get("color_match", 1.0)))
            rect, delta = faces.paste_delta(win[:real], dec[:real], faces._slice_crop(c, 0, real),
                                            scale=float(kw.get("mask_scale", 1.0)),
                                            feather=float(kw.get("feather", 0.3)), device=dev)
            if delta is not None:
                x0, y0, x1, y1 = rect
                wt = torch.as_tensor(weights[k][:real], dtype=delta.dtype, device=delta.device)[:, None, None, None]
                out[s:s + real, y0:y1, x0:x1] += (delta * wt).to(out.device, out.dtype)
            row = c.as_dict()
            row.update(window=k, cached=cached)
            report["crops"].append(row)
            del crop, dec, delta
            if pbar is not None:
                pbar.update(1)
        out.clamp_(0, 1)
        report["treated"] = len({c.track.id for _k, c in plan["jobs"]})
        return self._out(out, report)

    @staticmethod
    def redraw(model, clip, vae, noise, sampler, sigmas, crop, audio, text, frames, work, ref_image,
               hold_audio, index):
        """The close-up resampled from where it is: its latent, partly noised (the sigmas),
        with the sound held as a guide over the whole window."""
        from comfy.nested_tensor import NestedTensor
        h3, _cs = _n._native()
        v0 = vae.encode(crop[..., :3])
        if v0.dim() == 4:
            v0 = v0.unsqueeze(0)
        if ref_image is not None:
            positive = _n._unwrap(h3.MiniMaxH3ReferenceToVideo.execute(
                clip, text, work, work, frames, ref_image_size="match", vae=vae, audio_vae=None,
                ref_images={"ref_image_0": ref_image[:1]}))[0]
        else:
            positive = _n._unwrap(h3.MiniMaxH3ImageToVideo.execute(clip, vae, text, work, work, frames))[0]
        if audio is None:
            audio = torch.zeros(1, 32, 2, round(frames / 24.0 * 40.0))
        elif hold_audio:
            positive = _n._set_keyframes(positive, _n._keyframes(positive) +
                                         [{"resolved_frame_index": 0, "audio_latent": audio.clone()}])
        seed = getattr(noise, "seed", None)
        nz = _n._segment_noise(noise, (int(seed) + index) & 0xFFFFFFFFFFFFFFFF) if seed is not None else noise
        out = _n._sample_window(model, nz, sampler, sigmas, positive,
                                {"samples": NestedTensor((v0.float(), audio.float().to(v0.device)))})
        return out["samples"].tensors[0]

    @staticmethod
    def _load(folder, name):
        path = os.path.join(folder, name + ".safetensors")
        if not os.path.isfile(path):
            return None
        try:
            from safetensors.torch import load_file
            return load_file(path)["video"].float()
        except Exception:
            return None

    @staticmethod
    def _save(folder, name, v):
        try:
            os.makedirs(folder, exist_ok=True)
            from safetensors.torch import save_file
            tmp = os.path.join(folder, name + ".tmp")
            save_file({"video": v.detach().to("cpu", torch.float16).contiguous()}, tmp)
            os.replace(tmp, os.path.join(folder, name + ".safetensors"))
        except Exception as err:            # a full disk mustn't cost the run
            logger.warning("MiniMax H3 Face Refine: couldn't keep a redrawn face: %s", err)

    @staticmethod
    def _out(images, report):
        text = json.dumps(report)
        return {"ui": {"face_report": [text]}, "result": (images, text)}


class _Faces:
    """What faces.face_mask needs: a face box per frame and how present it is."""

    def __init__(self, face, presence):
        self.face, self.presence = face, presence


def _boxes_per_frame(rows, frames, sx=1.0, sy=1.0):
    """Face boxes [frames,4] (cx, cy, w, h) interpolated from report rows [frame, cx, cy, w, h]
    (every few frames), and presence (1 inside the rows' span, fading 6 frames outside)."""
    rows = sorted((r for r in rows or [] if len(r) >= 5), key=lambda r: r[0])
    if not rows:
        return None, None
    f = np.array([r[0] for r in rows], np.float64)
    grid = np.arange(frames, dtype=np.float64)
    box = np.stack([np.interp(grid, f, [r[1] * sx for r in rows]), np.interp(grid, f, [r[2] * sy for r in rows]),
                    np.interp(grid, f, [r[3] * sx for r in rows]), np.interp(grid, f, [r[4] * sy for r in rows])], 1)
    d = np.maximum(f[0] - grid, grid - (f[-1] + 3))
    presence = np.clip(1.0 - np.maximum(d, 0) / 6.0, 0.0, 1.0)
    return box, presence


class MiniMaxH3PasteFaces:
    """1.8.1: the faces of approved Face refine clips put into a decoded film, each over
    its Shot's frames. Only the area around each redrawn face changes (a soft mask a
    little bigger than the redraw's own), so the rest of the film stays as it was decoded.

    items: JSON [{file, at, from, frames, w, h, boxes: [[frame, cx, cy, w, h], ...]}]:
    file is in the output folder; its frames from .. from+frames-1 go over the film's
    frames at .. at+frames-1; boxes are the redrawn faces in the clip's own frames and
    pixels (w x h), one list per face as "faces": [[...], ...] or a single "boxes"."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "items": ("STRING", {"default": "[]", "multiline": True}),
        }, "optional": {
            "mask_scale": ("FLOAT", {"default": 1.35, "min": 0.5, "max": 3.0, "step": 0.05}),
            "feather": ("FLOAT", {"default": 0.35, "min": 0.02, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "MiniMax H3"

    @staticmethod
    def _frames(path, start, count, W, H):
        import subprocess
        ffmpeg = _n._ffmpeg_exe()
        if not ffmpeg:
            raise ValueError("ffmpeg isn't available. Install it (or pip install imageio-ffmpeg).")
        cmd = [ffmpeg, "-v", "error", "-i", path, "-vf",
               f"select=between(n\\,{int(start)}\\,{int(start + count - 1)}),scale={W}:{H}:flags=bicubic",
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        res = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if res.returncode != 0:
            raise ValueError(f"Couldn't read {os.path.basename(path)}: {res.stderr.decode('utf-8', 'replace')[-300:]}")
        n = len(res.stdout) // (W * H * 3)
        arr = np.frombuffer(res.stdout[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
        return torch.from_numpy(arr.copy()).float() / 255.0

    def paste(self, images, items="[]", mask_scale=1.35, feather=0.35):
        import folder_paths
        try:
            rows = json.loads(items) if str(items or "").strip() else []
        except ValueError:
            raise ValueError("MiniMax H3 Paste Faces: items isn't JSON.") from None
        if not rows:
            return (images,)
        root = os.path.realpath(folder_paths.get_output_directory())
        n, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        out = images.clone()
        for it in rows:
            rel = str(it.get("file") or "").replace("\\", "/").strip("/")
            path = os.path.realpath(os.path.join(root, *rel.split("/")))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                raise ValueError(f"The refined clip {rel} isn't there. Refine that Shot again (or set it to Original).")
            at, frm, cnt = int(it.get("at", 0)), int(it.get("from", 0)), int(it.get("frames", 0))
            if at < 0:
                frm, cnt, at = frm - at, cnt + at, 0
            cnt = min(cnt, n - at)
            if cnt <= 0:
                continue
            clip = self._frames(path, frm, cnt, W, H)
            cnt = min(cnt, clip.shape[0])
            sx = W / float(it.get("w") or W)
            sy = H / float(it.get("h") or H)
            mask = torch.zeros(cnt, H, W)
            for face in it.get("faces") or ([it["boxes"]] if it.get("boxes") else []):
                shifted = [[r[0] - frm] + list(r[1:5]) for r in face]
                box, presence = _boxes_per_frame(shifted, cnt, sx, sy)
                if box is None:
                    continue
                m = faces.face_mask(_Faces(box, presence), H, W, scale=float(mask_scale), feather=float(feather))
                mask = torch.maximum(mask, m)
            seg = out[at:at + cnt]
            out[at:at + cnt] = seg + (clip[:cnt].to(seg) - seg) * mask[..., None].to(seg)
        return (out.clamp_(0, 1),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3FaceRefine": MiniMaxH3FaceRefine, "MiniMaxH3PasteFaces": MiniMaxH3PasteFaces}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3FaceRefine": "MiniMax H3 Face Refine",
                              "MiniMaxH3PasteFaces": "MiniMax H3 Paste Faces"}
