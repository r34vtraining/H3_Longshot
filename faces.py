"""Face refine helpers (1.8.0): find faces in a decoded video, follow them over time,
cut a steady close-up out around each small face, and paste a redrawn close-up back.

Pure torch / numpy (no ComfyUI imports), so it's testable on its own.

Detection is MTCNN (Zhang et al. 2016), re-implemented here after facenet-pytorch
(MIT License, Copyright (c) 2019 Timothy Esler). Its three small weight files
(pnet.pt, rnet.pt, onet.pt, ~2 MB) aren't shipped with the pack: H3 Long Shot Studio
downloads them into models/facedetect/mtcnn (or put them there by hand).
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_FILES = ("pnet.pt", "rnet.pt", "onet.pt")
DETECTOR_SUBDIR = os.path.join("facedetect", "mtcnn")


# ---------------------------------------------------------------------------
# MTCNN
# ---------------------------------------------------------------------------

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        return self.conv4_2(x), self.softmax4_1(self.conv4_1(x))


class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.prelu3(self.conv3(x))
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.prelu4(self.dense4(x.view(x.shape[0], -1)))
        return self.dense5_2(x), self.softmax5_1(self.dense5_1(x))


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.pool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.prelu5(self.dense5(x.view(x.shape[0], -1)))
        return self.dense6_2(x), self.dense6_3(x), self.softmax6_1(self.dense6_1(x))


def weights_present(folder) -> bool:
    return bool(folder) and all(os.path.isfile(os.path.join(folder, f)) for f in WEIGHT_FILES)


def _nms(boxes, scores, iou, method="union"):
    """Plain NMS on [K,4] boxes (x1, y1, x2, y2); returns kept indices (best first)."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    if method == "union":
        try:
            from torchvision.ops import nms
            return nms(boxes.float(), scores.float(), iou)
        except Exception:     # no torchvision: the slow way below
            pass
    b = boxes.detach().float().cpu().numpy()
    s = scores.detach().float().cpu().numpy()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-s)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = w * h
        o = inter / np.minimum(area[i], area[rest]) if method == "min" else inter / (area[i] + area[rest] - inter)
        order = rest[o <= iou]
    return torch.as_tensor(np.array(keep, dtype=np.int64), device=boxes.device)


def _square(b):
    """Boxes made square around their centre (the longer side)."""
    b = b.clone()
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    l = torch.maximum(w, h)
    b[:, 0] = b[:, 0] + w * 0.5 - l * 0.5
    b[:, 1] = b[:, 1] + h * 0.5 - l * 0.5
    b[:, 2] = b[:, 0] + l
    b[:, 3] = b[:, 1] + l
    return b


def _regress(b, reg):
    b = b.clone()
    w = b[:, 2] - b[:, 0] + 1
    h = b[:, 3] - b[:, 1] + 1
    b[:, 0] += reg[:, 0] * w
    b[:, 1] += reg[:, 1] * h
    b[:, 2] += reg[:, 2] * w
    b[:, 3] += reg[:, 3] * h
    return b


def _patches(img, boxes, size):
    """Crops of `img` [3,H,W] (0..255) under `boxes`, padded with zeros outside, resized."""
    H, W = img.shape[1:]
    out = []
    for x1, y1, x2, y2 in boxes.round().long().tolist():
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            out.append(torch.zeros(1, 3, size, size, device=img.device, dtype=img.dtype))
            continue
        out.append(F.interpolate(img[None, :, y1:y2, x1:x2], size=(size, size), mode="area"))
    return torch.cat(out, 0)


class MTCNN:
    """Faces in RGB frames: boxes (x1, y1, x2, y2), scores and 5 landmarks
    (left eye, right eye, nose, left / right mouth corner)."""

    def __init__(self, folder, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.pnet, self.rnet, self.onet = PNet(), RNet(), ONet()
        for net, name in ((self.pnet, "pnet.pt"), (self.rnet, "rnet.pt"), (self.onet, "onet.pt")):
            sd = torch.load(os.path.join(folder, name), map_location="cpu", weights_only=True)
            net.load_state_dict(sd)
            net.to(self.device, dtype).eval()
        self.thresholds = (0.6, 0.7, 0.7)
        self.factor = 0.709

    @torch.no_grad()
    def detect(self, img, min_face=20):
        """One frame: [H,W,3] float 0..1 (or uint8). Returns (boxes [K,4], scores [K],
        points [K,5,2]) as numpy, in pixels."""
        x = img.to(self.device)
        x = (x.float() * 255.0 if not x.dtype == torch.uint8 else x.float()).permute(2, 0, 1).to(self.dtype)
        H, W = x.shape[1:]
        m = 12.0 / max(float(min_face), 12.0)
        minl = min(H, W) * m
        scales = []
        s = m
        while minl >= 12:
            scales.append(s)
            s *= self.factor
            minl *= self.factor
        cand = []
        for sc in scales:
            hs, ws = int(H * sc + 1), int(W * sc + 1)
            im = F.interpolate(x[None], size=(hs, ws), mode="area")
            reg, prob = self.pnet((im - 127.5) * 0.0078125)
            p = prob[0, 1]
            mask = p >= self.thresholds[0]
            if not mask.any():
                continue
            yx = mask.nonzero().to(x.dtype)
            score = p[mask]
            r = reg[0].permute(1, 2, 0)[mask]
            q1 = ((2 * yx.flip(1) + 1) / sc).floor()
            q2 = ((2 * yx.flip(1) + 12) / sc).floor()
            b = torch.cat([q1, q2], 1)
            keep = _nms(b, score, 0.5)
            cand.append((b[keep], score[keep], r[keep]))
        empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 5, 2), np.float32))
        if not cand:
            return empty
        b = torch.cat([c[0] for c in cand])
        score = torch.cat([c[1] for c in cand])
        r = torch.cat([c[2] for c in cand])
        keep = _nms(b, score, 0.7)
        b, score, r = b[keep], score[keep], r[keep]
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        b = torch.stack([b[:, 0] + r[:, 0] * w, b[:, 1] + r[:, 1] * h,
                         b[:, 2] + r[:, 2] * w, b[:, 3] + r[:, 3] * h], 1)
        b = _square(b)
        # second stage
        reg, prob = self.rnet((_patches(x, b, 24) - 127.5) * 0.0078125)
        ok = prob[:, 1] > self.thresholds[1]
        if not ok.any():
            return empty
        b, score, reg = b[ok], prob[ok, 1], reg[ok]
        keep = _nms(b, score, 0.7)
        b = _square(_regress(b[keep], reg[keep]))
        # third stage
        reg, pts, prob = self.onet((_patches(x, b, 48) - 127.5) * 0.0078125)
        ok = prob[:, 1] > self.thresholds[2]
        if not ok.any():
            return empty
        b, score, reg, pts = b[ok], prob[ok, 1], reg[ok], pts[ok]
        w = (b[:, 2] - b[:, 0] + 1)[:, None]
        h = (b[:, 3] - b[:, 1] + 1)[:, None]
        px = w * pts[:, :5] + b[:, 0:1] - 1
        py = h * pts[:, 5:] + b[:, 1:2] - 1
        points = torch.stack([px, py], 2)
        b = _regress(b, reg)
        keep = _nms(b, score, 0.7, "min")
        return (b[keep].float().cpu().numpy(), score[keep].float().cpu().numpy(),
                points[keep].float().cpu().numpy())


_DETECTORS = {}


def detector(folder, device="cpu"):
    key = (os.path.abspath(folder), str(device))
    det = _DETECTORS.get(key)
    if det is None:
        if not weights_present(folder):
            raise FileNotFoundError(
                f"The face detector isn't installed: {', '.join(WEIGHT_FILES)} are missing from {folder}. "
                "Download it in H3 Long Shot Studio's Face refine panel.")
        det = _DETECTORS[key] = MTCNN(folder, device)
    return det


def detect_frames(det, frames, stride=2, min_face=16, progress=None):
    """Faces on every `stride`-th frame (and the last one) of `frames` [N,H,W,3].
    Returns {frame: [(box[4], score, points[5,2]), ...]}."""
    n = frames.shape[0]
    idx = list(range(0, n, max(1, int(stride))))
    if idx and idx[-1] != n - 1:
        idx.append(n - 1)
    out = {}
    for k, i in enumerate(idx):
        b, s, p = det.detect(frames[i], min_face=min_face)
        out[i] = [(b[j], float(s[j]), p[j]) for j in range(len(s)) if s[j] >= 0.9]
        if progress:
            progress(k + 1, len(idx))
    return out


# ---------------------------------------------------------------------------
# Tracks: the same face from frame to frame
# ---------------------------------------------------------------------------

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _smooth(values, sigma):
    """Gaussian smoothing of a 1-D series (edges held)."""
    v = np.asarray(values, np.float64)
    if sigma <= 0 or v.size < 3:
        return v
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([np.full(r, v[0]), v, np.full(r, v[-1])])
    return np.convolve(padded, k, mode="valid")


class Track:
    """One face over a run of frames. Per frame (from first to last, gaps filled):
    centre cx, cy, width w, height h of the face box, and whether it was detected there."""

    def __init__(self, tid, first, cx, cy, w, h, seen, points):
        self.id, self.first = tid, first
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.seen = seen              # bool per frame: detected there (else filled in)
        self.points = points          # [T,5,2] landmarks (filled in too)

    @property
    def last(self):
        return self.first + len(self.cx) - 1

    def size(self):
        """The face's height in pixels (median over its frames)."""
        return float(np.median(self.h)) if len(self.h) else 0.0

    def at(self, f):
        i = int(np.clip(f - self.first, 0, len(self.cx) - 1))
        return self.cx[i], self.cy[i], self.w[i], self.h[i]

    def as_dict(self, every=4):
        rows = []
        for i in range(0, len(self.cx), every):
            rows.append([self.first + i, round(float(self.cx[i]), 1), round(float(self.cy[i]), 1),
                         round(float(self.w[i]), 1), round(float(self.h[i]), 1)])
        return {"id": self.id, "first": self.first, "last": self.last, "size": round(self.size(), 1),
                "boxes": rows}


def build_tracks(dets, n_frames, max_gap=12, min_hits=3, sigma=2.0):
    """Link per-frame detections into tracks (nearest box, enough overlap), fill gaps
    by interpolation and smooth them. Tracks seen on fewer than `min_hits`
    detected frames are dropped (stray detections)."""
    frames = sorted(dets)
    active, done = [], []
    for f in frames:
        faces = sorted(dets[f], key=lambda d: -d[1])
        used = set()
        for tr in active:
            last_box = tr["hits"][-1][1]
            best, best_score = None, 0.0
            for j, (box, score, pts) in enumerate(faces):
                if j in used:
                    continue
                ov = _iou(last_box, box)
                ca = np.array([(last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2])
                cb = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                size = max(last_box[2] - last_box[0], last_box[3] - last_box[1], 1.0)
                near = np.linalg.norm(ca - cb) / size
                ratio = (box[3] - box[1]) / max(last_box[3] - last_box[1], 1e-3)
                if (ov > 0.2 or near < 0.6) and 0.6 < ratio < 1.6:
                    m = ov + max(0.0, 1.0 - near)
                    if m > best_score:
                        best, best_score = j, m
            if best is not None:
                used.add(best)
                tr["hits"].append((f, faces[best][0], faces[best][2]))
        for tr in [t for t in active if f - t["hits"][-1][0] > max_gap]:
            active.remove(tr)
            done.append(tr)
        for j, (box, score, pts) in enumerate(faces):
            if j not in used:
                active.append({"hits": [(f, box, pts)]})
    done.extend(active)
    tracks = []
    for tr in done:
        hits = tr["hits"]
        if len(hits) < min_hits:
            continue
        fs = np.array([h[0] for h in hits], np.float64)
        boxes = np.array([h[1] for h in hits], np.float64)
        pts = np.array([h[2] for h in hits], np.float64)
        first, last = int(fs[0]), int(fs[-1])
        grid = np.arange(first, last + 1, dtype=np.float64)
        cx = np.interp(grid, fs, (boxes[:, 0] + boxes[:, 2]) / 2)
        cy = np.interp(grid, fs, (boxes[:, 1] + boxes[:, 3]) / 2)
        w = np.interp(grid, fs, boxes[:, 2] - boxes[:, 0])
        h = np.interp(grid, fs, boxes[:, 3] - boxes[:, 1])
        p = np.stack([np.stack([np.interp(grid, fs, pts[:, k, a]) for a in (0, 1)], -1)
                      for k in range(5)], 1)
        seen = np.isin(grid, fs)
        tracks.append(Track(0, first, _smooth(cx, sigma), _smooth(cy, sigma), _smooth(w, sigma * 2),
                            _smooth(h, sigma * 2), seen, p))
    tracks.sort(key=lambda t: (t.first, -t.size()))
    for k, t in enumerate(tracks):
        t.id = k
    return tracks


# ---------------------------------------------------------------------------
# Windows: pieces of a long video H3 can redraw in one go (17n+5 frames)
# ---------------------------------------------------------------------------

def valid_frames(n):
    """The largest H3 frame count (17k+5) not above n; 0 when n < 5."""
    if n < 5:
        return 0
    return n - ((n - 5) % 17)


def aligned_frames(n):
    """The smallest H3 frame count (17k+5) not below n."""
    n = max(5, int(n))
    return n + (-(n - 5)) % 17


def windows(n_frames, max_frames=209, overlap=22):
    """[(start, frames)] covering 0..n_frames-1, each 17k+5 frames. A video no longer than
    `max_frames` is one window (padded past its end when it isn't 17k+5 frames: the
    caller repeats the last frame). A longer one is cut into windows overlapping by at
    least `overlap` frames (for a crossfade), the last one ending on the last frame."""
    if n_frames <= 0:
        return []
    if n_frames <= max_frames or valid_frames(max_frames) == 0:
        return [(0, aligned_frames(n_frames))]
    w = valid_frames(int(max_frames))
    overlap = max(5, min(int(overlap), w // 2))
    out, start = [], 0
    while True:
        if start + w >= n_frames:
            out.append((n_frames - w, w))
            break
        out.append((start, w))
        start += w - overlap
    return out


def window_weights(wins, n_frames):
    """Per window, its weight on each of its frames that exist (start .. min(end,
    n_frames)): 1 alone, linear ramps where windows overlap, adding up to 1 on every frame."""
    cover = np.zeros(n_frames, np.float64)
    ramps = []
    for k, (s, n) in enumerate(wins):
        n = min(n, n_frames - s)
        r = np.ones(n, np.float64)
        if k > 0:
            ps, pn = wins[k - 1]
            ov = min(ps + pn - s, n)
            if ov > 0:
                r[:ov] = np.minimum(r[:ov], (np.arange(ov) + 1) / (ov + 1))
        if k + 1 < len(wins):
            ov = s + n - wins[k + 1][0]
            if ov > 0:
                r[n - ov:] = np.minimum(r[n - ov:], 1 - (np.arange(ov) + 1) / (ov + 1))
        ramps.append(r)
        cover[s:s + n] += r
    return [r / np.maximum(cover[s:s + len(r)], 1e-9) for r, (s, _n) in zip(ramps, wins)]


# ---------------------------------------------------------------------------
# Crops: what H3 redraws for one face in one window
# ---------------------------------------------------------------------------

class Crop:
    """One face in one window: a square region per frame (centre, side), the face box per
    frame (for its mask) and how present the face is on each frame (0..1)."""

    def __init__(self, track, start, frames, cx, cy, side, face, presence, follow):
        self.track, self.start, self.frames = track, start, frames
        self.cx, self.cy, self.side = cx, cy, side          # per frame (side is fixed)
        self.face = face                                    # [T,4]: cx, cy, w, h
        self.presence = presence                            # [T] 0..1
        self.follow = follow

    def as_dict(self):
        return {"track": self.track.id, "start": self.start, "frames": self.frames,
                "side": round(float(self.side), 1), "follow": bool(self.follow),
                "size": round(self.track.size(), 1),
                "box": [round(float(np.median(self.cx)), 1), round(float(np.median(self.cy)), 1)]}


def plan_crops(tracks, start, frames, *, max_face=150.0, min_face=12.0, which="all",
               padding=0.8, follow="auto", min_presence=0.25, ramp=6):
    """The faces to redraw in window start..start+frames-1: tracks present on at least
    `min_presence` of it whose face is below `max_face` px tall (and above `min_face`).
    which: "all" or "largest". padding: room around the face on each side, as a
    fraction of its size (0.8 -> a square 2.6x the face). follow: "auto" (a crop that
    moves when the face moves more than a little), "fixed" or "follow"."""
    end = start + frames
    picks = []
    for t in tracks:
        lo, hi = max(start, t.first), min(end - 1, t.last)
        if hi < lo:
            continue
        present = hi - lo + 1
        if present < max(5, min_presence * frames):
            continue
        sz = float(np.median(t.h[lo - t.first:hi - t.first + 1]))
        if sz >= max_face or sz < min_face:
            continue
        picks.append((t, lo, hi, sz))
    if which == "largest" and picks:
        picks = [max(picks, key=lambda p: p[3])]
    out = []
    for t, lo, hi, sz in picks:
        fr = np.arange(start, end)
        idx = np.clip(fr - t.first, 0, len(t.cx) - 1)
        fcx, fcy, fw, fh = t.cx[idx], t.cy[idx], t.w[idx], t.h[idx]
        face_side = float(np.median(np.maximum(fw, fh)[(fr >= lo) & (fr <= hi)]))
        side = face_side * (1.0 + 2.0 * padding)
        inside = (fr >= lo) & (fr <= hi)
        presence = inside.astype(np.float64)
        if ramp > 0:                        # fade in / out where the face enters or leaves
            k = np.arange(frames)
            d = np.minimum(np.abs(k - (lo - start)), np.abs(k - (hi - start)))
            edge = (k > lo - start) & (k < hi - start)
            presence = np.where(inside, np.where(edge, np.minimum(1.0, (d + 1) / ramp), 1.0 / ramp), 0.0)
            if lo == start:
                presence[:ramp] = np.where(inside[:ramp], 1.0, presence[:ramp])
            if hi == end - 1:
                presence[-ramp:] = np.where(inside[-ramp:], 1.0, presence[-ramp:])
        mx = float(np.ptp(fcx[inside])) if inside.any() else 0.0
        my = float(np.ptp(fcy[inside])) if inside.any() else 0.0
        moves = max(mx, my) > 0.12 * side
        use_follow = follow == "follow" or (follow == "auto" and moves)
        if use_follow:
            ccx, ccy = _smooth(fcx, 6.0), _smooth(fcy, 6.0)
        else:
            # one fixed square that holds the whole movement (grown if it has to)
            ccx = np.full(frames, (fcx[inside].min() + fcx[inside].max()) / 2)
            ccy = np.full(frames, (fcy[inside].min() + fcy[inside].max()) / 2)
            side = max(side, face_side + max(mx, my) + 2 * padding * face_side)
        out.append(Crop(t, start, frames, ccx, ccy, side,
                        np.stack([fcx, fcy, fw, fh], 1), presence, use_follow))
    return out


def _grid(cx, cy, side, size, H, W, device):
    """Sampling grid [T,size,size,2] (normalised to the frame) for squares of `side`
    centred on (cx, cy) per frame."""
    # continuous coordinates: pixel i covers [i, i+1); output cell j's centre sits at
    # cx - side/2 + (j + 0.5) * side / size (what paste_delta inverts)
    t = ((torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size - 0.5) * float(side)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    cx = torch.as_tensor(np.asarray(cx), dtype=torch.float32, device=device)[:, None, None]
    cy = torch.as_tensor(np.asarray(cy), dtype=torch.float32, device=device)[:, None, None]
    x = (cx + gx[None]) / W * 2 - 1
    y = (cy + gy[None]) / H * 2 - 1
    return torch.stack([x, y], -1)


def crop_rect(crop, H, W, scale=1.0):
    """The part of the picture a crop covers (and its paste can touch): (x0, y0, x1, y1)."""
    r = crop.side * 0.5 + 2
    x0 = int(max(0, math.floor(float(np.min(crop.cx)) - r)))
    y0 = int(max(0, math.floor(float(np.min(crop.cy)) - r)))
    x1 = int(min(W, math.ceil(float(np.max(crop.cx)) + r)))
    y1 = int(min(H, math.ceil(float(np.max(crop.cy)) + r)))
    return x0, y0, x1, y1


def extract(frames, crop, size, device=None):
    """The crop's frames [T,size,size,3] (float 0..1, on `device`) from `frames`
    [T,H,W,3] (the window's frames, anywhere), edges repeated where the square leaves
    the picture. Only the part of the picture the crop covers is moved to `device`."""
    T, H, W = frames.shape[:3]
    dev = torch.device(device) if device is not None else frames.device
    x0, y0, x1, y1 = crop_rect(crop, H, W)
    out = []
    for a in range(0, T, 16):
        b = min(T, a + 16)
        g = _grid(np.asarray(crop.cx[a:b]) - x0, np.asarray(crop.cy[a:b]) - y0, crop.side, size,
                  y1 - y0, x1 - x0, dev)
        src = frames[a:b, y0:y1, x0:x1].to(dev).movedim(-1, 1).float()
        out.append(F.grid_sample(src, g, mode="bicubic", padding_mode="border",
                                 align_corners=False).clamp(0, 1).movedim(1, -1))
    return torch.cat(out, 0)


def region_hash(frames, crop):
    """A cheap fingerprint of the picture under a crop (every 4th frame, 24 px): with
    the sound and the settings, it says whether a kept redraw still fits."""
    H, W = frames.shape[1:3]
    x0, y0, x1, y1 = crop_rect(crop, H, W)
    part = frames[::4, y0:y1, x0:x1].movedim(-1, 1).float()
    small = F.interpolate(part, size=(24, 24), mode="area")
    q = (small.clamp(0, 1) * 63).round().to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha1(q).hexdigest()[:16]


def _blur(x, sigma):
    """Gaussian blur of [N,C,H,W]."""
    if sigma <= 0.3:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def match_color(refined, original, strength=1.0, sigma=None):
    """Bring the redrawn crop's broad colour and shading back to the original's (the low
    frequencies of the original, the detail of the redraw). Both [T,S,S,3]."""
    if strength <= 0:
        return refined
    S = refined.shape[1]
    sigma = sigma or S / 16.0
    r = refined.movedim(-1, 1).float()
    o = original.movedim(-1, 1).float().to(r.device)
    out = r + strength * (_blur(o, sigma) - _blur(r, sigma))
    return out.clamp(0, 1).movedim(1, -1)


def face_mask(crop, H, W, *, scale=1.0, feather=0.3, device="cpu", rect=None):
    """Soft elliptical masks [T,h,w] over the face on each frame (inside `rect` =
    (x0, y0, x1, y1) of the picture, or all of it): 1 on the face, fading out over
    `feather` of its radius, times the face's presence."""
    x0, y0, x1, y1 = rect or (0, 0, W, H)
    ys = torch.arange(y0, y1, device=device, dtype=torch.float32)[None, :, None] + 0.5
    xs = torch.arange(x0, x1, device=device, dtype=torch.float32)[None, None, :] + 0.5
    f = torch.as_tensor(crop.face, dtype=torch.float32, device=device)
    cx, cy, w, h = f[:, 0, None, None], f[:, 1, None, None], f[:, 2, None, None], f[:, 3, None, None]
    # MTCNN's box runs brow to chin: the ellipse reaches a little above and to the sides
    a = 0.5 * w * 1.15 * scale
    b = 0.5 * h * 1.30 * scale
    cy = cy - 0.06 * h
    r = torch.sqrt(((xs - cx) / a) ** 2 + ((ys - cy) / b) ** 2)
    fe = max(0.02, float(feather))
    m = ((1.0 + fe * 0.5 - r) / fe).clamp(0, 1)
    m = m * m * (3 - 2 * m)                                 # smoothstep
    p = torch.as_tensor(crop.presence, dtype=torch.float32, device=device)[:, None, None]
    return m * p


def paste_delta(frames, refined, crop, *, scale=1.0, feather=0.3, device=None):
    """What pasting the redrawn crop `refined` [T,S,S,3] over the window's `frames`
    [T,H,W,3] changes: (rect, delta [T,h,w,3] on `device`) where frames[rect] + delta
    is the pasted picture. The crop is fitted back to its square on each frame
    (antialiased) and blended in under the face mask."""
    T, H, W = frames.shape[:3]
    S = refined.shape[1]
    x0, y0, x1, y1 = rect = crop_rect(crop, H, W, scale)
    if x1 <= x0 or y1 <= y0:
        return rect, None
    dev = torch.device(device) if device is not None else refined.device
    side = float(crop.side)
    small = max(8, int(round(side)))           # pre-shrink so sampling it back down doesn't alias
    ys = torch.arange(y0, y1, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(x0, x1, device=dev, dtype=torch.float32) + 0.5
    deltas = []
    for a in range(0, T, 16):
        b = min(T, a + 16)
        r = refined[a:b].movedim(-1, 1).float().to(dev)
        if small < S:
            r = F.interpolate(r, size=(small, small), mode="bicubic", antialias=True, align_corners=False)
        cx = torch.as_tensor(np.asarray(crop.cx[a:b]), dtype=torch.float32, device=dev)[:, None, None]
        cy = torch.as_tensor(np.asarray(crop.cy[a:b]), dtype=torch.float32, device=dev)[:, None, None]
        gx = ((xs[None, None, :] - cx) / side * 2).expand(-1, ys.numel(), -1)
        gy = ((ys[None, :, None] - cy) / side * 2).expand(-1, -1, xs.numel())
        g = torch.stack([gx, gy], -1)
        inside = ((g.abs() <= 1).all(-1)).float()
        patch = F.grid_sample(r, g, mode="bilinear", padding_mode="border", align_corners=False)
        patch = patch.clamp(0, 1).movedim(1, -1)
        m = face_mask(_slice_crop(crop, a, b), H, W, scale=scale, feather=feather, device=dev,
                      rect=rect) * inside
        orig = frames[a:b, y0:y1, x0:x1].to(dev).float()
        deltas.append((patch - orig) * m[..., None])
    return rect, torch.cat(deltas, 0)


def _slice_crop(crop, a, b):
    return Crop(crop.track, crop.start + a, b - a, crop.cx[a:b], crop.cy[a:b], crop.side,
                crop.face[a:b], crop.presence[a:b], crop.follow)
