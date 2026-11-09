import contextlib
import enum
import itertools
import os
import platform
import sys
import tempfile
import time
import zipfile
from collections import namedtuple
from datetime import datetime
from xml.sax.saxutils import escape

import cv2
import numpy as np

BALL_DIAMETER_CM = 2.0
PENDULUM_LENGTH_CM = 13.6
LENGTH_UNCERTAINTY_CM = 0.3
SIDE_STRING_CM = 13.0
BAR_LENGTH_CM = 15.0
BAR_GAP_CM = 11.0
G_LOCAL = 9.78

MAX_FRAME_W = 960
PLAYBACK_SPEED = 1.0
SNAP_INTERVAL_S = 1.0

MIN_RADIUS = 12
MIN_SAMPLES = 15
AXIS_MIN_DISPLACEMENT = 0.8
MIN_QUALITY = 0.6
TAPE_MIN_FRAC = 0.1
TAPE_DARKNESS = 0.5
TAPE_AREA_RANGE = (0.3, 3.0)
TAPE_SEARCH_RADII = 2.5
TAPE_MAX_JUMP_RADII = 1.2
ASPECT_TOL = 2.0
FILL_TOL = 0.6
AXIS_FRAMES = 3
WAIT_WARN_S = 6.0
STABILIZE = True
STAB_POINTS = 150
STAB_MIN_POINTS = 6
THR_SCALE_RANGE = (0.7, 1.3)
SEARCH_MARGIN = 70
MIN_SCORE = 0.6
LIVE_MIN_FPS = 12.0
SHOW_RED_LINE = True
NOMINAL_FPS = 30.0
MAX_DT_FACTOR = 8.0
STEADY_TOL = 0.1

SCREEN_W = 1080
SCREEN_H = 1920
WINDOW = "Con lac"
FONT = cv2.FONT_HERSHEY_SIMPLEX
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

click = {"pos": None}
DEFAULT_SIZE = [SCREEN_W, SCREEN_H]
BAR_BG = (235, 235, 235)
MENU_BG = (120, 90, 60)
TK_STATE = {"ok": None}


class Phase(enum.IntEnum):
    PICK = 0
    READY = 1
    WAIT = 2
    MEASURE = 3


TapeRef = namedtuple("TapeRef", "off_x off_y area thr bg aspect fill")


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pos"] = (x, y)


def put(img, text, x, y, color=(0, 255, 0), scale=0.5, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def window_size():
    with contextlib.suppress(cv2.error, AttributeError, IndexError, TypeError):
        r = cv2.getWindowImageRect(WINDOW)
        if r[2] > 200 and r[3] > 200:
            return int(r[2]), int(r[3])
    return DEFAULT_SIZE[0], DEFAULT_SIZE[1]


def is_phone():
    if "--phone" in sys.argv:
        return True
    if "--pc" in sys.argv:
        return False
    env = ("ANDROID_ROOT", "ANDROID_DATA", "ANDROID_STORAGE", "TERMUX_VERSION")
    if sys.platform == "android" or any(k in os.environ for k in env):
        return True
    if "android" in (platform.release() + platform.version()).lower():
        return True
    return os.path.isdir("/sdcard") or os.path.isdir("/storage/emulated")


def have_tk():
    if TK_STATE["ok"] is None:
        TK_STATE["ok"] = False
        try:
            import tkinter as tk
        except ImportError:
            return False
        try:
            root = tk.Tk()
            root.withdraw()
            DEFAULT_SIZE_PC[0] = root.winfo_screenwidth()
            DEFAULT_SIZE_PC[1] = root.winfo_screenheight()
            root.destroy()
            TK_STATE["ok"] = True
        except (tk.TclError, RuntimeError):
            TK_STATE["ok"] = False
    return TK_STATE["ok"]


DEFAULT_SIZE_PC = [1280, 800]


def open_window(phone):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if phone:
        DEFAULT_SIZE[0], DEFAULT_SIZE[1] = SCREEN_W, SCREEN_H
        with contextlib.suppress(cv2.error, AttributeError):
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        have_tk()
        w = min(1280, int(DEFAULT_SIZE_PC[0] * 0.85))
        h = min(800, int(DEFAULT_SIZE_PC[1] * 0.85))
        DEFAULT_SIZE[0], DEFAULT_SIZE[1] = w, h
        with contextlib.suppress(cv2.error, AttributeError):
            cv2.resizeWindow(WINDOW, w, h)
    cv2.setMouseCallback(WINDOW, on_mouse)


def layout(W, H, fw, fh):
    bar_h = max(70, int(H * 0.08))
    area = H - bar_h
    s = max(W / fw, area / fh)
    vw = min(float(fw), W / s)
    vh = min(float(fh), area / s)
    x0 = int((fw - vw) / 2)
    y0 = int((fh - vh) / 2)
    x1 = min(fw, x0 + round(vw))
    y1c = min(fh, y0 + round(vh))
    gap = int(W * 0.015)
    bw = (W - 4 * gap) // 3
    y1 = area + int(bar_h * 0.12)
    y2 = H - int(bar_h * 0.12)
    rects = {}
    for i, name in enumerate(("reset", "start", "stop")):
        bx = gap + i * (bw + gap)
        rects[name] = (bx, y1, bx + bw, y2)
    return {"area": area, "bar_h": bar_h, "crop": (x0, y0, x1, y1c), "rects": rects}


def to_frame(lay, W, tx, ty):
    x0, y0, x1, y1 = lay["crop"]
    return int(x0 + tx * (x1 - x0) / W), int(y0 + ty * (y1 - y0) / lay["area"])


def put_o(img, text, x, y, color, scale, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def hit(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def draw_button(canvas, rect, label, color, enabled, fs):
    x1, y1, x2, y2 = rect
    fill = color if enabled else (70, 70, 70)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
    thick = max(1, int(fs * 2))
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, thick)
    org = (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2)
    text_color = (255, 255, 255) if enabled else (150, 150, 150)
    cv2.putText(canvas, label, org, FONT, fs, text_color, thick, cv2.LINE_AA)


def draw_axes(frame, ox, oy):
    h, w = frame.shape[:2]
    n = int(min(w, h) * 0.18)
    cv2.arrowedLine(frame, (ox, oy), (min(w - 2, ox + n), oy), (0, 140, 255), 3, tipLength=0.2)
    cv2.arrowedLine(frame, (ox, oy), (ox, max(2, oy - n)), (0, 200, 0), 3, tipLength=0.2)
    cv2.circle(frame, (ox, oy), 5, (0, 255, 255), -1)
    put(frame, "O", ox + 8, oy + 24, (0, 255, 255), 0.8, 2)
    put(frame, "x", min(w - 20, ox + n + 6), oy + 24, (0, 140, 255), 0.8, 2)
    put(frame, "y", ox + 8, max(24, oy - n - 6), (0, 200, 0), 0.8, 2)


def shrink(frame):
    h, w = frame.shape[:2]
    if w <= MAX_FRAME_W:
        return frame
    s = MAX_FRAME_W / w
    return cv2.resize(frame, (MAX_FRAME_W, round(h * s)), interpolation=cv2.INTER_AREA)


class VideoSource:
    live = False

    def __init__(self, path, portrait=None):
        self.path = path
        self.portrait = portrait
        self.rot = False
        self.cap = None
        self.open()

    def fix(self, frame):
        if self.rot:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return shrink(frame)

    def decide_rotation(self, frame):
        if self.portrait is not None:
            h, w = frame.shape[:2]
            self.rot = (w < h) != self.portrait

    def open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.path)
        ok, frame = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ok:
            raise OSError(self.path)
        self.decide_rotation(frame)
        self.first = self.fix(frame)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else 30.0
        self.t = 0.0
        self.idx = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.idx += 1
        ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.t = ms if ms > self.t else self.t + 1.0 / self.fps
        return self.fix(frame)

    def rewind(self):
        self.open()

    def close(self):
        self.cap.release()


class CameraSource(VideoSource):
    live = True

    def __init__(self, index=0, portrait=None):
        self.index = index
        super().__init__("camera", portrait)

    def open(self):
        if self.cap is None:
            frame = None
            for i in dict.fromkeys([self.index, 0, 1, 2]):
                cap = cv2.VideoCapture(i)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                with contextlib.suppress(cv2.error, AttributeError):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(10):
                    ok, frame = cap.read()
                    if ok:
                        break
                if frame is not None and ok:
                    self.cap = cap
                    break
                cap.release()
            if self.cap is None:
                raise OSError("camera")
            self.decide_rotation(frame)
            self.first = self.fix(frame)
        self.fps = 30.0
        self.idx = 0
        self.t = 0.0
        self.t0 = time.perf_counter()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.idx += 1
        self.t = time.perf_counter() - self.t0
        if self.t > 1.0:
            self.fps = self.idx / self.t
        return self.fix(frame)


def crop_template(gray, cx, cy, half):
    h, w = gray.shape
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return gray[y1:y2, x1:x2].copy()


def track(gray, template, last_x, last_y, half):
    h, w = gray.shape
    x1 = max(0, last_x - half - SEARCH_MARGIN)
    y1 = max(0, last_y - half - SEARCH_MARGIN)
    x2 = min(w, last_x + half + SEARCH_MARGIN)
    y2 = min(h, last_y + half + SEARCH_MARGIN)
    window = gray[y1:y2, x1:x2]
    th, tw = template.shape
    if window.shape[0] < th or window.shape[1] < tw:
        return None, 0.0
    res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (x1 + loc[0] + tw // 2, y1 + loc[1] + th // 2), score


def crop_blur(gray, cx, cy, R):
    h, w = gray.shape
    x1, y1 = max(0, int(cx) - R), max(0, int(cy) - R)
    x2, y2 = min(w, int(cx) + R), min(h, int(cy) + R)
    win = gray[y1:y2, x1:x2]
    if win.shape[0] < 8 or win.shape[1] < 8:
        return None, x1, y1
    return cv2.GaussianBlur(win, (5, 5), 0), x1, y1


def window_level(blur):
    return float(np.median(blur[::4, ::4]))


def clean_mask(binary):
    return cv2.morphologyEx(binary.astype(np.uint8) * 255, cv2.MORPH_CLOSE, MORPH_KERNEL)


def refine(gray, pos, r):
    blur, x1, y1 = crop_blur(gray, pos[0], pos[1], int(r * 1.8))
    if blur is None:
        return pos
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    area0 = np.pi * r * r
    px, py = int(pos[0]) - x1, int(pos[1]) - y1
    best = None
    best_d = 1e9
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.3 * area0 or a > 2.5 * area0:
            continue
        d = float(np.hypot(cents[i][0] - px, cents[i][1] - py))
        if d < best_d:
            best_d = d
            best = cents[i]
    if best is None or best_d > r:
        return pos
    return (round(best[0] + x1), round(best[1] + y1))


def find_tape(gray, cx, cy, r):
    blur, x1, y1 = crop_blur(gray, cx, cy, int(r * 1.6))
    if blur is None:
        return None
    yy, xx = np.ogrid[y1:y1 + blur.shape[0], x1:x1 + blur.shape[1]]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    vals = blur[dist2 <= r * r]
    if vals.size < 50:
        return None
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(clean_mask(blur < thr))
    best = None
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        d = float(np.hypot(cents[i][0] + x1 - cx, cents[i][1] + y1 - cy))
        if a < TAPE_MIN_FRAC * np.pi * r * r or d > r:
            continue
        if best is None or a > best[1]:
            best = (i, a)
    if best is None:
        return None
    i, a = best
    ring = blur[dist2 > (1.3 * r) ** 2]
    if ring.size < 50 or float(blur[labels == i].mean()) > TAPE_DARKNESS * float(np.median(ring)):
        return None
    wide, _, _ = crop_blur(gray, cx, cy, int(r * TAPE_SEARCH_RADII))
    if wide is None:
        return None
    c = cents[i]
    bw = float(stats[i, cv2.CC_STAT_WIDTH])
    bh = float(stats[i, cv2.CC_STAT_HEIGHT])
    return TapeRef(
        float(c[0] + x1 - cx), float(c[1] + y1 - cy), a, float(thr), window_level(wide),
        bw / max(bh, 1.0), a / max(bw * bh, 1.0),
    )


def blob_track(gray, pred, r, ref, max_jump, relax=0):
    k = 1.0 + 0.5 * min(relax, 6)
    R = int(max(r * TAPE_SEARCH_RADII * (1.0 + 0.5 * min(relax, 3)), max_jump + 1.5 * r))
    blur, x1, y1 = crop_blur(gray, pred[0], pred[1], R)
    if blur is None:
        return None, 0.0
    scale = float(np.clip(window_level(blur) / ref.bg, *THR_SCALE_RANGE))
    n, _, stats, cents = cv2.connectedComponentsWithStats(clean_mask(blur < ref.thr * scale))
    tx = int(pred[0]) - x1 + ref.off_x
    ty = int(pred[1]) - y1 + ref.off_y
    best = None
    best_d = 1e9
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if a < TAPE_AREA_RANGE[0] * ref.area / k or a > TAPE_AREA_RANGE[1] * ref.area * k:
            continue
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        asp = bw / max(bh, 1.0)
        if not (ref.aspect / (ASPECT_TOL * k) <= asp <= ref.aspect * ASPECT_TOL * k):
            continue
        if a / max(bw * bh, 1.0) < FILL_TOL * ref.fill / k:
            continue
        d = float(np.hypot(cents[i][0] - tx, cents[i][1] - ty))
        if d < best_d:
            best_d = d
            best = (cents[i], a)
    if best is None or best_d > max_jump:
        return None, 0.0
    c, a = best
    score = min(a, ref.area) / max(a, ref.area)
    return (round(c[0] + x1 - ref.off_x), round(c[1] + y1 - ref.off_y)), score


class Stabilizer:
    def __init__(self, gray, center, r):
        self.origin = center
        self.r = r
        self.ref = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = np.full(gray.shape, 255, np.uint8)
        cv2.circle(mask, (int(center[0]), int(center[1])), int(3 * r), 0, -1)
        self.pts = cv2.goodFeaturesToTrack(self.ref, STAB_POINTS, 0.01, 8, mask=mask, blockSize=7)
        self.shift = (0.0, 0.0)
        self.n_used = 0

    def update(self, gray, ball):
        if not STABILIZE or self.pts is None or len(self.pts) < STAB_MIN_POINTS:
            return self.shift
        cur = cv2.GaussianBlur(gray, (3, 3), 0)
        lk = {
            "winSize": (31, 31),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        }
        init = self.pts + np.array(self.shift, np.float32)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.ref, cur, self.pts, init.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, self.ref, nxt, self.pts.copy(), **lk)
        err = np.linalg.norm((back - self.pts).reshape(-1, 2), axis=1)
        far = np.linalg.norm(nxt.reshape(-1, 2) - np.array(ball, np.float32), axis=1) > 2.2 * self.r
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (err < 1.0) & far
        if ok.sum() < STAB_MIN_POINTS:
            return self.shift
        d = (nxt - self.pts).reshape(-1, 2)[ok]
        med = np.median(d, axis=0)
        keep = np.linalg.norm(d - med, axis=1) < 3.0
        if keep.sum() < STAB_MIN_POINTS:
            return self.shift
        m = d[keep].mean(axis=0)
        self.n_used = int(keep.sum())
        self.shift = (float(m[0]), float(m[1]))
        return self.shift


class Tracker:
    def __init__(self, center, r, template, ref):
        self.origin = center
        self.r = r
        self.half = int(r * 1.2)
        self.template = template
        self.ref = ref
        self.last = center
        self.vel = (0, 0)
        self.lost = False
        self.lost_count = 0
        self.score = 0.0
        self.px_to_cm = BALL_DIAMETER_CM / (2.0 * r)
        self.stab = None
        self.shift = (0.0, 0.0)
        self.max_shift = 0.0
        self.t_prev = None

    def update(self, gray, moving, refresh, t=None):
        f = 1.0
        if t is not None:
            if self.t_prev is not None and t > self.t_prev:
                f = float(np.clip((t - self.t_prev) * NOMINAL_FPS, 1.0, MAX_DT_FACTOR))
            self.t_prev = t
        if self.ref is not None:
            pred = (self.last[0] + self.vel[0], self.last[1] + self.vel[1]) if moving and f <= 2.0 else self.last
            max_jump = TAPE_MAX_JUMP_RADII * self.r * f * (1 + self.lost_count)
            pos, score = blob_track(gray, pred, self.r, self.ref, max_jump, self.lost_count)
            lost = pos is None
        else:
            pos, score = track(gray, self.template, self.last[0], self.last[1], self.half)
            lost = pos is None or score < MIN_SCORE
            if not lost:
                pos = refine(gray, pos, self.r)
        self.lost = lost
        self.lost_count = self.lost_count + 1 if lost else 0
        self.score = score
        if lost:
            return None
        if moving:
            self.vel = (pos[0] - self.last[0], pos[1] - self.last[1])
        self.last = pos
        if refresh and self.ref is None and score > 0.9:
            nt = crop_template(gray, pos[0], pos[1], self.half)
            if nt is not None:
                self.template = nt
        return pos

    def reseed(self, pos, gray=None):
        self.last = (int(pos[0]), int(pos[1]))
        self.vel = (0, 0)
        self.lost_count = 0
        if gray is not None:
            ref = find_tape(gray, int(pos[0]), int(pos[1]), self.r)
            if ref is not None:
                self.ref = ref


PICK_ERROR = {"msg": ""}


def blob_too_big(gray, cx, cy, r):
    blur, x1, y1 = crop_blur(gray, cx, cy, int(r * 3.5))
    if blur is None:
        return False
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(clean_mask(blur < thr))
    best = None
    best_d = 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.1 * np.pi * r * r:
            continue
        d = float(np.hypot(cents[i][0] + x1 - cx, cents[i][1] + y1 - cy))
        if d < best_d:
            best_d = d
            best = i
    if best is None or best_d > r:
        return False
    a = float(stats[best, cv2.CC_STAT_AREA])
    return np.sqrt(a / np.pi) > 1.6 * r


def make_tracker(gray, center, edge):
    cx, cy = center
    r = int(np.hypot(edge[0] - cx, edge[1] - cy))
    PICK_ERROR["msg"] = ""
    if r < MIN_RADIUS:
        PICK_ERROR["msg"] = "Mep bi qua gan tam. Chon lai tu buoc 1"
        return None
    if blob_too_big(gray, cx, cy, r):
        PICK_ERROR["msg"] = "Bi to hon vong da chon. Cham MEP bi xa tam hon"
        return None
    template = crop_template(gray, cx, cy, int(r * 1.2))
    if template is None:
        return None
    tr = Tracker((cx, cy), r, template, find_tape(gray, cx, cy, r))
    tr.stab = Stabilizer(gray, (cx, cy), r)
    return tr


class Measurement:
    def __init__(self):
        self.clear()

    def clear(self):
        self.axis = None
        self.side = 0
        self.t0 = 0.0
        self.start_frame = 0
        self.times = []
        self.xs = []
        self.snaps = []
        self.next_snap = 0.0
        self.t_now = 0.0
        self.x_cm = 0.0
        self.y_cm = 0.0
        self.streak = 0
        self.cand = None
        self.xc = []
        self.yc = []

    def finalize(self):
        # kiem tra lai truc sau khi do: chon truc co bien do lon hon
        if len(self.xc) < 4:
            return
        rx = float(np.ptp(self.xc))
        ry = float(np.ptp(self.yc))
        ax = 0 if rx >= ry else 1
        if ax != self.axis:
            self.axis = ax
            self.xs = list(self.xc if ax == 0 else self.yc)

    def feed(self, phase, tracker, video_t, frame_idx):
        dx = tracker.last[0] - tracker.origin[0] - tracker.shift[0]
        dy = tracker.last[1] - tracker.origin[1] - tracker.shift[1]
        if phase == Phase.WAIT and tracker.score >= MIN_QUALITY:
            if self.axis is None:
                if max(abs(dx), abs(dy)) > max(10, tracker.r * AXIS_MIN_DISPLACEMENT):
                    cand = 0 if abs(dx) >= abs(dy) else 1
                    self.streak = self.streak + 1 if cand == self.cand else 1
                    self.cand = cand
                    if self.streak >= AXIS_FRAMES:
                        self.axis = cand
                        dev = dx if cand == 0 else dy
                        self.side = 1 if dev > 0 else -1
                else:
                    self.streak = 0
            elif (dx if self.axis == 0 else dy) * self.side <= 0:
                phase = Phase.MEASURE
                self.t0 = video_t
                self.start_frame = frame_idx
                self.times.clear()
                self.xs.clear()
                self.xc.clear()
                self.yc.clear()
                self.snaps.clear()
                self.next_snap = 0.0
        if phase == Phase.MEASURE:
            self.t_now = video_t - self.t0
            self.x_cm = dx * tracker.px_to_cm
            self.y_cm = -dy * tracker.px_to_cm
            li = self.x_cm if self.axis == 0 else self.y_cm
            self.times.append(self.t_now)
            self.xs.append(li)
            self.xc.append(self.x_cm)
            self.yc.append(self.y_cm)
            if self.t_now >= self.next_snap - 1e-9:
                self.snaps.append(
                    (self.t_now, video_t, frame_idx,
                     round(tracker.last[0] - tracker.shift[0]), round(tracker.last[1] - tracker.shift[1]),
                     self.x_cm, self.y_cm, li, tracker.r, float(tracker.score))
                )
                self.next_snap = (int(self.t_now / SNAP_INTERVAL_S + 1e-6) + 1) * SNAP_INTERVAL_S
        return phase


def step(phase, tracker, meas, gray, t, idx):
    pos = tracker.update(gray, phase >= Phase.WAIT, phase == Phase.MEASURE, t)
    if tracker.stab is not None:
        tracker.shift = tracker.stab.update(gray, tracker.last)
        tracker.max_shift = max(tracker.max_shift, float(np.hypot(*tracker.shift)))
    if pos is None or phase < Phase.WAIT:
        return phase
    return meas.feed(phase, tracker, t, idx)


def run_headless(path, center, edge):
    src = VideoSource(path)
    tracker = make_tracker(cv2.cvtColor(src.first, cv2.COLOR_BGR2GRAY), center, edge)
    if tracker is None:
        src.close()
        return None
    meas = Measurement()
    phase = Phase.WAIT
    while True:
        frame = src.read()
        if frame is None:
            break
        phase = step(phase, tracker, meas, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), src.t, src.idx)
    src.close()
    meas.finalize()
    return tracker, meas, src, phase == Phase.MEASURE


def zero_crossings(t, x):
    out = []
    for i in range(1, len(x)):
        if (x[i - 1] < 0) != (x[i] < 0):
            out.append(t[i - 1] + (t[i] - t[i - 1]) * (-x[i - 1]) / (x[i] - x[i - 1]))
    return np.array(out)


def steady_range(c):
    if len(c) < 4:
        return 0, len(c)
    hp = np.diff(c)
    med = float(np.median(hp))
    good = list(np.abs(hp - med) <= STEADY_TOL * med) + [False]
    best = (0, 0)
    start = None
    for i, ok in enumerate(good):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best[1] - best[0] < 3:
        return 0, len(c)
    return best[0], best[1] + 1


def fit_damped(t, x, T_guess):
    tc = float(np.mean(t))
    u = np.asarray(t, float) - tc
    y = np.asarray(x, float)
    n = len(u)
    if n < 20:
        return None
    w = 2.0 * np.pi / T_guess
    gam = 0.02

    def design(w, gam):
        e = np.exp(-gam * u)
        return np.column_stack([e * np.cos(w * u), e * np.sin(w * u), np.ones(n)])

    coef = np.linalg.lstsq(design(w, gam), y, rcond=None)[0]
    p = np.array([coef[0], coef[1], coef[2], w, gam])
    lam = 1e-3

    def model(p):
        a, b, c, w_, g_ = p
        return np.exp(-g_ * u) * (a * np.cos(w_ * u) + b * np.sin(w_ * u)) + c

    def jac(p):
        a, b, _, w_, g_ = p
        e = np.exp(-g_ * u)
        co, si = np.cos(w_ * u), np.sin(w_ * u)
        f0 = e * (a * co + b * si)
        return np.column_stack([e * co, e * si, np.ones(n), e * u * (-a * si + b * co), -u * f0])

    r = y - model(p)
    rss = float(r @ r)
    for _ in range(60):
        J = jac(p)
        A = J.T @ J
        gvec = J.T @ r
        try:
            dp = np.linalg.solve(A + lam * np.diag(np.diag(A)), gvec)
        except np.linalg.LinAlgError:
            return None
        q = p + dp
        r2 = y - model(q)
        rss2 = float(r2 @ r2)
        if rss2 < rss:
            done = abs(rss - rss2) < 1e-12 * max(rss, 1e-12)
            p, r, rss = q, r2, rss2
            lam = max(lam / 3.0, 1e-9)
            if done:
                break
        else:
            lam *= 5.0
            if lam > 1e8:
                break
    if p[3] <= 0:
        return None
    J = jac(p)
    s2 = rss / (n - 5)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        return None
    sw = float(np.sqrt(max(cov[3, 3], 0.0)))
    rho = 0.0
    if n > 3 and float(r[:-1] @ r[:-1]) > 0:
        rho = float(np.clip((r[:-1] @ r[1:]) / (r[:-1] @ r[:-1]), 0.0, 0.9))
    sw *= float(np.sqrt((1.0 + rho) / (1.0 - rho)))
    T = 2.0 * np.pi / p[3]
    return {"T": T, "sT": T * sw / p[3], "offset": float(p[2]), "rms": float(np.sqrt(rss / n)),
            "amp0": float(np.hypot(p[0], p[1])), "rho": rho, "gamma": float(p[4])}


def analyze(times, xs):
    t = np.asarray(times, float)
    x = np.asarray(xs, float)
    crossings = zero_crossings(t, x)
    if len(crossings) < 5:
        return None
    guess = 2.0 * float(np.median(np.diff(crossings)))
    fit = None
    sel_fit = (t >= crossings[0]) & (t <= crossings[-1])
    if sel_fit.sum() >= 20:
        fit = fit_damped(t[sel_fit], x[sel_fit], guess)
    if fit is not None:
        bad = abs(fit["T"] - guess) > 0.15 * guess or fit["rms"] > 0.35 * fit["amp0"]
        if bad:
            fit = None
    if fit is not None:
        x = x - fit["offset"]
        crossings = zero_crossings(t, x)
    i0, i1 = steady_range(crossings)
    cr = crossings[i0:i1]
    if len(cr) < 4:
        return None
    k = np.arange(len(cr), dtype=float)
    kc = k - k.mean()
    slope = float(np.sum(kc * (cr - cr.mean())) / np.sum(kc ** 2))
    res = cr - (cr.mean() + slope * kc)
    s2 = float(np.sum(res ** 2) / (len(cr) - 2))
    T = 2.0 * slope
    sT = 2.0 * float(np.sqrt(s2 / np.sum(kc ** 2)))
    T_cross, sT_cross = T, sT
    if fit is not None:
        T = fit["T"]
        sT = max(fit["sT"], 1e-5)

    peaks, peak_t = [], []
    for a, b in itertools.pairwise(cr):
        sel = (t >= a) & (t < b)
        if sel.sum() >= 3:
            j = int(np.argmax(np.abs(x[sel])))
            peaks.append(abs(float(x[sel][j])))
            peak_t.append(float(t[sel][j]))
    amp = float(np.mean(peaks)) if peaks else float(np.max(np.abs(x)))
    tau = Q = None
    if len(peaks) >= 4 and min(peaks) > 0:
        slope_ln = float(np.polyfit(peak_t, np.log(peaks), 1)[0])
        if slope_ln < 0:
            tau = -1.0 / slope_ln
            Q = float(np.pi * tau / T)

    hp = np.diff(cr)
    offset_mm = None
    if len(hp) >= 4:
        diff = abs(float(hp[0::2].mean() - hp[1::2].mean()))
        offset_mm = 10.0 * (2 * np.pi / T) * amp * diff / 4.0

    L = PENDULUM_LENGTH_CM
    r_cm = BALL_DIAMETER_CM / 2.0
    L_eff = L + 0.4 * r_cm ** 2 / L
    theta = float(np.arcsin(min(amp / L, 0.99)))
    factor = 1.0 + theta ** 2 / 16.0 + 11.0 * theta ** 4 / 3072.0
    T0 = T / factor
    g = 4 * np.pi ** 2 * (L_eff / 100.0) / T0 ** 2
    sg = g * float(np.hypot(LENGTH_UNCERTAINTY_CM / L_eff, 2.0 * sT / T))
    return {
        "crossings": crossings,
        "used": (i0, i1),
        "T": T,
        "sT": sT,
        "T_cross": T_cross,
        "sT_cross": sT_cross,
        "fit": fit,
        "scatter": float(np.sqrt(s2)),
        "amp": amp,
        "amp_max": float(np.max(np.abs(x))),
        "tau": tau,
        "Q": Q,
        "offset_mm": offset_mm,
        "theta_deg": float(np.degrees(theta)),
        "factor": factor,
        "T0": T0,
        "L_eff": L_eff,
        "g": float(g),
        "sg": sg,
        "T_th_simple": float(2 * np.pi * np.sqrt(L / 100.0 / G_LOCAL)),
        "T_th": float(2 * np.pi * np.sqrt(L_eff / 100.0 / G_LOCAL) * factor),
    }


def smooth(x):
    if len(x) < 3:
        return x
    return np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def draw_graph(t, x, res):
    W, H = 960, 720
    img = np.full((H, W, 3), 255, np.uint8)
    left, right, top, bottom = 80, W - 20, 60, H - 130
    amp = max(float(np.max(np.abs(x))), 0.1)
    t_max = max(float(t[-1]), 0.1)

    def px(v):
        return int(left + v / t_max * (right - left))

    def py(v):
        return int((top + bottom) / 2 - v / amp * ((bottom - top) / 2))

    black = (0, 0, 0)
    cv2.rectangle(img, (left, top), (right, bottom), black, 1)
    cv2.line(img, (left, py(0)), (right, py(0)), (150, 150, 150), 1)
    pts = np.array([[px(a), py(b)] for a, b in zip(t, smooth(x))], np.int32)
    cv2.polylines(img, [pts], False, (255, 0, 0), 2)
    i0, i1 = res["used"]
    for k, c in enumerate(res["crossings"]):
        color = (0, 0, 255) if i0 <= k < i1 else (150, 150, 150)
        cv2.circle(img, (px(c), py(0)), 5, color, -1)
    for k in range(6):
        tv = t_max * k / 5
        put(img, f"{tv:.1f}", px(tv) - 15, bottom + 25, black, 0.5)
    put(img, "x(t) - Li do (cm)", left, 35, black, 0.8, 2)
    put(img, f"+{amp:.1f}", 5, top + 12, black, 0.5)
    put(img, "0", 50, py(0) + 5, black, 0.5)
    put(img, f"-{amp:.1f}", 5, bottom, black, 0.5)
    put(img, "t (s)", right - 50, bottom + 50, black, 0.6)
    put(img, f"T = {res['T']:.4f} +- {res['sT']:.4f} s | T ly thuyet: {res['T_th']:.4f} s", left, bottom + 80, black, 0.65)
    put(img, f"g = {res['g']:.2f} +- {res['sg']:.2f} m/s2 | Bien do TB: {res['amp']:.2f} cm", left, bottom + 110, black, 0.65)
    return img


def col_name(i):
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def cell_xml(r, c, v):
    ref = f"{col_name(c)}{r}"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f'<c r="{ref}"><v>{float(v)!r}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'


def sheet_xml(rows, widths):
    cols = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>'
        for i, w in enumerate(widths)
    )
    body = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(cell_xml(r, c, v) for c, v in enumerate(row))
        body.append(f'<row r="{r}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<cols>{cols}</cols><sheetData>{''.join(body)}</sheetData></worksheet>"
    )


def write_xlsx(path, sheets):
    n = len(sheets)
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{overrides}</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheet_tags = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, (name, _, _) in enumerate(sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_tags}</sheets></workbook>"
    )
    wb_rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n + 1)
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{wb_rels}</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        for i, (_, rows, widths) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows, widths))


def or_na(v):
    return "không xác định" if v is None else float(v)


def build_sheets(path, tracker, meas, src, res):
    data_rows = [
        (
            "Thời gian đo t (s)", "Mốc video (s)", "Khung hình", "x (px)", "y (px)",
            "Tọa độ x (cm)", "Tọa độ y (cm)", "Li độ (cm)", "Bán kính bi (px)", "Độ khớp",
        )
    ]
    for sn in meas.snaps:
        data_rows.append(
            (
                round(float(sn[0]), 2), round(float(sn[1]), 2), int(sn[2]), int(sn[3]), int(sn[4]),
                round(float(sn[5]), 2), round(float(sn[6]), 2), round(float(sn[7]), 2),
                int(sn[8]), round(float(sn[9]), 2),
            )
        )
    i0, i1 = res["used"]
    summary_rows = [
        ("Đại lượng", "Giá trị"),
        ("Nguồn", "Camera trực tiếp" if getattr(src, "live", False) else os.path.basename(path)),
        ("Độ rung khung hình lớn nhất đã bù (px)", round(float(getattr(tracker, "max_shift", 0.0)), 1)),
        ("Khung hình xoay 90° cho vừa màn hình", "có" if getattr(src, "rot", False) else "không"),
        ("Gốc tọa độ O (px)", f"({tracker.origin[0]}, {tracker.origin[1]})"),
        ("Phương dao động", "ngang (trục x)" if meas.axis == 0 else "dọc (trục y)"),
        ("Mốc bắt đầu đo trên video (s)", float(meas.t0)),
        ("Khung hình bắt đầu", int(meas.start_frame)),
        ("Mốc kết thúc đo trên video (s)", float(src.t)),
        ("FPS video (trung bình)", float(src.fps)),
        ("Chu kỳ lưu thông số (s)", SNAP_INTERVAL_S),
        ("Số lần qua vị trí cân bằng dùng để khớp", int(i1 - i0)),
        ("Chu kỳ thực nghiệm T (s)", res["T"]),
        ("Phương pháp tính T", "khớp đường cong tắt dần" if res["fit"] else "hồi quy các lần qua cân bằng"),
        ("T theo hồi quy các lần qua cân bằng (s)", res["T_cross"]),
        ("Chênh lệch giữa hai phương pháp (ms)", 1000.0 * abs(res["T"] - res["T_cross"])),
        ("Vị trí cân bằng thực so với O (cm)", or_na(res["fit"]["offset"] if res["fit"] else None)),
        ("Sai số RMS của đường cong khớp (cm)", or_na(res["fit"]["rms"] if res["fit"] else None)),
        ("Đóng góp của L vào độ bất định g (m/s2)", res["g"] * LENGTH_UNCERTAINTY_CM / res["L_eff"]),
        ("Đóng góp của T vào độ bất định g (m/s2)", res["g"] * 2.0 * res["sT"] / res["T"]),
        ("Sai số thống kê của T (s)", res["sT"]),
        ("Độ lệch chuẩn mốc qua vị trí cân bằng (ms)", 1000.0 * res["scatter"]),
        ("Biên độ trung bình (cm)", res["amp"]),
        ("Biên độ lớn nhất (cm)", res["amp_max"]),
        ("Biên độ góc trung bình (độ)", res["theta_deg"]),
        ("Hệ số hiệu chỉnh biên độ lớn", res["factor"]),
        ("Chu kỳ quy về biên độ nhỏ T0 (s)", res["T0"]),
        ("Chiều dài con lắc L đo được (cm)", PENDULUM_LENGTH_CM),
        ("Độ bất định của L (cm)", LENGTH_UNCERTAINTY_CM),
        ("Chiều dài hiệu dụng L (cm)", res["L_eff"]),
        ("g tham chiếu (m/s2)", G_LOCAL),
        ("Chu kỳ lý thuyết, L đo, chưa hiệu chỉnh (s)", res["T_th_simple"]),
        ("Chu kỳ lý thuyết, có hiệu chỉnh (s)", res["T_th"]),
        ("g thực nghiệm (m/s2)", res["g"]),
        ("Độ bất định của g (m/s2)", res["sg"]),
        ("Chênh lệch g so với tham chiếu (%)", 100.0 * (res["g"] - G_LOCAL) / G_LOCAL),
        ("Hằng số tắt dần (s)", or_na(res["tau"])),
        ("Hệ số phẩm chất Q", or_na(res["Q"])),
        ("Độ lệch của gốc O so với vị trí cân bằng (mm)", or_na(res["offset_mm"])),
        ("Chiều dài mỗi bên dây (cm)", SIDE_STRING_CM),
        ("Chiều dài thanh (cm)", BAR_LENGTH_CM),
        ("Khoảng cách 2 thanh (cm)", BAR_GAP_CM),
        ("Đường kính bi (cm)", BALL_DIAMETER_CM),
        ("Số dòng dữ liệu", len(meas.snaps)),
    ]
    data_notes = [
        ("",),
        ("CHÚ THÍCH",),
        (f"Mỗi dòng là một lần ghi, cách nhau {SNAP_INTERVAL_S:.0f} giây, bắt đầu từ lúc bi qua vị trí ban đầu.",),
        ("Thời gian đo t (s): thời gian tính từ lúc bắt đầu đo (bi qua vị trí ban đầu lần đầu).",),
        ("Mốc video (s): thời điểm tương ứng trong video hoặc trong lúc quay.",),
        ("Khung hình: số thứ tự khung hình của video.",),
        ("x (px), y (px): vị trí tâm miếng băng keo trên ảnh, tính bằng điểm ảnh (ảnh đã xoay nếu cần).",),
        ("Tọa độ x, y (cm): vị trí so với gốc O đã chọn, đã bù rung, quy đổi theo đường kính bi.",),
        ("y dương hướng lên trên ảnh, x dương hướng sang phải. Tên trục theo ảnh hiển thị.",),
        ("Li độ (cm): độ lệch dọc theo phương con lắc dao động. Dấu cộng hoặc trừ chỉ là hai phía so với O.",),
        ("Bán kính bi (px): bán kính quả bi đã chọn, dùng để quy đổi điểm ảnh ra cm.",),
        ("Độ khớp: mức giống giữa miếng băng keo tìm được và mẫu lúc chọn, từ 0 đến 1. Thấp khi ảnh nhòe hoặc bị che.",),
    ]
    summary_notes = [
        ("",),
        ("CHÚ THÍCH",),
        ("Chu kỳ thực nghiệm T: thời gian một dao động toàn phần, lấy từ toàn bộ đường cong dao động.",),
        ("Sai số thống kê của T: độ lặp lại của phép đo T. Chưa gồm sai số hệ thống của video.",),
        ("Độ lệch chuẩn mốc qua vị trí cân bằng: độ phân tán của các lần bi đi qua vị trí cân bằng.",),
        ("Biên độ: li độ cực đại. Biên độ góc: góc lệch lớn nhất của dây so với phương thẳng đứng.",),
        ("Hệ số hiệu chỉnh biên độ lớn và T0: biên độ lớn làm chu kỳ dài hơn một chút. T0 là chu kỳ quy về biên độ nhỏ.",),
        ("Chiều dài hiệu dụng L: L đo được cộng hiệu chỉnh do quả bi có kích thước.",),
        (f"Độ bất định của L: {LENGTH_UNCERTAINTY_CM:.1f} cm, là giá trị đặt sẵn trong chương trình, không phải kết quả đo.",),
        ("g thực nghiệm: tính từ T0 và L theo g = 4π²L/T0².",),
        ("Độ bất định của g gồm hai phần: do L và do T. Thường phần do L lớn hơn nhiều.",),
        ("Chênh lệch g so với tham chiếu: so với g địa phương đặt sẵn. Nhỏ hơn độ bất định thì coi là phù hợp.",),
        ("Chu kỳ lý thuyết: T tính từ L và g tham chiếu để đối chiếu với T đo được.",),
        ("Hằng số tắt dần (s): thời gian biên độ giảm còn 1/e. Hệ số phẩm chất Q càng lớn thì dao động càng ít tắt.",),
        ("Độ lệch của gốc O so với vị trí cân bằng: O chọn lệch bao nhiêu so với vị trí cân bằng thực. Đã được bù khi tính T.",),
        ("Độ rung khung hình lớn nhất đã bù: độ dịch lớn nhất của ảnh do rung tay đã được trừ đi.",),
        ("Chiều dài dây, thanh, khoảng cách 2 thanh: chỉ ghi lại để tham khảo, không dùng để tính g.",),
    ]
    return [
        ("Du lieu", data_rows + data_notes, [20, 16, 12, 10, 10, 16, 16, 14, 18, 12]),
        ("Tong ket", summary_rows + summary_notes, [46, 24]),
    ]


def stamp_name():
    d = datetime.now()
    return f"{d.day}_{d.month}_{d.year}_{d.hour}h_{d.minute}m_{d.second}s"


def save_outputs(path, sheets, graph):
    base = stamp_name()
    dirs = []
    for d in (os.path.dirname(os.path.abspath(path)), os.getcwd(), tempfile.gettempdir()):
        if d not in dirs:
            dirs.append(d)
    for d in dirs:
        name = base
        k = 2
        while os.path.exists(os.path.join(d, name)):
            name = f"{base}_{k}"
            k += 1
        folder = os.path.join(d, name)
        try:
            os.makedirs(folder)
            xlsx = os.path.join(folder, "du_lieu.xlsx")
            write_xlsx(xlsx, sheets)
        except OSError:
            continue
        png = os.path.join(folder, "bieu_do.png")
        if not cv2.imwrite(png, graph):
            png = None
        return xlsx, png
    return None, None


def show_result(graph, lines):
    click["pos"] = None
    gh, gw = graph.shape[:2]
    while True:
        W, H = window_size()
        fs = W / 1300
        canvas = np.full((H, W, 3), 255, np.uint8)
        s = min(W / gw, (H * 0.75) / gh)
        nw, nh = int(gw * s), int(gh * s)
        ox = (W - nw) // 2
        canvas[0:nh, ox:ox + nw] = cv2.resize(graph, (nw, nh))
        y = nh + int(40 * fs)
        for line in lines:
            put(canvas, line, int(20 * fs), y, (0, 110, 0), fs * 0.6, 1)
            y += int(34 * fs)
        put(canvas, "Cham man hinh de thoat", int(20 * fs), H - int(30 * fs), (200, 80, 0), fs * 0.7, 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if click["pos"] is not None or key in (ord("q"), 27):
            break


def show_notice(title, lines):
    click["pos"] = None
    while True:
        W, H = window_size()
        fs = W / 1300
        canvas = np.full((H, W, 3), 255, np.uint8)
        put(canvas, title, int(24 * fs), int(80 * fs), (0, 0, 200), fs * 1.0, 2)
        y = int(150 * fs)
        for line in lines:
            put(canvas, line, int(24 * fs), y, (40, 40, 40), fs * 0.75, 1)
            y += int(46 * fs)
        put(canvas, "Cham man hinh de DO LAI  |  phim Q de thoat", int(24 * fs), H - int(40 * fs), (200, 80, 0), fs * 0.75, 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if click["pos"] is not None:
            click["pos"] = None
            return True


def run_menu(title, labels, note=""):
    click["pos"] = None
    while True:
        W, H = window_size()
        fs = W / 1300
        canvas = np.full((H, W, 3), MENU_BG, np.uint8)
        put(canvas, title, int(30 * fs), int(70 * fs), (255, 255, 255), fs * 1.0, 2)
        if note:
            put(canvas, note, int(30 * fs), int(110 * fs), (230, 230, 230), fs * 0.6, 1)
        n = len(labels)
        top = int(150 * fs)
        bh = max(50, min(int(H * 0.09), (H - top - 20) // max(n, 1) - 12))
        rects = []
        for i, lab in enumerate(labels):
            y1 = top + i * (bh + 12)
            rect = (int(W * 0.06), y1, int(W * 0.94), y1 + bh)
            rects.append(rect)
            draw_button(canvas, rect, lab[:34], (150, 100, 40), True, fs * 0.8)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        tap = click["pos"]
        click["pos"] = None
        if key in (ord("q"), ord("Q"), 27):
            return None
        if ord("1") <= key <= ord("9") and key - ord("1") < n:
            return key - ord("1")
        if tap is not None:
            for i, rect in enumerate(rects):
                if hit(rect, tap[0], tap[1]):
                    return i


def find_videos(phone):
    exts = (".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".3gp")
    script_dir = os.path.dirname(os.path.abspath(globals().get("__file__", ".")))
    dirs = [os.getcwd(), script_dir]
    home = os.path.expanduser("~")
    for sub in ("Downloads", "Videos", "Movies", "Desktop", "Download"):
        dirs.append(os.path.join(home, sub))
    if phone:
        for base in ("/sdcard", "/storage/emulated/0"):
            for sub in ("DCIM/Camera", "Movies", "Download", "Pictures"):
                dirs.append(os.path.join(base, sub))
    found = []
    for d in dict.fromkeys(dirs):
        try:
            for name in os.listdir(d):
                if name.lower().endswith(exts):
                    full = os.path.join(d, name)
                    found.append((os.path.getmtime(full), full))
        except OSError:
            pass
    found.sort(reverse=True)
    return list(dict.fromkeys(p for _, p in found))[:60]


def browse_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return ""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Chon tep video con lac",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.3gp"), ("Tat ca", "*.*")],
        )
        root.destroy()
    except (tk.TclError, RuntimeError):
        return ""
    return path


def type_path():
    print("Nhap duong dan tep video (vi du /sdcard/Download/video.mp4):")
    try:
        return input("> ").strip().strip('"')
    except EOFError:
        return ""


def choose_file(phone):
    files = find_videos(phone)
    page = 0
    per = 5
    while True:
        chunk = files[page * per:(page + 1) * per]
        labels = [os.path.basename(p) for p in chunk]
        acts = [("file", p) for p in chunk]
        if (page + 1) * per < len(files):
            labels.append("TRANG SAU >")
            acts.append(("next", None))
        if page > 0:
            labels.append("< TRANG TRUOC")
            acts.append(("prev", None))
        if have_tk():
            labels.append("DUYET TEP...")
            acts.append(("browse", None))
        labels.append("NHAP DUONG DAN")
        acts.append(("type", None))
        labels.append("QUAY LAI")
        acts.append(("back", None))
        note = f"Tim thay {len(files)} video" if files else "Khong tim thay video trong cac thu muc thuong dung"
        i = run_menu("CHON VIDEO", labels, note)
        if i is None:
            return None
        kind, val = acts[i]
        if kind == "file":
            return val
        if kind == "next":
            page += 1
        elif kind == "prev":
            page -= 1
        elif kind == "browse":
            path = browse_file()
            if path:
                return path
        elif kind == "type":
            path = type_path()
            if path:
                return path
        else:
            return ""


def choose_source(phone):
    while True:
        dev = "Dien thoai" if phone else "May tinh"
        i = run_menu("CON LAC - CHON NGUON DO", ["QUAY TRUC TIEP (camera)", "TAI VIDEO LEN"], f"Thiet bi: {dev} (tu nhan dien)")
        if i is None:
            return None
        if i == 0:
            return ("cam", 0)
        path = choose_file(phone)
        if path is None:
            return None
        if path:
            return ("file", path)


def run_gui(choice, phone):
    W, H = window_size()
    try:
        if choice[0] == "cam":
            src = CameraSource(choice[1], W < H)
        else:
            src = VideoSource(choice[1], W < H)
    except OSError:
        print("Khong mo duoc camera" if choice[0] == "cam" else "Khong mo duoc tep video")
        return None

    phase = Phase.PICK
    taps = []
    tracker = None
    meas = Measurement()
    bad_tap = 0.0
    pending_reset = False
    finished = False
    wait_t0 = None

    while True:
        t_loop = time.time()
        if phase >= Phase.WAIT or src.live:
            frame = src.read()
            if frame is None:
                finished = phase == Phase.MEASURE
                break
        else:
            frame = src.first.copy()
        h, w = frame.shape[:2]
        W, H = window_size()
        lay = layout(W, H, w, h)
        rects = lay["rects"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tap = click["pos"]
        click["pos"] = None
        stop = False
        do_reset = pending_reset
        pending_reset = False

        if tap is not None:
            tx, ty = tap
            in_video = ty < lay["area"]
            if hit(rects["reset"], tx, ty):
                do_reset = True
            elif hit(rects["start"], tx, ty):
                if phase == Phase.READY:
                    phase = Phase.WAIT
                    wait_t0 = None
                    meas.clear()
                    if src.live:
                        src.rewind()
            elif hit(rects["stop"], tx, ty):
                if phase >= Phase.WAIT:
                    stop = True
            elif phase == Phase.WAIT and tracker is not None and in_video:
                fx, fy = to_frame(lay, W, tx, ty)
                if 0 <= fx < w and 0 <= fy < h:
                    tracker.reseed((fx, fy), gray)
                    wait_t0 = None
            elif phase == Phase.PICK and in_video:
                fx, fy = to_frame(lay, W, tx, ty)
                if 0 <= fx < w and 0 <= fy < h:
                    taps.append((fx, fy))
                    if len(taps) == 2:
                        tracker = make_tracker(gray, taps[0], taps[1])
                        if tracker is None:
                            taps = []
                            bad_tap = time.time()
                        else:
                            phase = Phase.READY

        if do_reset:
            phase = Phase.PICK
            taps = []
            tracker = None
            meas.clear()
            src.rewind()

        if phase >= Phase.READY:
            phase = step(phase, tracker, meas, gray, src.t, src.idx)
            color = (255, 0, 0) if not tracker.lost else (0, 165, 255)
            cv2.circle(frame, tracker.last, tracker.half, color, 2)
            cv2.circle(frame, tracker.last, 3, (0, 0, 255), -1)
            ox = round(tracker.origin[0] + tracker.shift[0])
            oy = round(tracker.origin[1] + tracker.shift[1])
            lw = 3 if meas.axis is not None else 1
            if SHOW_RED_LINE:
                if meas.axis != 1:
                    cv2.line(frame, (ox, 0), (ox, h), (0, 0, 255), lw)
                if meas.axis != 0:
                    cv2.line(frame, (0, oy), (w, oy), (0, 0, 255), lw)
            draw_axes(frame, ox, oy)
        elif taps:
            cv2.circle(frame, taps[0], 4, (0, 255, 255), -1)

        x0, y0, x1, y1 = lay["crop"]
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:lay["area"]] = cv2.resize(frame[y0:y1, x0:x1], (W, lay["area"]))
        canvas[lay["area"]:] = BAR_BG

        fs = W / 1300
        if phase == Phase.PICK:
            if len(taps) == 0 and time.time() - bad_tap < 3.5 and PICK_ERROR["msg"]:
                status = PICK_ERROR["msg"]
            elif len(taps) == 0:
                status = "Buoc 1: cham vao TAM bi (= goc toa do O)"
            else:
                status = "Buoc 2: cham vao MEP bi"
            status_color = (0, 255, 255)
        elif phase == Phase.READY:
            status = "San sang. Bam BAT DAU" + (" de do" if src.live else " de phat video")
            status_color = (0, 200, 255)
        elif phase == Phase.WAIT:
            status = f"{'Cam' if src.live else 'Video'} {src.t:.1f}s. Cho bi qua vi tri ban dau..."
            status_color = (0, 255, 255)
            if wait_t0 is None:
                wait_t0 = src.t
            if src.t - wait_t0 > WAIT_WARN_S:
                status = "Chua thay bi qua O. Cham vao bi de bam lai, hoac CHON LAI"
        else:
            status = f"DO t={meas.t_now:.2f}s x={meas.x_cm:.1f} y={meas.y_cm:.2f}cm"
            status_color = (0, 255, 0)
        if tracker is not None and tracker.lost and phase >= Phase.READY:
            status += "  [MAT DAU]"
            status_color = (0, 165, 255)
        put_o(canvas, status, int(12 * fs), int(40 * fs), status_color, fs * 0.95, 2)
        if src.live and src.t > 2.0 and src.fps < LIVE_MIN_FPS:
            warn = f"Camera cham ({src.fps:.0f} FPS): them anh sang hoac quay video roi tai len"
            put_o(canvas, warn, int(12 * fs), int(80 * fs), (0, 0, 255), fs * 0.7, 2)

        draw_button(canvas, rects["reset"], "CHON LAI", (120, 80, 40), True, fs)
        draw_button(canvas, rects["start"], "BAT DAU", (60, 160, 60), phase == Phase.READY, fs)
        draw_button(canvas, rects["stop"], "KET THUC", (50, 50, 200), phase >= Phase.WAIT, fs)

        cv2.imshow(WINDOW, canvas)
        delay = 1
        if not src.live and phase >= Phase.WAIT and PLAYBACK_SPEED > 0:
            delay = max(1, int(1000.0 / (src.fps * PLAYBACK_SPEED) - (time.time() - t_loop) * 1000.0))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            stop = True
        elif key in (ord("r"), ord("R")):
            pending_reset = True

        if stop:
            finished = phase == Phase.MEASURE
            break

    src.close()
    return finished, tracker, meas, src, choice


def report(path, tracker, meas, src, res):
    graph = draw_graph(np.array(meas.times), np.array(meas.xs), res)
    xlsx, png = save_outputs(path, build_sheets(path, tracker, meas, src, res), graph)
    lines = [
        f"T = {res['T']:.4f} +- {res['sT']:.4f} s | T ly thuyet = {res['T_th']:.4f} s",
        f"g = {res['g']:.2f} +- {res['sg']:.2f} m/s2 | A = {res['amp']:.2f} cm",
    ]
    for line in lines:
        print(line)
    if xlsx is None:
        print("Khong luu duoc tep ket qua")
        lines.append("Khong luu duoc tep ket qua")
    else:
        print(f"Da luu: {xlsx}" + (f", {png}" if png else ""))
        lines.append(f"Da luu vao thu muc: {os.path.basename(os.path.dirname(xlsx))}")
    show_result(graph, lines)


def make_test_video(path, period, tape, fps=30.0, seconds=12.0):
    w, h = 640, 480
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        return False
    bg, ball = (242, 165) if tape else (235, 110)
    for i in range(int(fps * seconds)):
        t = i / fps
        img = np.full((h, w, 3), bg, np.uint8)
        for k in range(10):
            cv2.putText(img, "ab cd", (20 + k * 60, 40 + (k % 3) * 20), FONT, 0.4, (30, 30, 30), 1)
        u = max(0.0, t - 1.0)
        y = 240 + 55 * np.sin(2 * np.pi * u / period) * np.exp(-0.04 * u)
        cx, cy = 320, round(y)
        cv2.circle(img, (cx, cy), 20, (ball, ball, ball), -1)
        cv2.circle(img, (cx - 9, cy - 9), 4, (250, 250, 250), -1)
        if tape:
            cv2.rectangle(img, (cx - 2, cy - 15), (cx + 8, cy + 11), (20, 20, 20), -1)
        else:
            cv2.circle(img, (cx + 5, cy + 5), 4, (30, 30, 30), -1)
        vw.write(cv2.GaussianBlur(img, (3, 3), 0))
    vw.release()
    return True


def selftest():
    all_ok = True
    for tape, period in ((True, 0.74), (False, 0.85)):
        path = os.path.join(tempfile.gettempdir(), f"conlac_selftest_{int(tape)}.mp4")
        if not make_test_video(path, period, tape):
            print("Khong tao duoc video thu (thieu codec mp4v)")
            return False
        out = run_headless(path, (320, 240), (340, 240))
        res = None
        if out is not None and out[3]:
            res = analyze(out[1].times, out[1].xs)
        ok = res is not None and abs(res["T"] - period) / period < 0.01
        label = "co bang keo" if tape else "khong bang keo"
        if res is None:
            print(f"[HONG] {label}: khong do duoc")
        else:
            print(f"[{'DAT' if ok else 'HONG'}] {label}: T that {period:.3f} s, do duoc {res['T']:.4f} s")
        all_ok = all_ok and ok
        try:
            os.remove(path)
        except OSError:
            pass
    return all_ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    phone = is_phone()
    open_window(phone)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    choice = ("file", args[0]) if args else choose_source(phone)
    if choice is None:
        cv2.destroyAllWindows()
        print("Chua chon nguon do")
        return
    while True:
        out = run_gui(choice, phone)
        if out is None:
            cv2.destroyAllWindows()
            return
        finished, tracker, meas, src, choice = out
        res = None
        n = len(meas.times)
        if finished and n >= MIN_SAMPLES:
            meas.finalize()
            res = analyze(meas.times, meas.xs)
        if res is not None:
            break
        if meas.axis is None and n == 0:
            lines = ["Bi chua qua vi tri ban dau (O) nen chua bat dau do.",
                     "Hay de bi dao dong ro, roi bam BAT DAU va cho bi qua O."]
        elif not finished:
            lines = ["Chua bat dau do hoac ket thuc qua som.",
                     "Sau BAT DAU hay cho bi qua O, do it nhat 6 giay roi bam KET THUC."]
        else:
            lines = [f"Chi co {n} mau do, can it nhat {MIN_SAMPLES} mau va 5 lan qua can bang.",
                     "Hay do lau hon (6 giay tro len) va cho bi dao dong ro.",
                     "Camera cham thi quay video bang may roi chon TAI VIDEO LEN."]
        if choice[0] == "cam":
            lines.append(f"Camera thuc te: {src.fps:.0f} khung/giay")
        if not show_notice("CHUA CO KET QUA", lines):
            cv2.destroyAllWindows()
            return
    report(choice[1] if choice[0] == "file" else "camera", tracker, meas, src, res)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
