"""The Tkinter host UI: chat on the left, Holly + settings on the right.

Layout (the whole point of this module):

    +----------------------------------+----------------------+
    |                                  |   Holly animation    |
    |            chat log              |     (top half)       |
    |                                  +----------------------+
    |                                  |    settings panel    |
    +----------------------------------+----------------------+
    |            input line            |
    +----------------------------------+----------------------+
              left 2/3                        right 1/3

Everything the pygame shell guaranteed still holds here:

* the clock is the audio device's position, never wall time -- the tick polls
  `session.poll()` on a `root.after` timer exactly like the old `while running:` loop;
* only the mouth rect is redrawn, and only when `HollyFace` says the pose changed;
* the pump's sentences are drained on the UI thread (the only thread allowed to touch
  the face queue), and `session.say` keeps the face pinned to the audio tail.

The face is drawn on a Canvas: one static base image scaled to the pane, plus a small
mouth-rect image laid over it, redrawn in place on pose changes. Pillow does the scaling
(`ImageTk` is just the numpy -> Tk bridge); the component itself is untouched.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from .idle_motion import IdleMotion

# The dark palette the pygame shell had, kept on purpose.
BG = "#101012"
PANEL = "#18181c"
FIELD = "#202026"
EDGE = "#3a3a44"
FG = "#dcdcdc"
MUTED = "#9a9aa4"

_SPEAKER = {"you": "You", "holly": "Holly", "sys": "!"}


def _scaled(arr: np.ndarray, w: int, h: int):
    """numpy frame region -> PIL image at (w, h). NEAREST keeps the 8-bit look crisp."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
    if (w, h) != img.size:
        img = img.resize((max(1, w), max(1, h)), Image.NEAREST)
    return img


class FaceView:
    """A Canvas showing the baked face: static base image + a mouth-rect overlay.

    Mirrors the old host's blit rule: the base is drawn once per resize, and a pose
    change redraws only the (scaled) mouth rect via `update()`.
    """

    def __init__(self, canvas: tk.Canvas, face):
        self.canvas = canvas
        self.face = face
        self._w = self._h = 0
        self._fx = self._fy = 0
        self._scale = 1.0
        self._base_item = canvas.create_image(0, 0, anchor="nw", image="")
        self._mouth_item = canvas.create_image(0, 0, anchor="nw", image="")
        self._base_photo = None  # keep refs: Tk PhotoImages are garbage-collected eagerly
        self._mouth_photo = None
        canvas.bind("<Configure>", self._configure)

    def _configure(self, event) -> None:
        if (event.width, event.height) == (self._w, self._h):
            return
        self._w, self._h = event.width, event.height
        self._relayout()

    def _relayout(self) -> None:
        fw, fh = self.face.renderer.size
        if self._w < 16 or self._h < 16:
            return
        s = min(self._w / fw, self._h / fh)
        self._scale = s
        w, h = round(fw * s), round(fh * s)
        self._fx, self._fy = (self._w - w) // 2, (self._h - h) // 2
        self._base_photo = _photo(_scaled(self.face.renderer.frame, w, h))
        self.canvas.itemconfigure(self._base_item, image=self._base_photo)
        self.canvas.coords(self._base_item, self._fx, self._fy)
        self.update()  # the mouth overlay must land at the new scale too

    def update(self) -> None:
        """Redraw only the mouth rect -- the host's one cheap-draw rule, unchanged."""
        if self._w < 16:
            return  # before the first Configure there is no pane to draw into
        from PIL import ImageTk

        x, y, w, h = self.face.renderer.mouth_rect
        s = self._scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        region = _scaled(self.face.renderer.frame[y : y + h, x : x + w], x1 - x0, y1 - y0)
        self._mouth_photo = ImageTk.PhotoImage(region)
        self.canvas.itemconfigure(self._mouth_item, image=self._mouth_photo)
        self.canvas.coords(self._mouth_item, self._fx + x0, self._fy + y0)


def _photo(image):
    from PIL import ImageTk

    return ImageTk.PhotoImage(image)


# --------------------------------------------------------------------------- A1: interactive


class InteractiveUI:
    """Chat log + input on the left 2/3; Holly + settings on the right 1/3."""

    def __init__(self, root: tk.Tk, session, args, pump=None):
        self.session = session
        self.args = args
        self.pump = pump
        self.root = root
        self.running = True
        self.poll_ms = max(8, round(1000.0 / max(1.0, args.poll_hz)))
        self._open_para: str | None = None
        self._error_logged = False
        # E: opt-in idle breathing; the settings panel can turn it on and off live.
        self.breathe = IdleMotion(session.face, max_weight=args.idle_weight) if args.idle_motion else None

        root.configure(bg=BG)
        self._style()

        root.columnconfigure(0, weight=2)  # left 2/3
        root.columnconfigure(1, weight=1)  # right 1/3
        root.rowconfigure(0, weight=1)

        self._build_left(root)
        self._build_right(root)

        root.bind("<Escape>", lambda _e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
            style.configure(".", background=BG, foreground=FG)
            style.configure("TFrame", background=BG)
            style.configure("TLabel", background=BG, foreground=MUTED)
            style.configure("TLabelframe", background=BG, foreground=MUTED, bordercolor=EDGE)
            style.configure("TLabelframe.Label", background=BG, foreground=MUTED)
            style.configure("TButton", background=PANEL, foreground=FG, bordercolor=EDGE,
                            focuscolor=PANEL)
            style.map("TButton", background=[("active", EDGE)])
            style.configure("TCheckbutton", background=BG, foreground=FG)
            style.map("TCheckbutton", background=[("active", BG)])
            style.configure("TScale", background=BG, troughcolor=FIELD)
            style.configure("TScrollbar", background=PANEL, troughcolor=BG, bordercolor=EDGE,
                            arrowcolor=FG)
        except tk.TclError:
            pass  # a theme without these element options: default look, same layout

    # -- left: chat log + input ---------------------------------------------

    def _build_left(self, root: tk.Tk) -> None:
        left = tk.Frame(root, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        holder = tk.Frame(left, bg=BG)
        holder.grid(row=0, column=0, sticky="nsew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        self.log = tk.Text(holder, wrap="word", state="disabled", bg=PANEL, fg=FG,
                           relief="flat", padx=8, pady=6,
                           highlightthickness=1, highlightbackground=EDGE, highlightcolor=EDGE)
        bar = ttk.Scrollbar(holder, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.log.tag_configure("you", foreground="#9ecbff")
        self.log.tag_configure("holly", foreground=FG)
        self.log.tag_configure("sys", foreground="#e0a458", font=("TkDefaultFont", 9, "italic"))

        self.status = tk.Label(left, text="", bg=BG, fg=MUTED, anchor="w")
        self.status.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.entry = tk.Entry(left, bg=FIELD, fg=FG, insertbackground=FG, relief="flat",
                              highlightthickness=1, highlightbackground=EDGE, highlightcolor=EDGE,
                              font=("TkFixedFont", 12))
        self.entry.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.entry.focus_set()
        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<KP_Enter>", self._submit)
        self.entry.bind("<Tab>", self._tab)
        self.entry.bind("<space>", self._space)

    # -- right: Holly (top half) + settings (bottom half) --------------------

    def _build_right(self, root: tk.Tk) -> None:
        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)  # top half: the animation
        right.rowconfigure(1, weight=1)  # bottom half: the settings panel

        self.canvas = tk.Canvas(right, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.view = FaceView(self.canvas, self.session.face)

        settings = ttk.LabelFrame(right, text="Settings")
        settings.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        settings.columnconfigure(0, weight=1)

        self.idle_on = tk.BooleanVar(value=self.breathe is not None)
        self.idle_weight = tk.DoubleVar(value=float(self.args.idle_weight))

        ttk.Checkbutton(settings, text="Idle motion (breathing)", variable=self.idle_on,
                        command=self._toggle_idle).grid(row=0, column=0, sticky="w",
                                                        padx=10, pady=(8, 0))
        ttk.Label(settings, text="Breath depth").grid(row=1, column=0, sticky="w",
                                                      padx=10, pady=(6, 0))
        self.weight_scale = ttk.Scale(settings, from_=0.05, to=0.45, orient="horizontal",
                                      variable=self.idle_weight, command=self._weight_changed)
        self.weight_scale.grid(row=2, column=0, sticky="ew", padx=10)
        self.weight_value = ttk.Label(settings, text=f"{self.idle_weight.get():.2f}")
        self.weight_value.grid(row=3, column=0, sticky="e", padx=10)
        ttk.Button(settings, text="Stop Holly (Tab)", command=self._barge_in).grid(
            row=4, column=0, sticky="ew", padx=10, pady=(10, 8))

    # -- chat log ------------------------------------------------------------

    def _log(self, who: str, text: str) -> None:
        """Append to the chat log. Consecutive Holly sentences share one paragraph."""
        text = " ".join(str(text).split())
        if not text:
            return
        self.log.configure(state="normal")
        if who == "holly" and self._open_para == "holly":
            self.log.insert("end", " " + text, "holly")
        else:
            if self._open_para is not None:
                self.log.insert("end", "\n")
            label = _SPEAKER[who]
            if label:
                self.log.insert("end", f"{label}: ", who)
            self.log.insert("end", text, who)
            self._open_para = who
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    # -- input ---------------------------------------------------------------

    def _submit(self, _event=None) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._log("you", text)
        self._error_logged = False
        if self.pump is not None:
            if not self.pump.submit(text):  # chat thread: brain -> TTS -> queue
                self._set_status("Holly is still talking - press Tab to stop him")
        else:
            self.session.say(text)  # A1: speak exactly what was typed
        self._set_status("")

    def _tab(self, _event=None):
        self._barge_in()
        return "break"  # Tab interrupts, it never moves focus here

    def _space(self, _event=None):
        if not self.entry.get():
            self._barge_in()  # Space only types when the line already holds text
            return "break"

    def _barge_in(self) -> None:
        """E: stop playback, stop the queue, back to idle (the plan's barge-in, verbatim)."""
        if self.pump is not None:
            self.pump.interrupt()
        self.session.interrupt()
        self._set_status("")

    # -- settings panel --------------------------------------------------------

    def _toggle_idle(self) -> None:
        if self.idle_on.get():
            if self.breathe is None:
                self.breathe = IdleMotion(self.session.face, max_weight=self.idle_weight.get())
        elif self.breathe is not None:
            drew = self.breathe.poll(active=True)  # restores the pure idle pose
            self.breathe = None
            if drew:
                self.view.update()

    def _weight_changed(self, _value: str) -> None:
        self.weight_value.configure(text=f"{self.idle_weight.get():.2f}")
        if self.breathe is not None:
            self.breathe = IdleMotion(self.session.face, max_weight=self.idle_weight.get())

    # -- the tick --------------------------------------------------------------

    def start(self) -> None:
        self._tick()

    def _tick(self) -> None:
        if not self.running:
            return
        # Phase C: whatever the chat thread synthesised lands on the clock here, on the UI
        # thread -- the only thread allowed to touch the face queue or the audio tail.
        for said, samples, sample_rate in (self.pump.sentences() if self.pump else ()):
            self.session.say(said, samples=samples, rate=sample_rate)
            self._log("holly", said)

        # E: the idle overlay runs *before* the face poll: if speech has just arrived it
        # first restores the pure idle pose, which is what keeps the component's dirty
        # check truthful for the speech frames that follow.
        breathed = self.breathe.poll(active=self.session.face.pending > 0) if self.breathe else False

        # The face: redraw the mouth rect only when the pose changed.
        if self.session.poll()[2] or breathed:
            self.view.update()

        self._status_tick()
        if self.running:
            self.root.after(self.poll_ms, self._tick)

    def _status_tick(self) -> None:
        if self.pump is None:
            return
        if self.pump.error:
            if not self._error_logged:
                self._log("sys", "! " + self.pump.error)
                self._error_logged = True
            self._set_status("! " + self.pump.error)
        elif self.pump.busy and not self.pump.interrupted:
            self._set_status("thinking...")
        else:
            self._set_status("")

    def _quit(self) -> None:
        self.running = False
        self.root.quit()


def run_interactive(session, args, pump=None) -> int:
    """The interactive (A1) window. Same contract the pygame shell had: 0 on a clean exit."""
    root = tk.Tk()
    root.title("Holly host (type, Enter to speak)")
    root.geometry(_geometry(args.window))

    ui = InteractiveUI(root, session, args, pump)
    try:
        session.audio.start()
    except (ImportError, RuntimeError, OSError) as exc:  # no libportaudio / no device on this box
        print(f"audio output unavailable: {exc}\n  (try --dry, or install libportaudio2)", file=sys.stderr)
        root.destroy()
        return 1

    ui.start()
    root.mainloop()
    root.destroy()

    if pump is not None:
        pump.stop()
    session.audio.stop()
    return 0


# --------------------------------------------------------------------------- A0: batch


def run_batch(face, device, args) -> int:
    """The batch (A0) window: the face alone, mouth-only redraws, quit when the clock ends."""
    root = tk.Tk()
    root.title("Holly host - Phase A0")
    root.geometry(_geometry(args.window))
    root.configure(bg=BG)

    canvas = tk.Canvas(root, bg="#000000", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    view = FaceView(canvas, face)

    poll_ms = max(8, round(1000.0 / max(1.0, args.poll_hz)))
    started = time.perf_counter()

    def stop() -> None:
        root.quit()

    root.bind("<Escape>", lambda _e: stop())
    root.protocol("WM_DELETE_WINDOW", stop)

    def tick() -> None:
        changed = face.frame_at(device.t)[2]
        if device.finished:
            face.frame_at(device.t)  # let the return-to-rest pose settle
            stop()
            return
        if args.seconds is not None and time.perf_counter() - started > args.seconds:
            stop()
            return
        if changed:
            view.update()
        root.after(poll_ms, tick)

    device.start()
    root.after(poll_ms, tick)
    root.mainloop()
    root.destroy()

    device.stop()
    ds = face.stats
    print(f"polls {ds['polls']}  draws {ds['draws']}  skips {ds['skips']} "
          f"({face.draw_share():.1%})  underrun frames {getattr(device, 'underrun_frames', 0)}")
    return 0


def _geometry(spec: str) -> str:
    w, h = spec.lower().split("x")
    return f"{int(w)}x{int(h)}"
