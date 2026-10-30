"""Before/After zoom compare viewer for NpuStudio (Tk thread only)."""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

MINZ, MAXZ = 0.1, 16.0

class ZoomViewer:
    def __init__(self, root):
        self.root = root; self.win = None; self.before = None; self.after = None
        self.s = 1.0; self.ox = 0.0; self.oy = 0.0; self.fit = True
        self.split = False; self.div = 0.5; self.pending = False; self.drag = None; self.photos = []

    # ---------- public ----------
    def open(self, before, after):
        self.before, self.after = before, after
        self.f = (after.size[0] / before.size[0]) if after is not None else 1.0
        self.W, self.H = (after.size if after is not None else before.size)
        if self.win is None or not self.win.winfo_exists(): self._build()
        else: self.win.deiconify(); self.win.lift(); self.win.focus_force()
        self.win.title("NpuStudio - Compare" if after is not None else "NpuStudio - Before")
        self._layout(); self.fit = True
        self.win.after(60, self.do_fit)

    def set_zoom(self, z, cx=None, cy=None):
        """Zoom to z (1.0 = 100%) keeping pane point (cx,cy) fixed (default pane centre)."""
        w, h = self._pane_size()
        if cx is None: cx, cy = w / 2, h / 2
        z = max(MINZ, min(MAXZ, z))
        ix, iy = self.ox + cx / self.s, self.oy + cy / self.s
        self.s = z; self.ox = ix - cx / z; self.oy = iy - cy / z; self.fit = False
        self.redraw()

    def do_fit(self):
        w, h = self._pane_size()
        self.s = max(MINZ, min(MAXZ, min(w / self.W, h / self.H)))
        self.ox = (self.W - w / self.s) / 2; self.oy = (self.H - h / self.s) / 2; self.fit = True
        self.redraw()

    # ---------- ui ----------
    def _build(self):
        r = self.root; w = self.win = tk.Toplevel(r)
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w.geometry(f"{int(sw*0.75)}x{int(sh*0.75)}+{int(sw*0.12)}+{int(sh*0.1)}")
        bar = ttk.Frame(w); bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="Fit", command=self.do_fit).pack(side="left")
        for z in (1, 2, 4): ttk.Button(bar, text=f"{z*100}%", command=lambda z=z: self.set_zoom(z)).pack(side="left", padx=2)
        self.zl = ttk.Label(bar, text="", width=14, font=("Segoe UI", 10, "bold")); self.zl.pack(side="left", padx=10)
        self.splitv = tk.BooleanVar(master=w, value=False)
        self.splitcb = ttk.Checkbutton(bar, text="Split slider", variable=self.splitv, command=self._toggle_split)
        self.splitcb.pack(side="left", padx=6)
        ttk.Label(bar, text="Wheel = zoom   Drag = move   Double-click = Fit/100%", foreground="#666").pack(side="right")
        self.body = ttk.Frame(w); self.body.pack(fill="both", expand=True)
        self.cvs = []
        w.bind("<Configure>", lambda e: self._on_resize() if e.widget is w else None)
        w.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        self.win.destroy(); self.win = None; self.photos = []; self.before = self.after = None

    def _toggle_split(self):
        self.split = self.splitv.get(); self._layout(); self.win.after(30, self.do_fit if self.fit else self.redraw)

    def _layout(self):
        for c in self.cvs: c.destroy()
        self.cvs = []
        two = self.after is not None and not self.split
        self.splitcb.state(["!disabled"] if self.after is not None else ["disabled"])
        if self.after is None: self.split = False
        n = 2 if two else 1
        self.labels = ([f"Before ({self.f:g}x nearest)", "After (NPU)"] if two else
                       ["Before | After"] if self.split else ["Before"])
        for i in range(n):
            c = tk.Canvas(self.body, bg="#1e1e1e", highlightthickness=0, cursor="fleur")
            c.pack(side="left", fill="both", expand=True, padx=(0, 2 if i == 0 and n == 2 else 0))
            c.bind("<MouseWheel>", self._wheel); c.bind("<ButtonPress-1>", self._press)
            c.bind("<B1-Motion>", self._move); c.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))
            c.bind("<Double-Button-1>", self._dbl)
            self.cvs.append(c)

    def _pane_size(self):
        if not self.cvs: return 800, 600
        c = self.cvs[0]; c.update_idletasks()
        return max(c.winfo_width(), 50), max(c.winfo_height(), 50)

    def _on_resize(self):
        if self.fit: self.do_fit()
        else: self.redraw()

    # ---------- input ----------
    def _wheel(self, e):
        k = 1.25 if e.delta > 0 else 1 / 1.25
        self.set_zoom(self.s * k, e.x, e.y)

    def _press(self, e):
        if self.split:
            w, _ = self._pane_size()
            if abs(e.x - self.div * w) < 10: self.drag = ("div",); return
        self.drag = ("pan", e.x, e.y, self.ox, self.oy)

    def _move(self, e):
        if not self.drag: return
        if self.drag[0] == "div":
            w, _ = self._pane_size(); self.div = min(1, max(0, e.x / w)); self.redraw(); return
        _, x0, y0, ox, oy = self.drag
        self.ox = ox - (e.x - x0) / self.s; self.oy = oy - (e.y - y0) / self.s; self.fit = False; self.redraw()

    def _dbl(self, e):
        if self.fit: self.set_zoom(1.0, e.x, e.y)
        else: self.do_fit()

    # ---------- render ----------
    def redraw(self):
        if self.pending or self.win is None: return
        self.pending = True; self.win.after(15, self._render)

    def _region(self, img, scale, w, h):
        """Render the visible part of img (whose pixels are `scale` view-units wide) into a w x h pane."""
        s = self.s
        ix0, iy0 = max(0.0, self.ox), max(0.0, self.oy)
        ix1, iy1 = min(self.W, self.ox + w / s), min(self.H, self.oy + h / s)
        if ix1 <= ix0 or iy1 <= iy0: return None, 0, 0
        dx, dy = int(round((ix0 - self.ox) * s)), int(round((iy0 - self.oy) * s))
        dw, dh = max(1, int(round((ix1 - ix0) * s))), max(1, int(round((iy1 - iy0) * s)))
        box = (ix0 / scale, iy0 / scale, ix1 / scale, iy1 / scale)
        eff = s * scale  # screen px per source px
        rs = Image.NEAREST if eff >= 1 else Image.BILINEAR
        kw = {"reducing_gap": 2.0} if eff < 0.5 else {}
        return img.resize((dw, dh), rs, box=box, **kw), dx, dy

    def _render(self):
        self.pending = False
        if self.win is None or not self.win.winfo_exists(): return
        w, h = self._pane_size(); self.photos = []
        srcs = []
        if self.after is None: srcs = [(self.before, 1.0)]
        elif self.split: srcs = [None]
        else: srcs = [(self.before, self.f), (self.after, 1.0)]
        for i, c in enumerate(self.cvs):
            c.delete("all")
            if self.split:
                a, dx, dy = self._region(self.after, 1.0, w, h)
                b, _, _ = self._region(self.before, self.f, w, h)
                if a is not None:
                    cut = int(self.div * w) - dx
                    if cut > 0: a.paste(b.crop((0, 0, min(cut, b.size[0]), b.size[1])), (0, 0))
                    ph = ImageTk.PhotoImage(a); self.photos.append(ph); c.create_image(dx, dy, anchor="nw", image=ph)
                x = int(self.div * w)
                c.create_line(x, 0, x, h, fill="#ffcc00", width=2)
                c.create_oval(x - 9, h / 2 - 9, x + 9, h / 2 + 9, fill="#ffcc00", outline="")
                c.create_text(x - 12, 12, text="Before", anchor="ne", fill="white", font=("Segoe UI", 11, "bold"))
                c.create_text(x + 12, 12, text="After", anchor="nw", fill="white", font=("Segoe UI", 11, "bold"))
            else:
                img, sc = srcs[i]
                im, dx, dy = self._region(img, sc, w, h)
                if im is not None:
                    ph = ImageTk.PhotoImage(im); self.photos.append(ph); c.create_image(dx, dy, anchor="nw", image=ph)
                t = c.create_text(12, 18, text=self.labels[i], anchor="w", fill="white", font=("Segoe UI", 11, "bold"))
                x0, y0, x1, y1 = c.bbox(t); r_ = c.create_rectangle(x0 - 6, y0 - 3, x1 + 6, y1 + 3, fill="#000000", outline="")
                c.tag_lower(r_, t)
        self.zl.configure(text=f"Zoom: {self.s*100:.0f}%" + (" (fit)" if self.fit else ""))
