"""Display-only extra viewpoints; never changes recording cameras or physics."""

import contextlib
import queue
import time
from types import SimpleNamespace

import mujoco
import numpy as np

VIEW_NAMES = ("Front", "Overhead / top", "Side", "Diagonal")
CONTROLS = (
    ("Start [s]", "s"),
    ("Finish [f]", "f"),
    ("Keep [k]", "k"),
    ("Discard [d]", "d"),
    ("Reset [r]", "r"),
    ("Quit [q]", "q"),
)


class MultiViewRenderer:
    def __init__(self, model, height=240, width=320):
        self.height, self.width = height, width
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cameras = {}
        for name, azimuth, elevation in (
            ("Front", 180, -12),
            ("Side", 90, -12),
            ("Diagonal", 225, -35),
        ):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = [0.12, 0.04, 0.05 if name == "Front" else 0.02]
            camera.distance = 0.352
            camera.azimuth, camera.elevation = azimuth, elevation
            self.cameras[name] = camera

    def render(self, world):
        from PIL import Image

        images = {}
        for name, camera in self.cameras.items():
            self.renderer.update_scene(world.data, camera=camera)
            images[name] = self.renderer.render().copy()
        # Move only the display scene's stereo cameras, never model camera poses.
        self.renderer.update_scene(world.data, camera="overhead")
        for camera in self.renderer.scene.camera:
            camera.pos[2] -= 0.10
        overhead = Image.fromarray(self.renderer.render().copy())
        # Display-only central crop (~1.95x zoom); never edit dataset pixels.
        width, height = overhead.size
        margin_x, margin_y = round(width * 0.244), round(height * 0.244)
        images["Overhead / top"] = np.asarray(
            overhead.crop(
                (margin_x, margin_y, width - margin_x, height - margin_y)
            ).resize((self.width, self.height), Image.Resampling.BILINEAR)
        )
        return {name: images[name] for name in VIEW_NAMES}

    def close(self):
        self.renderer.close()


def _preview_process(model, snapshots, commands, stop):
    """Own the GUI and extra rendering independently of the control process."""
    import tkinter as tk

    from PIL import Image, ImageOps, ImageTk

    root = tk.Tk()
    root.title("SO-101 demonstration collection")
    root.geometry("1280x800")
    root.minsize(640, 480)
    root.configure(background="#222222")
    # Linux window managers support -zoomed; retain decorations for moving it.
    try:
        root.attributes("-zoomed", True)
    except tk.TclError:
        pass
    status = tk.StringVar(value="Starting collection…")
    canvases, photos, items = {}, {}, {}
    for index, name in enumerate(VIEW_NAMES):
        row, column = divmod(index, 2)
        root.rowconfigure(row, weight=1, uniform="views")
        root.columnconfigure(column, weight=1, uniform="views")
        panel = tk.Frame(root, background="#222222")
        panel.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        tk.Label(panel, text=name, background="#222222", foreground="white").grid(
            row=0, column=0
        )
        canvas = tk.Canvas(
            panel, background="#111111", highlightthickness=0, width=1, height=1
        )
        canvas.grid(row=1, column=0, sticky="nsew")
        canvases[name] = canvas
        items[name] = canvas.create_image(0, 0, anchor="center")
    tk.Label(root, textvariable=status, background="#222222", foreground="white").grid(
        row=2, column=0, columnspan=2, pady=5
    )
    controls = tk.Frame(root)
    controls.grid(row=3, column=0, columnspan=2, pady=5)

    def send(key):
        commands.put(key)

    for column, (label, key) in enumerate(CONTROLS):
        tk.Button(controls, text=label, command=lambda key=key: send(key)).grid(
            row=0, column=column, padx=3
        )

    def on_key(event):
        key = "q" if event.keysym == "Escape" else event.char.lower()
        if key in {key for _, key in CONTROLS}:
            send(key)

    root.bind("<KeyPress>", on_key)
    root.protocol("WM_DELETE_WINDOW", lambda: send("q"))
    frames = MultiViewRenderer(model)
    data = mujoco.MjData(model)
    world = SimpleNamespace(data=data)
    mask = mujoco.mjtState.mjSTATE_INTEGRATION

    def refresh():
        if stop.is_set():
            root.quit()
            return
        latest = None
        try:
            while True:
                latest = snapshots.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            state, message = latest
            mujoco.mj_setState(model, data, state, mask)
            mujoco.mj_forward(model, data)
            images = frames.render(world)
            status.set(message)
            for name, image in images.items():
                canvas = canvases[name]
                size = (max(1, canvas.winfo_width()), max(1, canvas.winfo_height()))
                fitted = ImageOps.contain(
                    Image.fromarray(image), size, Image.Resampling.BILINEAR
                )
                photos[name] = ImageTk.PhotoImage(fitted, master=root)
                canvas.itemconfigure(items[name], image=photos[name])
                canvas.coords(items[name], size[0] / 2, size[1] / 2)
        # Poll promptly; snapshot publication sets the display-rate ceiling.
        root.after(1 if latest is not None else 5, refresh)

    try:
        root.after(5, refresh)
        root.mainloop()
    finally:
        frames.close()
        root.destroy()


class MultiViewPreview:
    """Publish bounded display snapshots; never wait for GUI rendering."""

    def __init__(self, model, fps=30, on_command=None):
        import multiprocessing as mp

        context = mp.get_context("spawn")
        self.snapshots = context.Queue(maxsize=1)
        self.commands = context.Queue()
        self.stop = context.Event()
        self.period = 1 / fps
        self.last_render = -float("inf")
        self.on_command = on_command
        self.is_open = True
        self.process = context.Process(
            target=_preview_process,
            args=(model, self.snapshots, self.commands, self.stop),
            daemon=True,
        )
        self.process.start()

    def update(self, world, lock=None, status="Preview — s: start"):
        if not self.is_open:
            return
        if not self.process.is_alive():
            raise RuntimeError("Collection window stopped unexpectedly")
        try:
            while True:
                command = self.commands.get_nowait()
                if self.on_command is not None:
                    self.on_command(command)
        except queue.Empty:
            pass
        now = time.monotonic()
        if (
            now - self.last_render < self.period - min(0.001, self.period * 0.05)
            or self.snapshots.full()
        ):
            return
        mask = mujoco.mjtState.mjSTATE_INTEGRATION
        with lock if lock is not None else contextlib.nullcontext():
            state = np.empty(mujoco.mj_stateSize(world.model, mask))
            mujoco.mj_getState(world.model, world.data, state, mask)
        try:
            self.snapshots.put_nowait((state, status))
        except queue.Full:
            return
        self.last_render = now

    def close(self):
        if not self.is_open:
            return
        self.is_open = False
        self.stop.set()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for channel in (self.snapshots, self.commands):
            channel.cancel_join_thread()
            channel.close()
