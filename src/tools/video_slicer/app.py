from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    END,
    Entry,
    Frame,
    Label,
    Listbox,
    Radiobutton,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)

import cv2
from PIL import Image, ImageTk

from config import SlicerConfig
from extractor import FrameProcessingError, VideoFrameExtractor
from player import VideoPlayer


class VideoFrameSlicerApp:
    def __init__(self, config: SlicerConfig) -> None:
        self.root = Tk()
        self.root.title("Video Frame Slicer")
        self.root.geometry("1180x760")
        self.root.minsize(940, 620)

        self.config = config
        self.extractor = VideoFrameExtractor(config)
        self.player = VideoPlayer()
        self.video_files = []
        self.video_index = 0
        self.saved_count = 0
        self.is_running = False
        self.is_paused = True
        self.current_raw_frame = None
        self.current_processed_frame = None
        self.current_frame_id = 0
        self.photo = None
        self.after_id = None

        self.video_folder_var = StringVar(value=str(config.video_folder))
        self.output_folder_var = StringVar(value=str(config.output_folder))
        self.selected_files_var = StringVar(value="Selected files: 0")
        self.max_count_var = StringVar(value=str(config.max_screenshots_per_video))
        self.resize_mode_var = StringVar(value="resize" if config.resize_enabled else "original")
        self.width_var = StringVar(value=str(config.target_size[0]) if config.target_size else "1280")
        self.height_var = StringVar(value=str(config.target_size[1]) if config.target_size else "720")
        self.auto_enabled_var = BooleanVar(value=config.auto_enabled)
        self.auto_step_var = StringVar(value=str(config.auto_step))
        self.roi_var = BooleanVar(value=config.use_roi)
        self.status_var = StringVar(value="Choose folders and press Start.")

        self._build_ui()
        self._bind_keys()
        self._sync_resize_controls()
        self._refresh_selected_files()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        root_frame = Frame(self.root, padx=10, pady=10)
        root_frame.pack(fill="both", expand=True)

        controls = Frame(root_frame, width=320)
        controls.pack(side="left", fill="y", padx=(0, 12))
        controls.pack_propagate(False)

        preview = Frame(root_frame)
        preview.pack(side="right", fill="both", expand=True)

        self._build_path_row(controls, "Video folder", self.video_folder_var, self._choose_video_folder)
        self._build_video_files_controls(controls)
        self._build_path_row(controls, "Output folder", self.output_folder_var, self._choose_output_folder)

        Label(controls, text="Max saves per video").pack(anchor="w", pady=(14, 2))
        Entry(controls, textvariable=self.max_count_var).pack(fill="x")

        Label(controls, text="Frame size").pack(anchor="w", pady=(14, 2))
        Radiobutton(
            controls,
            text="Original",
            variable=self.resize_mode_var,
            value="original",
            command=self._sync_resize_controls,
        ).pack(anchor="w")
        Radiobutton(
            controls,
            text="Resize",
            variable=self.resize_mode_var,
            value="resize",
            command=self._sync_resize_controls,
        ).pack(anchor="w")

        size_frame = Frame(controls)
        size_frame.pack(fill="x", pady=(4, 0))
        self.width_entry = Entry(size_frame, textvariable=self.width_var, width=8)
        self.width_entry.pack(side="left")
        Label(size_frame, text=" x ").pack(side="left")
        self.height_entry = Entry(size_frame, textvariable=self.height_var, width=8)
        self.height_entry.pack(side="left")

        Checkbutton(controls, text="Use ROI", variable=self.roi_var).pack(anchor="w", pady=(14, 0))
        Button(controls, text="Select ROI", command=self._select_roi).pack(fill="x", pady=(4, 0))

        Checkbutton(controls, text="Auto save", variable=self.auto_enabled_var).pack(anchor="w", pady=(14, 0))
        Label(controls, text="Frame interval").pack(anchor="w", pady=(4, 2))
        Entry(controls, textvariable=self.auto_step_var).pack(fill="x")

        Button(controls, text="Start", command=self._start).pack(fill="x", pady=(20, 4))
        Button(controls, text="Pause / Play", command=self._toggle_pause).pack(fill="x", pady=4)
        Button(controls, text="Save frame (S)", command=self._save_manual).pack(fill="x", pady=4)

        step_frame = Frame(controls)
        step_frame.pack(fill="x", pady=(8, 0))
        Button(step_frame, text="<", command=lambda: self._step(-1)).pack(side="left", fill="x", expand=True)
        Button(step_frame, text=">", command=lambda: self._step(1)).pack(side="left", fill="x", expand=True)

        Button(controls, text="Stop", command=self._stop).pack(fill="x", pady=(16, 4))

        self.preview_label = Label(preview, bg="#111111")
        self.preview_label.pack(fill="both", expand=True)
        Label(preview, textvariable=self.status_var, anchor="w").pack(fill="x", pady=(8, 0))

    def _build_path_row(self, parent: Frame, label: str, variable: StringVar, command) -> None:
        Label(parent, text=label).pack(anchor="w", pady=(0, 2))
        row = Frame(parent)
        row.pack(fill="x", pady=(0, 8))
        Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        Button(row, text="...", command=command, width=3).pack(side="right", padx=(4, 0))

    def _build_video_files_controls(self, parent: Frame) -> None:
        Label(parent, textvariable=self.selected_files_var).pack(anchor="w", pady=(2, 2))
        self.selected_files_list = Listbox(parent, height=4)
        self.selected_files_list.pack(fill="x")

        row = Frame(parent)
        row.pack(fill="x", pady=(4, 8))
        Button(row, text="Add video files", command=self._choose_video_files).pack(side="left", fill="x", expand=True)
        Button(row, text="Clear", command=self._clear_video_files).pack(side="left", padx=(4, 0))

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda event: self._toggle_pause())
        self.root.bind("<KeyPress-p>", lambda event: self._toggle_pause())
        self.root.bind("<KeyPress-P>", lambda event: self._toggle_pause())
        self.root.bind("<KeyPress-s>", lambda event: self._save_manual())
        self.root.bind("<KeyPress-S>", lambda event: self._save_manual())
        self.root.bind("<Left>", lambda event: self._step(-1))
        self.root.bind("<Right>", lambda event: self._step(1))

    def _choose_video_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.video_folder_var.get() or ".")
        if folder:
            self.video_folder_var.set(folder)

    def _choose_video_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            initialdir=self.video_folder_var.get() or ".",
            filetypes=(
                ("Video files", "*.avi *.mov *.mp4 *.mkv *.m4v"),
                ("All files", "*.*"),
            ),
        )
        if not filenames:
            return

        known_files = set(self.config.selected_video_files)
        for filename in filenames:
            path = Path(filename)
            if path not in known_files:
                self.config.selected_video_files.append(path)
                known_files.add(path)
        self._refresh_selected_files()

    def _clear_video_files(self) -> None:
        self.config.selected_video_files.clear()
        self._refresh_selected_files()

    def _refresh_selected_files(self) -> None:
        self.selected_files_list.delete(0, END)
        for path in self.config.selected_video_files:
            self.selected_files_list.insert(END, path.name)
        self.selected_files_var.set(f"Selected files: {len(self.config.selected_video_files)}")

    def _choose_output_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get() or ".")
        if folder:
            self.output_folder_var.set(folder)

    def _sync_resize_controls(self) -> None:
        state = "normal" if self.resize_mode_var.get() == "resize" else "disabled"
        self.width_entry.configure(state=state)
        self.height_entry.configure(state=state)

    def _read_positive_int(self, value: str, field_name: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be an integer") from error
        if parsed <= 0:
            raise ValueError(f"{field_name} must be positive")
        return parsed

    def _apply_form_config(self) -> bool:
        try:
            max_count = self._read_positive_int(self.max_count_var.get(), "Max saves")
            auto_step = self._read_positive_int(self.auto_step_var.get(), "Frame interval")
            target_size = None
            if self.resize_mode_var.get() == "resize":
                width = self._read_positive_int(self.width_var.get(), "Width")
                height = self._read_positive_int(self.height_var.get(), "Height")
                target_size = (width, height)
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error))
            return False

        self.config.video_folder = Path(self.video_folder_var.get())
        self.config.output_folder = Path(self.output_folder_var.get())
        self.config.max_screenshots_per_video = max_count
        self.config.target_size = target_size
        self.config.auto_enabled = self.auto_enabled_var.get()
        self.config.auto_step = auto_step
        self.config.use_roi = self.roi_var.get()
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        if not self.config.use_roi:
            self.extractor.roi = None
        return True

    def _start(self) -> None:
        if not self._apply_form_config():
            return

        selected_roi = self.extractor.roi
        self.extractor = VideoFrameExtractor(self.config)
        if self.config.use_roi:
            self.extractor.roi = selected_roi
        self.video_files = self.extractor.list_videos()
        if not self.video_files:
            messagebox.showerror("No videos", self._no_videos_message())
            return

        self.video_index = 0
        self.saved_count = 0
        self.is_running = True
        self.is_paused = False
        self._open_current_video()
        self._schedule_next_frame()

    def _open_current_video(self) -> None:
        video_path = self.video_files[self.video_index]
        self.player.open(video_path)
        self.saved_count = 0
        self.current_raw_frame = None
        self.current_processed_frame = None
        self.current_frame_id = 0
        self.status_var.set(f"Processing {video_path.name}")

    def _stop(self) -> None:
        self.is_running = False
        self.is_paused = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.player.close()
        self.status_var.set("Stopped.")

    def _toggle_pause(self) -> None:
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.status_var.set(self._status_text())
        if not self.is_paused:
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if not self.is_running or self.is_paused:
            return
        delay = self._playback_delay_ms()
        self.after_id = self.root.after(delay, self._play_next_frame)

    def _playback_delay_ms(self) -> int:
        fps = 0
        if self.player.capture is not None:
            fps = self.player.capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            return max(1, int(1000 / fps))
        return 30

    def _play_next_frame(self) -> None:
        if not self.is_running or self.is_paused:
            return

        ok, raw_frame, frame_id = self.player.read_next()
        if not ok:
            self._next_video_or_finish()
            return

        self._set_current_frame(raw_frame, frame_id)
        self._auto_save_if_needed()
        self._schedule_next_frame()

    def _set_current_frame(self, raw_frame, frame_id: int) -> None:
        self.current_raw_frame = raw_frame
        self.current_frame_id = frame_id
        try:
            self.current_processed_frame = self.extractor.prepare_frame(raw_frame)
        except FrameProcessingError as error:
            self.is_paused = True
            messagebox.showerror("Frame error", str(error))
            return
        self._show_frame(self.current_processed_frame)
        self.status_var.set(self._status_text())

    def _show_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        max_width = max(self.preview_label.winfo_width(), 1)
        max_height = max(self.preview_label.winfo_height(), 1)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.photo)

    def _auto_save_if_needed(self) -> None:
        if not self.config.auto_enabled:
            return
        if self.saved_count >= self.config.max_screenshots_per_video:
            return
        if self.current_frame_id % self.config.auto_step != 0:
            return
        self._save_current_frame("auto")

    def _save_manual(self) -> None:
        if self.current_processed_frame is None:
            return
        self._save_current_frame("manual")

    def _save_current_frame(self, prefix: str) -> None:
        if self.player.video_path is None or self.current_processed_frame is None:
            return
        if self.saved_count >= self.config.max_screenshots_per_video:
            self.status_var.set("Save limit reached for current video.")
            return

        try:
            output_path = self.extractor.save_frame(
                self.player.video_path,
                self.current_frame_id,
                self.current_processed_frame,
                prefix,
            )
        except FrameProcessingError as error:
            messagebox.showerror("Save failed", str(error))
            return

        self.saved_count += 1
        self.status_var.set(f"Saved {output_path.name} ({self.saved_count}/{self.config.max_screenshots_per_video})")

    def _step(self, direction: int) -> None:
        if not self.is_running or not self.is_paused:
            return

        target = self.current_frame_id - 1 + direction
        ok, raw_frame, frame_id = self.player.read_at(target)
        if ok:
            self._set_current_frame(raw_frame, frame_id)

    def _select_roi(self) -> None:
        if not self._apply_form_config():
            return

        raw_frame = self.current_raw_frame
        if raw_frame is None:
            video_path = self._first_selected_video()
            if video_path is None:
                messagebox.showerror("No videos", self._no_videos_message())
                return
            capture = cv2.VideoCapture(str(video_path))
            ok, raw_frame = capture.read()
            capture.release()
            if not ok:
                messagebox.showerror("ROI failed", f"Failed to read first frame: {video_path}")
                return

        roi = self.extractor.select_roi_from_frame(raw_frame)
        self.roi_var.set(roi is not None)
        self.config.use_roi = roi is not None
        if self.current_raw_frame is not None:
            self._set_current_frame(self.current_raw_frame, self.current_frame_id)
        self.status_var.set("ROI selected." if roi else "ROI cancelled; full frame is used.")

    def _first_selected_video(self):
        self.extractor = VideoFrameExtractor(self.config)
        videos = self.extractor.list_videos()
        if not videos:
            return None
        return videos[0]

    def _next_video_or_finish(self) -> None:
        if self.video_index + 1 >= len(self.video_files):
            self.is_running = False
            self.is_paused = True
            self.player.close()
            self.status_var.set("Finished.")
            return

        self.video_index += 1
        self._open_current_video()
        self._schedule_next_frame()

    def _status_text(self) -> str:
        video_name = self.player.video_path.name if self.player.video_path else "-"
        state = "Paused" if self.is_paused else "Playing"
        total = self.player.frame_count or "?"
        return (
            f"{state}: {video_name} | frame {self.current_frame_id}/{total} | "
            f"saved {self.saved_count}/{self.config.max_screenshots_per_video}"
        )

    def _no_videos_message(self) -> str:
        if self.config.selected_video_files:
            return "No supported video files found in selected files."
        return f"No video files found in: {self.config.video_folder}"
