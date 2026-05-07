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
from settings import save_slicer_config


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
        self.space_advance_after_id = None
        self.space_is_down = False

        self.video_folder_var = StringVar(value=str(config.video_folder))
        self.output_folder_var = StringVar(value=str(config.output_folder))
        self.selected_files_var = StringVar(value="Selected files: 0")
        self.resize_mode_var = StringVar(value="resize" if config.resize_enabled else "original")
        self.width_var = StringVar(value=str(config.target_size[0]) if config.target_size else "1280")
        self.height_var = StringVar(value=str(config.target_size[1]) if config.target_size else "720")
        self.auto_enabled_var = BooleanVar(value=config.auto_enabled)
        self.auto_step_var = StringVar(value=str(config.auto_step))
        self.roi_var = BooleanVar(value=config.use_roi)
        self.status_var = StringVar(value="Choose folders and press Start.")

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._sync_resize_controls()
        self._refresh_selected_files()
        self._bind_settings_autosave()

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

        Checkbutton(
            controls,
            text="Use ROI",
            variable=self.roi_var,
            command=self._apply_roi_toggle,
        ).pack(anchor="w", pady=(14, 0))
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
        self.root.bind_all("<KeyPress-space>", self._handle_space_press)
        self.root.bind_all("<KeyRelease-space>", self._handle_space_release)
        self.root.bind_all("<KeyPress-p>", self._handle_pause_key)
        self.root.bind_all("<KeyPress-P>", self._handle_pause_key)
        self.root.bind_all("<KeyPress-s>", self._handle_save_key)
        self.root.bind_all("<KeyPress-S>", self._handle_save_key)
        self.root.bind_all("<Left>", self._handle_left_key)
        self.root.bind_all("<Right>", self._handle_right_key)
        self.root.bind_all("<Escape>", self._handle_escape_key)

    def _choose_video_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.video_folder_var.get() or ".")
        if folder:
            self.video_folder_var.set(folder)
            self._save_current_settings()

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
        self._save_current_settings()

    def _clear_video_files(self) -> None:
        self.config.selected_video_files.clear()
        self._refresh_selected_files()
        self._save_current_settings()

    def _refresh_selected_files(self) -> None:
        self.selected_files_list.delete(0, END)
        for path in self.config.selected_video_files:
            self.selected_files_list.insert(END, path.name)
        self.selected_files_var.set(f"Selected files: {len(self.config.selected_video_files)}")

    def _choose_output_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get() or ".")
        if folder:
            self.output_folder_var.set(folder)
            self._save_current_settings()

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
        self.config.target_size = target_size
        self.config.auto_enabled = self.auto_enabled_var.get()
        self.config.auto_step = auto_step
        self.config.use_roi = self.roi_var.get()
        self.config.roi = self.extractor.roi if self.config.use_roi else None
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        if not self.config.use_roi:
            self.extractor.roi = None
        self._save_current_settings()
        return True

    def _start(self) -> None:
        if self.is_running:
            self._apply_running_settings()
            return

        if not self._apply_form_config():
            return

        selected_roi = self.extractor.roi
        self.extractor = VideoFrameExtractor(self.config)
        if self.config.use_roi:
            self.extractor.roi = selected_roi
            self.config.roi = selected_roi
        self.video_files = self.extractor.list_videos()
        if not self.video_files:
            messagebox.showerror("No videos", self._no_videos_message())
            return

        self.video_index = 0
        self.saved_count = 0
        self.is_running = True
        self.is_paused = False
        self._open_current_video()
        self._play_next_frame()

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
        self._cancel_space_advance()
        self.player.close()
        self.status_var.set("Stopped.")

    def _toggle_pause(self) -> None:
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self._cancel_space_advance()
        self.status_var.set(self._status_text())
        if not self.is_paused:
            self._schedule_next_frame()

    def _handle_pause_key(self, event=None) -> str:
        self._toggle_pause()
        return "break"

    def _handle_save_key(self, event=None) -> str:
        self._save_manual()
        return "break"

    def _handle_left_key(self, event=None) -> str:
        self._step(-1)
        return "break"

    def _handle_right_key(self, event=None) -> str:
        self._step(1)
        return "break"

    def _handle_escape_key(self, event=None) -> str:
        self._skip_to_next_video()
        return "break"

    def _handle_space_press(self, event=None) -> str:
        if not self.is_running:
            return "break"

        if not self.is_paused:
            self._toggle_pause()
            return "break"

        if self.space_is_down:
            return "break"

        self.space_is_down = True
        self._step(1)
        self._schedule_space_advance()
        return "break"

    def _handle_space_release(self, event=None) -> str:
        self._cancel_space_advance()
        return "break"

    def _schedule_space_advance(self) -> None:
        if not self.is_running or not self.is_paused or not self.space_is_down:
            return

        delay = self._playback_delay_ms()
        self.space_advance_after_id = self.root.after(delay, self._play_space_advance)

    def _play_space_advance(self) -> None:
        self.space_advance_after_id = None
        if not self.is_running or not self.is_paused or not self.space_is_down:
            return

        self._step(1)
        self._schedule_space_advance()

    def _cancel_space_advance(self) -> None:
        self.space_is_down = False
        if self.space_advance_after_id is not None:
            self.root.after_cancel(self.space_advance_after_id)
            self.space_advance_after_id = None

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
        if self.current_frame_id % self.config.auto_step != 0:
            return
        self._save_current_frame("auto")

    def _save_manual(self) -> None:
        if self.current_processed_frame is None:
            return
        saved = self._save_current_frame("manual")
        if saved and self.is_running:
            self._advance_one_frame()

    def _save_current_frame(self, prefix: str) -> bool:
        if self.player.video_path is None or self.current_processed_frame is None:
            return False
        try:
            output_path = self.extractor.save_frame(
                self.player.video_path,
                self.current_frame_id,
                self.current_processed_frame,
                prefix,
            )
        except FrameProcessingError as error:
            messagebox.showerror("Save failed", str(error))
            return False

        self.saved_count += 1
        self.status_var.set(f"Saved {output_path.name} ({self.saved_count}) | {self._current_video_info_text()}")
        return True

    def _step(self, direction: int) -> None:
        if not self.is_running or not self.is_paused:
            return

        target = self.current_frame_id - 1 + direction
        ok, raw_frame, frame_id = self.player.read_at(target)
        if ok:
            self._set_current_frame(raw_frame, frame_id)

    def _advance_one_frame(self) -> None:
        if not self.is_running:
            return

        ok, raw_frame, frame_id = self.player.read_next()
        if not ok:
            self._next_video_or_finish()
            return

        self._set_current_frame(raw_frame, frame_id)

    def _select_roi(self) -> None:
        if not self._apply_form_config():
            return
        if self.is_running:
            self.is_paused = True
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            self._cancel_space_advance()

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
        self.config.roi = roi
        if self.current_raw_frame is not None:
            self._set_current_frame(self.current_raw_frame, self.current_frame_id)
        self.status_var.set("ROI selected." if roi else "ROI cancelled; full frame is used.")
        self._save_current_settings()

    def _apply_roi_toggle(self) -> None:
        self.config.use_roi = self.roi_var.get()
        if self.config.use_roi:
            self.extractor.roi = self.config.roi
        else:
            self.extractor.roi = None
            self.config.roi = None

        if self.current_raw_frame is not None:
            self._set_current_frame(self.current_raw_frame, self.current_frame_id)
        self._save_current_settings()

    def _apply_running_settings(self) -> None:
        was_paused = self.is_paused
        if not self._apply_form_config():
            return

        self.extractor.config = self.config
        if self.current_raw_frame is not None:
            self._set_current_frame(self.current_raw_frame, self.current_frame_id)
        self.is_paused = was_paused
        self.status_var.set(self._status_text())

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
        if self.is_paused:
            self._advance_one_frame()
        else:
            self._play_next_frame()

    def _skip_to_next_video(self) -> None:
        if not self.is_running:
            return

        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self._cancel_space_advance()
        self._next_video_or_finish()

    def _status_text(self) -> str:
        state = "Paused" if self.is_paused else "Playing"
        total = self.player.frame_count or "?"
        return (
            f"{state}: {self._current_video_info_text()} | frame {self.current_frame_id}/{total} | "
            f"saved {self.saved_count}"
        )

    def _current_video_info_text(self) -> str:
        video_name = self.player.video_path.name if self.player.video_path else "-"
        return f"{video_name} | {self._current_frame_size_text()}"

    def _current_frame_size_text(self) -> str:
        frame = self.current_processed_frame if self.current_processed_frame is not None else self.current_raw_frame
        if frame is None:
            return "size -"

        height, width = frame.shape[:2]
        label = "ROI" if self.extractor.roi is not None else "frame"
        return f"{label} {width}x{height}"

    def _no_videos_message(self) -> str:
        if self.config.selected_video_files:
            return "No supported video files found in selected files."
        return f"No video files found in: {self.config.video_folder}"

    def _close(self) -> None:
        if not self._save_current_settings():
            messagebox.showerror("Settings failed", "Failed to save settings.")
        self._stop()
        self.root.destroy()

    def _bind_settings_autosave(self) -> None:
        variables = (
            self.video_folder_var,
            self.output_folder_var,
            self.resize_mode_var,
            self.width_var,
            self.height_var,
            self.auto_enabled_var,
            self.auto_step_var,
            self.roi_var,
        )
        for variable in variables:
            variable.trace_add("write", lambda *args: self._save_current_settings())

    def _save_current_settings(self) -> bool:
        self._apply_config_without_validation()
        try:
            save_slicer_config(self.config)
        except OSError:
            return False
        return True

    def _apply_config_without_validation(self) -> None:
        self.config.video_folder = Path(self.video_folder_var.get())
        self.config.output_folder = Path(self.output_folder_var.get())
        self.config.auto_enabled = self.auto_enabled_var.get()
        self.config.auto_step = self._read_optional_positive_int(self.auto_step_var.get(), self.config.auto_step)
        self.config.use_roi = self.roi_var.get()
        self.config.roi = self.extractor.roi if self.config.use_roi else None

        if self.resize_mode_var.get() != "resize":
            self.config.target_size = None
            return

        width = self._read_optional_positive_int(self.width_var.get(), None)
        height = self._read_optional_positive_int(self.height_var.get(), None)
        if width is not None and height is not None:
            self.config.target_size = (width, height)

    def _read_optional_positive_int(self, value: str, default: int | None) -> int | None:
        try:
            parsed = int(value)
        except ValueError:
            return default
        if parsed <= 0:
            return default
        return parsed
