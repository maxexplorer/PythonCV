# ui.py

import tkinter as tk
from tkinter import filedialog


class MainWindow:
    """
    Окно для выбора видео.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Выберите видео")
        self.root.geometry("1280x720")
        self.root.resizable(False, False)

        self.video_path = tk.StringVar()
        self.selected_path: str | None = None

        tk.Label(self.root, text="Выберите видеофайл:").pack(pady=10)
        tk.Entry(self.root, textvariable=self.video_path, width=80).pack(pady=5)
        tk.Button(self.root, text="Обзор", command=self.browse_video, width=20).pack(pady=10)
        tk.Button(self.root, text="Запустить", command=self.start, width=20).pack(pady=10)

    def browse_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if path:
            self.video_path.set(path)

    def start(self) -> None:
        """Сохраняет выбранный путь и закрывает окно."""
        self.selected_path = self.video_path.get()
        self.root.destroy()  # <-- вот это закрывает окно полностью

    def run(self) -> str:
        self.root.mainloop()
        return self.selected_path