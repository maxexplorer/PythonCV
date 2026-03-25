import tkinter as tk
from tkinter import filedialog


class MainWindow:
    """
    Окно для выбора видео, типа детектора и опции записи.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Выберите видео")
        self.root.geometry("500x250")
        self.root.resizable(False, False)

        self.video_path = tk.StringVar()
        self.selected_path: str | None = None
        self.detector_choice: str = "YOLO"
        self.record_video: bool = False

        tk.Label(self.root, text="Выберите видеофайл:").pack(pady=5)
        tk.Entry(self.root, textvariable=self.video_path, width=60).pack(pady=5)
        tk.Button(self.root, text="Обзор", command=self.browse_video).pack(pady=5)

        # Выбор детектора
        tk.Label(self.root, text="Выберите детектор:").pack(pady=5)
        self.detector_var = tk.StringVar(value="YOLO")
        tk.Radiobutton(self.root, text="YOLO", variable=self.detector_var, value="YOLO").pack()
        tk.Radiobutton(self.root, text="OpenCV", variable=self.detector_var, value="OpenCV").pack()

        # Галочка записи видео
        self.record_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Записать видео", variable=self.record_var).pack(pady=5)

        tk.Button(self.root, text="Запустить", command=self.start).pack(pady=10)

    def browse_video(self) -> None:
        """Открывает диалог выбора видеофайла."""
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if path:
            self.video_path.set(path)

    def start(self) -> None:
        """Сохраняет выбранный путь и параметры и закрывает окно."""
        self.selected_path = self.video_path.get()
        self.detector_choice = self.detector_var.get()
        self.record_video = self.record_var.get()
        self.root.destroy()

    def run(self) -> tuple[str, str, bool]:
        """
        Запускает главное окно.
        Возвращает: (video_path, detector_choice, record_video)
        """
        self.root.mainloop()
        return self.selected_path, self.detector_choice, self.record_video