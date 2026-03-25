import tkinter as tk
from tkinter import filedialog


class MainWindow:
    """
    Главное окно для выбора видеофайла, типа детектора и опции записи видео.

    Атрибуты:
        video_path (tk.StringVar): Путь к выбранному видеофайлу.
        selected_path (str | None): Сохраненный путь к видео после закрытия окна.
        detector_choice (str): Выбранный тип детектора ("YOLO" или "OpenCV").
        record_video (bool): Флаг записи видео.
    """

    def __init__(self):
        """
        Инициализация интерфейса с виджетами для выбора видео, детектора и опции записи.
        """
        self.root = tk.Tk()
        self.root.title("Выберите видео")
        self.root.geometry("500x250")
        self.root.resizable(False, False)

        # Путь к видео
        self.video_path = tk.StringVar()
        self.selected_path: str | None = None

        # Параметры детектора и записи
        self.detector_choice: str = "YOLO"
        self.record_video: bool = False

        # Ввод пути к видео
        tk.Label(self.root, text="Выберите видеофайл:").pack(pady=5)
        tk.Entry(self.root, textvariable=self.video_path, width=60).pack(pady=5)
        tk.Button(self.root, text="Обзор", command=self.browse_video).pack(pady=5)

        # Выбор детектора через радиокнопки
        tk.Label(self.root, text="Выберите детектор:").pack(pady=5)
        self.detector_var = tk.StringVar(value="YOLO")
        tk.Radiobutton(self.root, text="YOLO", variable=self.detector_var, value="YOLO").pack()
        tk.Radiobutton(self.root, text="OpenCV", variable=self.detector_var, value="OpenCV").pack()

        # Галочка записи видео
        self.record_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Записать видео", variable=self.record_var).pack(pady=5)

        # Кнопка запуска обработки
        tk.Button(self.root, text="Запустить", command=self.start).pack(pady=10)

    def browse_video(self) -> None:
        """
        Открывает стандартный диалог для выбора видеофайла
        и сохраняет путь в self.video_path.
        """
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if path:
            self.video_path.set(path)

    def start(self) -> None:
        """
        Сохраняет выбранный путь к видео, параметры детектора и записи.
        Закрывает главное окно.
        """
        self.selected_path = self.video_path.get()
        self.detector_choice = self.detector_var.get()
        self.record_video = self.record_var.get()
        self.root.destroy()  # Закрываем окно после выбора

    def run(self) -> tuple[str, str, bool]:
        """
        Запускает главное окно и ожидает действия пользователя.

        Returns:
            tuple[str, str, bool]: Кортеж из:
                - video_path: путь к выбранному видео,
                - detector_choice: выбранный детектор ("YOLO" или "OpenCV"),
                - record_video: флаг записи видео (True/False)
        """
        self.root.mainloop()
        return self.selected_path, self.detector_choice, self.record_video