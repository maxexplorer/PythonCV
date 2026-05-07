# utils.py

import os


def find_first_video(folder: str, exts=('.avi', '.mp4', '.mov')) -> str:
    """
    Находит первый видеофайл в папке с указанными расширениями.

    :param folder: Путь к папке, где искать видео.
    :param exts: Кортеж расширений, которые нужно искать.
    :return: Полный путь к первому найденному видеофайлу.
    :raises FileNotFoundError: Если видеофайлы не найдены.
    """
    for file in os.listdir(folder):
        if file.lower().endswith(exts):
            return os.path.join(folder, file)
    raise FileNotFoundError(f"No video files with extensions {exts} found in {folder}")
