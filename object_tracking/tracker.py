import math


class EuclideanDistTracker:
    """
    Трекер объектов, основанный на евклидовом расстоянии между центрами объектов.

    Присваивает уникальные идентификаторы (ID) новым объектам и отслеживает их
    перемещение по кадрам на основе расстояния между центрами ограничивающих прямоугольников.
    """

    def __init__(self):
        # Словарь для хранения координат центров отслеживаемых объектов: {id: (cx, cy)}
        self.center_points = {}

        # Счётчик ID — увеличивается при обнаружении нового объекта
        self.id_count = 0

    def update(self, objects_rect):
        """
        Обновляет позиции объектов на текущем кадре и возвращает список прямоугольников с ID.

        :param objects_rect: Список прямоугольников объектов в формате [x, y, w, h]
        :return: Список объектов с ID в формате [x, y, w, h, id]
        """
        # Список обнаруженных объектов с ID
        objects_bbs_ids = []

        # Получение центра для каждого прямоугольника
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Проверяем, не отслеживается ли уже этот объект
            same_object_detected = False
            for object_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:  # объект считается тем же, если центр на близком расстоянии
                    self.center_points[object_id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            # Если объект новый — присваиваем ему новый ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Удаляем неактуальные ID (те, которых нет в текущем кадре)
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Обновляем хранимые центры
        self.center_points = new_center_points.copy()

        return objects_bbs_ids
