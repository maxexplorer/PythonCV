# HPolyMapper

Модель для построения преобразования координат между изображениями:

- top → door2
- bottom → door2

Используются методы:
- Homography (RANSAC)
- fallback: Affine transformation

## Установка

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Структура проекта
project/
│
├── main.py
├── train.py
├── predict.py
├── utils.py
│
├── artifacts/
│   ├── model.pkl
│   ├── metrics.json
│
└── train/
│
└── val/
    
### Запуск
TRAIN

В main.py:

mode = "train"

Запуск:
```
python main.py
```

## Результат

модель сохраняется в artifacts/model.pkl

## Пример результата
 
TOP → DOOR2    : 202.899
BOTTOM → DOOR2 : 196.339

## Запуск

```
python main.py
```

## Зависимости
numpy
opencv-python
scikit-learn

## Старт
### обучение
mode = "train"
```
python main.py
```

### проверка
mode = "predict"
```
python main.py
```