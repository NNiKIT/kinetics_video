# Kinetics Video Action Recognition

## Постановка задачи 
1. Обучить модель классификации видео (3DCNN, CNN-RNN, video transformer и тд)
2. Сравнить с моделью обученной на отдельных кадрах
3. Обучить модель классификации видео с другим подходом и провести сравнение

## Данные
Набор данных - датасет Kinetics 700-2020 с классами содержащими слово dancing
Ссылка на датасет: https://www.deepmind.com/open-source/kinetics
Ссылка на обрезанные по таймкодам ролики: https://disk.yandex.ru/d/FaFOKvRg7_baxA

class_name = [
    "belly_dancing",
    "breakdancing",
    "country_line_dancing",
    "dancing_ballet",
    "dancing_charleston",
    "dancing_gangnam_style",
    "dancing_macarena",
    "jumpstyle_dancing",
    "mosh_pit_dancing",
    "robot_dancing",
    "salsa_dancing",
    "square_dancing",
    "swing_dancing",
    "tango_dancing",
    "tap_dancing",
]


## Модели

Были выбранны модели SEResNet и ResNet 3D.
Запуск:
    !python main.py
    !python main_SE_ResNet.py

Для классификации использовался каждый 4 кадр видео(видео делилось на 32 фрейма), после чего усреднялся предикт по всем кадрам и выбираем наибольший.
Итоговые метрики
```
          Accuracy   F1-score
                            
SEResNet    0.54       0.51
ResNet 3D   0.35       0.3
```
