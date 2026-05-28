# Расширенное описание детекции человека с поддержкой track-id и pose-keypoints
# Используется как дополнение к стандартному vision_msgs/Detection2DArray
# См. Листинг А.3 ВКР Миргазева М.А., 2026

std_msgs/Header header

# Идентификатор трека от ByteTrack ('track-N' формат для совместимости с Foxy)
string track_id

# Координаты bounding box в пикселях изображения (левый верхний угол + размеры)
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

# Уверенность детектора
float32 confidence

# Класс (для текущей реализации всегда 'person')
string class_name

# 17 ключевых точек скелета (COCO Keypoints):
# [x_0, y_0, vis_0, x_1, y_1, vis_1, ..., x_16, y_16, vis_16]
# Если ключевые точки не вычислены, массив пустой.
float32[] keypoints
