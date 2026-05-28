# Идентификатор режима ухудшения видимости, опубликованного классификатором сцены
# Профиль QoS: RELIABLE + TRANSIENT_LOCAL (последнее значение доставляется новым подписчикам)
# См. Листинг А.2 ВКР Миргазева М.А., 2026

std_msgs/Header header

# Идентификатор режима (соответствует EnhancementMode в vision_pro_v5)
# 0 = OFF, 1 = NIGHT, 2 = FOG, 3 = SMOKE, 4 = RAIN
uint8 mode_id

# Человекочитаемое имя режима ('OFF', 'NIGHT', 'FOG', 'SMOKE', 'RAIN')
string mode_name

# Уверенность классификатора
# (1.0 для эвристического дерева правил; при апгрейде на ML — softmax)
float32 confidence

# Метаданные кадра, на котором принято решение
float32 brightness
float32 std_dev
float32 laplacian_var
float32 edge_density
