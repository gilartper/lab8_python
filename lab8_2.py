import cv2
import numpy as np

# Загрузка шаблона (метки) и изображения мухи
template = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
fly_img = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)  # Загружаем с альфа-каналом

if template is None:
    print("Ошибка: не удалось загрузить шаблон метки")
    exit()

if fly_img is None:
    print("Ошибка: не удалось загрузить изображение мухи")
    exit()

# Загрузка видео
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

# Получаем размеры шаблона
w, h = template.shape[::-1]

# Получаем размеры изображения мухи
fly_h, fly_w = fly_img.shape[:2]

# Параметры для метода сопоставления шаблонов
method = cv2.TM_CCOEFF_NORMED
threshold = 0.7  # Порог совпадения

# Состояние отслеживания метки
marker_inside = False
flip_state = False


def overlay_image(background, overlay, pos):
    x, y = pos
    # Вычисляем область наложения
    y1, y2 = max(0, y - fly_h // 2), min(background.shape[0], y + fly_h // 2)
    x1, x2 = max(0, x - fly_w // 2), min(background.shape[1], x + fly_w // 2)

    # Если изображение выходит за границы кадра
    if y1 >= y2 or x1 >= x2:
        return background

    # Обрезаем изображение мухи, если оно выходит за границы
    fly_y1 = max(0, fly_h // 2 - y)
    fly_y2 = fly_h - max(0, (y + fly_h // 2) - background.shape[0])
    fly_x1 = max(0, fly_w // 2 - x)
    fly_x2 = fly_w - max(0, (x + fly_w // 2) - background.shape[1])

    overlay_cropped = overlay[fly_y1:fly_y2, fly_x1:fly_x2]

    # Если есть альфа-канал
    if overlay_cropped.shape[2] == 4:
        alpha = overlay_cropped[:, :, 3] / 255.0
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (
                    alpha * overlay_cropped[:, :, c] +
                    (1 - alpha) * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay_cropped

    return background


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2
    square_size = 150
    square_half = square_size // 2

    square_left = center_x - square_half
    square_right = center_x + square_half
    square_top = center_y - square_half
    square_bottom = center_y + square_half

    cv2.rectangle(frame, (square_left, square_top),
                  (square_right, square_bottom), (255, 0, 255), 2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    thresh = cv2.erode(thresh, (3, 3))
    thresh = cv2.dilate(thresh, (7, 7))

    res = cv2.matchTemplate(gray_frame, template, method)
    loc = np.where(res >= threshold)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    current_marker_inside = False
    marker_pos = None

    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            marker_pos = (cx, cy)

            cv2.circle(frame, (cx, cy), 3, (255, 0, 255), 7)
            cv2.putText(frame, "Target", (cx - 10, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            current_marker_inside = (square_left <= cx <= square_right and
                                     square_top <= cy <= square_bottom)

    if current_marker_inside and not marker_inside:
        flip_state = not flip_state

    marker_inside = current_marker_inside

    if flip_state:
        frame = cv2.flip(frame, -1)
        # Если кадр перевернут, нужно также перевернуть координаты метки
        if marker_pos:
            marker_pos = (frame_width - marker_pos[0], frame_height - marker_pos[1])

    # Накладываем изображение мухи, если найдена метка
    if marker_pos:
        frame = overlay_image(frame, fly_img, marker_pos)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        cv2.putText(frame, "Marker", (pt[0], pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, "Flip: " + ("ON" if flip_state else "OFF"), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Marker Tracking', thresh)
    cv2.imshow("origin", gray_frame)
    cv2.imshow("cnt", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

