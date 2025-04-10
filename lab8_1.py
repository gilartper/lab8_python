import cv2


image = cv2.imread('variant-10 (1).jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Ошибка: изображение не найдено или путь неверный")
    exit()


_, thresholded = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)


cv2.imwrite('output_thresholded.jpg', thresholded)


cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image (150)', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()