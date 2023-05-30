import cv2 as cv
import numpy as np

INIT_ABS_THRESHOLD = 20
SEGMENT_AVG_THRESHOLD = 15
CENTRAL_ABS_THRESHOLD = 80
NEIGHBOURS_WIDTH = 1

original_image = cv.cvtColor(cv.imread('abdomen.png'), cv.COLOR_BGR2GRAY)


def pixel_neighbours(image, y: int, x: int, NEIGHBOURS_WIDTH: int, type='') -> list[tuple[int, int]]:
    h, w = image.shape
    y_start = y - NEIGHBOURS_WIDTH
    y_end = y_start + NEIGHBOURS_WIDTH * 2 + 1
    x_start = x - NEIGHBOURS_WIDTH
    x_end = x_start + NEIGHBOURS_WIDTH * 2 + 1

    return [
        (n_y, n_x) for n_y in range(y_start, y_end) for n_x in range(x_start, x_end)
        if n_y >= 0 and n_y < h and n_x >= 0 and n_x < w and not (n_y == y and n_x == x)
    ]


def in_segment_condition(image, segment_area, start_pixel_value: int,
                         neighbour_pixel_value: int, central_pixel_value: int, type='init_abs') -> bool:
    if type == 'init_abs':
        return np.abs(start_pixel_value - neighbour_pixel_value) <= INIT_ABS_THRESHOLD
    elif type == 'segment_avg':
        return np.abs(np.average(image[segment_area]) - neighbour_pixel_value) <= SEGMENT_AVG_THRESHOLD
    elif type == 'central_abs':
        return np.abs(central_pixel_value - neighbour_pixel_value) <= CENTRAL_ABS_THRESHOLD
    elif type == 'segment_var':
        return np.abs(start_pixel_value - neighbour_pixel_value) < np.var(image[segment_area])


def area_expansion_segmentation(image, start_y: int, start_x: int, NEIGHBOURS_WIDTH: int):
    h, w = image.shape
    segment_area = np.full((h, w), False)
    segment_area[start_y, start_x] = True
    visited = np.full((h, w), False)
    visited[start_y, start_x] = True
    central_pixels = [(start_y, start_x)]
    start_pixel_value = image[start_y, start_x]

    while len(central_pixels) > 0:
        central_pixel = central_pixels.pop()
        central_pixel_value = image[central_pixel[0], central_pixel[1]]
        neighbours = pixel_neighbours(
            image, central_pixel[0], central_pixel[1], NEIGHBOURS_WIDTH)
        for neighbour_y, neighbour_x in neighbours:
            neighbour_pixel_value = image[neighbour_y, neighbour_x]
            if not visited[neighbour_y, neighbour_x]:
                visited[neighbour_y, neighbour_x] = True
                if in_segment_condition(image, segment_area, start_pixel_value,
                                        neighbour_pixel_value, central_pixel_value, type='init_abs'):
                    segment_area[neighbour_y, neighbour_x] = True
                    central_pixels.append((neighbour_y, neighbour_x))

    return segment_area


def mouse_callback(event, x, y, flags, params):
    global original_image, NEIGHBOURS_WIDTH
    if event == 1:
        image = original_image
        image = cv.GaussianBlur(image, (3, 3), 0)
        kernel = np.ones((3, 3), np.uint8)
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        segment = area_expansion_segmentation(
            image, start_y=y, start_x=x, NEIGHBOURS_WIDTH=NEIGHBOURS_WIDTH).astype('uint8')
        print('finished area expansion!')
        contours, _ = cv.findContours(
            segment, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cv.drawContours(original_image, contours, -1, (0, 0, 0), 1)
        cv.imshow('image', original_image)


cv.imshow('image', original_image)
cv.setMouseCallback('image', mouse_callback)
cv.waitKey()
cv.destroyAllWindows()
