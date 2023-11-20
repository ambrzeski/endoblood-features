import numpy as np
import cv2 as cv


def detect(img):
    """
    :param img: color, deinterlaced image
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    flashes_text_mask = detect_text_and_flashes(img_gray)
    border_mask = detect_border(img, flashes_text_mask)
    color_bars = detect_color_bars(img)

    mask = cv.add(flashes_text_mask, border_mask)
    mask = cv.add(mask, color_bars)
    roi = 255 - mask
    return roi


def detect_color_bars(img):
    """
    :param img: color, deinterlaced image
    """
    rect3  = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    rect5  = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    rect13 = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))

    # Detect color bars without red color
    blue_bars = cv.inRange(img, np.uint8([200, 0, 0]), np.uint8([255, 50, 50]))
    blue_bars = cv.dilate(blue_bars, rect13)
    turq_bars = cv.inRange(img, np.uint8([200, 200, 0]), np.uint8([255, 255, 40]))
    turq_bars = cv.dilate(turq_bars, rect13)
    violet_bars = cv.inRange(img, np.uint8([200, 0, 200]), np.uint8([255, 30, 255]))
    violet_bars = cv.dilate(violet_bars, rect13)
    yellow_bars = cv.inRange(img, np.uint8([0, 220, 220]), np.uint8([20, 255, 255]))
    yellow_bars = cv.dilate(yellow_bars, rect13)
    green_bars = cv.inRange(img, np.uint8([0, 200, 0]), np.uint8([20, 255, 20]))
    green_bars = cv.dilate(green_bars, rect13)

    # Add to acquire a single mask
    color_bars_without_red = cv.add(blue_bars, turq_bars)
    color_bars_without_red = cv.add(color_bars_without_red, violet_bars)
    color_bars_without_red = cv.add(color_bars_without_red, yellow_bars)
    color_bars_without_red = cv.add(color_bars_without_red, green_bars)

    # Detect red color bars
    b, g, r = cv.split(img)
    #   Subtract B and G from R to get red map
    red_map = np.int16(2 * np.int16(r) - np.int16(b) - np.int16(g))
    red_map = red_map / 2
    red_map[red_map < 0] = 0
    red_map = np.uint8(red_map)
    #   Get high values from the red map
    _, red_map_thresh_basic = cv.threshold(red_map, 170, 255, type=0)
    red_map_thresh_adaptive = cv.adaptiveThreshold(red_map, 255, adaptiveMethod=0, thresholdType=0, blockSize=45, C=-70.0)
    red_map_thresh = cv.bitwise_and(red_map_thresh_basic, red_map_thresh_adaptive)
    #   Apply morph ops
    morphed = cv.morphologyEx(red_map_thresh, cv.MORPH_OPEN, rect3)
    morphed = cv.morphologyEx(morphed, cv.MORPH_ERODE, rect3)
    #   Find rectangular blobs
    red_color_bars = np.zeros_like(morphed)
    _, cnts, _ = cv.findContours(morphed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv.contourArea(cnt) < 600:
            tmpmask = np.zeros_like(morphed)
            cv.drawContours(tmpmask, [cnt], -1, (255), -1)
            box = cv.boundingRect(cnt)
            box_size = box[2] * box[3]
            if cv.countNonZero(tmpmask) * 1.0 / box_size > 0.7:
                cv.drawContours(red_color_bars, [cnt], -1, (255), -1)
    #   Apply morph ops
    red_color_bars = cv.morphologyEx(red_color_bars, cv.MORPH_OPEN, rect5)
    red_color_bars = cv.morphologyEx(red_color_bars, cv.MORPH_DILATE, rect13)

    color_bars = cv.add(color_bars_without_red, red_color_bars)
    return color_bars


def detect_text_and_flashes(img):
    """
    :param img: grayscale, deinterlaced image
    """
    rect9 = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))

    # Detect bright areas
    flashes = cv.adaptiveThreshold(img, maxValue=255.0, adaptiveMethod=1, thresholdType=0, blockSize=45, C=-45)
    flashes = cv.dilate(flashes, rect9, iterations=1)

    return flashes


def detect_text_and_flashes_v0(img):
    """
    :param img: grayscale, deinterlaced image
    """
    rect5 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    rect7 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    rect9 = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))

    # Detect edges
    edges = cv.Laplacian(img, ddepth=0, ksize=3, scale=2.0, delta=0, borderType=0)
    _, edges = cv.threshold(edges, thresh=254, maxval=255, type=0)
    edges = cv.dilate(edges, rect5, iterations=1)
    edges = cv.erode(edges, rect7, iterations=1)
    edges = cv.dilate(edges, rect7, iterations=1)
    _, cnts, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(edges.shape, np.uint8)
    for cnt in cnts:
        if cv.contourArea(cnt) > 3:
            cv.drawContours(edges, [cnt], 0, (255), cv.cv.CV_FILLED)

    # Detect bright areas
    flashes = cv.adaptiveThreshold(img, maxValue=255.0, adaptiveMethod=1, thresholdType=0, blockSize=15, C=-45)
    flashes = cv.dilate(flashes, rect9, iterations=1)

    # Get intersection
    return cv.bitwise_and(edges, flashes)


def detect_border(img, flashes_text_mask):
    """
    :param img: deinterlaced color image
    :return:
    """
    height = img.shape[0]
    width = img.shape[1]
    flashes_text_mask_color = cv.cvtColor(flashes_text_mask, cv.COLOR_GRAY2BGR)
    img = cv.subtract(img, flashes_text_mask_color)

    crop_left = 0
    crop_right = width - 1
    crop_top = 0
    crop_down = height - 1
    current_thresh = 50.0
    avg_thresh = 15.0

    # Evaluate crops
    while crop_left < width and (np.var(img[:, crop_left]) < current_thresh or np.mean(img[:, crop_left]) < avg_thresh):
        crop_left += 1

    while crop_right > 0 and (np.var(img[:, crop_right]) < current_thresh or np.mean(img[:, crop_right]) < avg_thresh):
        crop_right -= 1

    while crop_top < height and (np.var(img[crop_top, :]) < current_thresh or np.mean(img[crop_top, :]) < avg_thresh):
        crop_top += 1

    while crop_down > 0 and (np.var(img[crop_down, :]) < current_thresh or np.mean(img[crop_down, :]) < avg_thresh):
        crop_down -= 1

    # Check if crop is valid
    min_crop_size = 0.3
    if crop_right - crop_left < min_crop_size * width or crop_down - crop_top < min_crop_size * height:
        # if invalid, return fill image mask
        return np.zeros(img.shape[:2], np.uint8)

    # Evaluate corners to crop
    corner_size = 0
    max_corner_size = height * 0.25
    args = [crop_top, crop_down, crop_left, crop_right, img.shape[:2], current_thresh, avg_thresh]
    while corner_size < max_corner_size and _corner_stats_ok(img, corner_size + 1, *args):
        corner_size += 1

    # Extend crops by 2
    crop_left += 2
    crop_right -= 2
    crop_top += 2
    crop_down -= 2
    if corner_size > 0:
        corner_size += 2

    # Draw the border mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:, 0:crop_left] = 255
    mask[:, crop_right+1:width] = 255
    mask[0:crop_top, :] = 255
    mask[crop_down+1:height, :] = 255

    if corner_size:
        mask += _prepare_corner_mask(corner_size, crop_top, crop_down, crop_left, crop_right, img.shape[:2])

    return mask


def _prepare_corner_mask(corner_size, top, down, left, right, shape):
    corner_size -= 1
    mask = np.zeros(shape, np.uint8)
    pts_1 = np.array([[left, top], [left, top + corner_size], [left + corner_size, top]], np.int32)
    pts_2 = np.array([[right, top], [right, top + corner_size], [right - corner_size, top]], np.int32)
    pts_3 = np.array([[left, down], [left, down - corner_size], [left + corner_size, down]], np.int32)
    pts_4 = np.array([[right, down], [right, down - corner_size], [right - corner_size, down]], np.int32)

    cv.fillConvexPoly(mask, pts_1, (255))
    cv.fillConvexPoly(mask, pts_2, (255))
    cv.fillConvexPoly(mask, pts_3, (255))
    cv.fillConvexPoly(mask, pts_4, (255))
    return mask


def _corner_stats_ok(img, corner_size, top, down, left, right, shape, var_thresh, mean_thresh):
    corner_mask = _prepare_corner_mask(corner_size, top, down, left, right, shape)
    corner_mask_3ch = cv.cvtColor(corner_mask, cv.COLOR_GRAY2BGR)
    masked_img = np.ma.array(img, mask=255 - corner_mask_3ch)
    factor = 1.3
    return np.ma.var(masked_img) < var_thresh * factor or np.ma.mean(masked_img) < mean_thresh * factor
