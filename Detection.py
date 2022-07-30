import cv2
import numpy as np
import matplotlib.pyplot as plt


def corrdinates(image, parameters):
    slope, intercept = parameters
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def slope_intercept(image, lines):
    left = []
    right = []
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameter = np.polyfit((x1, x2), (y1, y2), 1)
            # print(para)
            slope = parameter[0]
            intercept = parameter[1]
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))
        # print(f'Slope of left line {left}')
        # print(f'Slope of Right line {right}')

        left_average = np.average(left, axis=0)
        right_average = np.average(right, axis=0)

        # print(f'Average Slope of left line {left_average}')
        # print(f'Average Slope of Right line {right_average}')

        left_line = corrdinates(image, left_average)
        right_line = corrdinates(image, right_average)
        return np.array([left_line, right_line])
    except:
        return None


def edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32([polygons]), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('Images/test_image.jpg')
# lane_image = np.copy(image)
# canny = edge(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_lines = slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, average_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('Result', combo_image)
# # cv2.imwrite("AfterDetection.jpg", combo_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



cap = cv2.VideoCapture('Images/test2.mp4')
# out = cv2.VideoWriter("Output.avi", -1, 20.0, (640,480))
while(cap.isOpened()):
    _, lane_image = cap.read()
    if not _:
        video = cv2.VideoCapture('Images/test2.mp4')
        continue
    canny = edge(lane_image)
    cropped_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=90, maxLineGap=15)
    average_lines = slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, average_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow('Result', combo_image)
    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()