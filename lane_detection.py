import cv2
import numpy as np
import pyttsx3
import threading

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

def region_of_interest(image):
    height, width = image.shape[:2]
    polygons = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if len(line) == 4:
                x1, y1, x2, y2 = line
                if all(isinstance(v, (int, np.integer)) for v in [x1, y1, x2, y2]):
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 8)
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return np.array([0, y1, 0, y2])
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if 0.5 < abs(slope) < 2.0:  # Filter extreme slopes
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    line_list = []
    if len(left_fit) > 0:
        left_avg = np.average(left_fit, axis=0)
        line_list.append(make_coordinates(image, left_avg))
    if len(right_fit) > 0:
        right_avg = np.average(right_fit, axis=0)
        line_list.append(make_coordinates(image, right_avg))

    return np.array(line_list)

def check_deviation(image, lines):
    height, width = image.shape[:2]
    vehicle_center = width // 2
    warning_text = ""

    if len(lines) == 2:
        left_x = lines[0][2]
        right_x = lines[1][2]
        lane_center = (left_x + right_x) // 2
        deviation = vehicle_center - lane_center

        if abs(deviation) > width * 0.05:
            warning_text = "⚠️ Vehicle deviating from lane!"
            cv2.putText(image, warning_text, (width // 4, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            speak(warning_text)

    return image

def detect_lane(frame):
    canny_image = canny(frame)
    roi = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        roi, 2, np.pi / 180, 100, np.array([]),
        minLineLength=50, maxLineGap=100
    )
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    combo_image = check_deviation(combo_image, averaged_lines)
    return combo_image
