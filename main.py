import cv2
import numpy as np

def nothing(_):
    pass

def create_hsv_trackbars():
    cv2.namedWindow("HSV Trackbars")
    cv2.createTrackbar("H min", "HSV Trackbars", 0, 179, nothing)
    cv2.createTrackbar("H max", "HSV Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S min", "HSV Trackbars", 0, 255, nothing)
    cv2.createTrackbar("S max", "HSV Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V min", "HSV Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V max", "HSV Trackbars", 255, 255, nothing)

def get_hsv_ranges():
    hmin = cv2.getTrackbarPos("H min", "HSV Trackbars")
    hmax = cv2.getTrackbarPos("H max", "HSV Trackbars")
    smin = cv2.getTrackbarPos("S min", "HSV Trackbars")
    smax = cv2.getTrackbarPos("S max", "HSV Trackbars")
    vmin = cv2.getTrackbarPos("V min", "HSV Trackbars")
    vmax = cv2.getTrackbarPos("V max", "HSV Trackbars")
    return (hmin, smin, vmin), (hmax, smax, vmax)

def build_kalman():
    # State: [x, y, vx, vy]
    # Measurement: [x, y]
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf

def find_largest_blob_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 500:  # ignore noise blobs
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), largest

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0)).")

    create_hsv_trackbars()

    kf = build_kalman()
    initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for easier interaction
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = get_hsv_ranges()
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Kalman predict every frame
        pred = kf.predict()
        pred_x, pred_y = int(pred[0]), int(pred[1])

        found = find_largest_blob_center(mask)
        if found is not None:
            (cx, cy), contour = found

            # draw contour and raw measurement
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)  # raw measurement (red)

            meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)

            if not initialized:
                # initialize state with first measurement
                kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                initialized = True
            else:
                kf.correct(meas)

        # filtered estimate after correct()
        est = kf.statePost
        est_x, est_y = int(est[0]), int(est[1])

        # draw prediction and filtered estimate
        cv2.circle(frame, (pred_x, pred_y), 6, (255, 0, 0), -1)   # prediction (blue)
        cv2.circle(frame, (est_x, est_y), 6, (0, 255, 0), -1)     # filtered (green)

        cv2.putText(frame, "Red=raw  Green=filtered  Blue=pred",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Tracking", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()