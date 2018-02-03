import cv2
from matplotlib import pyplot as plt


# Open Video file using VideoCapture
video_capture = cv2.VideoCapture('webcam_recording.mp4')

while video_capture.isOpened():
    # Capture frame by frame
    is_frame_ready, frame = video_capture.read()
    if is_frame_ready:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
video_capture.release()
cv2.destroyAllWindows()
