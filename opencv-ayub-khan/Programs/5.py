import cv2
from matplotlib import pyplot as plt


# Open the default camera
video_capture = cv2.VideoCapture(0)

# Verify that camera opened successfully
if not video_capture.isOpened():
    print 'Unable to open camera'
else:
    print 'Camera Open and ready to use'

# Get default resolutions for the frames.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_count = 20  # Frames per second

#Defind Video file extention(which codec to use to compress the frames)
file_type = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object.
video_out = cv2.VideoWriter(
    filename='webcam_recording.mp4',
    fourcc=file_type,
    fps=fps_count,
    frameSize=(frame_width, frame_height),
    isColor=True
)

#Record Video Frame by Frame and write it.
while video_capture.isOpened():
    is_frame_ready, frame = video_capture.read()
    if is_frame_ready:
        video_out.write(frame)
        cv2.imshow('Webcam Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
video_capture.release()
video_out.release()
cv2.destroyAllWindows()
