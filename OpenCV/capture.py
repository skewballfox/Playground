import cv2

feed = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
fps = 30
size = (
    int(feed.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(feed.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

video_writer = cv2.VideoWriter(
    "Output.mp4", cv2.VideoWriter_fourcc("X", "2", "6", "4"), fps, size
)
print("yeet")
success, frame = feed.read()
frames_remaining = 10 * fps - 1  # 10 seconds of frames
while success and frames_remaining > 0:
    video_writer.write(frame)
    sucess, frame = feed.read()
    frames_remaining -= 1
