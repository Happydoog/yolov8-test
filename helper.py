import cv2


def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
    return writer