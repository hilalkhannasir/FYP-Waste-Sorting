import cv2
import numpy as np


def process_video(input_video, output_video, threshold):

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    ret, first_rgb = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return

    first_gray = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2GRAY)

    progress = 0
    while True:
        ret, candidate_rgb = cap.read()
        if not ret:
            out.write(first_rgb)
            print(f"End of video. Final frame saved.")
            break


        candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_BGR2GRAY)

        abs_diff = cv2.absdiff(first_gray, candidate_gray)
        distance = np.mean(abs_diff)

        progress+=1
        if progress % 100 == 0:
            print(f"Progress: {progress}/{total_frames}")
        if distance > threshold:
            out.write(first_rgb)

            first_rgb = candidate_rgb
            first_gray = candidate_gray

        else:
            pass

    cap.release()
    out.release()

    print(f"Output saved to: {output_video}")


process_video(
    input_video="3.mp4",
    output_video="video_a3.mp4",
    threshold=30
)