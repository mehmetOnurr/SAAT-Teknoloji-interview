import av
import cv2
import numpy as np
import pyaudio
import threading
import torch

# Open the RTMP stream
container = av.open('rtmp url')

# Find the video and audio streams
video_stream = None
audio_stream = None
for s in container.streams:
    if isinstance(s, av.video.stream.VideoStream):
        video_stream = s
    elif isinstance(s, av.audio.stream.AudioStream):
        audio_stream = s

if video_stream is None:
    print('Error: No video stream found in RTMP stream')
    exit(1)

if audio_stream is None:
    print('Error: No audio stream found in RTMP stream')
    exit(1)

# Set up the OpenCV window
cv2.namedWindow('frame')

# Set up the PyAudio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=48000,
                output=True,
                frames_per_buffer=4098)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
object_names = ['sports ball', 'ball']

# Create a flag to signal the threads to stop processing
stop_event = threading.Event()

# Define a function to handle the video processing
def video_process(video_stream):
    i = 0
    filename = 'images/savedImage3.jpg'
    for packet in video_stream.demux():
        for frame in packet.decode():
            if isinstance(frame, av.video.frame.VideoFrame):
                # Convert the video frame to a format compatible with OpenCV
                frame = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', frame)
                i += 1
                print(frame.shape)
                print("görüntü okay")
                print(i)
                
                # Process the video frame with YOLOv5
                results = model(frame)
                if results.xyxy[0].tolist().count('sports ball') >= 2:
                    # Results
                    results.print()
                    # results.save()  # or .show()

                    print(results.xyxy[0])  # img1 predictions (tensor)
                    print(results.pandas().xyxy[0])

                if cv2.waitKey(1) == ord('q'):
                    stop_event.set()
                    return

                if stop_event.is_set():
                    return

# Define a function to handle the audio processing
def audio_process(audio_stream):
    for packet in audio_stream.demux():
        for frame in packet.decode():
            if isinstance(frame, av.audio.frame.AudioFrame):
                # Convert the audio frame to PyAudio format and play it
                audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.float32)
                stream.write(audio_data.tobytes())

                if stop_event.is_set():
                    return

# Start the video processing thread
video_thread = threading.Thread(target=video_process, args=(container,))
video_thread.start()

# Start the audio processing thread
audio_thread = threading.Thread(target=audio_process, args=(container,))
audio_thread.start()

# Wait for the user to press "q" to stop processing
while cv2.waitKey(1) != ord('q'):
    pass

# Set the stop event to signal the threads to stop processing
stop_event.set()

# Wait for the threads to finish processing
video_thread.join()
audio_thread.join()

# Stop the OpenCV window
cv2.destroyAllWindows()

# Stop the PyAudio stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()