import av
import cv2
import sounddevice as sd
import numpy as np

# Open the RTMP stream
container = av.open('rtmp url')

# Find the video and audio streams
video_stream = None
audio_stream = None
for s in container.streams:
    print(type(s))
    if type(s) == av.video.stream.VideoStream:
        video_stream = s
    elif type(s) == av.audio.stream.AudioStream:
        audio_stream = s

if video_stream is None:
    print('Error: No video stream found in RTMP stream')
    exit(1)

if audio_stream is None:
    print('Error: No audio stream found in RTMP stream')
    exit(1)

# Set up the OpenCV window
cv2.namedWindow('frame')

# Set up the sounddevice stream
def audio_callback(indata, frames, time, status):
    pass
i = 0
filename = 'images/savedImage3.jpg'
with sd.InputStream(callback=audio_callback):
    # Read frames from the video and audio streams
    for packet in container.demux():
        for frame in packet.decode():
            # Process video frames
            print(frame)
            if isinstance(frame, av.video.frame.VideoFrame):
                # Convert the frame to a format compatible with OpenCV
                frame = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)

                if i == 100:
                    cv2.imwrite(filename, frame)
                i+=1
                print("görüntü okay")
                print(i)
                # Display the frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Process audio frames
            if type(frame) == av.audio.frame.AudioFrame:
                print("ses okay")
                # Get the audio data as a NumPy array
                audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
                print(type(audio_data),audio_stream.rate,audio_data.shape)
                # Play the audio data using sounddevice
                sd.play(audio_data, audio_stream.rate)

        # Check if the user has pressed "q" to exit the program
        if cv2.waitKey(1) == ord('q'):
            break

# Stop the OpenCV window
cv2.destroyAllWindows()

# Stop the sounddevice stream
sd.stop()