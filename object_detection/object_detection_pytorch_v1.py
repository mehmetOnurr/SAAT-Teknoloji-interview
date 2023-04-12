import av
import cv2
import numpy as np
import pyaudio
import torch
import pandas as pd
# Open the RTMP stream

sentences = pd.DataFrame(columns=  ['Zaman','Cumle','Guncel Kelime Sayisi'])
obje = pd.DataFrame(columns = ['Zaman','Ekran Goruntusu Adi'])


if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu" 
device = torch.device(dev) 

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)
object_names = ['sports ball','ball']
container = av.open('rtmp://176.53.96.101/showLow/sportstv_fhd')

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
               )
# frames_per_buffer=4098
hop_size = 512
n_channels = 1
rate_factor = 2.0

i = 0
folder = 'target_saves/'
filename = 'result'
format = '.jpg'

# Play the video and audio frames
for packet in container.demux():
    for frame in packet.decode():
        if isinstance(frame, av.video.frame.VideoFrame):
            # Convert the video frame to a format compatible with OpenCV
            frame = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
            cv2.imshow('frame',frame)
            # if i == 100:
            #    cv2.imwrite(filename, frame)
            i+=1
            print(frame.shape)
            print("görüntü okay")
            print(i)
            if cv2.waitKey(1) == ord('q'):
                break
            print
            results = model(frame)
            if results.xyxy[0].tolist().count('sports ball') >=2:
                # Results
                cv2.imwrite(folder+filename+str(i)+format, frame)
                results.print()
                # results.save()  # or .show()
                
                print(results.xyxy[0] ) # img1 predictions (tensor)
                print(results.pandas().xyxy[0] )

           
        if isinstance(frame, av.audio.frame.AudioFrame):
            print("ses okay")
            # Convert the audio frame to PyAudio format and play it
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.float32)

            stream.write(audio_data.tobytes())

    # Check if the user has pressed "q" to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Stop the OpenCV window
cv2.destroyAllWindows()

# Stop the PyAudio stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()