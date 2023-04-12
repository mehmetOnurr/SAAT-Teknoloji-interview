import av
import pyaudio
import requests
import json
import time
import cv2
import numpy as np
import azure.cognitiveservices.speech as speechsdk
# Replace with your own subscription key and endpoint
subscription_key = 'YOUR_SUBSCRIPTION_KEY'
endpoint = 'YOUR_ENDPOINT'

# Set up the PyAudio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024)

# Set up the Azure Cognitive Services request headers
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'audio/wav; codec=audio/pcm; samplerate=16000'
}

# Initialize variables for speech recognition
current_audio = b''
silent_count = 0
silence_threshold = 15
speech_recognition_started = False

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
    
# Azure Speech-to-Text API endpoint
url = 'url'

# Setup speech recognizer
speech_config = speechsdk.SpeechConfig(subscription="key", region="westeurope")
speech_config.speech_recognition_language = "tr-TR"  # Set the language to Turkish

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# # Azure Speech-to-Text API subscription key
# subscription_key = 'your_subscription_key'

# # Request headers
# headers = {
#     'Accept': 'application/json',
#     'Content-Type': 'audio/wav',
#     'Ocp-Apim-Subscription-Key': subscription_key,
# }



# Play the video and audio frames
for packet in container.demux():
    for frame in packet.decode():
        if isinstance(frame, av.video.frame.VideoFrame):
            # Display the video frame
            frame = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
        if isinstance(frame, av.audio.frame.AudioFrame):
            # Convert the audio frame to bytes and append to current audio
            audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.float32)
            current_audio += audio_data.tobytes()
            
            # If speech recognition has not started and audio is above threshold, start speech recognition
            if not speech_recognition_started and np.abs(np.array(frame.samples)).max() > 800:
                speech_recognition_started = True
                print('Speech recognition started')
            
            # If speech recognition has started and audio is below threshold, increment silent count
            if speech_recognition_started and np.abs(np.array(frame.samples)).max() < 800:
                silent_count += 1
            
            # If silent count reaches threshold, end speech recognition and send request
            if speech_recognition_started and silent_count >= silence_threshold:
                speech_recognition_started = False
                silent_count = 0
                
                # Send the speech recognition request
                response = requests.post(url, headers=headers, data=current_audio)
                print("Sonuc",response)
                if response.status_code == 200:
                    print("sonuc")
                    result = json.loads(response.text)
                    if 'DisplayText' in result:
                        display_text = result['DisplayText']
                        print('Transcription:', display_text)
                    else:
                        print('Speech recognition failed')
                else:
                    print('Speech recognition failed')
                
                # Reset the current audio
                current_audio = b''
            
    # Check if the user has pressed "q" to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Stop the OpenCV window
cv2.destroyAllWindows()


# Stop the PyAudio stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()