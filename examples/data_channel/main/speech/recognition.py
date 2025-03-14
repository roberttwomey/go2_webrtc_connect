import speech_recognition as sr

# Install SpeechRecognition library: 'pip install SpeechRecognition pyaudio'

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable
def listen_for_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        audio_text = r.listen(source)
        print("Time over, thanks")
        # recoginze_() method will throw a request
        # error if the API is unreachable,
        # hence using exception handling
        
        try:
            command = r.recognize_google(audio_text)
            # using google speech recognition
            print("You said: "+command)
            return command.lower()
        except:
            print("Sorry, I did not get that")