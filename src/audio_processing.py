import soundfile as sf
import sounddevice as sd
import librosa
import numpy as np
import threading
import time

# Load the audio

"""Faster
audio, sr = librosa.load('../audio/029500_morning-rain-piano-65875.wav', sr=None)
audio_faster = librosa.effects.time_stretch(audio, rate=1.5)

sd.play(audio_faster, samplerate=sr)
sd.wait()"""

"""Volume
#audio, sr = sf.read('../audio/029500_morning-rain-piano-65875.wav')  # 'sr' = sample rate
# Half volume
quieter_audio = 0.5 * audio

sd.play(quieter_audio, samplerate=sr)
sd.wait()"""

"""Pitch shift
audio, sr = librosa.load('../audio/029500_morning-rain-piano-65875.wav', sr=None)


print(f"Audio shape: {audio.shape}")
print(f"Sampling rate: {sr}")
print(f"Duration (seconds): {audio.shape[0] / sr}")

# +4 semitones (pitch up)
audio_pitch_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)

# Play
sd.play(audio_pitch_up, samplerate=sr)
sd.wait()"""


# Load
audio, sr = librosa.load('../audio/029500_morning-rain-piano-65875.wav', sr=None)
audio_ptr = 0

# Settings
block_size = 2048  # Small block for streaming
is_playing = True
volume = 1.0
pitch_shift_steps = 0
speed_rate = 1.0


# Audio callback
def audio_callback(outdata, frames, time, status):
    global audio_ptr, volume, is_playing
    if not is_playing:
        outdata[:] = np.zeros((frames, 1))
        return

    # Take small block
    block = audio[audio_ptr:audio_ptr + frames]
    if block.shape[0] < frames:
        block = np.pad(block, (0, frames - block.shape[0]))

    # Apply volume
    block = volume * block

    # Output
    outdata[:] = block.reshape(-1, 1)

    # Update pointer
    audio_ptr += frames
    if audio_ptr >= len(audio):
        audio_ptr = 0  # Loop again or set is_playing = False

# Control thread
def control_audio():
    global volume, is_playing
    while True:
        command = input("Control (w=vol+, s=vol-, p=play/pause, x=exit): ")
        if command == 'w':
            volume = min(volume + 0.1, 2.0)
        elif command == 's':
            volume = max(volume - 0.1, 0.0)
        elif command == 'p':
            is_playing = not is_playing
        elif command == 'x':
            is_playing = False
            break
        print(f"Volume: {volume:.2f}")

# Launch audio stream and control
stream = sd.OutputStream(
    samplerate=sr, channels=1, callback=audio_callback, blocksize=block_size
)
stream.start()

threading.Thread(target=control_audio, daemon=True).start()

# Keep main thread alive
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop()
    stream.close()

