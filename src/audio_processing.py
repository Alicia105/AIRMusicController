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
stretch_buffer = np.array([]) 
block_size = 2048  # Small block for streaming
is_playing = True
volume = 1.0
pitch_shift_steps = 0
speed_rate = 1.0
position = 0


def audio_callback(outdata, frames, time, status):
    global position, audio, volume, pitch_shift_steps, speed_rate, stretch_buffer

    if not is_playing:
        outdata[:] = np.zeros((frames, 1))
        return

    # Fill stretch buffer if needed
    while len(stretch_buffer) < frames:
        # Load next block of original audio
        end_pos = min(position + block_size, len(audio))
        block = audio[position:end_pos]
        position = end_pos

        if block.size == 0:
            # End of audio
            outdata[:] = np.zeros((frames, 1))
            raise sd.CallbackStop()

        # Apply pitch shift
        if pitch_shift_steps != 0:
            block = librosa.effects.pitch_shift(block, sr=sr, n_steps=pitch_shift_steps)

        if speed_rate != 1.0:
            # 1. STFT
            stft = librosa.stft(block)
            # 2. Phase Vocoder
            stft_stretched = librosa.phase_vocoder(stft, rate=speed_rate, hop_length=512)
            # 3. Inverse STFT
            block = librosa.istft(stft_stretched, hop_length=512)

        # Append to buffer
        stretch_buffer = np.concatenate((stretch_buffer, block))

    # Take exactly `frames` samples
    out_chunk = stretch_buffer[:frames]
    stretch_buffer = stretch_buffer[frames:]

    # Volume
    out_chunk = volume * out_chunk

    # Clip to avoid overflow
    out_chunk = np.clip(out_chunk, -1.0, 1.0)

    # Reshape
    if out_chunk.ndim == 1:
        out_chunk = out_chunk[:, np.newaxis]

    outdata[:] = out_chunk

# Control thread
def control_audio():
    global volume, pitch_shift_steps, speed_rate
    while True:
        cmd = input("w=Vol+, s=Vol-, a=Pitch-, d=Pitch+, q=Speed-, e=Speed+: ")
        if cmd == 'w':
            volume = min(volume + 0.1, 2.0)
        elif cmd == 's':
            volume = max(volume - 0.1, 0.0)
        elif cmd == 'a':
            pitch_shift_steps -= 1
        elif cmd == 'd':
            pitch_shift_steps += 1
        elif cmd == 'q':
            speed_rate = max(0.5, speed_rate - 0.1)
        elif cmd == 'e':
            speed_rate = min(2.0, speed_rate + 0.1)

        print(f"Volume={volume:.2f}, Pitch steps={pitch_shift_steps}, Speed={speed_rate:.2f}")

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

