import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import pyrubberband as pyrb

# Load audio
audio, sr = sf.read('../audio/029500_morning-rain-piano-65875.wav')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # Force mono

# Settings
block_size = 1024
volume = 1.0
pitch_shift_steps = 0
speed_rate = 1.0

# Control flags
is_playing = True

# Buffer for processed audio (thread-safe)
processed_buffer = queue.Queue(maxsize=50)  # 50 blocks max to avoid RAM explosion
position = 0

def background_processing():
    global position, volume, pitch_shift_steps, speed_rate
    while is_playing:
        # Load next block of audio
        end_pos = min(position + block_size, len(audio))
        block = audio[position:end_pos]
        position = end_pos

        if block.size == 0:
            break  # End of file

        # Pitch shift
        if pitch_shift_steps != 0:
            block = pyrb.pitch_shift(block, sr, n_steps=pitch_shift_steps)

        # Time stretch
        if speed_rate != 1.0:
            block = pyrb.time_stretch(block, sr, speed_rate)

        # Volume control
        block = volume * block

        # Clip
        block = np.clip(block, -1.0, 1.0)

        # Pad block if too short
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))

        # Push to buffer
        try:
            processed_buffer.put(block, timeout=0.5)
        except queue.Full:
            # If the buffer is full, just wait
            pass

def audio_callback(outdata, frames, time_info, status):
    try:
        block = processed_buffer.get_nowait()
    except queue.Empty:
        outdata[:] = np.zeros((frames, 1))
        return

    if block.ndim == 1:
        block = block[:, np.newaxis]  # Reshape

    outdata[:] = block

def control_audio():
    global volume, pitch_shift_steps, speed_rate
    while is_playing:
        cmd = input("w=Vol+, s=Vol-, a=Pitch-, d=Pitch+, q=Speed-, e=Speed+, x=Exit: ")
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
        elif cmd == 'x':
            stop_stream()
            break

        print(f"Volume={volume:.2f}, Pitch steps={pitch_shift_steps}, Speed={speed_rate:.2f}")

def stop_stream():
    global is_playing
    is_playing = False
    stream.stop()
    stream.close()

# Launch background processor
threading.Thread(target=background_processing, daemon=True).start()

# Launch control thread
threading.Thread(target=control_audio, daemon=True).start()

# Start audio output
stream = sd.OutputStream(
    samplerate=sr,
    channels=1,
    blocksize=block_size,
    callback=audio_callback
)
stream.start()

# Keep main alive
try:
    while is_playing:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_stream()
    print("Stopped by user.")
