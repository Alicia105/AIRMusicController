import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import pyrubberband as pyrb

# Load audio
audio, sr = sf.read('../audio/0_oliver-colbentson_bwv1006_mov5.wav')
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


"""def background_processing(): 
    global position, volume, pitch_shift_steps, speed_rate
    while is_playing:

        processing_block_size = 16384  # Processing block size
        # Load next big processing block
        end_pos = min(position + processing_block_size, len(audio))
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

        # Now split into small playback blocks
        for i in range(0, len(block), block_size):
            small_block = block[i:i+block_size]
            if len(small_block) < block_size:
                small_block = np.pad(small_block, (0, block_size - len(small_block)))
            try:
                processed_buffer.put(small_block, timeout=0.5)
            except queue.Full:
                pass  # skip if full

"""

def background_processing():
    global position, volume, pitch_shift_steps, speed_rate

    processing_block_size = 16384  # Processing block size
    overlap_size = processing_block_size // 4  # 25% overlap
    #fade_in = np.linspace(0, 1, overlap_size)
    #fade_out = np.linspace(1, 0, overlap_size)
    fade = np.linspace(0, 1, overlap_size)
    fade_in = np.sqrt(fade)         # Nonlinear fade in
    fade_out = np.sqrt(1.0 - fade)  # Nonlinear fade out

    #fade = np.linspace(0, np.pi / 2, overlap_size)  # From 0 to 90° (π/2)
    #fade_in = np.sin(fade)  # Fade in: Sine curve (0 -> 1)
    #fade_out = np.sin(np.pi / 2 - fade)  # Fade out: Reverse sine (1 -> 0)

    previous_tail = None

    while is_playing:
        # Load big block with overlap
        end_pos = min(position + processing_block_size, len(audio))
        block = audio[position:end_pos]
        position = end_pos

        if block.size == 0:
            break  # End of file

        # Apply pitch shift
        if pitch_shift_steps != 0:
            block = pyrb.pitch_shift(block, sr, n_steps=pitch_shift_steps)

        # Apply time stretch
        if speed_rate != 1.0:
            block = pyrb.time_stretch(block, sr, speed_rate)

        # Volume control
        block = volume * block

        # Clip
        block = np.clip(block, -1.0, 1.0)

        # Handle overlap crossfade
        if previous_tail is not None:
            # Crossfade previous tail with current head
            head = block[:overlap_size]
            crossfaded = (previous_tail * fade_out) + (head * fade_in)

            # Replace the start of the current block with crossfaded version
            block[:overlap_size] = crossfaded

        # Save new tail for next block
        if len(block) >= overlap_size:
            previous_tail = block[-overlap_size:]
        else:
            previous_tail = block

        # Now slice into small playback blocks
        for i in range(0, len(block), block_size):
            small_block = block[i:i+block_size]
            if len(small_block) < block_size:
                small_block = np.pad(small_block, (0, block_size - len(small_block)))
            try:
                processed_buffer.put(small_block, timeout=0.5)
            except queue.Full:
                pass  # skip if full


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
    global volume, pitch_shift_steps, speed_rate,is_playing
    while is_playing:
        cmd = input("w=Vol+, s=Vol-, a=Pitch-, d=Pitch+, q=Speed-, e=Speed+,  p=play/pause, x=Exit: ")
        if cmd == 'w':
            volume = min(volume + 0.1, 2.0)
        elif cmd == 's':
            volume = max(volume - 0.1, 0.0)
        elif cmd == 'a':
            pitch_shift_steps -= 1
        elif cmd == 'd':
            pitch_shift_steps += 1
        elif cmd == 'p':
            is_playing = not is_playing
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

def start_audio_system(with_control=True):
    global stream

    threading.Thread(target=background_processing, daemon=True).start()
    
    if with_control:
        threading.Thread(target=control_audio, daemon=True).start()

    stream = sd.OutputStream(
        samplerate=sr,
        channels=1,
        blocksize=block_size,
        callback=audio_callback
    )
    stream.start()

    try:
        while is_playing:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_stream()
        print("Stopped by user.")


if __name__ == "__main__":
    start_audio_system(False)
