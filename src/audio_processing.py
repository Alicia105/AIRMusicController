import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import pyrubberband as pyrb
import shared_data

# Load audio
audio, sr = sf.read('../audio/0_oliver-colbentson_bwv1006_mov5.wav')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # Force mono

# Globals
stream = None
position = 0
processed_buffer = queue.Queue(maxsize=50)  # Avoid RAM explosion

def background_processing():
    global position, processed_buffer

    processing_block_size = 16384
    overlap_size = processing_block_size // 4

    fade = np.linspace(0, 1, overlap_size)
    fade_in = np.sqrt(fade)
    fade_out = np.sqrt(1.0 - fade)

    previous_tail = None

    while True:
        with shared_data.param_lock:
            if not shared_data.is_playing:
                break
            vol = shared_data.volume
            pitch_shift = shared_data.pitch_shift_steps
            speed = shared_data.speed_rate

        # Load block
        end_pos = min(position + processing_block_size, len(audio))
        block = audio[position:end_pos]
        position = end_pos

        if block.size == 0:
            break

        # Effects
        if pitch_shift != 0:
            block = pyrb.pitch_shift(block, sr, n_steps=pitch_shift)
        if speed != 1.0:
            block = pyrb.time_stretch(block, sr, speed)
        block *= vol
        block = np.clip(block, -1.0, 1.0)

        # Crossfade
        if previous_tail is not None and len(block) >= overlap_size:
            head = block[:overlap_size]
            crossfaded = (previous_tail * fade_out) + (head * fade_in)
            block[:overlap_size] = crossfaded

        previous_tail = block[-overlap_size:] if len(block) >= overlap_size else block

        # Slice into playback blocks
        for i in range(0, len(block), shared_data.block_size):
            small_block = block[i:i + shared_data.block_size]
            if len(small_block) < shared_data.block_size:
                small_block = np.pad(small_block, (0, shared_data.block_size - len(small_block)))
            try:
                processed_buffer.put(small_block, timeout=0.5)
            except queue.Full:
                pass

def audio_callback(outdata, frames, time_info, status):
    try:
        block = processed_buffer.get_nowait()
    except queue.Empty:
        outdata[:] = np.zeros((frames, 1))
        return

    if block.ndim == 1:
        block = block[:, np.newaxis]
    outdata[:] = block

def control_audio_keyboard():
    while True:
        with shared_data.param_lock:
            if not shared_data.is_playing:
                break

        cmd = input("w=Vol+, s=Vol-, a=Pitch-, d=Pitch+, q=Speed-, e=Speed+, p=Play/Pause, x=Exit: ")
        with shared_data.param_lock:
            if cmd == 'w':
                shared_data.volume = min(shared_data.volume + 0.1, 2.0)
            elif cmd == 's':
                shared_data.volume = max(shared_data.volume - 0.1, 0.0)
            elif cmd == 'a':
                shared_data.pitch_shift_steps -= 1
            elif cmd == 'd':
                shared_data.pitch_shift_steps += 1
            elif cmd == 'q':
                shared_data.speed_rate = max(0.5, shared_data.speed_rate - 0.1)
            elif cmd == 'e':
                shared_data.speed_rate = min(2.0, shared_data.speed_rate + 0.1)
            elif cmd == 'p':
                shared_data.is_playing = not shared_data.is_playing
            elif cmd == 'x':
                shared_data.is_playing = False
                stop_stream()
                break

            print(f"Volume={shared_data.volume:.2f}, Pitch={shared_data.pitch_shift_steps}, Speed={shared_data.speed_rate:.2f}")

def control_audio_vision(vol, pitch, speed, is_playing):
    with shared_data.param_lock:
        shared_data.volume = vol
        shared_data.pitch_shift_steps = pitch
        shared_data.speed_rate = speed
        shared_data.is_playing = is_playing

    print(f"Volume={vol:.2f}, Pitch steps={pitch}, Speed={speed:.2f}")

def stop_stream():
    global stream
    with shared_data.param_lock:
        shared_data.is_playing = False
    if stream:
        stream.stop()
        stream.close()

def start_audio_system(with_control=True):
    global stream
    with shared_data.param_lock:
        shared_data.is_playing = True

    threading.Thread(target=background_processing, daemon=True).start()

    if with_control:
        threading.Thread(target=control_audio_keyboard, daemon=True).start()

    stream = sd.OutputStream(
        samplerate=sr,
        channels=1,
        blocksize=shared_data.block_size,
        callback=audio_callback
    )
    stream.start()

    try:
        while True:
            with shared_data.param_lock:
                if not shared_data.is_playing:
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_stream()
        print("Stopped by user.")

if __name__ == "__main__":
    start_audio_system(True)
