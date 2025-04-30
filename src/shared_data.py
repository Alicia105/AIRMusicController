import threading
# Settings
is_playing = True
volume = 1.0
pitch_shift_steps = 0
speed_rate = 1.0
block_size = 1024
param_lock = threading.Lock()