import soundfile as sf
import sounddevice as sd
import librosa

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
