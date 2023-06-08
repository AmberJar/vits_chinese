from pydub import AudioSegment
from noisereduce import reduce_noise
import pyloudnorm as pyln
import librosa
import numpy as np
import soundfile as sf

trim_db = 30
wav_db = -20.0


def remove_noise(wavt_path, tmp_path, out_path, sampling_rate):
    # Load the .wav file using pydub
    audio = AudioSegment.from_wav(wavt_path)

    # Convert the audio to a numpy array
    audio_array = audio.get_array_of_samples()

    # Perform noise reduction on the audio array
    reduced_noise = reduce_noise(audio_array, audio.frame_rate)

    # Create a new AudioSegment from the reduced noise array
    reduced_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    # Export the reduced noise audio as a .wav file
    reduced_audio.export(tmp_path, format="wav")
    process_wav(tmp_path, out_path, sampling_rate)


def process_wav(filename, file_o, sr):
    meter = pyln.Meter(sr)
    wav, sr = librosa.load(filename, sr=sr)
    loudness = meter.integrated_loudness(wav)
    wav = pyln.normalize.loudness(wav, loudness, wav_db)
    if np.abs(wav).max() > 1.0:
        wav = wav / np.abs(wav).max()

    sf.write(file_o, wav, sr)