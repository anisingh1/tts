import nltk
import numpy as np
import os

nltk.download('punkt_tab')
os.environ["SERP_ENABLE_MPS"] = "1"

from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE
from scipy.io.wavfile import write as wav_write


class TTS:
    def __init__(self):
        semantic_path = "models/bark/semantic_output/pytorch_model.bin"
        coarse_path = "models/bark/coarse_output/pytorch_model.bin"
        fine_path = "models/bark/fine_output/pytorch_model.bin"
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            text_model_path=semantic_path,
            coarse_use_gpu=True,
            coarse_use_small=False,
            coarse_model_path=coarse_path,
            fine_use_gpu=True,
            fine_use_small=False,
            fine_model_path=fine_path,
            codec_use_gpu=True,
            force_reload=False,
            path="models"
        )
        self.silence = np.zeros(int(0.25 * SAMPLE_RATE))
    
    
    def generate_audio(self, text: str, locale: str, speaker: str = '6'):
        speaker = "v2/" + locale.split('-')[0] + "_speaker_" + speaker
        text = text.replace("\n", " ").strip()
        sentences = nltk.sent_tokenize(text)
        pieces = []
        for sentence in sentences:
            audio_array = generate_audio(sentence, history_prompt=speaker)
            pieces += [audio_array, self.silence.copy()]
        return np.concatenate(pieces)
    
    
    def save_audio(self, audio, filename):
        wav_write(filename, rate=SAMPLE_RATE, data=audio)
