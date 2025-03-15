import torch
import torchaudio
import numpy as np

from bark.generation import load_codec_model
from encodec.utils import convert_audio

from hubert.hubert_manager import HuBERTManager
hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer


class Clone:
    def __init__(self):
        self.device, use_gpu = self._best_device()
        self.hubert_model = CustomHubert(checkpoint_path='models/hubert/hubert.pt').to(self.device)
        self.tokenizer = CustomTokenizer.load_from_checkpoint('models/hubert/tokenizer.pth').to(self.device)
        self.model = load_codec_model(use_gpu=use_gpu)
        self.hubert_model.eval()
        self.tokenizer.eval()
    
    
    def _best_device(self, use_gpu=True):
        use_gpu = False
        if torch.cuda.device_count() > 0 and use_gpu:
            device = "cuda"
        elif torch.backends.mps.is_available() and use_gpu:
            device = "mps"
            use_gpu = False
        else:
            device = "cpu"
            use_gpu = False
        return device, use_gpu
    
    
    def clone(self, audio_filepath: str, voice_name: str):
        wav, sr = torchaudio.load(audio_filepath)
        wav = convert_audio(wav, sr, self.hubert_model.sample_rate, self.hubert_model.channels)
        wav = wav.to(self.device)
        
        semantic_vectors = self.hubert_model.forward(wav, input_sample_hz=self.hubert_model.sample_rate)
        semantic_tokens = self.tokenizer.get_token(semantic_vectors)
        
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

        # move codes to cpu
        codes = codes.cpu().numpy()
        # move semantic tokens to cpu
        semantic_tokens = semantic_tokens.cpu().numpy()
        
        output_path = 'assets/prompts/' + voice_name + '.npz'
        np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
        return output_path


obj = Clone()
obj.clone('/Users/anisingh/Downloads/audio.wav', 'en_male')