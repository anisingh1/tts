import whisper


class STT:
    def __init__(self):
        model_name = 'large-v3'
        self.model = whisper.load_model(name=model_name, download_root='models/whisper', in_memory=True)
    
    
    def transcribe_audio(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        return result['text']