import whisper


class STT:
    def __init__(self):
        model_name = 'large-v3'
        self.model = whisper.load_model(name=model_name, download_root='models/whisper', in_memory=True)
    
    
    def transcribe_audio(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, fp16=False, word_timestamps=True)
        return result
    
    
    def calculate_wpm(self, transcription_result: dict) -> float:
        total_duration = 0.0
        total_words = 0
        for segment in transcription_result['segments']:
            words = segment['words']
            if not words:
                return 0.0
            
            total_duration = words[-1]['end'] - words[0]['start']
            total_words = len(words)
            
        total_duration_minutes = total_duration / 60.0
        wpm = total_words / total_duration_minutes
        return {
            "wpm": wpm,
            "total_duration": total_duration,
            "total_words": total_words
        }