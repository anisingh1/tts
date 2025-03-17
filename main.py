import sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.utils.prefs import pref
import pandas as pd
from tqdm import tqdm
import uuid

from app.tts import TTS
from app.stt import STT

tts = TTS()
stt = STT()

file = pref().getPref('input_data', 'label')
workers = 1

if not os.path.exists(file):
    print(f"Could not find {file}")
    sys.exit(1)


def get_audio(text, locale):
    id = str(uuid.uuid4())
    audio = tts.generate_audio(text=text, locale=locale)
    tts.save_audio(audio, filename=f"{id}.wav")
    data = stt.transcribe_audio(f"{id}.wav")
    wpm = stt.calculate_wpm(data)
    ret = {
        "filepath": f"{id}.wav",
        "details": data,
        "wpm": wpm
    }
    return ret
    

def process_text(df):
    data = df.to_dict('records')
    columns = []
    columns.extend(data[0].keys())
    input = pref().getPref('text_field', 'label')
    locale = pref().getPref('locale_field', 'label')

    out = []
    row_count = len(data)
    tbar = tqdm(total=row_count, desc='Processing', leave=True, unit='texts')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_audio, str(row[input]), str(row[locale])): row for row in data}
        for future in as_completed(futures):
            out_row = futures[future]
            results = future.result()
            if 'wpm' in results:
                out_row['words'] = results['wpm']['total_words']
                out_row['duration'] = results['wpm']['total_duration']
                out_row['wpm'] = results['wpm']['wpm']
            out.append(out_row)
            tbar.update(n=1)
            tbar.refresh()
    tbar.refresh()
    tbar.close()

    out_df = pd.DataFrame.from_records(out)
    return out_df


if __name__ == "__main__":
    if str(file).endswith('xlsx'):
        df = pd.read_excel(file, header=0)
    elif str(file).endswith('csv'):
        df = pd.read_csv(file)
    df = df.where(pd.notnull(df), None)
    df = process_text(df)
    filepath = os.path.dirname(file)
    filename = os.path.basename(file)
    df.to_excel(os.path.join(filepath, 'out_' + filename), index=False)
    sys.exit(0)