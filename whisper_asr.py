#encoding = 'utf-8'

from faster_whisper import WhisperModel
import audio_utils
import json
from vad_utils import SpeechVadFrontend


class WhisperAsrInstance:
    def __init__(self,
            model_path, 
            processor_path, 
            multilingual = True,
            use_gpu = True,
            device = "cuda",
            max_dec_length = 400,
            max_batch_size = 1, 
            vad_enable = True
            ):
        self.use_gpu      = use_gpu
        self.device = device
        self.multilingual = multilingual
        self.max_dec_len  = max_dec_length
        self.model_path   = model_path
        self.processor    = processor_path
        self.vad_enable   = vad_enable
        if device.startswith('cuda'):
            index = int(device[-1])
            device = 'cuda'
        else:
            index = 0
            device = 'cpu'

        if self.vad_enable:
            self.vad_conf = {
                'vad_type': 'webrtcvad',
                'vad_level': 1,
                'frame_length': 30,
                'window_size' : 10,
                'seg_thres' : 0.9,
                'max_speech_len' : 30,
                'min_speech_len' : 5,
                'merge_sil_thres' : 2
                }

        self.model = WhisperModel(model_path, device=device, device_index=index, compute_type="float32")
    
    def transcribe_audio_file(self, audio_path: str, lang=None, word_timestamps=False):
        # load
        if audio_path.startswith('http') and ':' in audio_path:
            wf, sr = audio_utils.load_from_http_url(audio_path)
        else:
            wf, sr = audio_utils.load_from_local_path(audio_path)
        import pdb
        pdb.set_trace()
        wf = wf.squeeze()
        total_dur = wf.size()[0] / sr 

        if self.vad_enable:
            vad_frontend = SpeechVadFrontend(**self.vad_conf)
            vad_segments, vad_segment_lens, vad_segment_times = vad_frontend.get_all_speech_segments(wf, sr)
            segments = []
            for vad_wf, vad_time in zip(vad_segments, vad_segment_times):
                vad_wf = vad_wf.numpy()
                tmp_rslt, sub_info = self.model.transcribe(
                        vad_wf, 
                        language=lang, 
                        word_timestamps=word_timestamps,
                        temperature=0.1,
                        condition_on_previous_text = False
                        )
                for _s in tmp_rslt:
                    xx = {start:_s.start, end:_s.end, text:_s.text}
                    xx.start = vad_time[0] + xx.start
                    xx.end   = vad_time[0] + xx.end
                    segments.append(xx)
        else:
            wf = wf.numpy()
            segments, info = self.model.transcribe(
                    wf, 
                    language=lang, 
                    word_timestamps=word_timestamps,
                    temperature=0.1,
                    condition_on_previous_text = False
                    )
            segments = [s for s in segments]
        #segments_text = ''.join([s.text for s in segments])

        #total_dur = info.duration 
        seg_num   = len(segments)
        
        data = []
        for i, _seg in enumerate(segments):
            seg = {'seg_idx':i+1, 'begin':_seg.start*1000, 'end':_seg.end*1000, 'transcript':_seg.text}
            if word_timestamps:
                words = []
                for j, _word in enumerate(_seg.words):
                    word = {'word_idx':j+1, 'begin':_word.start*1000, 'end':_word.end*1000, 'word':_word.word}
                    words.append(word)
                seg['words'] = words
            data.append(seg)
        
        result = {'total_dur':total_dur*1000, 'seg_num':seg_num, 'data':data}
        result_str = json.dumps(result, ensure_ascii=False)
        
        return result_str

def main(
        model_path,
        processor_path,
        multilingual = True,
        use_gpu = True,
        max_dec_length = 400,
        max_batch_size = 4,
        vad_level = 3,
        max_vad_length = 30, # max_asr_length: seconds
        lang = 'en',
        wav_file = None,
        wav_scp  = None,
        lang_wav_scp = None,
        dataset = None,
        ):

    # 
    if wav_scp is None and wav_file is None and lang_wav_scp is None:
        logging.error("please set one of --wav_scp and --wav_file and --lang_wav_scp")
        exit(1)
    # 
    asr_inst = WhisperAsrInstance(
            model_path,
            processor_path, 
            multilingual = multilingual,
            use_gpu = use_gpu,
            max_dec_length = max_dec_length,
            max_batch_size = max_batch_size,
            word_timestamps = False
            )

    #
    wav_list = []
    lang_list = []
    if wav_scp is not None:
        with open(wav_scp, 'r', encoding='utf8') as fp:
            wav_list = [ l.strip() for l in fp.readlines() ]
            lang_list = [ lang for _ in range(len(wav_list)) ]
    elif wav_file is not None:
        wav_list.append(wav_file)
        lang_list.append(lang)
    elif lang_wav_scp is not None:
        with open(lang_wav_scp, 'r', encoding='utf8') as fp:
            for l in fp.readlines():
                l, w = l.strip().split()
                wav_list.append(w)
                lang_list.append(l)

    from tqdm import tqdm
    for l, w in tqdm(zip(lang_list, wav_list)):
        trans, _, _ = asr_inst.transcribe_audio_file(w, lang = l)
        print('{}\t{}\t{}'.format(w, l, ' '.join(trans)))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
