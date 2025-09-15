import os
import time
import threading
import queue
import platform
from typing import Optional, List, Callable
import numpy as np
import soundfile as sf
import torch
import sounddevice as sd
from huggingface_hub import hf_hub_download
import json
# 配置

LANGUAGE_MAP = {
    'english': {'full_path': 'en/en_GB/cori/high/en_GB-cori-high.onnx'},
    # en/en_GB/cori/high/en_GB-cori-high.onnx
    # en/en_US/ryan/high/en_US-ryan-high.onnx
    'german': {'full_path': 'de/de_DE/thorsten/high/de_DE-thorsten-high.onnx'},  # 高品质，男声，沉稳
    'french': {'full_path': 'fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx'},  # medium，女声，语调自然
    'spanish': {'full_path': 'es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx'},  # medium，男声，衔接流畅
    'italian': {'full_path': 'it/it_IT/riccardo_fasol/medium/it_IT-riccardo_fasol-medium.onnx'},  # medium，男声
    'portuguese': {'full_path': 'pt/pt_BR/edresson/medium/pt_BR-edresson-medium.onnx'},  # medium，男声
    'dutch': {'full_path': 'nl/nl_NL/mls_5809/medium/nl_NL-mls_5809-medium.onnx'},  # medium，混合性别
    'russian': {'full_path': 'ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx'},  # medium，女声
    'polish': {'full_path': 'pl/pl_PL/mc_speech/medium/pl_PL-mc_speech-medium.onnx'},  # medium
    'swedish': {'full_path': 'sv/sv_SE/jonas/medium/sv_SE-jonas-medium.onnx'},  # medium
    'finnish': {'full_path': 'fi/fi_FI/harri_tapani_ylilammi/medium/fi_FI-harri_tapani_ylilammi-medium.onnx'},  # medium
    'norwegian': {'full_path': 'no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx'},  # medium
    'danish': {'full_path': 'da/da_DK/talesyntese/medium/da_DK-talesyntese-medium.onnx'},  # medium
    'czech': {'full_path': 'cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx'},  # medium
    'hungarian': {'full_path': 'hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx'},  # medium
    'romanian': {'full_path': 'ro/ro_RO/mihai/medium/ro_RO-mihai-medium.onnx'},  # medium
    'serbian': {'full_path': 'sr/sr_RS/serbski_institut/medium/sr_RS-serbski_institut-medium.onnx'},  # medium
    'ukrainian': {'full_path': 'uk/uk_UA/lada/medium/uk_UA-lada-medium.onnx'},  # medium
    'greek': {'full_path': 'el/el_GR/rapunzelina/medium/el_GR-rapunzelina-medium.onnx'},  # medium
    'catalan': {'full_path': 'ca/ca_ES/upc_ona/medium/ca_ES-upc_ona-medium.onnx'},  # medium
    'esperanto': {'full_path': 'eo/eo/marytts/medium/eo-marytts-medium.onnx'},  # medium
    'icelandic': {'full_path': 'is/is_IS/bui/medium/is_IS-bui-medium.onnx'},  # medium
    'luxembourgish': {'full_path': 'lb/lb_LU/marylux/medium/lb_LU-marylux-medium.onnx'},  # medium
    'swahili': {'full_path': 'sw/sw_CD/rehema/medium/sw_CD-rehema-medium.onnx'},  # medium
    'welsh': {'full_path': 'cy/cy_GB/gwryw_gogleddol/medium/cy_GB-gwryw_gogleddol-medium.onnx'},  # medium
    'arabic': {'full_path': 'ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx'},  # medium
    'kazakh': {'full_path': 'kk/kk_KZ/isekeev/medium/kk_KZ-isekeev-medium.onnx'},  # medium
    'kyrgyz': {'full_path': 'ky/ky_KG/ermektursunov/medium/ky_KG-ermektursunov-medium.onnx'},  # medium
    'nepali': {'full_path': 'ne/ne_NP/google/medium/ne_NP-google-medium.onnx'},  # medium
    'tajik': {'full_path': 'tg/tg_TJ/gulnur/medium/tg_TJ-gulnur-medium.onnx'},  # medium
    'georgian': {'full_path': 'ka/ka_GE/natia/medium/ka_GE-natia-medium.onnx'},  # medium
    'vietnamese': {'full_path': 'vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx'},  # medium
}

class AudioPlayer:
    def __init__(self):
        """初始化播放器"""
        self.audio_file = None
        self.current_position = 0
        self.sample_rate = None
        self.audio_data = None
        self.stream = None
        self.__status = 'finished' 
        # playing: 音频正在播放
        # paused: 音频被暂停，但还没结束
        # finished: 音频播放完毕或未开始播放，此时resume和pause无效
        self.lock = threading.Lock()

    def set_audio_file(self, filepath: str):
        """设置音频文件"""
        self.audio_file = filepath
        self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
        self.current_position = 0
        print(f"Audio file set: {filepath}")
        self.__status = 'finished'

    def play(self, finish_callback=None):
        """从头开始播放音频"""
        if self.audio_data is None or self.sample_rate is None:
            print("No audio file set. Use set_audio_file() first.")
            return

        def callback(outdata, frames, time, status):
            with self.lock:
                if self.__status != 'playing':
                    outdata.fill(0)
                    return
                start = self.current_position
                end = start + frames
                if end >= len(self.audio_data): # 播放完毕
                    end = len(self.audio_data)
                    outdata[:end - start] = self.audio_data[start:end]
                    outdata[end - start:] = 0
                    self.__status = 'finished'  
                    self.current_position = 0  # 重置位置
                    print("Audio playback finished.")
                    # ==== CASCADE PATCH: robust finish_callback guard ====
                    if callable(finish_callback):
                        try:
                            finish_callback()
                        except Exception as e:
                            print(f"finish_callback error: {e}")
                    raise sd.CallbackStop
                outdata[:] = self.audio_data[start:end]
                self.current_position = end

        # Ensure audio data is 2D (mono audio needs to be reshaped)
        if len(self.audio_data.shape) == 1:
            self.audio_data = self.audio_data[:, np.newaxis]

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.audio_data.shape[1],
            callback=callback
        )
        self.__status = 'playing'
        self.stream.start()
        print("Audio playback started.")

    def pause(self):
        """暂停播放"""
        if self.__status != 'finished':
            with self.lock:
                self.__status = 'paused'
            print("Audio playback paused.")

    def resume(self):
        """继续播放"""
        if self.__status != 'finished':
            with self.lock:
                self.__status = 'playing'
            print("Audio playback resumed.")

    def stop(self):
        """停止播放"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.current_position = 0
        self.__status = 'finished'
        print("Audio playback stopped.")
    
    def status(self):
        return self.__status

class AudioGenerator:
    def __init__(self, model_cache_dir, length_scale=1.1):
        """初始化，自动检测 GPU 可用性"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        self.model_cache_dir = model_cache_dir
        self.model_path = None
        self.piper_voice = None
        self.audio_dir = "generated_audios"
        os.makedirs(self.audio_dir, exist_ok=True)
        self.sample_rate = 22050  # 默认值

        self.length_scale = length_scale
        self.silence_duration = 0.3
        self.audio_player = AudioPlayer()  # 添加播放器实例
        self.stream_player_thread = None # 用于流式播放的线程
        self.stop_flag = threading.Event()  # 用于安全终止线程
        
    def download_model(self, language: str = 'english'):
        """下载模型"""
        os.makedirs(self.model_cache_dir, exist_ok=True)
        full_path = LANGUAGE_MAP.get(language.lower(), LANGUAGE_MAP['english'])['full_path']
        repo_id = "rhasspy/piper-voices"
        
        try:
            # 下载模型文件 (.onnx)
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=full_path,
                local_dir=self.model_cache_dir,
                local_dir_use_symlinks=False,
                revision="main"
            )
            
            # 下载配置文件 (.onnx.json)
            config_file = hf_hub_download(
                repo_id=repo_id,
                filename=full_path + '.json',
                local_dir=self.model_cache_dir,
                local_dir_use_symlinks=False,
                revision="main"
            )
            
            # 验证文件存在
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            print(f"Downloaded model: {model_file}")
            print(f"Downloaded config: {config_file}")
            return model_file, config_file
            
        except Exception as e:
            print(f"Failed to download model for {language}: {e}")
            print("Please manually download the model and config files from:")
            print(f"  Model: https://huggingface.co/{repo_id}/resolve/main/{full_path}")
            print(f"  Config: https://huggingface.co/{repo_id}/resolve/main/{full_path}.json")
            print(f"Place them in: {os.path.join(self.model_cache_dir, os.path.dirname(full_path))}")
            raise
    
    def init_tts_pipeline(self, language: str = 'english'):
        """初始化TTS，预热模型，指定语言"""
        model_path, config_path = self.download_model(language)
        from piper import PiperVoice
        import json
        
        # 修改config json的length_scale
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['inference']['length_scale'] = self.length_scale
        temp_config_path = config_path + '.temp.json'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        
        self.piper_voice = PiperVoice.load(model_path, config_path=temp_config_path, use_cuda=(self.device == "cuda"))
        self.sample_rate = self.piper_voice.config.sample_rate
        # 验证 CUDA
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' not in providers and self.device == "cuda":
            print("Warning: CUDAExecutionProvider not available, falling back to CPU")
        
        # 预热模型
        audio = self.piper_voice.synthesize("Test")
        print(f"Synthesize return type: {type(audio)}")
        self.piper_voice.synthesize("Test")
        print(f"Piper TTS pipeline initialized successfully for {language} (sample_rate: {self.sample_rate})")
        return True
    
    def generate_audio_chunks(self, text: str, language: str = 'english') -> List[np.ndarray]:
        """生成音频chunks，输入整段文本由模型自动分割，后处理插入静音"""
        if self.piper_voice is None:
            self.init_tts_pipeline(language)
        
        audio_chunks = []
        silence_samples = int(self.sample_rate * self.silence_duration)  # 500ms 静音
        silence_chunk = np.zeros(silence_samples, dtype=np.float32)  # 生成静音数组
        # ==== CASCADE PATCH START: precompute synthesize results to avoid repeated generation ====
        synth_results = list(self.piper_voice.synthesize(text))
        total = len(synth_results)
        for idx, audio_chunk in enumerate(synth_results, 1):
            print(f"Generating audio for chunk {idx}/{total}")
            audio_chunks.append(
                np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            if idx < total:
                audio_chunks.append(silence_chunk)
        # ==== CASCADE PATCH END ====
        
        if not audio_chunks:
            print("No valid audio chunks generated")
            return []
        
        return audio_chunks
    
    def save_complete_audio(self, audio_chunks: List[np.ndarray], filename: str, sample_rate: int = None) -> str:
        """保存完整音频"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        complete_audio = np.concatenate(audio_chunks)
        filepath = os.path.join(self.audio_dir, f"{filename}.wav")
        sf.write(filepath, complete_audio, sample_rate)
        return filepath

    def generate_and_play_audio(self, text: str, save_filename: Optional[str] = None, play_audio: bool = True,
                                first_chunk_time_callback: Optional[Callable[[], None]] = None, language: str = 'english',
                                callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
        """音频生成pipeline，支持无缝流式播放，后处理插入静音"""
        if self.piper_voice is None:
            if not self.init_tts_pipeline(language):
                return None

        audio_queue = queue.Queue(maxsize=10)
        audio_chunks = []
        buffer = np.array([], dtype=np.float32)
        chunk_count = 0
        silence_samples = int(self.sample_rate * self.silence_duration)
        silence_chunk = np.zeros(silence_samples, dtype=np.float32)

        def producer():
            """生产者线程：流式生成 audio chunks，插入静音"""
            nonlocal chunk_count
            try:
                synth_results = list(self.piper_voice.synthesize(text))
                total = len(synth_results)
                for chunk_idx, audio_chunk in enumerate(synth_results, 1):
                    if self.stop_flag.is_set():
                        break
                    chunk_count = chunk_idx
                    audio_queue.put(
                        np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    if chunk_idx < total:
                        audio_queue.put(silence_chunk)
                        chunk_count += 1
            finally:
                # Signal termination
                audio_queue.put(None)

        def consumer():
            """消费者线程：启动音频流"""
            try:
                stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=stream_callback,
                    blocksize=1024
                )
                if callable(first_chunk_time_callback):
                    try:
                        first_chunk_time_callback()
                    except Exception as e:
                        print(f"first_chunk_time_callback error: {e}")
                with stream:
                    timeout = 30
                    start_time = time.time()
                    while (producer_thread.is_alive() or audio_queue.qsize() > 0 or len(buffer) > 0) \
                        and not self.stop_flag.is_set():
                        if time.time() - start_time > timeout:
                            print("Consumer timeout, stopping stream")
                            break
                        time.sleep(0.01)
            except Exception as e:
                print(f"Stream error: {e}")

        def stream_callback(outdata, frames, time_info, status):
            """sounddevice 流回调：从缓冲区或队列获取音频"""
            nonlocal buffer
            if status:
                print(f"Stream status: {status}")
            
            while len(buffer) < frames:
                try:
                    audio_data = audio_queue.get_nowait()
                    if audio_data is None:
                        if len(buffer) == 0:
                            outdata[:] = 0
                            raise sd.CallbackStop
                        break
                    buffer = np.concatenate((buffer, audio_data))
                    audio_chunks.append(audio_data)
                except queue.Empty:
                    if not producer_thread.is_alive() and audio_queue.qsize() == 0:
                        if len(buffer) == 0:
                            outdata[:] = 0
                            raise sd.CallbackStop
                    break
            
            if len(buffer) >= frames:
                outdata[:, 0] = buffer[:frames]
                buffer = buffer[frames:]
            else:
                outdata[:len(buffer), 0] = buffer
                outdata[len(buffer):, 0] = 0
                buffer = np.array([], dtype=np.float32)

        producer_thread = threading.Thread(target=producer, daemon=True)
        consumer_thread = threading.Thread(target=consumer, daemon=True)

        producer_thread.start()
        consumer_thread.start()

        self.stream_player_thread = consumer_thread

        producer_thread.join()
        consumer_thread.join()

        saved_path = None
        if save_filename and audio_chunks:
            saved_path = self.save_complete_audio(audio_chunks, save_filename)
            # ==== CASCADE PATCH: single callback invocation ====
            if callable(callback):
                try:
                    callback(saved_path)
                except Exception as e:
                    print(f"callback error: {e}")

        return saved_path

    def async_generate_and_play_audio(self, text: str, save_filename: Optional[str] = None, play_audio: bool = True,
                                      first_chunk_time_callback: Optional[Callable[[], None]] = None,
                                      callback: Optional[Callable[[str], None]] = None, language: str = 'english') -> threading.Thread:
        """异步音频生成"""
        def audio_task():
            # ==== CASCADE PATCH: callback is handled inside generate_and_play_audio to avoid double-calling ====
            self.generate_and_play_audio(text, save_filename, play_audio, first_chunk_time_callback, language, callback)

        thread = threading.Thread(target=audio_task, daemon=True)
        thread.start()
        return thread

    

    def stop_stream_player(self):
        """终止流式音频的播放，但不阻碍生成。"""
        # 设置停止标志
        self.stop_flag.set()

        # 等待stream player线程结束
        if self.stream_player_thread and self.stream_player_thread.is_alive():
            self.stream_player_thread.join()
        self.stream_player_thread = None
        print("All sub-threads stopped.")
        self.stop_flag.clear()
