subgen_version = '2026.02.10'

"""
Bazarr-focused Subgen runtime.

This build intentionally keeps only the Bazarr Whisper provider API surface:
- POST /asr
- POST /detect-language
- GET /status
- GET /

Legacy media-server webhooks and filesystem scanning flows were removed to reduce
maintenance and runtime overhead.
"""

import asyncio
import ast
import ctypes
import ctypes.util
import gc
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from threading import Event, Lock, Timer
from typing import Optional, Union

import faster_whisper
import ffmpeg
import numpy as np
import stable_whisper
import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from stable_whisper import Segment

from language_code import LanguageCode


def convert_to_bool(in_bool):
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')


def get_env_with_fallback(new_name: str, old_name: str, default_value=None, convert_func=None):
    value = os.getenv(new_name) or os.getenv(old_name)
    if value is None:
        value = default_value
    if convert_func and value is not None:
        return convert_func(value)
    return value


# ==========================================================================
# CONFIGURATION
# ==========================================================================

whisper_model = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cpu').lower()
if transcribe_device == 'gpu':
    transcribe_device = 'cuda'

webhookport = get_env_with_fallback('WEBHOOK_PORT', 'WEBHOOKPORT', 9000, int)
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
model_location = os.getenv('MODEL_PATH', './models')
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
reload_script_on_change = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')
detect_language_length = int(os.getenv('DETECT_LANGUAGE_LENGTH', 30))
detect_language_offset = int(os.getenv('DETECT_LANGUAGE_OFFSET', 0))
model_cleanup_delay = int(os.getenv('MODEL_CLEANUP_DELAY', 30))
asr_timeout = int(os.getenv('ASR_TIMEOUT', 18000))
force_detected_language_to = LanguageCode.from_string(os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''))

try:
    kwargs = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
except (ValueError, SyntaxError):
    kwargs = {}


# ==========================================================================
# APP + MODEL STATE
# ==========================================================================

app = FastAPI()
model = None
model_cleanup_timer = None
model_cleanup_lock = Lock()
model_init_lock = Lock()

in_docker = os.path.exists('/.dockerenv')
docker_status = 'Docker' if in_docker else 'Standalone'


# ==========================================================================
# LOGGING
# ==========================================================================


class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        patterns = [
            'Compression ratio threshold is not met',
            'Processing segment at',
            'Log probability threshold is',
            'Reset prompt',
            'Attempting to release',
            'released on ',
            'Attempting to acquire',
            'acquired on',
            'header parsing failed',
            'timescale not set',
            'misdetection possible',
            'srt was added',
        ]
        return not any(pattern in record.getMessage() for pattern in patterns)


level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(
    stream=sys.stderr,
    level=level,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger()
logger.setLevel(level)
for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('watchfiles').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)


# ==========================================================================
# TASK RESULT STORAGE (for blocking endpoints)
# ==========================================================================


class TaskResult:
    def __init__(self):
        self.result = None
        self.error = None
        self.done = Event()

    def set_result(self, result):
        self.result = result
        self.done.set()

    def set_error(self, error):
        self.error = error
        self.done.set()

    def wait(self, timeout=None):
        return self.done.wait(timeout)


# task_id -> TaskResult
task_results = {}
task_results_lock = Lock()


def get_or_create_task_result(task_id: str) -> TaskResult:
    with task_results_lock:
        task_result = task_results.get(task_id)
        if task_result is None:
            task_result = TaskResult()
            task_results[task_id] = task_result
        return task_result


def cleanup_task_result(task_id: Optional[str]) -> None:
    if not task_id:
        return
    with task_results_lock:
        task_result = task_results.get(task_id)
        if task_result and task_result.done.is_set():
            task_results.pop(task_id, None)


def cleanup_task_result_if_done(task_id: Optional[str], expected_result: Optional[TaskResult]) -> None:
    """
    Cleanup helper for worker threads.
    Removes only if the dict still points to the same completed result object.
    """
    if not task_id or expected_result is None or not expected_result.done.is_set():
        return

    with task_results_lock:
        current = task_results.get(task_id)
        if current is expected_result and current.done.is_set():
            task_results.pop(task_id, None)


# ==========================================================================
# HELPERS
# ==========================================================================


def generate_audio_hash(audio_content: bytes, task: Optional[str] = None, language: Optional[str] = None) -> str:
    """Stable dedup hash with optional request-shape parameters."""
    hasher = hashlib.sha256()
    hasher.update(audio_content)

    if task:
        hasher.update(b'|task:')
        hasher.update(task.encode('utf-8'))
    if language:
        hasher.update(b'|lang:')
        hasher.update(language.encode('utf-8'))

    return hasher.hexdigest()[:16]


class ProgressHandler:
    def __init__(self, filename):
        self.filename = filename
        self.start_time = time.time()
        self.last_print_time = 0
        self.interval = 5

    def __call__(self, seek, total):
        if docker_status != 'Docker' and not debug:
            return

        current_time = time.time()
        if self.last_print_time != 0 and (current_time - self.last_print_time) < self.interval:
            return

        self.last_print_time = current_time
        pct = int((seek / total) * 100) if total > 0 else 0
        elapsed = current_time - self.start_time
        speed = seek / elapsed if elapsed > 0 else 0
        eta = (total - seek) / speed if speed > 0 else 0

        def fmt_t(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f'{h}:{m:02d}:{s:02d}'
            return f'{m:02d}:{s:02d}'

        proc = len(task_queue.get_processing_tasks())
        queued = len(task_queue.get_queued_tasks())
        clean_name = (self.filename[:37] + '..') if len(self.filename) > 40 else self.filename

        logging.info(
            f"[ {clean_name:<40}] {pct:>3}% | "
            f"{int(seek):>5}/{int(total):<5}s "
            f"[{fmt_t(elapsed):>5}<{fmt_t(eta):>5}, {speed:>5.2f}s/s] | "
            f"Jobs: {proc} processing, {queued} queued"
        )


TIME_OFFSET = 5


def append_line(result):
    if not append:
        return

    segments = getattr(result, 'segments', None)
    if not segments:
        return

    last_segment = segments[-1]
    date_time_str = datetime.now().strftime('%d %b %Y - %H:%M:%S')
    appended_text = f'Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}'

    new_segment = Segment(
        start=last_segment.start + TIME_OFFSET,
        end=last_segment.end + TIME_OFFSET,
        text=appended_text,
        words=[],
        id=last_segment.id + 1,
    )
    segments.append(new_segment)


def extract_audio_segment_from_content(audio_content: bytes, start_time: int, duration: int) -> bytes:
    """Extract a short audio section using ffmpeg; falls back to original bytes."""
    try:
        out, _ = (
            ffmpeg
            .input('pipe:0', ss=start_time, t=duration)
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)
            .run(input=audio_content, capture_stdout=True, capture_stderr=True)
        )
        if out:
            return out
        logging.warning('FFmpeg returned empty chunk for language detection, using original audio bytes')
        return audio_content
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        logging.error(f'FFmpeg error extracting segment: {stderr}')
        return audio_content
    except Exception as e:
        logging.error(f'Error extracting audio segment: {e}')
        return audio_content


def render_asr_output(result, output: str, word_timestamps: bool) -> str:
    output = (output or 'srt').lower()

    if output in ('srt', 'vtt'):
        if output == 'vtt':
            try:
                return result.to_srt_vtt(filepath=None, vtt=True, word_level=word_timestamps)
            except TypeError:
                # Older stable-ts signature fallback
                return result.to_srt_vtt(filepath=None, word_level=word_timestamps)
        return result.to_srt_vtt(filepath=None, word_level=word_timestamps)

    if output == 'txt' and hasattr(result, 'to_txt'):
        return result.to_txt(filepath=None)

    if output == 'tsv' and hasattr(result, 'to_tsv'):
        return result.to_tsv(filepath=None)

    if output == 'json':
        if hasattr(result, 'to_dict'):
            return json.dumps(result.to_dict(), ensure_ascii=False)
        if hasattr(result, 'to_json'):
            return result.to_json()

    logging.warning(f"Unsupported output '{output}', falling back to srt")
    return result.to_srt_vtt(filepath=None, word_level=word_timestamps)


# ==========================================================================
# QUEUE + WORKERS
# ==========================================================================


class DeduplicatedQueue(queue.PriorityQueue):
    def __init__(self):
        super().__init__()
        self._queued = set()
        self._processing = set()
        self._lock = Lock()

    def put(self, item, block=True, timeout=None):
        with self._lock:
            task_id = item['path']
            if task_id in self._queued or task_id in self._processing:
                return False

            task_type = item.get('type', 'asr')
            priority = 0 if task_type == 'detect_language' else 1
            super().put((priority, time.time(), item), block, timeout)
            self._queued.add(task_id)
            return True

    def get(self, block=True, timeout=None):
        _, _, item = super().get(block, timeout)
        with self._lock:
            task_id = item['path']
            self._queued.discard(task_id)
            self._processing.add(task_id)
        return item

    def mark_done(self, item):
        with self._lock:
            self._processing.discard(item['path'])

    def is_idle(self):
        with self._lock:
            return self.empty() and len(self._processing) == 0

    def get_queued_tasks(self):
        with self._lock:
            return list(self._queued)

    def get_processing_tasks(self):
        with self._lock:
            return list(self._processing)


task_queue = DeduplicatedQueue()


def asr_task_worker(task_data: dict) -> None:
    task_id = task_data.get('path', 'unknown')
    result_container = task_data.get('result_container')

    try:
        start_model()

        task = task_data['task']
        language = task_data['language']
        video_file = task_data.get('video_file')
        file_content = task_data['audio_content']
        encode = task_data['encode']
        output = task_data.get('output', 'srt')
        word_timestamps = task_data.get('word_timestamps', False)

        args = {
            'progress_callback': ProgressHandler(os.path.basename(video_file) if video_file else task_id),
        }

        if encode:
            args['audio'] = file_content
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        if custom_regroup and custom_regroup.lower() != 'default':
            args['regroup'] = custom_regroup

        args.update(kwargs)

        result = model.transcribe(task=task, language=language, **args, verbose=None)
        append_line(result)

        if result_container:
            payload = render_asr_output(result, output, word_timestamps or word_level_highlight)
            result_container.set_result(payload)

    except Exception as e:
        logging.error(f'Error processing ASR (ID: {task_id}): {e}', exc_info=True)
        if result_container:
            result_container.set_error(str(e))


def detect_language_from_upload(task_data: dict) -> None:
    task_id = task_data.get('path', 'unknown')
    result_container = task_data.get('result_container')

    try:
        start_model()

        video_file = task_data.get('video_file')
        file_content = task_data['audio_content']
        encode = task_data['encode']
        detect_lang_length = max(1, int(task_data['detect_lang_length']))
        detect_lang_offset = max(0, int(task_data['detect_lang_offset']))

        logging.info(
            f"Detecting language for '{video_file}' ({detect_lang_length}s at +{detect_lang_offset}s) - ID: {task_id}"
            if video_file
            else f'Detecting language ({detect_lang_length}s at +{detect_lang_offset}s) - ID: {task_id}'
        )

        args = {}
        if encode:
            args['audio'] = extract_audio_segment_from_content(
                file_content,
                detect_lang_offset,
                detect_lang_length,
            )
            args['input_sr'] = 16000
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        args.update(kwargs)
        detected_language = LanguageCode.from_name(model.transcribe(**args, verbose=None).language)
        payload = {
            'detected_language': detected_language.to_name(),
            'language_code': detected_language.to_iso_639_1(),
        }

        if result_container:
            result_container.set_result(payload)

    except Exception as e:
        logging.error(f'Error detecting language (ID: {task_id}): {e}', exc_info=True)
        if result_container:
            result_container.set_error(str(e))


def transcription_worker():
    while True:
        task = None
        try:
            task = task_queue.get(block=True, timeout=1)
            task_type = task.get('type', 'asr')
            task_name = task.get('video_file') or task.get('path', 'unknown')
            display_name = os.path.basename(str(task_name))

            proc_count = len(task_queue.get_processing_tasks())
            queue_count = len(task_queue.get_queued_tasks())
            logging.info(
                f'WORKER START : [{task_type.upper():<15}] {display_name:^40} '
                f'| Jobs: {proc_count} processing, {queue_count} queued'
            )

            start_time = time.time()
            if task_type == 'asr':
                asr_task_worker(task)
            elif task_type == 'detect_language':
                detect_language_from_upload(task)
            else:
                raise ValueError(f'Unsupported task type: {task_type}')

            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            remaining_queued = len(task_queue.get_queued_tasks())
            logging.info(
                f'WORKER FINISH: [{task_type.upper():<15}] {display_name:^40} '
                f'in {m}m {s}s | Remaining: {remaining_queued} queued'
            )

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f'Error processing task: {e}', exc_info=True)
        finally:
            if task:
                task_queue.task_done()
                task_queue.mark_done(task)
                delete_model()
                cleanup_task_result_if_done(task.get('path'), task.get('result_container'))


for _ in range(concurrent_transcriptions):
    threading.Thread(target=transcription_worker, daemon=True).start()


# ==========================================================================
# MODEL LIFECYCLE
# ==========================================================================


def start_model():
    global model

    if model is not None:
        return

    with model_init_lock:
        if model is None:
            logging.debug('Model was purged, loading model')
            model = stable_whisper.load_faster_whisper(
                whisper_model,
                download_root=model_location,
                device=transcribe_device,
                cpu_threads=whisper_threads,
                num_workers=concurrent_transcriptions,
                compute_type=compute_type,
            )


def schedule_model_cleanup():
    global model_cleanup_timer

    with model_cleanup_lock:
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()

        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()


def perform_model_cleanup():
    global model, model_cleanup_timer

    with model_cleanup_lock:
        if clear_vram_on_complete and task_queue.is_idle():
            if model:
                try:
                    model.model.unload_model()
                    del model
                    model = None
                    logging.info('Model unloaded from memory')
                except Exception as e:
                    logging.error(f'Error unloading model: {e}')

            if transcribe_device == 'cuda' and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f'Error clearing CUDA cache: {e}')

        if os.name != 'nt':
            gc.collect()
            libc_name = ctypes.util.find_library('c')
            if libc_name:
                ctypes.CDLL(libc_name).malloc_trim(0)

        model_cleanup_timer = None


def delete_model():
    if not clear_vram_on_complete:
        return

    if task_queue.is_idle():
        schedule_model_cleanup()


# ==========================================================================
# API
# ==========================================================================


@app.get('/asr')
@app.get('/detect-language')
def handle_get_request():
    return {
        'message': 'You accessed this endpoint incorrectly via GET. Configure Bazarr Whisper provider to send POST requests.'
    }


@app.get('/')
def webui():
    return {'message': 'Bazarr-only Subgen build. Configure via environment variables.'}


@app.get('/status')
def status():
    return {
        'version': (
            f'Subgen {subgen_version}, '
            f'stable-ts {stable_whisper.__version__}, '
            f'faster-whisper {faster_whisper.__version__} ({docker_status})'
        )
    }


@app.post('/asr')
async def asr(
    task: Union[str, None] = Query(default='transcribe', enum=['transcribe', 'translate']),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description='Encode audio first through ffmpeg'),
    output: Union[str, None] = Query(default='srt', enum=['txt', 'vtt', 'srt', 'tsv', 'json']),
    word_timestamps: bool = Query(default=False, description='Word-level timestamps'),
):
    _ = initial_prompt  # reserved for compatibility
    task_id = None

    try:
        logging.info(
            f"ASR {task.capitalize()} received for file '{video_file}'"
            if video_file
            else f'ASR {task.capitalize()} received'
        )

        file_content = await audio_file.read()
        if not file_content:
            return {'status': 'error', 'message': 'Audio file is empty'}

        hash_task = f'{task}:{output}:{int(bool(word_timestamps))}:{int(bool(encode))}'
        audio_hash = generate_audio_hash(file_content, hash_task, language or '')
        task_id = f'asr-{audio_hash}'

        final_language = language
        if force_detected_language_to:
            final_language = force_detected_language_to.to_iso_639_1()

        task_result = get_or_create_task_result(task_id)
        asr_task_data = {
            'path': task_id,
            'type': 'asr',
            'task': task,
            'language': final_language,
            'video_file': video_file,
            'audio_content': file_content,
            'encode': encode,
            'output': output,
            'word_timestamps': word_timestamps,
            'result_container': task_result,
        }

        if task_queue.put(asr_task_data):
            logging.info(f'ASR task {task_id} queued')
        else:
            logging.info(f'ASR task {task_id} already queued/processing - waiting for result')

        completed = await asyncio.to_thread(task_result.wait, asr_timeout)
        if not completed:
            logging.error(f'ASR task {task_id} timed out')
            return {
                'status': 'timeout',
                'task_id': task_id,
                'message': f'ASR processing timed out after {asr_timeout} seconds',
            }

        if task_result.error:
            logging.error(f'ASR task {task_id} failed: {task_result.error}')
            return {
                'status': 'error',
                'task_id': task_id,
                'message': f'ASR processing failed: {task_result.error}',
            }

        media_type = 'application/json' if (output or '').lower() == 'json' else 'text/plain'
        return StreamingResponse(
            iter([task_result.result]),
            media_type=media_type,
            headers={'Source': f'{task.capitalize()}d using stable-ts from Subgen!'},
        )

    except Exception as e:
        logging.error(f'Error in ASR endpoint: {e}', exc_info=True)
        return {'status': 'error', 'message': f'Error: {e}'}
    finally:
        await audio_file.close()
        cleanup_task_result(task_id)


@app.post('/detect-language')
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True),
    video_file: Union[str, None] = Query(default=None),
    detect_lang_length: int = Query(default=detect_language_length),
    detect_lang_offset: int = Query(default=detect_language_offset),
):
    task_id = None

    if force_detected_language_to:
        await audio_file.close()
        return {
            'detected_language': force_detected_language_to.to_name(),
            'language_code': force_detected_language_to.to_iso_639_1(),
        }

    try:
        file_content = await audio_file.read()
        if not file_content:
            return {'detected_language': 'Unknown', 'language_code': 'und', 'status': 'error'}

        detect_lang_length = max(1, int(detect_lang_length))
        detect_lang_offset = max(0, int(detect_lang_offset))

        hash_task = f'detect:{int(bool(encode))}:{detect_lang_offset}:{detect_lang_length}'
        audio_hash = generate_audio_hash(file_content, hash_task, '')
        task_id = f'detect-{audio_hash}'

        task_result = get_or_create_task_result(task_id)
        detect_task_data = {
            'path': task_id,
            'type': 'detect_language',
            'video_file': video_file,
            'audio_content': file_content,
            'encode': encode,
            'detect_lang_length': detect_lang_length,
            'detect_lang_offset': detect_lang_offset,
            'result_container': task_result,
        }

        if task_queue.put(detect_task_data):
            logging.info(
                f"Detect-language task {task_id} queued"
                + (f" for '{video_file}'" if video_file else '')
            )
        else:
            logging.info(f'Detect-language task {task_id} already queued/processing - waiting for result')

        completed = await asyncio.to_thread(task_result.wait, asr_timeout)
        if not completed:
            logging.error(f'Detect-language task {task_id} timed out')
            return {
                'detected_language': 'Unknown',
                'language_code': 'und',
                'status': 'timeout',
                'message': f'Detect-language processing timed out after {asr_timeout} seconds',
            }

        if task_result.error:
            logging.error(f'Detect-language task {task_id} failed: {task_result.error}')
            return {
                'detected_language': 'Unknown',
                'language_code': 'und',
                'status': 'error',
                'message': f'Detect-language processing failed: {task_result.error}',
            }

        return task_result.result

    except Exception as e:
        logging.error(f'Error in detect-language endpoint: {e}', exc_info=True)
        return {'detected_language': 'Unknown', 'language_code': 'und', 'status': 'error'}
    finally:
        await audio_file.close()
        cleanup_task_result(task_id)


if __name__ == '__main__':
    import uvicorn

    logging.info(f'Subgen v{subgen_version}')
    logging.info(f'Threads: {whisper_threads}, Concurrent transcriptions: {concurrent_transcriptions}')
    logging.info(f'Transcribe device: {transcribe_device}, Model: {whisper_model}')

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    uvicorn.run(
        '__main__:app',
        host='0.0.0.0',
        port=int(webhookport),
        reload=reload_script_on_change,
        use_colors=True,
    )
