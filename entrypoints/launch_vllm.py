import argparse
import datetime
import json
import os
import random
import socket
import subprocess
import sys
from typing import Dict, List, Tuple
import uuid

import pynvml


try:
    import fcntl
    def lock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except ImportError:
    import msvcrt
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


class FileLock:

    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'a+') # 'a+'确保文件存在
        self.file.seek(0)
        lock_file(self.file)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            unlock_file(self.file)
            self.file.close()
            self.file = None


class LogRedirector:

    def __init__(self, log_file_handle):
        self.log_file_handle = log_file_handle
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.log_file_handle
        sys.stderr = self.log_file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


def get_local_ip() -> str:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        ip = '127.0.0.1'
        print(f'[Launcher] Cannot get local ip, error msg: {e}')
    finally:
        if s:
            s.close()
    return ip


def find_free_port(min_port=30001, max_port=65535) -> int:
    if min_port > max_port:
        raise ValueError('min_port must be less than or equal to max_port')

    ports_to_try = list(range(min_port, max_port + 1))
    random.shuffle(ports_to_try)

    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    raise IOError(f"No available port found within the range [{min_port}, {max_port}]")


def find_free_gpus(num_gpus: int) -> List[int]:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_gpus = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                free_gpus.append(i)
        except pynvml.NVMLError as e:
            print(f'[Launcher] Could not query processes for GPU {i}: {e}')
    
    pynvml.nvmlShutdown()

    if len(free_gpus) < num_gpus:
        raise ValueError(
            f'Not enough free GPUs. Found {len(free_gpus)}, but need {num_gpus}. '
            f'Available GPUs: {free_gpus}'
        )

    return free_gpus[:num_gpus]


def get_current_time() -> str:
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y/%m/%d %H:%M:%S")
    return formatted_time


def get_launcher(args) -> List[str]:
    if args.port is None:
        args.port = find_free_port()
        print(f'[Launcher] No port specified. Found and using free port: {args.port}')

    vllm_args = [
        'vllm.entrypoints.openai.api_server', 
        '--model', args.model,
        '--tensor-parallel-size', str(args.tp),
        '--max-model-len', str(args.max_model_len),
        '--max-num-seqs', str(args.max_num_seqs),
        '--dtype', 'auto',
        '--host', '0.0.0.0',
        '--port', str(args.port),
        '--trust-remote-code',
        '--gpu-memory-utilization', '0.95',
    ]

    if args.served_model_name:
        vllm_args.extend(['--served-model-name', args.served_model_name])

    if 'glm-4.5v' in args.model.lower():
        vllm_args.extend([
            '--tool-call-parser', 'glm45',  
            '--reasoning-parser', 'glm45',  
            '--enable-auto-tool-choice',
            '--allowed-local-media-path', '/',
            '--media-io-kwargs', '{"video": {"num_frames": -1}}',
        ])

    if 'qwen3-vl' in args.model.lower():
        vllm_args.extend([
            '--mm-encoder-tp-mode', 'data',
            '--enable-expert-parallel',
            '--quantization', 'fp8',
            '--async-scheduling',
            '--limit-mm-per-prompt.video', '0',
            '--distributed-executor-backend', 'mp',
            '--mm-processor-cache-gb', '50',
        ])
        if 'thinking' in args.model.lower():
            vllm_args.extend(['--reasoning-parser', 'qwen3'])

    print(f'[Launcher] {" ".join(vllm_args)}')

    launcher = [sys.executable, '-m'] + vllm_args
    return launcher


def prepare_envs(num_gpus: int) -> Tuple[Dict[str, str], List[int]]:
    env = os.environ.copy()

    # set visible gpus
    try:
        selected_gpus = find_free_gpus(num_gpus)
        print(f'[Launcher] Found {len(selected_gpus)} free GPUs: {selected_gpus}')
    except Exception as e:
        print(f'[Launcher] Error finding free GPUs: {e}')
        raise
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
    return env, selected_gpus


def setup_record(
    serve_file: str, 
    lock_file: str, 
    args: argparse.Namespace, 
    uid: str, 
    pid: str,
    gpus: List[int],
) -> None:
    with FileLock(lock_file):
        if os.path.exists(serve_file):
            with open(serve_file, 'r', encoding='utf-8') as f:
                serve_dict = json.load(f)
        else:
            serve_dict = {}
        
        model_key = args.model if args.served_model_name is None \
            else args.served_model_name

        if model_key not in serve_dict:
            serve_dict[model_key] = {}

        serve_dict[model_key][uid] = {
            'pid': pid,
            'ip': get_local_ip(),
            'port': str(args.port),
            'tp': str(args.tp),
            'gpus': gpus,
            'max_model_len': str(args.max_model_len),
            'max_num_seqs': str(args.max_num_seqs),
            'create_time': get_current_time(),
        }

        with open(serve_file, 'w', encoding='utf-8') as f:
            json.dump(serve_dict, f, indent=2, ensure_ascii=False)


def cleanup_record(
    serve_file: str, 
    lock_file: str, 
    model: str, 
    uid: str
) -> None:
    with FileLock(lock_file):
        try:
            with open(serve_file, 'r+', encoding='utf-8') as f:
                serve_dict = json.load(f)

                if model in serve_dict and uid in serve_dict[model]:
                    del serve_dict[model][uid]
                    if not serve_dict[model]: 
                        del serve_dict[model]

                    f.seek(0)
                    f.truncate()
                    json.dump(serve_dict, f, indent=2, ensure_ascii=False)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f'[Launcher] Cleanup skipped, file might be missing, empty or entry not found: {e}')
            pass


def launch_vllm_server(args: argparse.Namespace):
    uid = str(uuid.uuid4())

    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'serve_{uid}.log')

    serve_file = os.path.join(log_dir, 'serve.json')
    lock_file = serve_file + '.lock'

    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        with LogRedirector(f):
            print(f'--- Launcher Log for Service UID: {uid} ---')
            launcher = get_launcher(args)
            envs, selected_gpus = prepare_envs(args.tp)

            process = None
            model_key = args.model if args.served_model_name is None \
                else args.served_model_name
            try:
                # 1. Launch the subprocess
                process = subprocess.Popen(
                    launcher,
                    stdout=f,
                    stderr=f,
                    env=envs,
                )
                pid = process.pid
                print(f'[Launcher] vLLM server (PID: {pid}) for model "{model_key}" started.')

                # 2. Register the service
                setup_record(serve_file, lock_file, args, uid, str(pid), selected_gpus)

                # 3. Wait for the process to complete
                process.wait()

            finally:
                if process:
                    print(f'[Launcher] vLLM server (PID: {pid}) has terminated. Cleaning up record.')
                    cleanup_record(serve_file, lock_file, model_key, uid)
                else:
                    print(f'[Launcher] Process failed to launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('vLLM Model Launcher')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--served_model_name', type=str, default=None)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=65536)
    parser.add_argument('--max_num_seqs', type=int, default=16)
    args = parser.parse_args()

    launch_vllm_server(args)
