from __future__ import annotations

import glob
import os
import time
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download as hf_snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from minisgl.utils.logger import init_logger
from modelscope import snapshot_download as ms_snapshot_download

logger = init_logger(__name__)
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict

def _log_cuda_env(tag: str, device: torch.device) -> None:
    # 只在 CUDA 情况下打印
    if not torch.cuda.is_available():
        logger.info(f"[{tag}] torch.cuda not available")
        return
    try:
        dev_count = torch.cuda.device_count()
        cur = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cur) if dev_count > 0 else None
        logger.info(
            f"[{tag}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"torch_cuda_device_count={dev_count} current_device={cur} "
            f"requested_device={device} "
            f"torch_version={torch.__version__} cuda_version={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}"
        )
        if props is not None:
            logger.info(
                f"[{tag}] current_device_name={props.name} "
                f"capability={props.major}.{props.minor} "
                f"total_mem_GiB={props.total_memory/1024**3:.2f}"
            )
        # 打印 memory snapshot（轻量）
        logger.info(
            f"[{tag}] mem_alloc_GiB={torch.cuda.memory_allocated(cur)/1024**3:.2f} "
            f"mem_reserved_GiB={torch.cuda.memory_reserved(cur)/1024**3:.2f}"
        )
        # 打印默认流句柄（帮助看是否变化）
        ds = torch.cuda.default_stream(cur)
        logger.info(f"[{tag}] default_stream={ds}")
    except Exception as e:
        logger.info(f"[{tag}] failed to log cuda env: {e}")


def load_weight(
    model_path: str,
    device: torch.device,
    source: str = "huggingface",
    use_mma: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Unified model weight loading function.
    :param source: "huggingface" or "modelscope"
    :param use_mma: Use MMA for accelerated CPU-GPU data transfer
    """
    if os.path.isdir(model_path):
        model_folder = model_path
    else:
        try:
            if source == "huggingface":
                model_folder = hf_snapshot_download(
                    model_path,
                    allow_patterns=["*.safetensors"],
                    tqdm_class=DisabledTqdm,
                )
            elif source == "modelscope":
                model_folder = ms_snapshot_download(
                    model_path,
                    allow_file_pattern=["*.safetensors"],
                )
            else:
                raise ValueError(
                    f"Unknown source: {source}, expected 'huggingface' or 'modelscope'"
                )
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid {source} model ID"
            )

    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    # Log device transfer info
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    src_device = next(iter(state_dict.values())).device if state_dict else "N/A"
    logger.info(
        f"Transferring {len(state_dict)} tensors ({total_bytes / 1e9:.2f} GB) "
        f"from {src_device} to {device} (use_mma={use_mma})"
    )

    start_time = time.perf_counter()
    '''
    if use_mma:
        import mma

        if device.type == "cuda":
        
            torch.cuda.set_device(device)

    
        mma.init()
        new_state_dict = {}
        for k, v in state_dict.items():
            v = v.contiguous()
            gpu_tensor = torch.empty_like(v, device=device)
            if v.dtype == torch.bfloat16:
                cpu_data = v.view(torch.int16).numpy()
                mma.memcpy(gpu_tensor.view(torch.int16), cpu_data)
            else:
                cpu_data = v.numpy()
                mma.memcpy(gpu_tensor, cpu_data)
            torch.cuda.synchronize(device)
            new_state_dict[k] = gpu_tensor
        state_dict = new_state_dict

    else:
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    '''

    if use_mma:
        import mma

        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Pre-allocate GPU buffers with PyTorch first, then init MMA and memcpy.
        # This avoids "invalid resource handle" if MMA init changes CUDA/stream state.
        cpu_state_dict: Dict[str, torch.Tensor] = {}
        new_state_dict: Dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            v = v.contiguous()
            cpu_state_dict[k] = v
            new_state_dict[k] = torch.empty_like(v, device=device)
        
        prev_dev = torch.cuda.current_device() 
        #start_time = time.perf_counter()

        _log_cuda_env("before_mma_init", device)
        #start_time = time.perf_counter()
        mma.init()
        start_time = time.perf_counter()
        
        _log_cuda_env("after_mma_init", device)
        
        torch.cuda.set_device(prev_dev)
        
        #Synchronous
        for k, v in cpu_state_dict.items():
            gpu_tensor = new_state_dict[k]
            if v.dtype == torch.bfloat16:
                cpu_data = v.view(torch.int16).numpy()
                mma.memcpy(gpu_tensor.view(torch.int16), cpu_data)
            else:
                cpu_data = v.numpy()
                mma.memcpy(gpu_tensor, cpu_data)

            torch.cuda.set_device(prev_dev)
            #torch.cuda.synchronize(prev_dev)
            #torch.cuda.set_stream(torch.cuda.default_stream(prev_dev)) 
            #torch.cuda.synchronize(device)
            new_state_dict[k] = gpu_tensor

        state_dict = new_state_dict
        
        #batch
        '''
        # 1) 收集 batch
        bf16_gpu_tensors = []
        bf16_cpu_arrays = []

        other_gpu_tensors = []
        other_cpu_arrays = []

        for k, v in cpu_state_dict.items():
            gpu_tensor = new_state_dict[k]

            if v.dtype == torch.bfloat16:
                bf16_gpu_tensors.append(gpu_tensor.view(torch.int16))
                bf16_cpu_arrays.append(v.view(torch.int16).numpy())
            else:
                other_gpu_tensors.append(gpu_tensor)
                other_cpu_arrays.append(v.numpy())

        # 2) 批量传输（可分块，避免一次太大）
        BATCH_CHUNK = 32

        def _chunked_batch_h2d(gpus, cpus):
            for i in range(0, len(gpus), BATCH_CHUNK):
                mma.batch_h2d(gpus[i:i+BATCH_CHUNK], cpus[i:i+BATCH_CHUNK])
                torch.cuda.set_device(prev_dev)

        if bf16_gpu_tensors:
            _chunked_batch_h2d(bf16_gpu_tensors, bf16_cpu_arrays)

        if other_gpu_tensors:
            _chunked_batch_h2d(other_gpu_tensors, other_cpu_arrays)

        # 3) ✅ 你问的这句：放在 batch 全部结束之后
        state_dict = new_state_dict
        '''
        
        #Asynchronous
        '''
        # 建议：在进入 use_mma 分支后就把 device 拉回目标卡
        torch.cuda.set_device(prev_dev)

        # 用一个 CUDA stream 来承载这些 async copy
        copy_stream = torch.cuda.Stream(device=prev_dev)
        i=0
        for k, v in cpu_state_dict.items():
            gpu_tensor = new_state_dict[k]

            if v.dtype == torch.bfloat16:
                # bfloat16 按 int16 原样搬运（2 bytes/elem）
                src = v.contiguous().view(torch.int16).numpy()
                dst = gpu_tensor.view(torch.int16)
                nbytes = src.nbytes
            else:
                src = v.contiguous().numpy()
                dst = gpu_tensor
                nbytes = src.nbytes
            logger.info(i)
            i+=1

            # ✅ 异步 H2D：显式传 size（bytes）
            # 关键点：stream 通常要传底层句柄 copy_stream.cuda_stream
            mma.h2d_async(dst, src, nbytes, copy_stream)
            torch.cuda.set_device(prev_dev)
            #torch.cuda.synchronize(prev_dev)

        state_dict = new_state_dict
        '''


        _log_cuda_env("after_mma_copy", device)

        #torch.cuda.set_device(prev_dev)
        #torch.cuda.synchronize(prev_dev)
        torch.cuda.set_stream(torch.cuda.default_stream(prev_dev))

        #_log_cuda_env("after_xiugai", device)
    else:
        state_dict = {k: v.to(device) for k, v in state_dict.items()}

    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Transfer completed in {elapsed:.2f}s ({total_bytes / elapsed / 1e9:.2f} GB/s)")
    return _merge_state_dict(state_dict)


# Backward compatibility
def load_hf_weight(
    model_path: str, device: torch.device, use_mma: bool = False
) -> Dict[str, torch.Tensor]:
    return load_weight(model_path, device, source="huggingface", use_mma=use_mma)


def load_ms_weight(
    model_path: str, device: torch.device, use_mma: bool = False
) -> Dict[str, torch.Tensor]:
    return load_weight(model_path, device, source="modelscope", use_mma=use_mma)
