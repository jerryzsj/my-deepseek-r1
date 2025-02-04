# Local deployment of Deepseek-R1 with RTX3090/RTX4090
With distilled models (qwen-7B and llama-8B)

## Description
Locally deployment of Deepseek-R1 distilled models (qwen-7B and llama-8B at RTX3090). 
This project provides scripts and instructions for setting up and running Deepseek-R1 models on a local machine with an RTX3090/RTX4090 GPU.

## Installation
Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/jerryzsj/my-deepseek-r1.git

# Navigate to the project directory
cd my-deepseek-r1
```

## Usage
Instructions on how to use the project.

## Model Download
Deepseek models should be downloaded and placed within the workspace folder beforehand. 
It is recommended to use git to download models. 

```bash
# Install git-lfs (Make sure you have git-lfs installed to download large files with git)
git lfs install

# Clone the models
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Setup proxy for git if needed
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

## Docker Compose
This project uses Docker Compose to create and run the SGLang server. An example `compose.yaml` is provided.

```bash
# Modify 'compose.yaml' for your PC identically
# Start the SGLang server using Docker Compose
docker-compose up -d

# if everything works well, you will see the following in the Docker-Containers-sglang-Logs:
2025-02-04 21:21:33 [2025-02-04 05:21:33] server_args=ServerArgs(model_path='/sgl-workspace/models/DeepSeek-R1-Distill-Qwen-7B', tokenizer_path='/sgl-workspace/models/DeepSeek-R1-Distill-Qwen-7B', tokenizer_mode='auto', load_format='auto', trust_remote_code=False, dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, quantization=None, context_length=None, device='cuda', served_model_name='/sgl-workspace/models/DeepSeek-R1-Distill-Qwen-7B', chat_template=None, is_embedding=False, revision=None, skip_tokenizer_init=False, host='0.0.0.0', port=30000, mem_fraction_static=0.88, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=2048, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, cpu_offload_gb=0, prefill_only_one_req=False, tp_size=1, stream_interval=1, stream_output=False, random_seed=625144628, constrained_json_whitespace_pattern=None, watchdog_timeout=300, download_dir=None, base_gpu_id=0, log_level='info', log_level_http=None, log_requests=False, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, file_storage_pth='sglang_storage', enable_cache_report=False, dp_size=1, load_balance_method='round_robin', ep_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', lora_paths=None, max_loras_per_batch=8, attention_backend='flashinfer', sampling_backend='flashinfer', grammar_backend='outlines', speculative_draft_model_path=None, speculative_algorithm=None, speculative_num_steps=5, speculative_num_draft_tokens=64, speculative_eagle_topk=8, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, disable_radix_cache=False, disable_jump_forward=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, disable_mla=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_ep_moe=False, enable_torch_compile=False, torch_compile_max_bs=32, cuda_graph_max_bs=8, cuda_graph_bs=None, torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, allow_auto_truncate=False, enable_custom_logit_processor=False, tool_call_parser=None, enable_hierarchical_cache=False)
2025-02-04 21:21:40 [2025-02-04 05:21:40 TP0] Init torch distributed begin.
2025-02-04 21:21:40 [2025-02-04 05:21:40 TP0] Load weight begin. avail mem=22.76 GB
2025-02-04 21:21:41 
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
2025-02-04 21:26:20 
Loading safetensors checkpoint shards:  50% Completed | 1/2 [04:38<04:38, 278.99s/it]
2025-02-04 21:29:17 
Loading safetensors checkpoint shards: 100% Completed | 2/2 [07:36<00:00, 219.04s/it]
2025-02-04 21:29:17 
Loading safetensors checkpoint shards: 100% Completed | 2/2 [07:36<00:00, 228.04s/it]
2025-02-04 21:29:17 
2025-02-04 21:29:17 [2025-02-04 05:29:17 TP0] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=8.37 GB
2025-02-04 21:29:17 [2025-02-04 05:29:17 TP0] KV Cache is allocated. K size: 2.82 GB, V size: 2.82 GB.
2025-02-04 21:29:17 [2025-02-04 05:29:17 TP0] Memory pool end. avail mem=1.68 GB
2025-02-04 21:29:18 [2025-02-04 05:29:18 TP0] Capture cuda graph begin. This can take up to several minutes.
2025-02-04 21:29:19 
  0%|          | 0/4 [00:00<?, ?it/s]
 25%|██▌       | 1/4 [00:01<00:03,  1.14s/it]
 50%|█████     | 2/4 [00:01<00:01,  1.73it/s]
 75%|███████▌  | 3/4 [00:01<00:00,  2.47it/s]
100%|██████████| 4/4 [00:01<00:00,  3.10it/s]
100%|██████████| 4/4 [00:01<00:00,  2.33it/s]
2025-02-04 21:29:19 [2025-02-04 05:29:19 TP0] Capture cuda graph end. Time elapsed: 1.79 s
2025-02-04 21:29:20 [2025-02-04 05:29:20 TP0] max_total_num_tokens=105531, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=2049, context_len=131072
2025-02-04 21:29:20 [2025-02-04 05:29:20] INFO:     Started server process [1]
2025-02-04 21:29:20 [2025-02-04 05:29:20] INFO:     Waiting for application startup.
2025-02-04 21:29:20 [2025-02-04 05:29:20] INFO:     Application startup complete.
2025-02-04 21:29:20 [2025-02-04 05:29:20] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
2025-02-04 21:29:21 [2025-02-04 05:29:21 TP0] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue-req: 0
2025-02-04 21:29:23 [2025-02-04 05:29:23] The server is fired up and ready to roll!
2025-02-04 21:29:21 [2025-02-04 05:29:21] INFO:     127.0.0.1:35342 - "GET /get_model_info HTTP/1.1" 200 OK
2025-02-04 21:29:23 [2025-02-04 05:29:23] INFO:     127.0.0.1:35348 - "POST /generate HTTP/1.1" 200 OK
2025-02-04 21:29:42 [2025-02-04 05:29:42] INFO:     127.0.0.1:41026 - "GET /health HTTP/1.1" 200 OK
```

## CUDA Support
To support CUDA, please install the NVIDIA Container Toolkit. 
Before anything, please install CUDA and CUDNN. 
(I was using cuda_12.6.0 with 560.76 driver)

### For Windows Users
Please use WSL2 for the Docker engine and specify your ideal WSL2 distro for Docker (in Docker-Settings-Resources-WSL integration). 
Run the following commands in your WSL2 distro to install the NVIDIA Container Toolkit:

```bash
# Install the NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Afterwards, add the following to the Docker Engine configuration page:

```json
"runtimes": {
    "nvidia": {
        "args": [],
        "path": "nvidia-container-runtime"
    }
}
```

## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.

## Contact
Contact information for the project maintainer.

---

**System Information:**
- Processor: 12th Gen Intel(R) Core(TM) i9-12900K 3.19 GHz
- Memory: 64.0 GB (63.7 GB usable)
- Graphics: RTX 3090

## Acknowledgements
This project uses [SGLang](https://github.com/sgl-project/sglang) as the server engine. Special thanks to the contributors and the open-source community for their support and tools.

Acknowledgements to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) with MIT license, and [Huggingface](https://huggingface.co/) for providing the platform and community for LLM.