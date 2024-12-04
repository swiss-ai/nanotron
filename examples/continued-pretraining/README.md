# Continued pretraining of TinyLlama_v1.1 1B on a single GPU

I am assuming that you run these commands from the root folder of the nanotron repo.

1. Download the model from huggingface
`pip install huggingface_hub[cli]`
`huggingface-cli download TinyLlama/TinyLlama_v1.1`
This will download the model into your huggingface cache to a path like `~/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1/snapshots/ff3c701f2424c7625fdefb9dd470f45ef18b02d6` (the output of the command). Move the model somewhere else or note down this path, e.g., by doing `export HF_MODEL=~/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1/snapshots/ff3c701f2424c7625fdefb9dd470f45ef18b02d6` or download the model via git lfs instead. 

2. Convert the model into nanotron format
`torchrun --nproc_per_node=1 examples/llama/convert_hf_to_nanotron.py --checkpoint_path=$HF_MODEL --save_path=models/llama1b`

3. Create the training config file. The main thing to consider here is to initialise the model from the checkpoint we just created. This can be done in the training config by changing 
```
model:
  init_method:
    std: 0.025
```
to 
```
model:
  init_method:
    path: models/llama1b
```
Additionally, the remaining model hyperparameters need to be updated to match the ones in `models/llama1b/model_config.json`. This file should look like this 
```
{"bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 5632, "is_llama_config": true, "max_position_embeddings": 2048, "num_attention_heads": 32, "num_hidden_layers": 22, "num_key_value_heads": 4, "pad_token_id": null, "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000.0, "rope_interleaved": true, "tie_word_embeddings": false, "use_cache": true, "vocab_size": 32000}
```
We provide an updated training config yaml in `examples/continued-pretraining/config_1gpu_tiny_llama.yaml`. We also updated the maximum learning rate to 5% of the learning rate and the minimum learning rate to 10% of the learning rate used during pretraining TinyLlama.

4. Launch your training
```
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 run_train.py --config-file examples/continued-pretraining/config_1gpu_tiny_llama.yaml
```

5. We configured our training in a way that also serializes the snapshot before training. We can use this snapshot to compare nanotron checkpoint generations with the ones from the corresponding huggingface checkpoint. To do so, run
`pip install accelerate`. Then you can generate using `torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/0/ &> nano-gen.log` and `python examples/continued-pretraining/hf_example_generations.py &> hf-gen.log` and compare the resulting log files containing their outputs.
