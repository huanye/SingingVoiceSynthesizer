## Quick Start

0. Make env
    ```bash
    pip install -r requirements.txt
    ```

1. Build binary data
    ```bash
    PYTHONPATH=. python datasets/tts/ccom/gen_fs.py --config configs/tts/ccom/fs.yaml
    ```

2. Train
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/fs.py --config configs/tts/ccom/fs.yaml --exp_name fs_exp1
    ```
    Note: Change the variable 'raw_data_dir' in 'configs/tts/ccom/fs.yaml' to your own data dir, firstly.

3. Download pre-trained vocoder
    ```bash
    mkdir wavegan_pretrained
    ```
    download `checkpoint-1000000steps.pkl`, `config.yaml`, `stats.h5` from https://drive.google.com/open?id=1XRn3s_wzPF2fdfGshLwuvNHrbgD0hqVS to `wavegan_pretrained/`

    a good suggestion is to use your own data to train a new vocoder based on https://github.com/kan-bayashi/ParallelWaveGAN

4. Infer
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/fs.py --config configs/tts/ccom/fs.yaml --exp_name fs_exp1 --infer
    ```