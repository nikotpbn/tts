import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig,
)
from TTS.utils.manage import ModelManager

# 1. Define Output Paths
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_run")
os.makedirs(OUT_PATH, exist_ok=True)
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# 2. Configure the Dataset (Pointed exactly at your S3 data)
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="thrall",
    path="/root/tts/data/processed/thrall/",
    meta_file_train="/root/tts/data/processed/thrall/metadata.csv",
    language="en",
)

# 3. Download Base Model Weights for Fine-Tuning
dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
mel_norm_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
tokenizer_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
xtts_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

ModelManager._download_model_files(
    [dvae_link, mel_norm_link, tokenizer_link, xtts_link],
    CHECKPOINTS_OUT_PATH,
    progress_bar=True,
)

# 4. Model Architecture & Audio Configuration
model_args = GPTArgs(
    max_conditioning_length=132300,
    min_conditioning_length=66150,
    max_wav_length=255995,
    max_text_length=200,
    mel_norm_file=os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth"),
    dvae_checkpoint=os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth"),
    xtts_checkpoint=os.path.join(CHECKPOINTS_OUT_PATH, "model.pth"),
    tokenizer_file=os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json"),
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_mic=False,
)

audio_config = XttsAudioConfig(
    sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
)

# 5. Training Loop Configuration
config = GPTTrainerConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name="thrall_xtts_finetune",
    project_name="XTTS_trainer",
    run_description="Fine tuning XTTS on the Thrall dataset",
    audio=audio_config,
    batch_size=2,  # Kept low to prevent out-of-memory errors on 16GB VRAM
    eval_batch_size=2,
    num_loader_workers=4,
    epochs=100,
    print_step=50,
    save_step=1000,
    save_n_checkpoints=2,
    optimizer="AdamW",
    lr=5e-06,
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={
        "milestones": [50000, 150000, 300000],
        "gamma": 0.5,
        "last_epoch": -1,
    },
)

# 6. Initialize Model and Start Trainer
train_samples, eval_samples = load_tts_samples(
    config_dataset, eval_split=True, eval_split_size=0.1
)

model = GPTTrainer.init_from_config(config)

trainer = Trainer(
    TrainerArgs(),
    config,
    OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
