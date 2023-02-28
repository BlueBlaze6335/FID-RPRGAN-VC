# FID-RPRGAN-VC

## Datasets

* <a href="https://datashare.ed.ac.uk/handle/10283/3061"> VCC2018 Non-Parallel Dataset </a>
* <a href="http://festvox.org/cmu_arctic/"> CMU Arctic </a>
* <a href="http://neurolab.unife.it/easycallcorpus/"> EasyCall Corpus Dysarthric dataset </a>

## Generated Samples
### VCC2018
* <a href="https://drive.google.com/drive/folders/1sRbAT83uYoESGERmKvSTghqkNelxiH7l?usp=sharing"> Female to Female </a>
* <a href="https://drive.google.com/drive/folders/1yw6gocfqm15sqVdyX5FotAYlYxoKwjmQ?usp=share_link"> Female to Male </a>
* <a href="https://drive.google.com/drive/folders/1STo0pgYLPdG4z-pPaVwb_L3Fz_7EHH9B?usp=share_link"> Male to Female </a>
* <a href="https://drive.google.com/drive/folders/1f8jwaqYS_7nz_XcUNBzDbWD_EpMIEdmZ?usp=share_link"> Male to Male </a>

### CMU Arctic
* <a href="https://drive.google.com/drive/folders/1oeQU2LfBZDWwhZ3soansOWj8pRz6Lbg-?usp=share_link"> Female to Female </a>
* <a href="https://drive.google.com/drive/folders/1JxeDFEm-kBR9jnjcc1NiidrjBynP5hjP?usp=share_link"> Female to Male </a>
* <a href="https://drive.google.com/drive/folders/1qIbJPKDl_UESwklZzpIjmghLqgu4M8tz?usp=share_link"> Male to Female </a>
* <a href="https://drive.google.com/drive/folders/13ere0oRjq-PioqJqCE2J4BM8W7kgyXhJ?usp=share_link"> Male to Male </a>

### EasyCall Corpus Dysarthric dataset
* <a href="https://drive.google.com/drive/folders/1UZxRYlzUuL2E0wiYa3RBflDNTZyyQR7W?usp=share_link"> Normal Female to Dysarthric Female </a>
* <a href="https://drive.google.com/drive/folders/1hpIDNC-zJ7mVgNxDlC-bTzHjBVhsFqdY?usp=share_link"> Normal Female to Dysarthric Male </a>
* <a href="https://drive.google.com/drive/folders/13N3Ckp9QwuYfkIchmQRhq5C51ioj-uXV?usp=share_link"> Normal Male to Dysarthric Female </a>
* <a href="https://drive.google.com/drive/folders/1GO2ZlsMJKKpzA_4PvKdiKfFVvsXhTjj1?usp=share_link"> Normal Male to Dysarthric Male </a>

More samples - https://sites.google.com/mtech.nitdgp.ac.in/fid-rprganvc/home

# Code

### Prerequisites
- Linux, macOS or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Data Preprocessing

To expedite training, we preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. We convert waveforms to spectrograms using a [melgan vocoder](https://github.com/descriptinc/melgan-neurips) to ensure that you can decode voice converted spectrograms to waveform and listen to your samples during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2
```


## Training

Train FID-RPRGAN-VC to convert between `<speaker_A_id>` and `<speaker_B_id>`. You should start to get excellent results after only several hundred epochs.
```
python -W ignore::UserWarning -m fid_rprgan_vc.train \
    --name mask_cyclegan_vc_<speaker_id_A>_<speaker_id_B> \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ \
    --speaker_A_id <speaker_A_id> \
    --speaker_B_id <speaker_B_id> \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 6172 \
    --batch_size 1 \
    --lr 5e-4 \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```

To continue training from a previous checkpoint in the case that training is suspended, add the argument `--continue_train` while keeping all others the same. The model saver class will automatically load the most recently saved checkpoint and resume training.

Launch Tensorboard in a separate terminal window.
```
tensorboard --logdir results/logs
```

## Testing

Test your trained FID-RPRGAN-VC by converting between `<speaker_A_id>` and `<speaker_B_id>` on the evaluation dataset. Your converted .wav files are stored in `results/<name>/converted_audio`.

```
python -W ignore::UserWarning -m fid_rprgan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data1/cycleGAN_VC3/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts \
    --load_epoch 500 \
    --model_name generator_A2B \
```
## Acknowledgments
Our code is heavily inspired by [MaskCycleGAN-VC](https://github.com/GANtastic3/MaskCycleGAN-VC).

