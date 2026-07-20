# C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection [![arXiv](https://img.shields.io/badge/arXiv-2408.09647-b31b1b.svg)](https://arxiv.org/abs/2408.09647)


[Chuangchuang Tan](https://scholar.google.com/citations?user=ufR1PmMAAAAJ&hl=zh-CN), [Renshuai Tao](https://rstao-bjtu.github.io/), [Huan Liu](), [Guanghua Gu](), [Baoyuan Wu](), [Yao Zhao](https://scholar.google.com/citations?hl=zh-CN&user=474TbQYAAAAJ), [Yunchao Wei](https://weiyc.github.io/)

Beijing Jiaotong University, YanShan University, CUHK


:star: If our code is helpful to you, please help star this repo. Thanks! :hugs:

## News 🆕
- [Pretrained models & Text links & Dataset link](https://drive.google.com/drive/folders/1WZStlW2zpH85NZit1-JADMvzovEEZEj9?usp=sharing)

# **Overall Pipeline**
<p align="center">
<img src="./assets/C2P-CLIP.png" width="950px" alt="overall pipeline", align="center">
</p>

---

## 🛠️ Installation
### 1) Main Environment (Training & Detection)

```bash
conda create -n c2pclip python=3.10.14 -y
conda activate c2pclip
pip install -r requirements.txt
```

## 📂 Data Preparation

- Prepare your dataset (e.g., **GenImage**, **UniversalFakeDetect**).
- Download **Genimage_CNNDetection_CLIP_prefix_caption.tar.gz** from the provided [Google Drive link](https://drive.google.com/drive/folders/1WZStlW2zpH85NZit1-JADMvzovEEZEj9?usp=sharing).
- Download CLIP weights (ViT-L/14) from [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14).

---

## 🚀 Usage

### 1) Training

Train C2P-CLIP on GenImage and UniversalFakeDetect.


```bash
conda activate c2pclip

./train_genimage.sh

./train_UniversalFakeDetect.sh
```

New local-feature experiments use an image-adaptive residual gate. Initialize
the global LoRA and classifier from a matched baseline checkpoint, freeze them,
and train only the patch residual and gate. The gate starts at `0.01`, so the
initial prediction remains close to the protected baseline:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
python scripts/train.py \
  --dataroot ./ForenSynths_train_val_19test \
  --textroot ./prefix_caption \
  --classes car,cat,chair,horse \
  --clip ./clip-vit-large-patch14 \
  --gpu_ids 0,1 --batch_size 64 --keep_last_batch --niter 1 \
  --total_steps 2251 --eval_freq 0 --lr 0.0002 --claloss 8.0 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --delr 0.9 --delr_freq 10 \
  --init_baseline_checkpoint ./c2p_checkpoints/baseline/model.pth \
  --freeze_global_branch \
  --use_local_features \
  --local_layer 12 --local_dim 256 \
  --local_dropout 0.1 --local_pool mean_std \
  --local_fusion adaptive_residual --local_gate_init 0.01 \
  --rank_loss_weight 1.0 \
  --preserve_loss_weight 0.1 \
  --gate_loss_weight 0.01 \
  --local_candidate_loss_weight 1.0 \
  --gate_supervision_weight 1.0 \
  --gate_target_margin 0.1 \
  --name c2p_local_relative_gate
```

The initialization checkpoint must be a non-local model with matching CLIP and
LoRA dimensions. `concat` and scalar `residual_gate` modes remain available
only for loading and reproducing earlier experiments. Training logs report all
auxiliary losses, the mean adaptive gate, and its relative-reliability target.
The local candidate loss trains a full correction before gate attenuation; the
gate target is nonzero only when that correction reduces per-image BCE versus
the protected baseline. This suppresses local intervention on confident global
predictions while retaining a strong signal on baseline mistakes.
Experiment directories use a compact name capped at 180 UTF-8 bytes; the full
configuration remains available in each directory's `opt.txt`.

### 2) Inference / Testing

```bash
conda activate c2pclip

python inference.py \
  --dataroot ./datasets/GenImage/test/ \
  --model_path ./checkpoints/c2p_clip_genimage/last_model.pth
```

Evaluate the official raw state dictionary on all direct and nested generators
under `CNN_synth_testset`:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_airplane_official.py \
  --dataroot ./CNN_synth_testset \
  --model_path ./C2P_CLIP_release_20240901.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 \
  --gpu 0 \
  --num_workers 4 \
  --predictions_csv ./official_cnn_synth_predictions.csv
```

Evaluate a self-trained baseline LoRA checkpoint through the same recursive,
image-only dataset and preprocessing pipeline:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 --gpu 0 --num_workers 4 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --predictions_csv ./baseline_cnn_synth_predictions.csv
```

Add the matched local-feature architecture flags for a local checkpoint:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./c2p_checkpoints/local/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 --gpu 0 --num_workers 4 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --use_local_features \
  --local_layer 12 --local_dim 256 \
  --local_dropout 0.1 --local_pool mean_std \
  --local_fusion auto \
  --predictions_csv ./local_cnn_synth_predictions.csv
```

`--local_fusion auto` detects legacy `concat`, scalar `residual_gate`, and new
`adaptive_residual` checkpoints from their state-dict keys. An explicit
mismatched fusion mode fails before strict state-dict loading.

Both scripts report ACC, real/fake accuracy, AP, AUROC, ECE, Brier score,
raw-logit class statistics, macro means, and overall metrics. Gated local-model
CSVs additionally contain `global_logit`, `local_logit`, and `gate`; forced-gate
runs also retain `learned_gate`. To diagnose one checkpoint without retraining,
repeat the test with `--gate_override 0`, `0.01`, or `learned`.

### Logit distribution analysis for self-trained LoRA checkpoints

Baseline model on `my_first_test`:

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --save ./baseline_logit_distribution.png
```

Local-feature model on `my_first_test`:

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/local/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --use_local_features \
  --local_layer 12 --local_dim 256 \
  --local_dropout 0.1 --local_pool mean_std \
  --local_fusion auto \
  --save ./local_logit_distribution.png
```

Compare matched baseline and local-feature checkpoints with shared bins:

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --checkpoint_label Baseline \
  --compare_checkpoint ./c2p_checkpoints/local/model.pth \
  --compare_label Local \
  --compare_use_local_features \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --local_layer 12 --local_dim 256 \
  --local_dropout 0.1 --local_pool mean_std \
  --compare_local_fusion auto \
  --save ./baseline_vs_local_logits.png
```


### 3) Feature Analysis (Decoding & Visualization)

```bash
conda activate c2pclip

# Decode features to text
python decode_clipfeature_image.py \
  --image_path ./assets/DALLE/DALLE_2_Cowboy_In_Swamp_Close_Up_Outpaint_1.png \
  --cal_detection_feat

# Visualization (t-SNE)
CUDA_VISIBLE_DEVICES=1 python draw_tsne_kmean.py \
  --draw_data_path ./tsne_png \
  --image_path ./stylegan_tsne_data  \
  --save_name stylegan_test \
  --legend stylegan-bedroom-real stylegan-bedroom-fake stylegan-car-real stylegan-car-fake stylegan-cat-real stylegan-cat-fake \
  --do_extract --do_fit --draw_text 0
```



## 📝 Citation

If you find this code or paper helpful, please cite:

```bibtex
@inproceedings{tan2025c2p,
  title={C2p-clip: Injecting category common prompt in clip to enhance generalization in deepfake detection},
  author={Tan, Chuangchuang and Tao, Renshuai and Liu, Huan and Gu, Guanghua and Wu, Baoyuan and Zhao, Yao and Wei, Yunchao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7184--7192},
  year={2025}
}
```

## 🙏 Acknowledgments

This repository borrows partially from the [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption), [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).
