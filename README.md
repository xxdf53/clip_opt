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

Two training-only objectives that improved the matched GAN protocol remain
supported. Symmetric Logit Anchor (SLAR) keeps real/fake logits around fixed
targets on opposite sides of zero. Counterfactual Prompt Decomposition (CPD)
aligns the LoRA visual residual with the real/fake direction of paired captions:

```bash
python scripts/train.py [baseline arguments] \
  --logit_anchor 3.0 --anchor_loss_weight 0.5 \
  --cpd_direction_weight 0.5 --cpd_content_weight 0.0 \
  --cpd_direction_margin 0.1 \
  --cpd_start_step 400 --cpd_warmup_steps 400
```

Both are disabled by default and add no inference inputs. Their options are
kept because failure on the diffusion protocol does not invalidate the matched
GAN results.

For multi-GPU training, optional global contrastive regularization gathers the
aligned image and text embeddings from every replica before constructing one
full-batch similarity matrix. The original local contrastive loss remains the
main contrastive term; the global loss is only an auxiliary regularizer, so
classification BCE and inference remain unchanged:

```bash
python scripts/train.py [baseline arguments] --global_contrastive_weight 0.1
```

The weight is disabled by default, changes neither inference nor checkpoint
structure, and supports uneven final batches because only feature matrices are
gathered across GPUs.

Experiment directories use a compact name capped at 180 UTF-8 bytes and record
all active objectives. Failed GenImage paths (GAlC, AGDRO,
gradient-accumulation emulation, PRH, SPH, RVIB, RTR, EMA and degradation
consistency) were removed from the active implementation. Their checkpoints
require an older Git revision when they changed the model structure; baseline,
SLAR, CPD and standard LoRA checkpoints remain compatible.

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

Evaluate a self-trained baseline or experimental LoRA checkpoint through the
same recursive, image-only dataset and preprocessing pipeline:

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./c2p_checkpoints/c2p_experiment/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --batch_size 64 --gpu 0 --num_workers 4 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --predictions_csv ./cnn_synth_predictions.csv
```

Both scripts report ACC, real/fake accuracy, AP, AUROC, ECE, Brier score,
raw-logit class statistics, macro means, and overall metrics. Prediction CSVs
contain the generator, image path, label, raw logit, and sigmoid score.

Pass multiple compatible checkpoints after one `--checkpoint` flag to evaluate
each model sequentially and report a uniform raw-logit ensemble without loading
all models into GPU memory at once:

```bash
python scripts/test_checkpoint.py \
  --dataroot ./CNN_synth_testset \
  --checkpoint ./seed42.pth ./seed123.pth ./seed2024.pth \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8
```

### Logit distribution analysis for self-trained LoRA checkpoints

One model on `my_first_test`:

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/c2p_experiment/model.pth \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --save ./logit_distribution.png
```

Compare matched baseline and experimental checkpoints with shared bins:

```bash
python scripts/plot_logit_dist.py \
  --dataroot ./my_first_test \
  --checkpoint ./c2p_checkpoints/baseline/model.pth \
  --checkpoint_label Baseline \
  --compare_checkpoint ./c2p_checkpoints/c2p_experiment/model.pth \
  --compare_label Experiment \
  --clip_path ./clip-vit-large-patch14 \
  --lora_r 6 --lora_alpha 6 --lora_dropout 0.8 \
  --save ./baseline_vs_experiment_logits.png
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
