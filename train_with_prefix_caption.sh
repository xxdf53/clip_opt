#!/bin/bash
# =============================================================================
# C2P-CLIP 训练脚本 — UniversalFakeDetect + prefix_caption
# =============================================================================
# 用法: bash train_with_prefix_caption.sh
#
# 数据布局要求:
#   UniversalFakeDetect/          ← 图片根目录（19个类别，每类含 0_real/ 1_fake/）
#   prefix_caption/train/         ← 标注文本根目录（train 下设相同19个类别）
#   prefix_caption/val/           ← 验证集标注（可选）
#   clip-vit-large-patch14/       ← CLIP ViT-L/14 模型权重（需预先下载）
#
# 自动创建 train/test 软链接分割:
#   训练集: car, cat, chair, horse (4 类)
#   测试集: 其余 15 类
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ------------------------------ 参数配置 --------------------------------------
DATAROOT="${PROJECT_ROOT}/UniversalFakeDetect"
TEXTROOT="${PROJECT_ROOT}/prefix_caption"
CLIP_PATH="${PROJECT_ROOT}/clip-vit-large-patch14"
CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints"
NAME="c2p_clip_UniversalFakeDetect"

TRAIN_CATEGORIES=("car" "cat" "chair" "horse")
TEST_CATEGORIES=(
    "airplane" "bicycle" "bird" "boat" "bottle" "bus"
    "cow" "diningtable" "dog" "motorbike" "person"
    "pottedplant" "sheep" "sofa" "tvmonitor"
)

TOTAL_STEPS=800
NITER=100
BATCH_SIZE=64
LR=0.0001
GPU_IDS=0

# ------------------------------ 创建 train/test 目录（软链接） -----------------
echo "=== 创建 train/test 目录结构 ==="

# 清理旧链接
rm -rf "${DATAROOT}/train" "${DATAROOT}/test" "${DATAROOT}/val"

mkdir -p "${DATAROOT}/train" "${DATAROOT}/test"

# 训练集软链接
for cat in "${TRAIN_CATEGORIES[@]}"; do
    if [ -d "${DATAROOT}/${cat}" ]; then
        ln -s "${DATAROOT}/${cat}" "${DATAROOT}/train/${cat}"
        echo "  train/${cat} -> ${cat} ✅"
    else
        echo "  train/${cat} -> 目录不存在，跳过 ⚠️"
    fi
done

# 测试集软链接
for cat in "${TEST_CATEGORIES[@]}"; do
    if [ -d "${DATAROOT}/${cat}" ]; then
        ln -s "${DATAROOT}/${cat}" "${DATAROOT}/test/${cat}"
        echo "  test/${cat} -> ${cat} ✅"
    else
        echo "  test/${cat} -> 目录不存在，跳过 ⚠️"
    fi
done

# prefix_caption 中创建 test -> train 软链接
# 原因：测试时图片路径为 UniversalFakeDetect/test/{category}/...,
# 路径替换后得到 prefix_caption/test/{category}/...，
# 但 caption 实际在 prefix_caption/train/{category}/...，所以用软链接桥接
if [ ! -e "${TEXTROOT}/test" ]; then
    ln -s "${TEXTROOT}/train" "${TEXTROOT}/test"
    echo "  prefix_caption/test -> train ✅"
fi

# ------------------------------ 检查必需文件 ----------------------------------
echo ""
echo "=== 环境检查 ==="

if [ ! -d "${DATAROOT}" ]; then
    echo "❌ 图片目录不存在: ${DATAROOT}"
    exit 1
fi
echo "✅ 图片目录: ${DATAROOT}"

if [ ! -d "${TEXTROOT}/train" ]; then
    echo "❌ 标注目录不存在: ${TEXTROOT}/train"
    exit 1
fi
echo "✅ 标注目录: ${TEXTROOT}"

if [ ! -d "${CLIP_PATH}" ]; then
    echo "❌ CLIP 模型不存在: ${CLIP_PATH}"
    echo "   请运行: huggingface-cli download openai/clip-vit-large-patch14 --local-dir ${CLIP_PATH}"
    exit 1
fi
echo "✅ CLIP 模型: ${CLIP_PATH}"

# 检查 GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ CUDA 可用"
else
    echo "⚠️  CUDA 不可用，将尝试使用 CPU（速度会很慢）"
    GPU_IDS="-1"
fi

# ------------------------------ 启动训练 --------------------------------------
echo ""
echo "=== 启动训练 ==="
echo "训练类别: ${TRAIN_CATEGORIES[*]}"
echo "测试类别: ${TEST_CATEGORIES[*]}"
echo "总步数: ${TOTAL_STEPS}"
echo "Epoch 数: ${NITER}"
echo "Batch Size: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo ""

python scripts/train.py \
    --dataroot "${DATAROOT}/" \
    --textroot "${TEXTROOT}/" \
    --clip "${CLIP_PATH}/" \
    --name "${NAME}" \
    --total_steps ${TOTAL_STEPS} \
    --niter ${NITER} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --gpu_ids "${GPU_IDS}"

echo ""
echo "=== 训练完成 ==="
echo "模型保存在: ${CHECKPOINTS_DIR}/${NAME}/"
