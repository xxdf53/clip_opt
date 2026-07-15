"""
=============================================================================
 train.py — C2P-CLIP 训练主脚本
=============================================================================
 功能：基于 CLIP ViT-L/14 + LoRA + 类别感知提示词的深度伪造检测模型训练。
 架构：双损失 = 对比损失（图文对齐） + 分类损失（真/假 BCE）。

 调用栈：
   main
   ├── seed_torch()       — 全局函数：锁定所有随机性来源，保证可复现
   ├── get_val_opt()      — 全局函数：构造验证配置（供外部脚本导入使用）
   ├── testmodel()        — 闭包函数：按配置步数评估所有测试子集
   └── 训练循环
       ├── set_input()    — 将数据从 CPU 搬运到 GPU
       ├── optimize_parameters() — 前向 + 计算双损失 + 反向 + AdamW 更新
       ├── adjust_learning_rate() — lr *= delr 衰减
       └── save_networks()       — 保存 .pth + HuggingFace 格式
=============================================================================
"""

import os
import sys
# 将项目根目录加入 sys.path，确保能 import networks / data / options / utils 等模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn
import numpy as np
from validate import validate          # 评估函数：返回 acc, ap, r_acc, f_acc, y_true, y_pred
from data import create_dataloader     # 构造 DataLoader，每条数据: (path, img, text, input_ids, attention_mask, label)
from networks.trainer import Trainer   # 训练器：包含 CLIPModel_lora + 优化器 + 损失计算逻辑
from options.train_options import TrainOptions  # 训练参数解析
from options.test_options import TestOptions    # 测试参数解析
from utils.util import Logger          # stdout → 文件 tee
from utils.evaluation_schedule import (
    should_evaluate,
    should_run_final_evaluation,
)
import random


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       全 局 函 数                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def seed_torch(seed=1029):
    """
    [函数] 锁定 Python / NumPy / PyTorch / cuDNN 的所有随机性来源。
    
    逐项作用：
      random.seed(seed)              — Python 标准库随机数
      PYTHONHASHSEED = str(seed)     — dict/set 遍历顺序（Python 3.3+）
      np.random.seed(seed)           — NumPy 随机数
      torch.manual_seed(seed)        — PyTorch CPU 随机数
      torch.cuda.manual_seed(seed)   — 当前 GPU 随机数
      torch.cuda.manual_seed_all()   — 所有 GPU 随机数
      cudnn.benchmark = False        — 关闭自动算法搜索（否则引入非确定性）
      cudnn.deterministic = True     — 强制使用确定性卷积算法
      cudnn.enabled = False          — ⚠️ 完全禁用 cuDNN，性能大幅下降但精度波动归零
    
    注意：cudnn.enabled = False 极为激进，通常 deterministic=True 已足够；
          这会显著拖慢训练速度。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# 文档字符串声明了该函数的使用前提：jpg_prob 和 blur_prob 必须为 0 或 1
"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    """
    [函数] 从训练配置构造验证/测试时的 Options 对象。
    
    关键设计：将数据增强参数（blur 范围、JPEG 质量范围）坍缩为单值中点，
    确保验证时无随机扰动。
    
    用途：仅供外部脚本导入使用（如 notebooks / 独立评估脚本）。
    train.py 主流程中并未调用此函数——testmodel() 用的是 TestOptions。
    
    步骤：
      1. 从命令行读取训练配置作为基础
      2. dataroot 切到 val 子目录
      3. isTrain=False → 不启用随机数据增强
      4. serial_batches=True → 顺序读取，不打乱
      5. jpg_method 固定为 'pil'
      6. blur_sig / jpg_qual 范围取中点（如 [0.5, 1.5] → [1.0]）
    """
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = os.path.join(val_opt.dataroot, val_opt.val_split, '')
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       主 入 口                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # ─── 阶段 ①：初始化 ────────────────────────────────────────────────────
    # 解析命令行参数 → 自动生成带时间戳的实验名（格式见 base_options.py:45）
    opt = TrainOptions().parse()
    # 锁定所有随机性
    seed_torch(opt.seed)

    # test 目录路径：用于多子集评估（如 airplane/, bicycle/, car/ …）
    Test_dataroot = os.path.join(opt.dataroot, 'test')
    # 训练数据路径切到 train 子目录
    opt.dataroot = os.path.join(opt.dataroot, opt.train_split, '')
    # 将 stdout 同时写入日志文件（tee 效果），日志路径: checkpoints/<name>/log.log
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    # 构造测试 Option 对象（用于后续 testmodel 闭包）
    Testopt = TestOptions().parse(print_options=False)
    # Test_vals = test 目录下所有子文件夹名（每个是一个独立测试任务，如 sdv4, midjourney, …）
    Test_vals = os.listdir(Test_dataroot)
    # 构造训练 DataLoader（内部: ImageFolder2，每条数据 6 元组）
    data_loader = create_dataloader(opt)
    # 初始化模型（CLIP ViT-L/14 + LoRA）+ 优化器（Adam/AdamW）
    model = Trainer(opt)
    
    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║         闭包函数: testmodel() — 按配置步数做多子集评估                 ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    def testmodel(epoch=0):
        """
        [闭包函数] 在测试集的所有子集上逐一评估，汇总结果。
        
        流程:
          1. 遍历 Test_vals（test 目录下的每个子文件夹，如 airplane/, car/, …）
          2. 将 Testopt.dataroot 切换到当前子文件夹
          3. 调用 validate() → 返回 (acc, ap, r_acc, f_acc, y_true, y_pred)
             validate() 内部: DataLoader → model(img, None, None, cla=True)
             → 只走分类头（跳过对比损失分支） → 收集 y_true/y_pred
             → 计算 accuracy_score + average_precision_score
          4. 汇总所有子集的 acc/ap，打印均值
        
        返回: 所有子集 acc 的均值（四舍五入到 4 位小数）
        """
        print('*' * 25)
        accs = []   # 每个子集的 accuracy
        aps = []    # 每个子集的 average precision
        logs = [f"Testing end of {epoch}"]
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        
        for v_id, val in enumerate(Test_vals):
            # 将测试路径切换到当前子文件夹
            Testopt.dataroot = os.path.join(Test_dataroot, val)
            # 注：被注释的行原本支持多类别模式（multiclass）
            # Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.loadSize = opt.cropSize
            Testopt.cropSize = opt.cropSize
            Testopt.no_resize = False
            Testopt.no_crop = False
            Testopt.classes = ''   # 空字符串 = 加载该子文件夹下所有图片
            # 核心评估调用：validate() 定义在 scripts/validate.py
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc)
            aps.append(ap)
            logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(
                v_id, val, acc * 100, ap * 100))
            print(logs[-1])
        
        # 打印所有子集的均值
        logs.append("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(
            v_id + 1, 'Mean',
            np.array(accs).mean() * 100,
            np.array(aps).mean() * 100))
        print(logs[-1])
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        # 返回值用于 save_networks 文件名中的 testacc 字段
        return round(np.array(accs).mean() * 100, 4)

    def evaluate_and_save(epoch):
        """Evaluate all held-out subsets and save the corresponding checkpoint."""
        model.eval()
        testacc = testmodel(epoch)
        model.save_networks(
            f'{str(epoch)}_total_steps_{str(model.total_steps)}_testacc_{str(testacc)}')
        print('saving the latest model %s (epoch %d, model.total_steps %d)' %
              (opt.name, epoch, model.total_steps))
        model.train()
        return model.total_steps

    # ─── 阶段 ②：训练循环 ──────────────────────────────────────────────────
    model.train()
    last_eval_step = None
    last_epoch = 0
    for epoch in range(opt.niter):
        last_epoch = epoch
        # epoch_start_time / iter_data_time 当前未使用（可能最初为计时预留）
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0   # 当前 epoch 已处理的样本数

        for i, data in enumerate(data_loader):
            # data = (path, img, text, input_ids, attention_mask, label) — 6 元组
            if model.total_steps >= opt.total_steps:
                break

            model.total_steps += 1
            
            epoch_iter += opt.batch_size
            
            # 将数据从 CPU 搬运到 GPU（见 trainer.py:113）
            model.set_input(data)
            # 核心训练步骤：前向 → 对比损失 + 分类损失 → 反向 → AdamW 更新
            # (详见 trainer.py:133 optimize_parameters)
            model.optimize_parameters()
            
            # 每隔 loss_freq 步打印一次三种损失
            #   loss1 = 对比损失（图文对齐）  loss2 = 分类损失（BCE）
            #   loss  = loss1 + claloss * loss2
            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} loss1: {} loss2-cla: {} at step: {} lr {}".format(
                          model.loss, model.loss1, model.loss2,
                          model.total_steps, model.lr))

            if should_evaluate(model.total_steps, opt.eval_freq):
                print(f'==========total_steps {model.total_steps}=================')
                last_eval_step = evaluate_and_save(epoch)

        if model.total_steps >= opt.total_steps:
            break

        # ─── 阶段 ③：学习率衰减（每个 delr_freq epoch，跳过 epoch 0） ─────
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            # lr *= delr（默认 delr=0.8），见 trainer.py:105
            model.adjust_learning_rate()

    if should_run_final_evaluation(model.total_steps, last_eval_step):
        print(f'==========final total_steps {model.total_steps}=================')
        evaluate_and_save(last_epoch)
