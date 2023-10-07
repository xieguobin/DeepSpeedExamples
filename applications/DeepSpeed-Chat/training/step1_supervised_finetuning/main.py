#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    # 1 训练数据集的路径
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['wangrui6/Zhihu-KOL'], # Dahoas/rm-static
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    # 2 phase1/2/3数据的划分比例
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    # 3 只用于SFT阶段的数据集路径
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    # 4 数据相关文件的存储路径
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    # 5 预训练模型的路径
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    # 6 每个设备上的训练批次大小
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    # 7 每个设备上的评估批次大小
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    # 8 最大序列长度
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # 9 初始学习率
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    # 10 权重衰减
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    # 11 进行的训练轮数
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    # 12 在执行反向传播之前累积的更新步骤数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 13 调度器类型
    # 作用：根据预设的策略来动态调整学习率
    # ① linear: 线性调度器在每个训练步骤中将学习率按线性规则降低。
    # ② cosine: 余弦调度器根据余弦退火调度来调整学习率。学习率将在周期内从初始学习率线性减少到最小学习率，然后在下一个周期内重复这个过程。
    # ③ cosine_with_restarts: 这是余弦退火调度的一个变种，它在每个周期结束后重启学习率，使得每个周期的开始学习率总是高于结束时的学习率。
    # ④ polynomial: 多项式调度器将根据一个多项式规则来降低学习率，通常是一个衰减因子乘以训练步数的某个幂。
    # ⑤ constant: 常数调度器使学习率保持不变，这意味着在整个训练过程中，学习率不会被调整。
    # ⑥ constant_with_warmup: 这是常数调度器的一个变种，在一开始的几个步骤（即预热阶段）内，学习率会被线性增加到预设的学习率，然后在剩余的训练步骤中保持不变。
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    # 14 学习率调度器中的warmup步骤数
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    # 15 存储模型的位置
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="Where to store the model.")
    # 16 用于可复制训练的seed
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    # 17 用于在GPU上进行分布式训练的local rank
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    # 18 是否启用梯度checkpoint
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # 19 是否禁用模型的dropout
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')

    # deepspeed features
    # 20 是否启用ZeRO Offload技术
    # 是否使用ZeRO Offload技术。如果为True，那么将模型参数和优化器状态offload到CPU，否则不使用offload。
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    # 21 用于Actor模型（和clones）的ZeRO优化阶段
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    ## LoRA for efficient training setting
    # 22 如果大于0，使用LoRA进行高效训练
    parser.add_argument("--lora_dim",
                        type=int,
                        default=128,
                        help="If > 0, use LoRA for efficient training.")
    # 23 LoRA的范围
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    # 24 只优化LoRA参数
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    # 如果启用了gradient_checkpointing并且lora_dim大于0，那么必须禁用只优化LoRA参数。
    # 这是因为梯度检查点和只优化LoRA参数这两个选项在同时启用时可能会引发冲突。
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    # 第1步：超参数配置
    args = parse_args()

    if args.local_rank == -1:
        # 单机版的CUDA
        device = torch.device("cuda")
    else:
        # 设置了当前进程中的默认device，确保每个进程在正确的device上运行
        torch.cuda.set_device(args.local_rank)
        # 确保tensor被创建或移动到正确的device上
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        # 初始化分布式训练环境
        deepspeed.init_distributed()
    # 获取当前运行设备在分布式训练环境中的rank
    args.global_rank = torch.distributed.get_rank()
    # 根据输入参数返回一个训练数据集的配置字典
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    # micro_batch训练是一种分布式训练技术，可以将一个大批次的数据分解成多个小批次，以适应设备的内存限制。
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # 每个训练步骤中处理的数据总量
    # gradient_accumulation(梯度积累)是另一种应对内存限制的技术，它会在多个步骤中积累梯度，然后再一次性更新模型参数。
    # torch.distributed.get_world_size() ：在分布式训练环境中的节点（设备）数量
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # PyTorch分布式训练中的一个同步工具，它确保所有分布式进程都达到了这个阻塞点，
    # 然后再继续执行后面的代码，以避免出现某些进程快于其他进程的情况。
    torch.distributed.barrier()
    # 加载预训练模型tokenizer，fast_tokenizer=True表示使用优化过的、速度更快的tokenizer。
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # 将tokenizer的pad_token设置为eos_token，意味着模型将认为这些填充部分是句子的结束。
    tokenizer.pad_token = tokenizer.eos_token
    # 将tokenizer的填充方向设置为'right'，表示在序列的右侧（末尾）添加填充符号。
    tokenizer.padding_side = 'right'

    # 第2步：创建actor模型
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)

    if args.lora_dim > 0:
        # 将模型中指定的线性层转换为LoRA层
        # lora_module_name指定了要转换的模块的名称
        # lora_dim指定了LoRA的维度
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            # 只优化LoRA层的参数
            # 将模型中非LoRA层的参数的requires_grad属性设为False，这样在训练过程中只有LoRA层的参数会被更新。
            model = only_optimize_lora_parameters(model)

    # 第3步：准备数据集（训练集和验证集）
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)

    # DataLoaders creation:
    # # 不在分布式训练环境下，因此我们将使用随机采样和顺序采样
    if args.local_rank == -1:
        # 在训练过程中，将随机选择样本进行训练，防止模型过拟合。
        train_sampler = RandomSampler(train_dataset)
        # 在评估过程中，将按照数据集中的顺序选择样本进行评估，验证集的顺序通常对模型的性能评估没有影响。
        eval_sampler = SequentialSampler(eval_dataset)
    # 在分布式训练环境中，将使用分布式采样
    else:
        # 创建一个用于训练集的分布式采样器，会在所有的训练节点上对样本进行均匀的分布，
        # 确保每个节点处理的样本是独立且均匀的，从而提高分布式训练的效率和稳定性。
        train_sampler = DistributedSampler(train_dataset)
        # 创建一个用于评估集的分布式采样器
        eval_sampler = DistributedSampler(eval_dataset)
    # 创建用于训练集的数据加载器
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator, # 要作用是将一批数据进行整合，使得它们可以整齐地堆叠在一起。
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    # 创建用于评估集的数据加载器
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    # 第4步：将模型参数分组、创建优化器 和 学习率调度器
    # 1. 将模型的参数分为两组，一组应用权重衰减，另一组不应用
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    # 根据是否使用DeepSpeed的CPU offload功能来选择优化器，优化器定义了如何更新模型的参数以最小化损失函数。
    # DeepSpeedCPUAdam : 为了配合DeepSpeed的CPU offload功能设计的，在DeepSpeed中，CPU offload可以将模型参数、
    #                    优化器状态和梯度数据在CPU和GPU之间进行切换，以减轻GPU的内存压力。
    # FusedAdam : 它将一些计算操作融合在一起（fused），以减少计算时间和内存使用量。
    #             FusedAdam主要是为了提高在GPU上的运算效率。
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    # 使用选择的优化器进行实例化
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    # 计算每个epoch需要进行的更新步数，等于训练数据集大小除以梯度累积步数（对结果向上取整）
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    # 创建学习率调度器
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, # 总的训练步数等于训练的epoch数乘以每个epoch的更新步数
    )

    # 第5步：deepspeed初始化，创建模型、优化器、学习率调度器
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config, # DeepSpeed的配置信息
        lr_scheduler=lr_scheduler,
        dist_init_required=True # 需要进行分布式训练的初始化
    )
    # 否启用了梯度检查点
    if args.gradient_checkpointing:
        # 梯度检查点是一种可以减少训练过程中内存使用的技术，通过牺牲一部分计算效率来换取内存占用的降低。
        model.gradient_checkpointing_enable()

    # 第6步：模型验证
    # 模型评估
    def evaluation(model, eval_dataloader):
        # 将模型设置为评估模式
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                # 用当前批次的输入数据去前向传播模型，并得到模型的输出。
                outputs = model(**batch)
            # 提取模型输出中的损失值
            loss = outputs.loss
            # 累积整个评估过程中的损失值
            losses += loss.float()
        # 计算平均损失值
        losses = losses / (step + 1)
        try:
            # 计算模型的困惑度，这是评估语言模型性能的常用指标。困惑度的计算方法是对平均损失值取指数。
            # 如果这步运算没有发生溢出，那么困惑度的值就是torch.exp(losses)。
            # 当损失值过大时，指数运算可能会导致溢出
            perplexity = torch.exp(losses)
        except OverflowError:
            # 如果上一步的指数运算发生了溢出，那么将困惑度的值设为无穷大。
            perplexity = float("inf")
        try:
            # 如果这是一个分布式设置，该函数会计算所有device上的平均困惑度
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        # 返回最后的困惑度作为模型在给定评估数据集上的性能指标
        return perplexity

    # 第7步：模型训练
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)

    # 计算困惑度
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        # 只在rank为0的进程中打印出开始训练的信息
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            # 将输入的批次数据（batch）移到指定的device上
            batch = to_device(batch, device)
            # use_cache 通常用于控制是否使用缓存来加速计算
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # 调用模型的backward方法进行反向传播，计算损失函数关于模型参数的梯度。
            model.backward(loss)
            # 更新模型的参数
            model.step()

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        # 计算在验证集上的困惑度
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        # 更新了模型的内部计时器，表示一个epoch已经完成。
        model.tput_timer.update_epoch_count()

    # 第8步：训练结束后保存模型
    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        # 将模型中的LoRA层转换为标准的线性层，这样使得模型在保存后可以在没有LoRA层代码的环境中加载和使用
        model = convert_lora_to_linear_layer(model)
        # 是否在主进程中
        if args.global_rank == 0:
            # 以HuggingFace格式保存模型和tokenizer
            save_hf_format(model, tokenizer, args)

        # 是否使用了Zero Redundancy Optimizer的第三阶段（ZeRO-3）
        # ZeRO-3是一种内存优化策略，可以大大减少模型训练中所需的GPU内存，但同时也意味着模型的各部分在不同的GPU之间分布。
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
