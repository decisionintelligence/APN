import time
import json
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.globals import logger, accelerator

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if val_loss in [np.nan, torch.nan, float("nan")]:
            logger.warning(f"Validation loss is nan, stopping...")
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.debug(f'Validation loss decreased ({self.val_loss_min:.2e} --> {val_loss:.2e}).  Saving model ...')
        accelerator.save_model(
            model, 
            path, 
            safe_serialization=False
        )
        self.val_loss_min = val_loss


def test_params_and_flops(
        model: torch.nn.Module,
        dummy_inputs: dict,
        model_id: str,
        task_key: str
):
    """
    测试模型的参数量和FLOPs。
    使用一个通用的包装器来处理需要多个命名参数输入的模型。
    - dummy_inputs: 一个字典，键是模型forward函数所需的参数名(str)，值是对应形状的虚拟张量(torch.Tensor)。
                    字典中的第一个键值对将被视为主输入。
    """
    from ptflops import get_model_complexity_info

    class GenericModelWrapper(torch.nn.Module):
        def __init__(self, model, dummy_inputs):
            super().__init__()
            self.model = model
            self.primary_input_key = next(iter(dummy_inputs))
            self.fixed_kwargs = {k: v for k, v in dummy_inputs.items() if k != self.primary_input_key}

        def forward(self, primary_input):
            kwargs = {self.primary_input_key: primary_input, **self.fixed_kwargs}
            return self.model(**kwargs)

    model.eval().cuda()

    wrapped_model = GenericModelWrapper(model, dummy_inputs)

    # ==================== 唯一的修改在这里 ====================
    # ptflops 期望输入形状不包含批次维度。
    # 我们获取主输入的形状，然后切片去掉第一个元素（批次维度）。
    primary_input_tensor = dummy_inputs[next(iter(dummy_inputs))]
    primary_input_shape_no_batch = tuple(primary_input_tensor.shape[1:])
    # ========================================================

    macs, params = get_model_complexity_info(
        wrapped_model,
        primary_input_shape_no_batch,  # 传递修正后的、不含batch维度的shape
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    if params is None or macs is None:
        logger.error("Failed to calculate FLOPs and parameters. Please check the model's forward pass compatibility.")
        return

    params_in_million = float(params.split(' ')[0])

    logger.info(f"--- Model Complexity Analysis for {model_id} ---")
    for name, tensor in dummy_inputs.items():
        logger.info(f"Input shape ({name}): {tensor.shape}")
    logger.info(f"{'Computational complexity (MACs)':<30}  {macs}")
    logger.info(f"{'Number of parameters (M)':<30}  {params_in_million:.2f}M")
    logger.info("-------------------------------------------------")


def test_peak_gpu_memory(
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        criterion: torch.nn.Module,  # <--- 新增参数
        model_id: str,
        gpu_id: int
):
    """
    测试模型单次前向+反向传播的峰值GPU显存占用。
    (已修复：现在使用项目特定的criterion，而不是通用MSELoss)
    """
    logger.info(f"--- GPU Memory Test for {model_id} ---")
    model.train().cuda(gpu_id)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(gpu_id)

    try:
        # --- 核心修改部分 ---
        # 1. 执行前向传播
        outputs = model(exp_stage="train", **batch)

        # 2. 使用正确的、项目特定的criterion来计算损失
        #    这与 train() 循环中的逻辑完全一致，保证了兼容性
        loss_dict = criterion(exp_stage="train", model=model, **outputs)
        loss = loss_dict["loss"]

        # 3. 执行反向传播
        loss.backward()
        # --- 修改结束 ---

    except Exception as e:
        logger.error(f"Error during memory test forward/backward pass: {e}")
        return

    peak_memory_gb = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)
    logger.info(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")
    logger.info("-----------------------------------------")
    torch.cuda.empty_cache()


def test_training_step_time(
        model: torch.nn.Module,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_id: str,
        gpu_id: int,
        warmup_steps=10,
        test_steps=100
):
    """
    测试模型单个训练步的平均耗时。
    (最终版：完全自适应，能处理任何大小的数据集)
    """
    logger.info(f"--- Training Step Time Test for {model_id} ---")
    model.train().cuda(gpu_id)

    num_batches = len(dataloader)

    # --- 核心修改部分：实现完全自适应的步数分配 ---
    MIN_BATCHES_FOR_TEST = 2
    if num_batches < MIN_BATCHES_FOR_TEST:
        logger.warning(
            f"Dataloader has only {num_batches} batch(es), which is not enough for both warmup and testing. "
            "Timing test aborted."
        )
        return

    # 动态分配预热步数
    if num_batches <= 20:  # 对于小型数据集，只预热1步以节省数据
        actual_warmup_steps = 1
    else:
        actual_warmup_steps = warmup_steps

    # 确保预热步数不超过总步数的一半
    actual_warmup_steps = min(actual_warmup_steps, num_batches // 2)

    actual_test_steps = num_batches - actual_warmup_steps
    # --- 修改结束 ---

    times = []
    data_iter = iter(dataloader)

    logger.info(f"Total batches: {num_batches}. Warming up for {actual_warmup_steps} step(s)...")
    for i in tqdm(range(actual_warmup_steps), desc="Warmup"):
        batch = next(data_iter)
        batch = {k: v.float().to(f"cuda:{gpu_id}") for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(exp_stage="train", **batch)
        loss = criterion(**outputs)["loss"]
        loss.backward()
        optimizer.step()

    logger.info(f"Testing for the remaining {actual_test_steps} step(s)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for i in tqdm(range(actual_test_steps), desc="Testing Train Time"):
        batch = next(data_iter)
        batch = {k: v.float().to(f"cuda:{gpu_id}") for k, v in batch.items()}

        starter.record()
        optimizer.zero_grad()
        outputs = model(exp_stage="train", **batch)
        loss = criterion(**outputs)["loss"]
        loss.backward()
        optimizer.step()
        ender.record()

        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        times.append(curr_time)

    if not times:
        logger.error("No valid timing results were recorded.")
        return

    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)

    logger.info(
        f"Average training step time (based on {actual_test_steps} samples): {avg_time_ms:.3f} ms ± {std_time_ms:.3f} ms")
    logger.info("-------------------------------------------------")


def test_inference_step_time(
        model: torch.nn.Module,
        dataloader: DataLoader,
        model_id: str,
        gpu_id: int,
        warmup_steps=10,
        test_steps=100
):
    """
    测试模型单个推理步的平均耗时。
    (最终版：完全自适应，能处理任何大小的数据集)
    """
    logger.info(f"--- Inference Step Time Test for {model_id} ---")
    model.eval().cuda(gpu_id)

    num_batches = len(dataloader)

    # --- 核心修改部分：实现完全自适应的步数分配 ---
    MIN_BATCHES_FOR_TEST = 2
    if num_batches < MIN_BATCHES_FOR_TEST:
        logger.warning(
            f"Dataloader has only {num_batches} batch(es), which is not enough for both warmup and testing. "
            "Timing test aborted."
        )
        return

    if num_batches <= 20:
        actual_warmup_steps = 1
    else:
        actual_warmup_steps = warmup_steps

    actual_warmup_steps = min(actual_warmup_steps, num_batches // 2)

    actual_test_steps = num_batches - actual_warmup_steps
    # --- 修改结束 ---

    times = []
    data_iter = iter(dataloader)

    logger.info(f"Total batches: {num_batches}. Warming up for {actual_warmup_steps} step(s)...")
    with torch.no_grad():
        for i in tqdm(range(actual_warmup_steps), desc="Warmup"):
            batch = next(data_iter)
            batch = {k: v.float().to(f"cuda:{gpu_id}") for k, v in batch.items()}
            model(exp_stage="test", **batch)

    logger.info(f"Testing for the remaining {actual_test_steps} step(s)...")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in tqdm(range(actual_test_steps), desc="Testing Inference Time"):
            batch = next(data_iter)
            batch = {k: v.float().to(f"cuda:{gpu_id}") for k, v in batch.items()}

            starter.record()
            model(exp_stage="test", **batch)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times.append(curr_time)

    if not times:
        logger.error("No valid timing results were recorded.")
        return

    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)

    logger.info(
        f"Average inference step time (based on {actual_test_steps} samples): {avg_time_ms:.3f} ms ± {std_time_ms:.3f} ms")
    logger.info("--------------------------------------------------")

def linear_interpolation(x):
    # Linear interpolation function
    # Assuming x is a tensor of shape (batch_size, sequence_length, input_size)
    # Interpolate n-1 values between n original values
    batch_size, time_length, n_variables = x.shape
    x_interpolated = torch.zeros(batch_size, 2 * time_length - 1, n_variables, device=x.device)
    x_interpolated[:, 0] = x[:, 0]
    interpolated_values = (x[:, 1:] + x[:, :-1]) / 2
    # for i in range(batch_size):
    for j in range(time_length - 1):
        x_interpolated[:, 2 * j + 1] = interpolated_values[:, j]
        x_interpolated[:, 2 * j] = x[:, j]

    return x_interpolated