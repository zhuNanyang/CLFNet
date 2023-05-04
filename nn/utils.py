import torch
import GPUtil


import numpy as np
import torch.nn as nn
import torch.nn.init as init
import logging
logger = logging.getLogger(__name__)
def is_model_parallel(model: torch.nn.Module) -> bool:
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False


def save_model(model, model_path, model_name):
    """存储不含有显卡信息的state_dict或model"""
    if is_model_parallel(model):
        model = model.module
    model_path = model_path + "/" + f"{model_name}.pkl"
    torch.save(model, model_path)


def save_checkpoint(model_state: dict, model_path, model_name: str) -> None:
    """
    保存模型的权重和训练状态(包括epoch step optimizer lr performance等)
    保存训练状态的目的是为了恢复训练(和fine-tune不同)
    """
    model_state_path = model_path + "/" + f"{model_name}_checkpoint.pth"
    torch.save(model_state, model_state_path)


def move_data_to_device(obj, device: torch.device, non_blocking=False):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, dict):
        return {
            key: move_data_to_device(value, device, non_blocking)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [move_data_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(
            *[move_data_to_device(item, device, non_blocking) for item in obj]
        )
    elif isinstance(obj, tuple):
        return tuple([move_data_to_device(item, device, non_blocking) for item in obj])
    else:
        return obj


def auto_choose_gpu():
    gpu_ids = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.5, maxMemory=0.5)
    logger.info(f"Get all available gpu ids {gpu_ids} (load & memory usage <= 50%)")
    gpu_id = GPUtil.getFirstAvailable(order="memory")[0]
    logger.info(f"Choose gpu:{gpu_id} with the minimal memory usage")
    return gpu_ids, gpu_id


def seq_len_to_mask(seq_len, max_len=None):
    """
    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert (
            len(np.shape(seq_len)) == 1
        ), f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert (
            seq_len.dim() == 1
        ), f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    return mask



def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.
     # 为什么要进行初始化： csdn 华仔168168  pytorch默认参数初始化以及自定义参数初始化
    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == "xavier_uniform":
        init_method = init.xavier_uniform_
    elif initial_method == "xavier_normal":
        init_method = init.xavier_normal_
    elif initial_method == "kaiming_normal" or initial_method == "msra":
        init_method = init.kaiming_normal_
    elif initial_method == "kaiming_uniform":
        init_method = init.kaiming_uniform_
    elif initial_method == "orthogonal":
        init_method = init.orthogonal_
    elif initial_method == "sparse":
        init_method = init.sparse_
    elif initial_method == "normal":
        init_method = init.normal_
    elif initial_method == "uniform":
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv1d)
            or isinstance(m, nn.Conv3d)
        ):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif (
            m is not None
            and hasattr(m, "weight")
            and hasattr(m.weight, "requires_grad")
        ):
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)
