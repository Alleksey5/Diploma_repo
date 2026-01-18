# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections.abc import Iterable

COLOR_RED = "91m"
COLOR_GREEN = "92m"
COLOR_YELLOW = "93m"


def colorful_print(fn_print, color=COLOR_RED):
    """A decorator to print text in the specified terminal color by wrapping the given print function."""

    def actual_call(*args, **kwargs):
        print(f"\033[{color}", end="")
        fn_print(*args, **kwargs)
        print("\033[00m", end="")

    return actual_call


prRed = colorful_print(print, color=COLOR_RED)
prGreen = colorful_print(print, color=COLOR_GREEN)
prYellow = colorful_print(print, color=COLOR_YELLOW)


def clever_format(nums, format="%.2f"):
    """Formats numbers into human-readable strings with units (K for thousand, M for million, etc.)."""
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    return clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)


if __name__ == "__main__":
    prRed("hello", "world")
    prGreen("hello", "world")
    prYellow("hello", "world")

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


def _count_rnn_cell(input_size, hidden_size, bias=True):
    """Calculate the total operations for an RNN cell given input size, hidden size, and optional bias."""
    total_ops = hidden_size * (input_size + hidden_size) + hidden_size
    if bias:
        total_ops += hidden_size * 2

    return total_ops


def count_rnn_cell(m: nn.RNNCell, x: torch.Tensor, y: torch.Tensor):
    """Counts the total RNN cell operations based on input tensor, hidden size, bias, and batch size."""
    total_ops = _count_rnn_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_gru_cell(input_size, hidden_size, bias=True):
    """Counts the total operations for a GRU cell based on input size, hidden size, and bias configuration."""
    total_ops = 0
    # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2

    # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    # r hadamard : r * (~)
    total_ops += hidden_size

    # h' = (1 - z) * n + z * h
    # hadamard hadamard add
    total_ops += hidden_size * 3

    return total_ops


def count_gru_cell(m: nn.GRUCell, x: torch.Tensor, y: torch.Tensor):
    """Calculates and updates the total operations for a GRU cell in a mini-batch during inference."""
    total_ops = _count_gru_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def _count_lstm_cell(input_size, hidden_size, bias=True):
    """Counts LSTM cell operations during inference based on input size, hidden size, and bias configuration."""
    total_ops = 0

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = o * \tanh(c') \\
    total_ops += hidden_size

    return total_ops


def count_lstm_cell(m: nn.LSTMCell, x: torch.Tensor, y: torch.Tensor):
    """Counts and updates the total operations for an LSTM cell in a mini-batch during inference."""
    total_ops = _count_lstm_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_rnn(m: nn.RNN, x, y):
    """Calculate and update the total number of operations for each RNN cell in a given batch."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    elif m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_rnn_cell(hidden_size * 2, hidden_size, bias) * 2
            if m.bidirectional
            else _count_rnn_cell(hidden_size, hidden_size, bias)
        )
    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_gru(m: nn.GRU, x, y):
    """Calculates total operations for a GRU layer, updating the model's operation count based on batch size."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    elif m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_gru_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_gru_cell(input_size, hidden_size, bias)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_gru_cell(hidden_size * 2, hidden_size, bias) * 2
            if m.bidirectional
            else _count_gru_cell(hidden_size, hidden_size, bias)
        )
    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_lstm(m: nn.LSTM, x, y):
    """Calculate total operations for LSTM layers, including bidirectional, updating model's total operations."""
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if isinstance(x[0], PackedSequence):
        batch_size = torch.max(x[0].batch_sizes)
        num_steps = x[0].batch_sizes.size(0)
    elif m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias)

    for _ in range(num_layers - 1):
        total_ops += (
            _count_lstm_cell(hidden_size * 2, hidden_size, bias) * 2
            if m.bidirectional
            else _count_lstm_cell(hidden_size, hidden_size, bias)
        )
    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    m.total_ops += torch.DoubleTensor([int(total_ops)])

    # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import warnings

import numpy as np
import torch


def l_prod(in_list):
    """Compute the product of all elements in the input list."""
    res = 1
    for _ in in_list:
        res *= _
    return res


def l_sum(in_list):
    """Calculate the sum of all numerical elements in a list."""
    return sum(in_list)


def calculate_parameters(param_list):
    """Calculate the total number of parameters in a list of tensors using the product of their shapes."""
    return sum(torch.DoubleTensor([p.nelement()]) for p in param_list)


def calculate_zero_ops():
    """Initializes and returns a tensor with all elements set to zero."""
    return torch.DoubleTensor([0])


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    """Calculate FLOPs for a Conv2D layer using input/output sizes, kernel size, groups, and the bias flag."""
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def calculate_conv(bias, kernel_size, output_size, in_channel, group):
    """Calculate FLOPs for convolutional layers given bias, kernel size, output size, in_channels, and groups."""
    warnings.warn("This API is being deprecated.")
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size + bias)])


def calculate_norm(input_size):
    """Compute the L2 norm of a tensor or array based on its input size."""
    return torch.DoubleTensor([2 * input_size])


def calculate_relu_flops(input_size):
    """Calculates the FLOPs for a ReLU activation function based on the input tensor's dimensions."""
    return 0


def calculate_relu(input_size: torch.Tensor):
    """Convert an input tensor to a DoubleTensor with the same value (deprecated)."""
    warnings.warn("This API is being deprecated")
    return torch.DoubleTensor([int(input_size)])


def calculate_softmax(batch_size, nfeatures):
    """Compute FLOPs for a softmax activation given batch size and feature count."""
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])


def calculate_avgpool(input_size):
    """Calculate the average pooling size for a given input tensor."""
    return torch.DoubleTensor([int(input_size)])


def calculate_adaptive_avg(kernel_size, output_size):
    """Calculate FLOPs for adaptive average pooling given kernel size and output size."""
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])


def calculate_upsample(mode: str, output_size):
    """Calculate the operations required for various upsample methods based on mode and output size."""
    total_ops = output_size
    if mode == "bicubic":
        total_ops *= 224 + 35
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "linear":
        total_ops *= 5
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return torch.DoubleTensor([int(total_ops)])


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for given input feature and number of elements."""
    return torch.DoubleTensor([int(in_feature * num_elements)])


def counter_matmul(input_size, output_size):
    """Calculate the total number of operations for matrix multiplication given input and output sizes."""
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]


def counter_mul(input_size):
    """Calculate the total number of operations for element-wise multiplication given the input size."""
    return input_size


def counter_pow(input_size):
    """Computes the total scalar multiplications required for power operations based on input size."""
    return input_size


def counter_sqrt(input_size):
    """Calculate the total number of scalar operations required for a square root operation given an input size."""
    return input_size


def counter_div(input_size):
    """Calculate the total number of scalar operations for a division operation given an input size."""
    return input_size

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging

import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1

def count_parameters(m, x, y):
    """Calculate and return the total number of learnable parameters in a given PyTorch model."""
    m.total_params[0] = calculate_parameters(m.parameters())


def zero_ops(m, x, y):
    """Incrementally add zero operations to the model's total operations count."""
    m.total_ops += calculate_zero_ops()


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    """Calculate and add the number of convolutional operations (FLOPs) for a ConvNd layer to the model's total ops."""
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
    )
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )


def count_convNd_ver2(m: _ConvNd, x, y: torch.Tensor):
    """Calculates and updates total operations (FLOPs) for a convolutional layer in a PyTorch model."""
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros(y.size()[:1] + y.size()[2:]).numel()
    # # Cout x Cin x Kw x Kh
    # kernel_ops = m.weight.nelement()
    # if m.bias is not None:
    #     # Cout x 1
    #     kernel_ops += + m.bias.nelement()
    # # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    # m.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])
    m.total_ops += calculate_conv(m.bias.nelement(), m.weight.nelement(), output_size)


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    """Calculate and add the FLOPs for a batch normalization layer, including elementwise and affine operations."""
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if getattr(m, "affine", False) or getattr(m, "elementwise_affine", False):
        flops *= 2
    m.total_ops += flops


# def count_layer_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


# def count_instance_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


def count_prelu(m, x, y):
    """Calculate and update the total operation counts for a PReLU layer using input element number."""
    x = x[0]

    nelements = x.numel()
    if not m.training:
        m.total_ops += calculate_relu(nelements)


def count_relu(m, x, y):
    """Calculate and update the total operation counts for a ReLU layer."""
    x = x[0]
    m.total_ops += calculate_relu_flops(list(x.shape))


def count_softmax(m, x, y):
    """Calculate and update the total operation counts for a Softmax layer in a PyTorch model."""
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += calculate_softmax(batch_size, nfeatures)


def count_avgpool(m, x, y):
    """Calculate and update the total number of operations (FLOPs) for an AvgPool layer based on the output elements."""
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)


def count_adap_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AdaptiveAvgPool layer using kernel and element counts."""
    kernel = torch.div(torch.DoubleTensor([*(x[0].shape[2:])]), torch.DoubleTensor([*(y.shape[2:])]))
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)


# TODO: verify the accuracy
def count_upsample(m, x, y):
    """Update total operations counter for upsampling layers based on the mode used."""
    if m.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning(f"mode {m.mode} is not implemented yet, take it a zero op")
        m.total_ops += 0
    else:
        x = x[0]
        m.total_ops += calculate_upsample(m.mode, y.nelement())


# nn.Linear
def count_linear(m, x, y):
    """Counts total operations for nn.Linear layers using input and output element dimensions."""
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

default_dtype = torch.float64

register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
    nn.Sequential: zero_ops,
    nn.PixelShuffle: zero_ops,
    nn.SyncBatchNorm: count_normalization,
}


def profile_origin(model, inputs, custom_ops=None, verbose=True, report_missing=False):
    """Profiles a PyTorch model's operations and parameters, applying either custom or default hooks."""
    handler_collection = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        verbose = True

    def add_hooks(m):
        if list(m.children()):
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params"):
            logging.warning(
                f"Either .total_ops or .total_params is already defined in {str(m)}. "
                "Be careful, it might change your code's behavior."
            )

        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))
        m.register_buffer("total_params", torch.zeros(1, dtype=default_dtype))

        for p in m.parameters():
            m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
        else:
            if m_type not in types_collection and report_missing:
                prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params.")

        if fn is not None:
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        types_collection.add(m_type)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if list(m.children()):  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if list(m.children()):
            continue
        if "total_ops" in m._buffers:
            m._buffers.pop("total_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_ops, total_params


def profile(
    model: nn.Module,
    inputs,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
):
    """Profiles a PyTorch model, returning total operations, parameters, and optionally layer-wise details."""
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        """Registers hooks to a neural network module to track total operations and parameters."""
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Customize rule {fn.__qualname__}() {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Register {fn.__qualname__}() for {m_type}.")
        else:
            if m_type not in types_collection and report_missing:
                prRed(f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params.")

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t"):
        """Recursively counts the total operations and parameters of the given PyTorch module and its submodules."""
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params
        # print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params