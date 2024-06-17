import gc
from logging import getLogger

import accelerate
import torch
from accelerate.utils import find_tied_parameters
from tqdm import tqdm

from ..nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear
from ..nn_modules.qlinear.qlinear_marlin import _get_perms, unpack_qzeros
from ..quantization import FORMAT, BaseQuantizeConfig
from .marlin_sparse24_utils import repack_gptq_to_marlin_sparse24, repack_scales_to_marlin_sparse24
from .modeling_utils import recurse_getattr, recurse_setattr


logger = getLogger(__name__)


def prepare_model_for_marlin_load(
    model,
    quantize_config: BaseQuantizeConfig,
    quant_linear_class,
    torch_dtype,
    current_model_save_name,
    device_map,
    is_sparse24=False,
):
    # The model (e.g. model.safetensors) is already serialized in the Marlin format, load it directly.
    if quantize_config.format == FORMAT.MARLIN:
        model_save_name = current_model_save_name
        logger.info(f"Loading a GPTQ model, detected Marlin serialized format at {model_save_name}.")
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=False, is_sparse24=is_sparse24)
    else:
        # Loading the GPTQ checkpoint to do the conversion.
        # TODO: Avoid loading the model with wrong QuantLinear, and directly use
        # Marlin ones. The repacking can be done directly on the safetensors, just
        # as for AWQ checkpoints.
        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            dtype=torch_dtype,  # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
            checkpoint=current_model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )
        # Convert model to marlin, repacking weights into Marlin format.
        model = convert_to_marlin(model, quant_linear_class, quantize_config, repack=True, is_sparse24=is_sparse24)

        # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
        tied_params = find_tied_parameters(model)

        for weight_group in tied_params:
            for param_name in weight_group:
                if isinstance(recurse_getattr(model, param_name), torch.nn.Parameter):
                    recurse_setattr(
                        model,
                        param_name,
                        torch.nn.Parameter(recurse_getattr(model, param_name).clone()),
                    )
                else:
                    recurse_setattr(
                        model,
                        param_name,
                        recurse_getattr(model, param_name).clone(),
                    )

    return model


# Validate marlin support
def _validate_marlin_device_support() -> bool:
    """
    Validates if the current device is compatible for Marlin.
    ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

    Returns:
        bool: indicates if CUDA device is compatible for Marlin
    """
    return torch.cuda.get_device_capability()[0] >= 8


# Adapted from https://github.com/rib-2/marlin/tree/conversion
def _validate_marlin_compatibility(cfg: BaseQuantizeConfig):
    if cfg.bits != 4 and cfg.bits != 8:
        return f"The quantized model uses a bitwidth different than 4 or 8 (found {cfg.bits})"
    if cfg.group_size != 128 and cfg.group_size != -1:
        return "The quantized model uses a group size that is not 128 or -1 (found quantization_config.group_size)"
    if not cfg.sym:
        return "The quantized model uses asymmetric quantization"
    # if cfg.desc_act:
    #     return "The quantized model uses act-order (also called desc-act) scheme"
    return None


@torch.no_grad()
def convert_to_marlin_original(
    model, model_quantlinear, quantization_config: BaseQuantizeConfig, repack: bool, strict: bool = False
):
    """
    Converts GPTQ-packed weights to the Marlin format. This assumes that the model already meets Marlin kernel constraints.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the Marlin's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with Marlin original (dense) kernel..."
    else:
        # TODO: load directly Marlin QuantLinear.
        message = "Overriding QuantLinear layers to use Marlin's QuantLinear"

    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when
        # loading weights from checkpoints holding zero bias.
        with torch.device("meta"):
            new_module = MarlinQuantLinear(
                bits=4,
                group_size=module.group_size,
                infeatures=module.infeatures,
                outfeatures=module.outfeatures,
                bias=module.bias is not None,
                is_sparse24=False,
                trainable=False,
            )

        # workspace is never in the state_dict, thus we need to allocate it manually.
        new_module.workspace = torch.zeros(module.outfeatures // 128 * 16, dtype=torch.int, device=module.device)

        # Dequantize the weight.
        if repack:
            import autogptq_marlin_cuda

            marlin_repacked_weight = autogptq_marlin_cuda.gptq_repack(module.qweight)

            if strict:
                dequantized_qzeros = unpack_qzeros(module.qzeros)

                if not torch.all(dequantized_qzeros == 8):
                    raise ValueError(
                        "Marlin kernel is compatible only with checkpoints using symmetric quantization."
                        "Found non-symmetric quantization for the weight {name}."
                    )

            _, _scale_perm, _scale_perm_single = _get_perms()

            s = module.scales.data.clone()
            if module.group_size != module.infeatures:
                s = s.reshape((1, -1))
                s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            else:
                s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
            s = s.reshape((-1, module.outfeatures)).contiguous()

            new_module.B = marlin_repacked_weight
            new_module.s = s
            new_module.bias = module.bias

            new_module = new_module.to(module.device)

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del module
        if repack:
            del marlin_repacked_weight
        gc.collect()

    # Set quantization config to be Marlin.
    quantization_config.format = FORMAT.MARLIN

    return model


@torch.no_grad()
def convert_to_marlin_sparse24(
    model, model_quantlinear, quantization_config: BaseQuantizeConfig, repack: bool, strict: bool = False
):
    """
    Converts GPTQ-packed weights to the Marlin format. This assumes that the model already meets Marlin kernel constraints.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the Marlin's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with Marlin_Sparse24 (sparse) kernel..."
    else:
        # TODO: load directly Marlin QuantLinear.
        message = "Overriding QuantLinear layers to use Marlin's QuantLinear..."

    for name, module in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
        if not isinstance(module, model_quantlinear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        layer_name = name[len(parent_name) + 1 :]

        # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when
        # loading weights from checkpoints holding zero bias.
        with torch.device("meta"):
            new_module = MarlinQuantLinear(
                bits=quantization_config.bits,
                group_size=module.group_size,
                infeatures=module.infeatures,
                outfeatures=module.outfeatures,
                bias=module.bias is not None,
                is_sparse24=True,
                trainable=False,
            )

        # workspace is never in the state_dict, thus we need to allocate it manually.
        new_module.workspace = torch.zeros(module.outfeatures // 128 * 16, dtype=torch.int, device=module.device)

        # Dequantize the weight.
        if repack:
            marlin_sparse24_weight, marlin_sparse24_meta, marlin_w_ref = repack_gptq_to_marlin_sparse24(
                module.qweight,
                module.scales,
                module.infeatures,
                module.outfeatures,
                quantization_config.bits,
                module.group_size,
            )

            if strict:
                dequantized_qzeros = unpack_qzeros(module.qzeros)

                if not torch.all(dequantized_qzeros == 8):
                    raise ValueError(
                        "Marlin_Sparse24 kernel is compatible only with checkpoints using symmetric quantization."
                        "Found non-symmetric quantization for the weight {name}."
                    )

            marlin_sparse24_scales = repack_scales_to_marlin_sparse24(
                module.scales, quantization_config.bits, module.group_size, module.infeatures, module.outfeatures
            )

            new_module.B_24 = marlin_sparse24_weight
            new_module.B_meta = marlin_sparse24_meta.resize_(
                marlin_sparse24_meta.shape[1] // 2, marlin_sparse24_meta.shape[0] * 2
            )
            # new_module.B_ref = marlin_w_ref
            new_module.s = marlin_sparse24_scales
            new_module.bias = module.bias

            new_module = new_module.to(module.device)

        # Save to parent.
        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, layer_name, new_module)

        # Free cuda memory.
        del module
        if repack:
            del marlin_sparse24_weight
            del marlin_sparse24_meta
            del marlin_w_ref
            del marlin_sparse24_scales
        gc.collect()

    # Set quantization config to be Marlin_Sparse24
    quantization_config.checkpoint_format = FORMAT.MARLIN_SPARSE24

    return model


@torch.no_grad()
def convert_to_marlin(
    model,
    model_quantlinear,
    quantization_config: BaseQuantizeConfig,
    repack: bool,
    strict: bool = False,
    is_sparse24: bool = False,
):
    if is_sparse24:
        return convert_to_marlin_sparse24(model, model_quantlinear, quantization_config, repack, strict)
    else:
        return convert_to_marlin_original(model, model_quantlinear, quantization_config, repack, strict)
