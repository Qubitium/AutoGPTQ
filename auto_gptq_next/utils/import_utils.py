from logging import getLogger
from typing import Optional

logger = getLogger(__name__)


def dynamically_import_QuantLinear(
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: Optional[bool] = None,
        use_marlin: bool = False,
):
    if use_triton:
        logger.info("Using tritonv2 for GPTQ")
        from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    else:
        if bits == 4 and use_marlin:
            from ..nn_modules.qlinear.qlinear_marlin import QuantLinear
        elif bits == 4 and not disable_exllama:  # only use exllama(v1) for packing
            from ..nn_modules.qlinear.qlinear_exllamav2 import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear


def dynamically_import_QuantLinear_for_packing(
        use_triton: bool,
        desc_act: bool,
        group_size: int,
        bits: int,
        disable_exllama: Optional[bool] = None,
        disable_exllamav2: bool = False,
        use_marlin: bool = False,
):
    if disable_exllama is None:
        disable_exllama = not disable_exllamav2

    return dynamically_import_QuantLinear(
        use_triton=use_triton,
        desc_act=desc_act,
        group_size=group_size,
        bits=bits,
        disable_exllama=disable_exllama,
        use_marlin=use_marlin,
    )
