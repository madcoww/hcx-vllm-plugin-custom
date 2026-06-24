from vllm import ModelRegistry

def register():
    from transformers import AutoConfig
    from .configuration_hyperclovax import HyperCLOVAXConfig
    try:
        AutoConfig.register("hyperclovax", HyperCLOVAXConfig)
    except ValueError:
        pass  # already registered

    from .vllm_hyperclovax import HyperCLOVAXForCausalLM

    if "HyperCLOVAXForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("HyperCLOVAXForCausalLM", HyperCLOVAXForCausalLM)
