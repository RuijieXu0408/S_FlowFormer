def build_flowformer(cfg, device):
    from .transformer import FlowFormer
    return FlowFormer(cfg["latentcostformer"], device=device)
