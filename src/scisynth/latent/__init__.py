# Import built-in families so their @register_family decorators run.
from scisynth.latent import families  # noqa: F401
from scisynth.latent.base import FAMILY_REGISTRY, LatentData, generate, register_family

__all__ = ["FAMILY_REGISTRY", "LatentData", "generate", "register_family"]
