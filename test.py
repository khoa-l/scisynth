from scisynth.latent.base import FAMILY_REGISTRY, generate_latent
from scisynth.observed.operators import (
    OPERATOR_REGISTRY,
    GaussianNoise,
    MCARMissingness,
    RandomSubsample,
)
from scisynth.spec import OperatorSpec, Spec

s = Spec(
    family="test",
    n=10,
    seed=1,
    operators=[OperatorSpec(name="noise", params={"sigma": 0.1})],
)
assert Spec.from_json(s.to_json()) == s
print(s.spec_hash)

spec = Spec(
    family="independent_features",
    family_params={"p": 4, "correlated": True},
    n=100,
    seed=0,
)
latent = generate_latent(spec)
print(latent.X.shape, latent.family)


assert "independent_features" in FAMILY_REGISTRY


assert set(OPERATOR_REGISTRY) == {
    "gaussian_noise",
    "mcar_missingness",
    "random_subsample",
}
