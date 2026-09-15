"""Fixed model, optimizer, and input identities for the recorded comparison."""
from dataclasses import dataclass
from .config import ATTENTION_ONLY_PROFILE, SDM_ONLY_PROFILE, CANONICAL_PROFILE, MemoryGeometry, ModelProfile
from access_curriculum.schedule import CELLS, HEADLINE_ARMS, WIKI_STEPS

CAMPAIGN_ID = "access-curriculum-wikitext103-seed0-v1"
TOTAL_STEPS = WIKI_STEPS
PASSES = 3
WARMUP_STEPS = 540
EXPECTED_MANIFEST_SHA256 = "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6"
EXPECTED_PAYLOAD_SHA256 = "f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e"
EXPECTED_INVENTORY_SHA256 = "efa0a0b857d184dccdecf6adafda5d646ec45bae91f01a15cd1165f9fe9ad6b3"
EXPECTED_COMMON_SDM_SHA256 = {
    "BBBBBBBB": "47f42b13c4b3cd3ebcb7a01b42b96d15d2d482a00ffd2debc1cab1cbc9b654fa",
    "BBBBBBBA": "c5ebb90d8a51e7a3ff584d77ac8d63c9f3e3a995571469116e7068b5931f3241",
}
GEOMETRY = MemoryGeometry("p2_c32_r8_w8", factors=2, codebook_size=32, reads=8, writes=8)

@dataclass(frozen=True)
class Arm:
    identifier: str
    label: str
    profile: ModelProfile
    router: str
    micro_batch_size: int
    expected_trainable_parameters: int

    @property
    def has_sdm(self):
        return bool(self.profile.memory_layers)

ARM_ORDER = HEADLINE_ARMS


def arm_definition(name):
    dense = name == "dense_attention"
    hybrid = name.startswith("hybrid_")
    native = name.removeprefix("hybrid_") == "native_k8"
    profile = ATTENTION_ONLY_PROFILE if dense else CANONICAL_PROFILE if hybrid else SDM_ONLY_PROFILE
    router = "not_applicable" if dense else "unmodified" if native else "residualized_read_write"
    parameters = (14_965_120 if dense else
                  (15_029_660 if native else 15_087_452) if hybrid else
                  (15_038_880 if native else 15_104_928))
    return Arm(name, name.replace("_", " "), profile, router, 8 if dense else 1, parameters)


ARMS = {name: arm_definition(name) for name in (*HEADLINE_ARMS, *CELLS)}
