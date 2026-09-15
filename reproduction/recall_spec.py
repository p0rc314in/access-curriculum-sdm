"""Historical Adaptive Recall protocol, including its original BF16 AdamW."""
from reproduction.spec import ARMS, ARM_ORDER, GEOMETRY
from access_curriculum.schedule import stages

PROTOCOL_ID = "adaptive-recall-seed102337-v1"
MANIFEST_SHA256 = "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800"
COMMON_SDM_SHA256 = "c02d9d732e5fe8ad539ac74076acca10ebfce546e1cd246f305d5f1723d689f4"
COMMON_BY_LAYOUT = {
    "BBBBBBBB": COMMON_SDM_SHA256,
    "BBBBBBBA": "6af00f5df2bae992e8a6cb3a0238ad3c1bef3567a72778e862540242a3560c88",
}
STEPS = 30_000
STREAM_SEED = 102337
BATCH_SIZE = 32
EVAL_EXAMPLES = 2048
EVAL_BATCH_SIZE = 8
# WikiText has two untied 50,257 x 128 lexical tables. Recall replaces them
# with a 14,592-parameter semantic adapter and a 192 x 128 output projection.
PARAMETERS = {name: arm.expected_trainable_parameters - 2 * 50_257 * 128 + 14_592 + 192 * 128
              for name, arm in ARMS.items()}


def learning_rate(arm: str, step: int) -> float:
    if not 1 <= step <= STEPS:
        raise ValueError("step lies outside the recorded Recall schedule")
    return 3e-4 * min(step / 100, 1)


def config(arm):
    return {
        "benchmark": "adaptive_recall", "protocol_id": PROTOCOL_ID,
        "evidence_tier": "small_wikitext_plus_recall", "arm": arm,
        "model": ARMS[arm].profile.as_dict(), "router": ARMS[arm].router,
        "reads": 8, "writes": 8, "logical_rows": 1024, "seed": 0,
        "steps": STEPS, "stream_seed": STREAM_SEED, "batch_size": BATCH_SIZE,
        "micro_batch_size": 32, "eval_examples": EVAL_EXAMPLES,
        "eval_batch_size": EVAL_BATCH_SIZE, "learning_rate": 3e-4,
        "warmup_steps": 100,
        "schedule": "warmup_constant",
        "adam_betas": [.9, .95], "weight_decay": .01, "gradient_clip": 1.,
        "activation_dtype": "bfloat16", "optimizer": "released_lingua_adamw",
        "optimizer_parameter_and_moment_dtype": "bfloat16",
        "access_stages": [list(row) for row in stages(arm, "recall")],
    }
