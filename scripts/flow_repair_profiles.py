"""Explicit identity labels for flow-only repair experiments, never admission."""

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class FlowRepairProfile:
    experiment: str
    kind: str
    flow_kind: str
    objective: str
    force_loss: bool

    @property
    def prefix(self):
        return self.experiment.replace("-", "_")

    def status(self, suffix):
        return self.prefix + "_" + suffix


PROFILES = {
    "FC-P028": FlowRepairProfile(
        "FC-P028", "FC_P028_FLOW_ROLLOUT_REPAIR",
        "FC_P028_FLOW_ROLLOUT_CHECKPOINT", "ten_equal_masked_normalized_state_MSE", False,
    ),
    "FC-P029": FlowRepairProfile(
        "FC-P029", "FC_P029_CONTROL_AWARE_FLOW_REPAIR",
        "FC_P029_CONTROL_AWARE_FLOW_CHECKPOINT",
        "half_parent_scaled_field_MSE_plus_half_parent_scaled_four_force_MSE", True,
    ),
}


def repair_profile(experiment="FC-P028"):
    if type(experiment) is not str or experiment not in PROFILES:
        raise ValueError("unsupported flow repair experiment")
    return PROFILES[experiment]


def validate_p029_result_binding(result, protocol, manifest):
    """Cross-bind saved effective scales; never infer them from a prior experiment."""
    scales = protocol.get("fixed_scales")
    if (not isinstance(scales, dict) or set(scales) != {"field", "force"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in scales.values())):
        raise ValueError("P029 finite positive fixed scales required")
    digest = protocol.get("scales_receipt_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("P029 scales receipt SHA required")
    base_protocol = {k: v for k, v in protocol.items()
                     if k not in ("fixed_scales", "scales_receipt_sha256")}
    if (result.get("fixed_scales") != scales or manifest.get("fixed_scales") != scales
            or manifest.get("scales_receipt_sha256") != digest
            or result.get("source_spec", {}).get("scales_receipt", {}).get("sha256") != digest
            or result.get("protocol") != base_protocol
            or result.get("source_spec", {}).get("protocol") != base_protocol):
        raise ValueError("P029 result/protocol/manifest/approved scales binding differs")
