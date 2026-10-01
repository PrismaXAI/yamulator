"""Make checkpoints from the training fork loadable in the inference venv (lerobot 0.5.1), INCLUDING relative-action ones.

The fork registers RelativeActionsProcessorStep as "relative_actions_processor"; lerobot 0.5.1 ships the identical
class (same to_relative/to_absolute maths, same _build_mask, same cached-state pairing with
"absolute_actions_processor", same factory re-link) but registers it as "delta_actions_processor". Registering the
real class under the fork's name replaces the old no-op shim (which asserted enabled=false), so a checkpoint saved
with use_relative_actions=true now runs: pre = ... to_batch -> relative (caches the RAW state) -> normalize ...,
post = unnormalize -> absolute (adds the cached raw state on the masked dims) -> device.
NOTE: with action_names=None the mask is all-True (grippers become relative too); a correct relative checkpoint
must carry action_names so exclude_joints=["gripper"] applies.
"""
from lerobot.processor import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import (  # noqa: F401
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)

try:
    ProcessorStepRegistry.get("relative_actions_processor")
except KeyError:
    ProcessorStepRegistry.register(name="relative_actions_processor")(RelativeActionsProcessorStep)
