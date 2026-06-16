"""CoSMo3D adapter (placeholder until wired to its inference API).

Will run CoSMo3D's SigLIP open-vocab segmenter on each CoMPaT instance and save
per-point predictions aligned to the GT point order. Run under the cosmo3d env.
"""

from __future__ import annotations


def run_subset(keys, by_key, *, overwrite: bool = False) -> int:
    raise NotImplementedError(
        "cosmo3d adapter not implemented yet; see benchmark TODO"
    )
