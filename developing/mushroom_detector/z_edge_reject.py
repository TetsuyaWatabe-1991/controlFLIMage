# -*- coding: utf-8 -*-
"""Z-edge force-reject helpers (no heavy imaging dependencies)."""

from __future__ import annotations

# Force-reject spines whose head Z falls in this many slices at each stack end.
# Example: n_z=15 and z_edge_reject_slices=1 rejects Z=0 and Z=14.
Z_EDGE_REJECT_SLICES = 0


def z_index_in_edge_reject_slices(
    z_idx: int | float,
    n_z: int,
    n_edge_slices: int = Z_EDGE_REJECT_SLICES,
) -> bool:
    """
    True when Z is within ``n_edge_slices`` of either end of the stack.

    For n_z=15 and n_edge_slices=1, rejects indices 0 and 14.
    ``n_edge_slices <= 0`` disables the check.
    """
    n_edge = int(n_edge_slices)
    if n_edge <= 0 or int(n_z) <= 0:
        return False
    z = int(round(float(z_idx)))
    return z < n_edge or z >= int(n_z) - n_edge


def ini_excluded_for_auto_rating(
    auto_rating: int | None,
    *,
    manual_excluded: int | None = None,
    force_excluded: bool = False,
    auto_accept_min_rating: int = 4,
) -> int:
    """
    Decide ini excluded flag (0=accepted, 1=rejected/pending).

    Manual accept/reject (manual_excluded) always wins. Batch default: auto_rating>=4
    is accepted; rating<3 is rejected; rating 3 or unrated stays pending (excluded=1).
    """
    if force_excluded:
        return 1
    if manual_excluded is not None:
        return int(manual_excluded)
    if auto_rating is not None and int(auto_rating) >= int(auto_accept_min_rating):
        return 0
    if auto_rating is not None and int(auto_rating) < 3:
        return 1
    return 1
