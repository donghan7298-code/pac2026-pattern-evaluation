"""Explicit conversion from the reference palletizing_core grid representation."""
from .contracts import number


def reference_candidate(candidate, grid_mm, candidate_id=None):
    """Accept a reference Candidate object or its dictionary; output pallet-frame m."""
    if not number(grid_mm, positive=True): raise ValueError("grid_mm must be positive")
    def get(key): return candidate[key] if isinstance(candidate, dict) else getattr(candidate, key)
    orientation = get("o")
    if type(orientation) is not int or orientation not in (0, 1):
        raise ValueError("Reference side-lying orientations 2..5 are not supported by the upright core")
    return dict(candidate_id=str(candidate_id if candidate_id is not None else f"grid:{get('i')}:{get('j')}:{get('z')}:{orientation}"),
                position_m=[get("i")*grid_mm/1000., get("j")*grid_mm/1000., get("z")/1000.],
                yaw_deg=90*orientation)


def reference_box(box):
    """Convert measured reference Box dimensions in mm; load capacity stays in N."""
    def get(key): return box[key] if isinstance(box, dict) else getattr(box, key)
    orientations = get("orientations")
    if not orientations or any(type(o) is not int or o not in (0, 1) for o in orientations):
        raise ValueError("Explicitly resolve side-lying orientation constraints before conversion")
    return dict(box_id=get("box_id"), sku=get("type_id"), dimensions_m=[get(k)/1000. for k in ("w", "d", "h")],
                mass_kg=get("mass"), top_load_capacity_N=get("max_load"),
                allowed_yaw_deg=[90*o for o in orientations], visual_damage_observed=get("damaged"), measurement_valid=True)
