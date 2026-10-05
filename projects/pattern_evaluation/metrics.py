"""Explainable static pattern measurements, not a dynamic collapse probability."""
import math

from pacdata.packing import G, bounds, dims


def pattern_metrics(pallet, boxes):
    capacity = pallet["length_m"]*pallet["width_m"]*pallet["max_height_m"]
    volume = sum(math.prod(b["dimensions_m"]) for b in boxes)
    mass = sum(b["mass_kg"] for b in boxes)
    centers = {b["box_id"]: [b["position_m"][a]+dims(b, b["yaw_deg"])[a]/2 for a in range(3)] for b in boxes}
    cog = [sum(b["mass_kg"]*centers[b["box_id"]][a] for b in boxes)/mass for a in range(3)] if mass else [pallet["length_m"]/2, pallet["width_m"]/2, 0.]
    by_id = {b["box_id"]: b for b in boxes}
    children = {b["box_id"]: [] for b in boxes}
    for b in boxes:
        if b["supporter_id"] != "FLOOR": children[b["supporter_id"]].append(b["box_id"])
    contacts = []; loads = []; quadrant = [0., 0., 0., 0.]
    for box in boxes:
        lo, hi = bounds(box); footprint = (hi[0]-lo[0])*(hi[1]-lo[1])
        parent = box["supporter_id"]
        blo, bhi = ([0., 0., 0.], [pallet["length_m"], pallet["width_m"], 0.]) if parent == "FLOOR" else bounds(by_id[parent])
        contact_lo = [max(lo[a], blo[a]) for a in (0, 1)]
        contact_hi = [min(hi[a], bhi[a]) for a in (0, 1)]
        area = math.prod(max(0., contact_hi[a]-contact_lo[a]) for a in (0, 1))
        subtree = []; queue = [box["box_id"]]
        while queue:
            identity = queue.pop(); subtree.append(by_id[identity]); queue.extend(children[identity])
        supported_mass = sum(b["mass_kg"] for b in subtree)
        subtree_cog = [sum(b["mass_kg"]*centers[b["box_id"]][a] for b in subtree)/supported_mass for a in range(3)]
        edge = min(subtree_cog[a]-contact_lo[a] for a in (0, 1))
        edge = min(edge, *(contact_hi[a]-subtree_cog[a] for a in (0, 1)))
        height = subtree_cog[2]-lo[2]
        contacts.append(dict(box_id=box["box_id"], supporter_id=parent, contact_area_m2=area,
            support_ratio=area/footprint, supported_mass_kg=supported_mass,
            supported_cog_m=subtree_cog, cog_contact_edge_margin_m=edge,
            rigid_tipping_acceleration_proxy_m_s2=G*max(0., edge)/max(height, 1e-9)))
        upper = box["upper_load_N"]; top_capacity = box["top_load_capacity_N"]
        loads.append(dict(box_id=box["box_id"], upper_load_N=upper, capacity_N=top_capacity,
                          margin_N=top_capacity-upper,
                          margin_ratio=(top_capacity-upper)/top_capacity if top_capacity else (1. if not upper else 0.)))
        if parent == "FLOOR":
            # Only a coarse load map: uniform pressure over each bottom box's footprint.
            for yi in (0, 1):
                for xi in (0, 1):
                    qlo = [xi*pallet["length_m"]/2, yi*pallet["width_m"]/2]
                    qhi = [(xi+1)*pallet["length_m"]/2, (yi+1)*pallet["width_m"]/2]
                    overlap = math.prod(max(0., min(hi[a], qhi[a])-max(lo[a], qlo[a])) for a in (0, 1))
                    quadrant[yi*2+xi] += supported_mass*G*overlap/footprint
    floor_area = sum(math.prod(dims(b, b["yaw_deg"])[:2]) for b in boxes if b["supporter_id"] == "FLOOR")
    return dict(box_count=len(boxes), volume_utilization=volume/capacity,
        floor_coverage_ratio=floor_area/(pallet["length_m"]*pallet["width_m"]),
        total_mass_kg=mass, pallet_mass_margin_kg=pallet["max_mass_kg"]-mass,
        max_height_m=max((bounds(b)[1][2] for b in boxes), default=0.),
        cog_m=cog, cog_normalized=[cog[0]/pallet["length_m"], cog[1]/pallet["width_m"], cog[2]/pallet["max_height_m"]],
        load_distribution=dict(method="UNIFORM_BOTTOM_FOOTPRINT_PROXY", quadrant_order=["x0_y0", "x1_y0", "x0_y1", "x1_y1"],
                               quadrant_load_N=quadrant, boxes=loads),
        stability=dict(method="RIGID_STATIC_FULL_SINGLE_SUPPORT", contacts=contacts,
            minimum_support_ratio=min((c["support_ratio"] for c in contacts), default=1.),
            minimum_contact_cog_margin_m=min((c["cog_contact_edge_margin_m"] for c in contacts), default=0.),
            collapse_probability=None, dynamic_validation="NOT_CHECKED"))
