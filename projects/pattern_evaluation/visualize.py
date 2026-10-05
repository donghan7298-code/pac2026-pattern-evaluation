"""Dependency-free top view, isometric 3D view and a simulator scene export."""
import copy
from html import escape

from pacdata.packing import apply_placement, bounds
from .planner import prepare_observation


def scene_data(observation, result):
    obs = prepare_observation(observation)
    boxes = copy.deepcopy(obs["placed_boxes"])
    action = result["action"]; proposed = None
    if action["type"] in ("PLACE_CURRENT", "RETRIEVE_BUFFER"):
        proposed = action["box_id"]
        box = obs["current_box"] if action["type"] == "PLACE_CURRENT" else next(b for b in obs["buffer_boxes"] if b["box_id"] == proposed)
        boxes = apply_placement(obs["pallet"], boxes, box, action["candidate"])
    return dict(schema_version="ahead-scene-1.0", coordinate_frame="pallet", units="m", pallet=obs["pallet"],
                boxes=[dict(b, status="PROPOSED" if b["box_id"] == proposed else "OBSERVED") for b in boxes],
                request_id=obs["request_id"], robot_trajectory=None)


def render_svg(observation, result):
    scene = scene_data(observation, result); p = scene["pallet"]; boxes = scene["boxes"]
    L, W, H = p["length_m"], p["width_m"], p["max_height_m"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="670" viewBox="0 0 1100 670" role="img">',
             '<title>Pallet placement: top view and isometric 3D view</title>',
             '<rect width="1100" height="670" fill="#f4f7fb"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#16324a} .small{font-size:14px} .label{font-size:12px}</style>']
    def text(x, y, value, size=18):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')
    def polygon(points, color, opacity=1):
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(f'<polygon points="{coords}" fill="{color}" fill-opacity="{opacity}" stroke="#35516b" stroke-width="1.2"/>')
    text(34, 40, "AHEAD / Pattern evaluation", 26)
    text(34, 66, f"{result['action']['type']} | {result.get('robot_feasibility', 'NOT_CHECKED')}", 15)
    text(34, 111, "Top view / pallet frame", 20)
    text(578, 111, "3D geometry / isometric projection", 20)
    s = min(450/L, 285/W); ox, oy = 40, 432
    def top(x, y): return ox+x*s, oy-y*s
    polygon([top(0, 0), top(L, 0), top(L, W), top(0, W)], "#e0e8f0")
    iso_scale = min(420/(L+W), 280/(H+.35*(L+W)))
    def iso(x, y, z): return 800+(x-y)*iso_scale, 418-(x+y)*.35*iso_scale-z*iso_scale
    polygon([iso(0, 0, 0), iso(L, 0, 0), iso(L, W, 0), iso(0, W, 0)], "#d7e1eb")
    for b in sorted(boxes, key=lambda b: b["position_m"][2]):
        lo, hi = bounds(b); color = "#edaa4b" if b["status"] == "PROPOSED" else "#83b7d6"
        polygon([top(lo[0], lo[1]), top(hi[0], lo[1]), top(hi[0], hi[1]), top(lo[0], hi[1])], color, .85)
        x, y = top(lo[0], hi[1]); text(x+3, y+14, b["box_id"][:14], 11)
    for b in sorted(boxes, key=lambda b: (-(b["position_m"][0]+b["position_m"][1]), b["position_m"][2])):
        lo, hi = bounds(b); x, y, z = lo; xx, yy, zz = hi
        color = "#edaa4b" if b["status"] == "PROPOSED" else "#83b7d6"
        polygon([iso(x,y,z), iso(xx,y,z), iso(xx,y,zz), iso(x,y,zz)], color, .72)
        polygon([iso(x,y,z), iso(x,yy,z), iso(x,yy,zz), iso(x,y,zz)], color, .9)
        polygon([iso(x,y,zz), iso(xx,y,zz), iso(xx,yy,zz), iso(x,yy,zz)], color)
    text(35, 465, "Blue: observed boxes    Amber: proposed placement    Axes: x / y on pallet, z upward", 14)
    m = result.get("metrics", {}).get("after", {})
    if m:
        text(35, 503, f"Volume utilization: {m['volume_utilization']:.2%}     Floor coverage: {m['floor_coverage_ratio']:.2%}", 17)
        text(35, 533, f"CoG [m]: {', '.join(f'{x:.3f}' for x in m['cog_m'])}     Support: {m['stability']['minimum_support_ratio']:.0%}", 17)
    d = result.get("diagnostics", {})
    text(35, 565, f"Valid candidates: {d.get('valid_candidates', 0)}     Detailed rollouts: {d.get('detailed_rollouts', 0)}     Evaluation: {result.get('decision_ms', 0):.1f} ms", 16)
    text(35, 610, "Geometry preview only. Motion, friction and dynamic collapse require external simulation.", 14)
    text(35, 638, f"Request: {scene['request_id']}", 12)
    parts.append("</svg>")
    return "\n".join(parts)
