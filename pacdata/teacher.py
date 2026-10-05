"""Teacher scores use only policy observations, never oracle future order."""
import math
from .packing import dims

FEATURE_NAMES=(['candidate_x','candidate_y','candidate_z','yaw_90','box_l','box_w','box_h','box_mass',
    'post_cog_x','post_cog_y','post_cog_z','state_volume_ratio','placed_count','verified_preview_count',
    'unordered_hint_count','unseen_count','sensor_sigma','remaining_nominal_volume_ratio']+
    [f'preview_{i}_{name}' for i in range(12) for name in ['l','w','h','mass','present']]+
    [f'post_height_{y}_{x}' for y in range(5) for x in range(5)])


def features(obs,candidate):
    p=obs['pallet'];b=obs['current_box'];d=dims(b,candidate['yaw_deg']);xyz=candidate['position_m']
    limits=[p['length_m'],p['width_m'],p['max_height_m']];volume=math.prod(limits)
    boxes=list(obs['placed_boxes'])+[dict(b,position_m=xyz,yaw_deg=candidate['yaw_deg'])]
    total=sum(box['mass_kg'] for box in boxes)
    cog=[sum(box['mass_kg']*(box['position_m'][a]+dims(box,box.get('yaw_deg',0))[a]/2) for box in boxes)/total for a in range(3)]
    nominal=sum(n*math.prod(obs['catalog'][sku]['dimensions_m']) for sku,n in obs['unseen_inventory'].items())
    v=[xyz[i]/limits[i] for i in range(3)]+[float(candidate['yaw_deg']==90)]+[d[i]/limits[i] for i in range(3)]+[b['mass_kg']/p['max_mass_kg']]
    v += [cog[i]/limits[i] for i in range(3)]+[sum(math.prod(box['dimensions_m']) for box in obs['placed_boxes'])/volume,
        len(obs['placed_boxes'])/100,len(obs['observed_preview'])/12,len(obs['unordered_visible_hints'])/12,
        sum(obs['unseen_inventory'].values())/100,obs['camera']['dimension_sigma_m']/.02,nominal/volume]
    for i in range(12):
        if i<len(obs['observed_preview']):
            t=obs['observed_preview'][i]
            v += [t['dimensions_m'][a]/limits[a] for a in range(3)]+[t['nominal_mass_kg']/p['max_mass_kg'],1.]
        else:v += [0.]*5
    for gy in range(5):
        for gx in range(5):
            x=(gx+.5)*p['length_m']/5;y=(gy+.5)*p['width_m']/5;h=0.
            for box in boxes:
                bd=dims(box,box.get('yaw_deg',0));lo=box['position_m']
                if lo[0]<=x<=lo[0]+bd[0] and lo[1]<=y<=lo[1]+bd[1]:h=max(h,lo[2]+bd[2])
            v.append(h/p['max_height_m'])
    assert len(v)==len(FEATURE_NAMES)
    return [round(value,6) for value in v]

