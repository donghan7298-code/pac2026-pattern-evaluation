"""Generate grouped episodes. Oracle sequences are never policy inputs."""
import copy
import math
import random
from collections import Counter
from .io import opaque_id, stable_seed

MODES = ['random', 'large_last', 'heavy_last', 'sku_blocks', 'alternating_size',
         'tall_flat_mix', 'heavy_weak_mix', 'fragile_rotation', 'damage_route',
         'measurement_recheck', 'occlusion_gap', 'pallet_resize']
CAMERAS = [
    dict(name='preview2', max_tracks=2, fov_length_m=1.1),
    dict(name='topview4', max_tracks=4, fov_length_m=2.0),
    dict(name='topview8', max_tracks=8, fov_length_m=3.8),
    dict(name='topview12', max_tracks=12, fov_length_m=5.6)]
TEMPLATES = [
    ('S', [.20,.15,.10], 1., 80.), ('M', [.30,.20,.20], 3., 150.),
    ('L', [.40,.30,.25], 6., 250.), ('FLAT', [.45,.30,.10], 2., 60.),
    ('TALL', [.20,.20,.40], 4., 90.), ('LONG', [.60,.18,.12], 3.5, 110.),
    ('WIDE', [.50,.40,.15], 5., 160.), ('DENSE', [.25,.20,.15], 9., 280.),
    ('WEAK', [.40,.30,.20], 1.5, 25.), ('CUBE', [.25,.25,.25], 4., 180.),
    ('THIN', [.35,.25,.06], 1.2, 35.), ('LARGE', [.60,.40,.30], 8., 220.)]


def split_groups(groups, seed):
    indices = list(range(groups)); random.Random(seed).shuffle(indices)
    ntrain, nval = int(groups*.7), int(groups*.15)
    return {idx: ('train' if rank < ntrain else 'validation' if rank < ntrain+nval else 'test')
            for rank, idx in enumerate(indices)}


def make_group(index, config):
    rng = random.Random(stable_seed(config['seed'], index, 'inventory'))
    catalog = {}
    for sku, dims, mass, cap in TEMPLATES:
        scale = rng.choice([.85, 1., 1.1])
        catalog[sku] = dict(sku=sku, dimensions_m=[round(x*scale,4) for x in dims],
            mass_kg=round(mass*rng.uniform(.9,1.1),3), top_load_capacity_N=round(cap*rng.uniform(.85,1.15),2),
            allowed_yaw_deg=[0] if rng.random()<.15 else [0,90])
    count = rng.randint(config['min_boxes'], config['max_boxes'])
    easy_sku='S' if index%10==0 else 'M' if index%10==1 else None
    inventory = [easy_sku]*count if easy_sku else list(catalog) + [rng.choice(list(catalog)) for _ in range(count-len(catalog))]
    instances = []
    for i, sku in enumerate(inventory):
        item = copy.deepcopy(catalog[sku])
        item['box_id'] = opaque_id(config['seed'], index, i, 'box')
        if not easy_sku:
            item['dimensions_m'] = [round(x*rng.uniform(.99,1.01),4) for x in item['dimensions_m']]
            item['mass_kg'] = round(item['mass_kg']*rng.uniform(.98,1.02),3)
        item['damaged'] = False
        item['station_fault'] = None
        item['gap_after_m'] = round(rng.uniform(.05,.20),3)
        instances.append(item)
    p = dict(length_m=rng.choice([1.0,1.2,1.4]), width_m=rng.choice([.8,1.,1.2]),
             max_height_m=rng.choice([.6,.9,1.2]), max_mass_kg=rng.choice([120.,200.,300.]))
    if easy_sku:p=dict(length_m=1.4,width_m=1.2,max_height_m=1.2,max_mass_kg=300.)
    return catalog, instances, p


def order_instances(instances, mode, seed):
    rng = random.Random(seed); values=copy.deepcopy(instances); rng.shuffle(values)
    volume=lambda b: math.prod(b['dimensions_m'])
    if mode=='large_last': values.sort(key=volume)
    elif mode=='heavy_last': values.sort(key=lambda b:b['mass_kg'])
    elif mode=='sku_blocks': values.sort(key=lambda b:b['sku'])
    elif mode=='alternating_size':
        ordered=sorted(values,key=volume); values=[]
        while ordered:
            values.append(ordered.pop(0))
            if ordered: values.append(ordered.pop())
    elif mode=='tall_flat_mix': values.sort(key=lambda b: 0 if b['sku'] in ('TALL','FLAT','THIN') else 1)
    elif mode=='heavy_weak_mix': values.sort(key=lambda b: 0 if b['sku'] in ('WEAK','DENSE') else 1)
    elif mode=='fragile_rotation':
        for b in values:
            if b['sku'] in ('WEAK','FLAT','THIN'):
                b['allowed_yaw_deg']=[0]; b['top_load_capacity_N'] *= .5
    elif mode=='damage_route':
        for b in values: b['damaged']=rng.random()<.10
        values[2]['damaged']=True
    elif mode=='measurement_recheck':
        for b in values:
            if rng.random()<.12: b['station_fault']=rng.choice(['missing_mass','unreliable_dimensions'])
        values[1]['station_fault']='missing_mass'
    return values


def episodes_for_group(index, config, split):
    catalog, instances, p=make_group(index,config)
    group_id=opaque_id(config['seed'],index,'base_group')
    for mode in MODES:
        arrivals=order_instances(instances,mode,stable_seed(config['seed'],index,mode))
        episode_catalog=copy.deepcopy(catalog)
        if mode=='fragile_rotation':
            for sku in ('WEAK','FLAT','THIN'):
                episode_catalog[sku]['allowed_yaw_deg']=[0]
                episode_catalog[sku]['top_load_capacity_N']*=.5
        for camera in CAMERAS:
            ep_id=opaque_id(config['seed'],index,mode,camera['name'],'episode')
            quality = 'degraded' if mode=='occlusion_gap' else ('moderate' if index%3==0 else 'clean')
            sensor=dict(camera, belt_width_m=.8, quality=quality,
                dimension_sigma_m=.002 if quality=='clean' else .008 if quality=='moderate' else .015,
                miss_probability=0 if quality=='clean' else .04 if quality=='moderate' else .20,
                occlusion_probability=0 if quality=='clean' else .03 if quality=='moderate' else .15,
                seed=stable_seed(config['seed'],index,mode,'shared_sensor_noise'))
            events=[]
            if mode=='pallet_resize':
                expanded=dict(p,length_m=round(p['length_m']+.2,3),width_m=round(p['width_m']+.2,3))
                events=[dict(at_processed=10,type='pallet_resize',new_pallet=expanded)]
            yield dict(schema_version='2.0',source='SYNTHETIC_DEVELOPMENT_ONLY',
                episode_id=ep_id,base_group=group_id,split=split,scenario=mode,
                inventory_profile='homogeneous_control' if index%10 in (0,1) else 'mixed_stress',
                pallet=p,catalog=episode_catalog,camera=sensor,initial_inventory=dict(Counter(b['sku'] for b in arrivals)),
                oracle=dict(arrivals=arrivals,events=events),
                robot_feasibility='NOT_CHECKED')
