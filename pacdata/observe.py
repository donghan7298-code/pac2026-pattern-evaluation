"""Allow-listed observation builder. No unseen arrival order leaves this module."""
import copy
import random
from collections import Counter
from .io import stable_seed


def observe(episode, state):
    step=state['processed']; arrivals=episode['oracle']['arrivals']
    inventory=Counter(b['sku'] for b in arrivals[step:])
    current=None
    if step<len(arrivals):
        source=arrivals[step]
        current={k:copy.deepcopy(source[k]) for k in
                 ['box_id','sku','dimensions_m','mass_kg','top_load_capacity_N','allowed_yaw_deg']}
        current.update(measurement_valid=True,visual_damage_observed=source['damaged'],
                       dimension_source='station_measurement',mass_source='station_scale')
        if step not in state.get('remeasured',[]) and source['station_fault']:
            current['measurement_valid']=False
            if source['station_fault']=='missing_mass': current['mass_kg']=None
        inventory[current['sku']]-=1
    sensor=episode['camera'];tracks=[];prefix=[];gap_seen=False
    distance=arrivals[step]['gap_after_m'] if current else 0.
    for rank, source in enumerate(arrivals[step+1:step+1+sensor['max_tracks']]):
        length=source['dimensions_m'][0]
        if distance+length>sensor['fov_length_m']: break
        rng=random.Random(stable_seed(sensor['seed'],step,source['box_id'],'frame'))
        missed=rng.random()<sensor['miss_probability']
        occluded=rng.random()<sensor['occlusion_probability']
        if not missed:
            sigma=sensor['dimension_sigma_m']
            confidence=round(max(.25, .99-(.4 if occluded else 0)-rng.uniform(0,.08)),3)
            known=episode['catalog'][source['sku']]
            estimated=[round(max(.01,source['dimensions_m'][i]+rng.gauss(0,sigma)),4) for i in (0,1)]
            track=dict(track_id=source['box_id'],sku=source['sku'],sku_source='assumed_WMS_identity',
                dimensions_m=estimated+[known['dimensions_m'][2]],
                nominal_mass_kg=known['mass_kg'],top_load_capacity_N=known['top_load_capacity_N'],
                allowed_yaw_deg=known['allowed_yaw_deg'],confidence=confidence,
                occluded=occluded,dimension_sigma_m=sigma,dimension_source='synthetic_topview_xy_plus_catalog_height',
                mass_source='catalog_not_camera',
                belt_distance_m=round(distance,4),
                bbox_normalized=[round(distance/sensor['fov_length_m'],5),.15,
                    round(min(1,(distance+estimated[0])/sensor['fov_length_m']),5),
                    round(min(.95,.15+estimated[1]/sensor['belt_width_m']),5)])
            tracks.append(track)
            if not gap_seen and not occluded and confidence>=.85:
                prefix.append(dict(track,verified_order_rank=len(prefix)+1))
                inventory[source['sku']]-=1
            else: gap_seen=True
        else: gap_seen=True
        distance+=length+source['gap_after_m']
    known_ids={track['track_id'] for track in prefix}
    return dict(schema_version='2.0',request_id=f'{episode["episode_id"]}:{step}',
        pallet=copy.deepcopy(state['pallet']),placed_boxes=copy.deepcopy(state['placed']),
        state_verified=True,current_box=current,observed_preview=prefix,
        unordered_visible_hints=[track for track in tracks if track['track_id'] not in known_ids],
        unseen_inventory={sku:count for sku,count in inventory.items() if count},
        catalog=copy.deepcopy(episode['catalog']),
        camera=dict(fov_length_m=sensor['fov_length_m'],max_tracks=sensor['max_tracks'],
                    coverage_complete=not gap_seen,ordering_assumption='single_lane_no_overtaking',
                    coverage_source='assumed_temporal_tracker_coverage_flag',
                    dimension_sigma_m=sensor['dimension_sigma_m']),
        robot_feasibility='NOT_CHECKED')
