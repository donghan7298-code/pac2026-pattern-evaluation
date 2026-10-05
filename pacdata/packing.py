"""Conservative box geometry, finite candidate search and a stepping environment.

Full support by one lower box; no heavier box on a lighter supporter. This is
not a physical simulator or an exhaustive packing optimizer.
"""
import copy
import math
from .observe import observe

EPS=1e-8
G=9.81


def dims(box,yaw=0):
    d=box['dimensions_m'];return [d[1],d[0],d[2]] if yaw==90 else list(d)


def bounds(box):
    d=dims(box,box.get('yaw_deg',0));lo=box['position_m']
    return lo,[lo[i]+d[i] for i in range(3)]


def mask(pallet,placed,box,candidate,_context=None):
    if box is None: return dict(status='DONE',reasons=['NO_CURRENT'])
    if box.get('visual_damage_observed'):return dict(status='ROUTE',reasons=['DAMAGE'])
    if not box.get('measurement_valid',True) or box.get('mass_kg') is None:
        return dict(status='HOLD',reasons=['REMEASURE'])
    if len(box.get('dimensions_m',[]))!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in box['dimensions_m']) or not math.isfinite(box['mass_kg']) or box['mass_kg']<=0:
        return dict(status='HOLD',reasons=['INVALID_MEASUREMENT'])
    if len(candidate.get('position_m',[]))!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in candidate['position_m']):
        return dict(status='REJECT',reasons=['INVALID_CANDIDATE'])
    if candidate['yaw_deg'] not in box['allowed_yaw_deg']:
        return dict(status='REJECT',reasons=['ROTATION'])
    d=dims(box,candidate['yaw_deg']);lo=candidate['position_m'];hi=[lo[i]+d[i] for i in range(3)]
    reasons=[]
    limits=[pallet['length_m'],pallet['width_m'],pallet['max_height_m']]
    if any(lo[i]<-EPS or hi[i]>limits[i]+EPS for i in (0,1)):reasons.append('BOUNDARY')
    if lo[2]<-EPS or hi[2]>limits[2]+EPS:reasons.append('HEIGHT')
    if reasons:return dict(status='REJECT',reasons=reasons)
    context=_context or dict(bounded=[(b,*bounds(b)) for b in placed],mass=sum(b['mass_kg'] for b in placed),by_id={b['box_id']:b for b in placed})
    if context['mass']+box['mass_kg']>pallet['max_mass_kg']+EPS:
        return dict(status='REJECT',reasons=['PALLET_MASS'])
    for existing,blo,bhi in context['bounded']:
        if min(hi[0],bhi[0])-max(lo[0],blo[0])>EPS and min(hi[1],bhi[1])-max(lo[1],blo[1])>EPS and min(hi[2],bhi[2])-max(lo[2],blo[2])>EPS:
            reasons.append('OVERLAP');break
    if reasons:return dict(status='REJECT',reasons=reasons)
    parent=None
    if abs(lo[2])<EPS:parent='FLOOR'
    else:
        for base,blo,bhi in context['bounded']:
            if abs(bhi[2]-lo[2])<EPS and all(blo[i]<=lo[i]+EPS and hi[i]<=bhi[i]+EPS for i in (0,1)):
                parent=base['box_id'];break
    if parent is None:reasons.append('SUPPORT')
    by_id=context['by_id']
    chain=[]
    if parent not in (None,'FLOOR'):
        if box['mass_kg']>by_id[parent]['mass_kg']+EPS:reasons.append('HEAVY_ON_LIGHT')
        while parent!='FLOOR':
            if parent in chain:raise ValueError('Invalid support cycle')
            chain.append(parent);base=by_id[parent]
            if base['top_load_capacity_N'] is None:
                return dict(status='HOLD',reasons=['LOAD_UNKNOWN'])
            if base.get('upper_load_N',0)+box['mass_kg']*G>base['top_load_capacity_N']+EPS:
                reasons.append('TOP_LOAD')
            parent=base.get('supporter_id','FLOOR')
    return dict(status='REJECT' if reasons else 'ALLOW',reasons=sorted(set(reasons)),support_chain=chain)


def generate_candidates(pallet,placed,box,max_candidates=32):
    if box is None or not box.get('measurement_valid',True) or box.get('visual_damage_observed'):
        return []
    context=dict(bounded=[(b,*bounds(b)) for b in placed],mass=sum(b['mass_kg'] for b in placed),by_id={b['box_id']:b for b in placed})
    if box.get('mass_kg') is None or context['mass']+box['mass_kg']>pallet['max_mass_kg']+EPS:return []
    candidates=[];seen=set()
    for yaw in box['allowed_yaw_deg']:
        d=dims(box,yaw)
        anchors=[(0,0,0),(pallet['length_m']-d[0],0,0),
                 (0,pallet['width_m']-d[1],0),(pallet['length_m']-d[0],pallet['width_m']-d[1],0)]
        for base in placed:
            lo,hi=bounds(base)
            anchors.extend([(hi[0],lo[1],0),(lo[0],hi[1],0),(hi[0],hi[1],0),
                (lo[0]-d[0],lo[1],0),(lo[0],lo[1]-d[1],0),
                (lo[0],lo[1],hi[2]),(hi[0]-d[0],lo[1],hi[2]),
                (lo[0],hi[1]-d[1],hi[2]),(hi[0]-d[0],hi[1]-d[1],hi[2])])
        for xyz in anchors:
            key=tuple(round(v,6) for v in xyz)+(yaw,)
            if key in seen:continue
            seen.add(key)
            candidate=dict(candidate_id=':'.join(map(str,key)),position_m=list(key[:3]),yaw_deg=yaw)
            candidates.append(candidate)
    candidates.sort(key=lambda c:(c['position_m'][2],c['position_m'][1],c['position_m'][0],c['yaw_deg']))
    # Cap AFTER the mask so early invalid candidates do not erase all valid ones.
    valid=[]
    for candidate in candidates:
        verdict=mask(pallet,placed,box,candidate,context)
        if verdict['status']=='ALLOW':
            valid.append(candidate)
            if len(valid)>=max_candidates:break
    return valid


def apply_placement(pallet,placed,box,candidate):
    verdict=mask(pallet,placed,box,candidate)
    if verdict['status']!='ALLOW':raise ValueError(f'Invalid placement: {verdict}')
    result=[dict(b,dimensions_m=list(b['dimensions_m']),position_m=list(b['position_m']),allowed_yaw_deg=list(b['allowed_yaw_deg'])) for b in placed]
    by_id={b['box_id']:b for b in result}
    for identity in verdict['support_chain']:by_id[identity]['upper_load_N']+=box['mass_kg']*G
    placed_box={k:copy.deepcopy(box[k]) for k in
        ['box_id','sku','dimensions_m','mass_kg','top_load_capacity_N','allowed_yaw_deg']}
    placed_box.update(position_m=list(candidate['position_m']),yaw_deg=candidate['yaw_deg'],upper_load_N=0.,
        supporter_id=verdict['support_chain'][0] if verdict['support_chain'] else 'FLOOR')
    result.append(placed_box)
    return result


def immediate_score(pallet,placed,box,candidate):
    d=dims(box,candidate['yaw_deg']);x,y,z=candidate['position_m']
    return -(z+d[2])/pallet['max_height_m']-.03*y/pallet['width_m']-.02*x/pallet['length_m']


def greedy(pallet,placed,box,max_candidates=16):
    values=generate_candidates(pallet,placed,box,max_candidates)
    return max(values,key=lambda c:immediate_score(pallet,placed,box,c)) if values else None


class PackingEnv:
    """reset()/step() Python environment. Raw episode stays private to observer."""
    def __init__(self,episode):
        self._episode=copy.deepcopy(episode);self.state=None;self.terminated=False

    def reset(self):
        self.state=dict(processed=0,pallet=copy.deepcopy(self._episode['pallet']),placed=[],remeasured=[],routed=0)
        self.terminated=False
        return observe(self._episode,self.state)

    def observation(self):return observe(self._episode,self.state)

    def step(self,action):
        if self.terminated:raise ValueError('Episode finished; reset first')
        obs=self.observation();box=obs['current_box'];kind=action.get('type')
        if box is None:
            self.terminated=True;return obs,0.,True,dict(reason='COMPLETE')
        if kind=='REMEASURE':
            if box['measurement_valid']:raise ValueError('Current box is already measured')
            self.state['remeasured'].append(self.state['processed'])
            return self.observation(),-.01,False,dict(reason='REMEASURED',robot_feasibility='NOT_CHECKED')
        if kind=='ROUTE_DAMAGE':
            if not box['visual_damage_observed']:raise ValueError('Cannot route a normal box as damaged')
            self.state['routed']+=1;reward=0.;reason='ROUTED_DAMAGE'
        elif kind=='PLACE':
            self.state['placed']=apply_placement(self.state['pallet'],self.state['placed'],box,action['candidate'])
            reward=math.prod(box['dimensions_m'])/(self.state['pallet']['length_m']*self.state['pallet']['width_m']*self.state['pallet']['max_height_m'])
            reason='PLACED'
        elif kind=='STOP':
            if generate_candidates(self.state['pallet'],self.state['placed'],box,1):
                raise ValueError('STOP requested while an allowed candidate exists')
            self.terminated=True;return obs,-.1,True,dict(reason='NO_VALID_SEARCH_CANDIDATE')
        else:raise ValueError('Unknown action')
        self.state['processed']+=1
        for event in self._episode['oracle']['events']:
            if event['at_processed']==self.state['processed'] and event['type']=='pallet_resize':
                self.state['pallet']=copy.deepcopy(event['new_pallet'])
        self.terminated=self.state['processed']==len(self._episode['oracle']['arrivals'])
        return self.observation(),reward,self.terminated,dict(reason=reason,robot_feasibility='NOT_CHECKED')
