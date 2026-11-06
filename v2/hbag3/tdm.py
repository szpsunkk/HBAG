"""Synchronized half-duplex optical link schedules with store-and-forward queues.

Constant full-active terminal power is an explicit upper-bound assumption.
Per-transition DC energy is a sensitivity parameter, not a measurement.
"""
import numpy as np
from .energy import energy_drift


def schedule_tdm(paths,rates,capacities,dt_s,frames,rate_bps=1e8):
    """Per edge, consecutive directed path-hop blocks; both endpoints wake together."""
    blocks={edge:[] for edge in capacities};queues=[];flow=[];offered=0.
    frame=dt_s/frames
    for f,items in enumerate(paths):
        for p,route in enumerate(items):
            if route is None or rates[f,p]<=1e-14:continue
            amount=float(rates[f,p]*rate_bps*dt_s);offered+=amount
            ident=len(queues);queues.append([amount]+[0.]*(len(route)-1));flow.append((f,p))
            for hop,(u,v) in enumerate(zip(route[:-1],route[1:])):
                edge=tuple(sorted((u,v)));duration=rates[f,p]*rate_bps*frame/capacities[edge]
                blocks[edge].append((float(duration),ident,hop,int(u),int(v)))
    events=[];max_util=0.
    for edge,parts in blocks.items():
        used=sum(b[0] for b in parts);max_util=max(max_util,used/frame)
        if used>frame+1e-9:raise ValueError('TDM capacity overflow')
        for cycle in range(frames):
            offset=cycle*frame
            for duration,ident,hop,u,v in parts:
                # End events precede start events at equal times. Data becomes
                # available only after transmission ends (finite hop delay).
                tag=(cycle,ident,hop)
                events.append((offset,1,tag,ident,hop,u,v,duration,capacities[edge]))
                events.append((offset+duration,0,tag,ident,hop,u,v,duration,capacities[edge]))
                offset+=duration
    events.sort(key=lambda e:(e[0],e[1],e[2]))
    in_flight={};delivered=0.;node_events=[];transitions=0
    for at,kind,tag,ident,hop,u,v,duration,capacity in events:
        if kind==1:
            sent=min(queues[ident][hop],duration*capacity);queues[ident][hop]-=sent;in_flight[tag]=sent
            node_events.append((at,u,v,1));transitions+=2
        else:
            sent=in_flight.pop(tag);queues[ident][hop+1]+=sent
            if hop+1==len(queues[ident])-1:delivered+=sent
            node_events.append((at,u,v,-1));transitions+=2
    return dict(node_events=node_events,offered_schedule_bits=offered,delivered_bits=delivered,
                pipeline_loss_bits=offered-delivered,max_link_utilization=max_util,
                endpoint_transitions=transitions,frames=frames)


def execute_schedule(schedule,initial_j,harvest_segments,mandatory_w,increment_w,dt_s,battery,switch_j=0.):
    """Integrate exact piecewise-constant loads/harvest, with capacity spill."""
    energy=np.asarray(initial_j,dtype=float).copy();active=np.zeros(energy.size,int)
    events={}
    for at,u,v,delta in schedule['node_events']:events.setdefault(float(at),[]).append((u,v,delta))
    for at,h in harvest_segments:events.setdefault(float(at),[])
    events.setdefault(dt_s,[])
    last=0.;h=np.broadcast_to(harvest_segments[0][1],energy.shape)
    harvest={float(at):np.broadcast_to(value,energy.shape) for at,value in harvest_segments}
    minimum=energy.copy();spill=np.zeros_like(energy);switching=0.;maxactive=0
    for at,items in sorted(events.items()):
        if at>dt_s+1e-7:raise ValueError('Schedule extends past slot')
        raw=energy+(at-last)*energy_drift(h,mandatory_w+increment_w*active,battery)
        minimum=np.minimum(minimum,raw);spill+=np.maximum(raw-battery.capacity_j,0.);energy=np.minimum(raw,battery.capacity_j)
        # Charge a conservative transition impulse for both wake/sleep or role
        # changes, even if adjacent blocks could keep the same hardware active.
        for u,v,delta in items:
            active[u]+=delta;active[v]+=delta
            energy[[u,v]]-=switch_j/battery.discharge_efficiency;switching+=2*switch_j
        minimum=np.minimum(minimum,energy);maxactive=max(maxactive,int(active.max(initial=0)))
        if at in harvest:h=harvest[at]
        last=at
    return dict(min_within_j=minimum.tolist(),final_j=energy.tolist(),capacity_spill_j=float(spill.sum()),
                switch_bus_j=switching,min_margin_j=float(minimum.min()-battery.minimum_j),
                max_active_terminals=maxactive,delivered_bits=schedule['delivered_bits'],
                max_link_utilization=schedule['max_link_utilization'])


def chord_increment_budget(energy,target,harvest_lower,mandatory,maximum,dt,battery,frames,switch_bound_j=0.):
    """Chord lower bound on drift, plus one-frame worst discharge and switch margin.

    Concavity gives f(h,L) >= chord(L) on [mandatory,maximum]. Repetition makes
    the chord bound valid at each frame end. One frame's maximum discharge and
    all assumed transition energy protect every intermediate time as well.
    """
    f0=energy_drift(harvest_lower,mandatory,battery);fm=energy_drift(harvest_lower,maximum,battery)
    slope=(fm-f0)/(maximum-mandatory)
    buffer=dt/frames*np.maximum(-fm,0.)+np.asarray(switch_bound_j)/battery.discharge_efficiency
    budget=((np.asarray(target)+buffer-np.asarray(energy))/dt-f0)/slope
    return np.clip(budget,0.,maximum-mandatory),buffer
