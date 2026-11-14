#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a reproducible synthetic 3-bin grain dataset for GraphSAGE+GRU.

Model: synchronized explicit heat balance in each bin with neighbour coupling,
room exchange, forced aeration, self-heating and moisture sorption/evaporation.
The numeric constitutive laws for heat release/moisture transfer are *assumptions*,
not experimentally validated equations. See README.md and assumptions.json.

Requires: numpy (no PyTorch needed to generate data).
Example: python generate_grain_dataset.py --scenarios 500 --out grain_dataset
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

GRAINS = ["wheat", "corn", "barley", "rye", "oats"]
RHO_BULK = np.array([750., 720., 620., 710., 450.], dtype=float)  # kg/m3
CP_12 = np.array([1906., 1805., 1516., 1866., 1621.], dtype=float)  # J/(kg K) at 12%
HEAT_FACTORS = np.array([1., 1.10, .85, .95, .80], dtype=float)  # hypothetical
CP_WATER = 4180.0
AIR_RHO = 1.2
AIR_CP = 1005.0
ETA = 0.20  # air-to-grain effective heat exchange (assumed)
K_NEIGHBOUR = .026  # W/K for 1m stagnant air and 1m2; ignores convection/radiation
H_ROOM = 2.0  # W/(m2 K) assumed
LATENT_HEAT = 2.45e6  # J/kg
RH_FIXED = .60  # humidity is not a learned input; fixed in all simulations
MOISTURE_EQ_PCT = 8.0 + 9.0 * RH_FIXED  # toy EMC, fixed at 13.4%; NOT grain-specific
MOISTURE_TAU_HOURS = 48.0  # toy exponential time constant at airflow 1m3/s
DT_SECONDS_DEFAULT = 60
FEATURES = [
    'T_grain_C', 'W_grain_pct',
    'grain_wheat', 'grain_corn', 'grain_barley', 'grain_rye', 'grain_oats',
    'mass_kg', 'T_room_C', 'T_vent_C', 'air_flow_m3_s'
]
EDGE_INDEX = [[0,1,1,2], [1,0,2,1]]


def heat_capacity(grain_ids, w_pct):
    """Illustrative water + dry grain mixture, adjusted to reference Cp at 12% wet-basis."""
    w = w_pct / 100.0
    c_dry = (CP_12[grain_ids] - .12 * CP_WATER) / .88
    return c_dry * (1.0-w) + CP_WATER * w


def simulate(rng, hours, record_minutes, dt_seconds):
    """Return (steps+1,3,11) features, scenario metadata, and temperatures."""
    grain = rng.integers(0, 5, size=3)
    mass = rng.uniform(300., 2000., size=3)
    volume = mass / RHO_BULK[grain]
    surface = 1.5 * volume ** (2.0/3.0)  # exposed area approximation
    temp = rng.uniform(8.,30.,size=3)
    moisture = rng.uniform(10.,20.,size=3)  # wet-basis percent
    flow = rng.uniform(.5,2.,size=3)
    vent_base = rng.uniform(5.,30.,size=3)
    vent_after = float(rng.uniform(5.,30.))
    # Align the ventilation step with stored timestamps so the model sees it.
    event_frame = int(rng.integers(int(6*60/record_minutes),
                                   int(min(24., hours-1.)*60/record_minutes)+1))
    event_hour = event_frame * record_minutes / 60.0
    room_base = float(rng.uniform(10.,40.))
    room_amp = float(rng.uniform(0., 1.5))
    room_phase = float(rng.uniform(0., 2*np.pi))
    total_steps = int(round(hours*3600/dt_seconds))
    record_every = int(round(record_minutes*60/dt_seconds))
    assert total_steps % record_every == 0
    n_records = total_steps//record_every + 1
    output = np.empty((n_records,3,len(FEATURES)), dtype=np.float32)
    onehot = np.eye(5,dtype=np.float64)[grain]
    max_abs_temp = 0.

    def external_values(t):
        room = room_base + room_amp * np.sin(2*np.pi*t/(24*3600.) + room_phase)
        vent = vent_base.copy()
        if t >= event_hour*3600.:
            vent[1] = vent_after  # step only in middle bin
        return room,vent

    def store(k,t):
        room,vent = external_values(t)
        data = np.column_stack((temp, moisture, onehot, mass,
                                np.full(3,room), vent, flow))
        output[k] = data.astype(np.float32)

    store(0,0.)
    for step in range(total_steps):
        t = step*dt_seconds
        room,vent=external_values(t)
        cp=heat_capacity(grain,moisture)
        # Heat flows [W]. Evaluate ALL from the same old state.
        q_neighbour=np.array([
            K_NEIGHBOUR*(temp[1]-temp[0]),
            K_NEIGHBOUR*(temp[0]-temp[1])+K_NEIGHBOUR*(temp[2]-temp[1]),
            K_NEIGHBOUR*(temp[1]-temp[2])
        ])
        q_room=H_ROOM*surface*(room-temp)
        q_vent=ETA*AIR_RHO*AIR_CP*flow*(vent-temp)
        # Hypothetical self-heating: W/kg, small and positive in whole domain.
        q_specific=.002 * HEAT_FACTORS[grain] * np.exp(
            np.clip(.06*(temp-20.)+.18*(moisture-17.), -8.,8.))
        q_self=mass*q_specific
        # Hypothetical moisture relaxation. Relative humidity is FIXED.
        # More airflow speeds it up. Calculate water mass change using wet basis.
        tau_s=MOISTURE_TAU_HOURS*3600. / flow
        moisture_new = MOISTURE_EQ_PCT + (moisture-MOISTURE_EQ_PCT)*np.exp(-dt_seconds/tau_s)
        moisture_new = np.clip(moisture_new, 5.,25.)
        w0=moisture/100.
        w1=moisture_new/100.
        dry_mass=mass*(1.-w0)  # approximate dry mass at step start
        water_lost_kg=dry_mass*(w0/(1.-w0)-w1/(1.-w1))
        q_latent=LATENT_HEAT*water_lost_kg/dt_seconds
        q_total=q_neighbour+q_room+q_vent+q_self-q_latent
        next_temp=temp + dt_seconds*q_total/(mass*cp)
        if not np.isfinite(next_temp).all() or (np.abs(next_temp)>200).any():
            raise RuntimeError('Numerical instability; decrease --dt-seconds')
        temp=next_temp
        moisture=moisture_new
        max_abs_temp=max(max_abs_temp, float(np.max(np.abs(temp))))
        if (step+1)%record_every==0:
            store((step+1)//record_every,(step+1)*dt_seconds)

    meta={
        'grain_type_ids_1to5':(grain+1).tolist(),
        'grain_types':[GRAINS[i] for i in grain],
        'masses_kg':mass.round(4).tolist(),
        'volumes_m3':volume.round(4).tolist(),
        'bulk_densities_kg_m3':RHO_BULK[grain].tolist(),
        'initial_grain_temp_C':output[0,:,0].astype(float).tolist(),
        'initial_moisture_pct':output[0,:,1].astype(float).tolist(),
        'air_flows_m3_s':flow.round(5).tolist(),
        'vent_before_C':vent_base.round(4).tolist(),
        'vent_middle_after_C':round(vent_after,4),
        'vent_middle_step_at_hour':round(event_hour,4),
        'T_room_base_C':round(room_base,4),
        'max_absolute_temperature_C':round(max_abs_temp,4),
    }
    return output,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenarios',type=int,default=500)
    p.add_argument('--out',type=Path,default=Path('grain_dataset'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hours',type=int,default=48)
    p.add_argument('--record-minutes',type=int,default=15)
    p.add_argument('--dt-seconds',type=int,default=DT_SECONDS_DEFAULT)
    p.add_argument('--window-steps',type=int,default=48)
    p.add_argument('--stride-steps',type=int,default=4)
    args=p.parse_args()
    if args.scenarios<7 or args.hours<25 or args.dt_seconds<=0 or args.record_minutes<=0:
        p.error('Require scenarios>=7, hours>=25, and positive steps')
    if (args.record_minutes*60)%args.dt_seconds !=0 or (args.hours*3600)%(args.record_minutes*60)!=0:
        p.error('dt must divide record interval, and record interval must divide duration')
    n_frames=args.hours*60//args.record_minutes+1
    if args.window_steps >=n_frames or args.stride_steps<1:
        p.error('Window must be shorter than a trajectory; stride must be positive')
    rng=np.random.default_rng(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    data=np.lib.format.open_memmap(args.out/'trajectories.npy',mode='w+',dtype='float32',
                                   shape=(args.scenarios,n_frames,3,len(FEATURES)))
    metas=[]
    for sid in range(args.scenarios):
        traj,meta=simulate(rng,args.hours,args.record_minutes,args.dt_seconds)
        data[sid]=traj
        meta['scenario_id']=sid
        metas.append(meta)
    data.flush()
    np.save(args.out/'edge_index.npy',np.array(EDGE_INDEX,dtype=np.int64))
    np.save(args.out/'edge_weight_W_per_K.npy', np.full((4,),K_NEIGHBOUR,dtype=np.float32))
    with (args.out/'scenarios.jsonl').open('w',encoding='utf8') as f:
        for meta in metas:
            f.write(json.dumps(meta,ensure_ascii=False)+'\n')
    ids=rng.permutation(args.scenarios)
    ntrain=int(.70*args.scenarios)
    nval=int(.15*args.scenarios)
    splits={'train':ids[:ntrain], 'val':ids[ntrain:ntrain+nval], 'test':ids[ntrain+nval:]}
    index_counts={}
    for split,scenario_ids in splits.items():
        with (args.out/f'{split}_windows.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.writer(f)
            writer.writerow(['scenario_id','start_frame','last_input_frame','target_frame'])
            count=0
            for sid in scenario_ids:
                for start in range(0,n_frames-args.window_steps,args.stride_steps):
                    last=start+args.window_steps-1
                    writer.writerow([int(sid),start,last,last+1]);count+=1
            index_counts[split]=count
    manifest={
        'purpose':'Synthetic pretraining on approximate physical model (NOT experimentally validated)',
        'shape_trajectories':list(data.shape),'dtype':'float32',
        'feature_order':FEATURES,
        'graph_edges_directed':EDGE_INDEX,
        'one_window_x_shape':[args.window_steps,3,len(FEATURES)],
        'one_window_target_shape':[3],
        'target_definition':'delta_T_C = trajectories[sid,target_frame,:,0] - trajectories[sid,last_input_frame,:,0]',
        'prediction_horizon_minutes':args.record_minutes,
        'window_duration_nominal_hours':args.window_steps*args.record_minutes/60,
        'window_stride_minutes':args.stride_steps*args.record_minutes,
        'index_counts':index_counts,
        'seed':args.seed,'scenarios':args.scenarios,
        'numeric_dt_seconds':args.dt_seconds,
        'references':'Parameters of the previously discussed approximate heat balance; assumptions below are modelling choices.',
        'assumptions':{
            'bulk_densities_kg_m3':dict(zip(GRAINS,RHO_BULK.tolist())),
            'cp_reference_at_12pct_J_kg_K':dict(zip(GRAINS,CP_12.tolist())),
            'cp_moisture':'linear mixture of dry grain and water; illustrative approximation',
            'air_rho_kg_m3':AIR_RHO,'air_cp_J_kg_K':AIR_CP,
            'vent_exchange_efficiency_eta':ETA,'heat_link_W_K':K_NEIGHBOUR,
            'room_heat_transfer_W_m2_K':H_ROOM,
            'exposed_surface_m2':'1.5*V**(2/3), heuristic',
            'latent_heat_J_kg':LATENT_HEAT,
            'relative_humidity':'FIXED at 60%; not included as feature',
            'equilibrium_moisture_pct':MOISTURE_EQ_PCT,
            'moisture_relaxation_tau_hours':'48/air_flow_m3_s, hypothetical',
            'self_heat_W_kg':'0.002 * cultivar_factor * exp(0.06*(T-20)+0.18*(W-17)); hypothetical, not calibrated',
            'thermal_communication':'1--2 and 2--3; no direct 1--3 connection',
            'step_change':'ventilation temperature of middle bin changes once between 6 and 24 h',
            'air_flow_caution':'0.5--2 m3/s per 300--2000kg zone is unusually high; user-specified experimental range',
            'heat_link_caution':'0.026 W/K assumes stagnant air, 1m, 1m2; strongly underestimates convection/radiation if present',
            'validation_caution':'Generated values depend on unvalidated coefficients; compare with measured data before scientific claims.'
        }
    }
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Saved to:', args.out.resolve())
    print('trajectory shape:', tuple(data.shape))
    print('window counts:',index_counts)
    print('sample T first,last:', data[0,0,:,0], data[0,-1,:,0])

if __name__=='__main__':
    main()
