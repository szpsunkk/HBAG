"""Small CPU parameter-shared PPO with an explicitly disclosed routing/safety layer."""
from pathlib import Path
import time,json
import numpy as np
from .environment import ROOT
from .model import RATE
_MODEL=None
LEVELS=np.array([0.,.25,.5,.75,1.])

def observation(env,t,energy):
    cat=env['catalogues'][t];m=env['shell'].size;b=env['battery']
    valid=cat.sources>=0;load=np.bincount(cat.sources[valid],weights=env['demand_bps'][t][valid]/(50*RATE),minlength=m)
    return np.column_stack([energy/b.capacity_j,env['harvest'][t]/env['config'].solar_w,np.clip(load,0,10),np.clip((energy-env['reserve'][t])/(b.capacity_j-b.minimum_j),-2,2)]).astype(np.float32)

def network():
    import torch
    class ActorCritic(torch.nn.Module):
        def __init__(self):
            super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh());self.actor=torch.nn.Linear(32,5);self.critic=torch.nn.Linear(32,1)
        def forward(self,x):
            h=self.body(x);return self.actor(h),self.critic(h).squeeze(-1)
    return ActorCritic()

def fractions(env,t,energy):
    global _MODEL
    import torch
    if _MODEL is None:
        torch.set_num_threads(1);_MODEL=network();checkpoint=torch.load(ROOT/'results/E13/model.pt',map_location='cpu');_MODEL.load_state_dict(checkpoint['state_dict']);_MODEL.eval()
    with torch.no_grad():logits,_=_MODEL(torch.from_numpy(observation(env,t,energy)));prob=torch.softmax(logits,dim=-1).numpy()
    return prob@LEVELS  # Prespecified deterministic expectation during evaluation.

def train(steps=4096,output_dir=None):
    import torch
    from dataclasses import replace
    from .model import Config
    from .environment import build_environment
    from .solvers import make_problem,node_power
    from .optimization import solve_distributed
    from .energy import step
    from .runner import write_json,source_hashes
    root=Path(output_dir) if output_dir is not None else ROOT/'results/E13';root.mkdir(parents=True,exist_ok=True)
    if (root/'model.pt').exists():
        old=json.loads((root/'training.json').read_text())
        if old['training_env_steps']!=steps:raise RuntimeError('Existing PPO checkpoint has a different training budget')
        return old
    torch.set_num_threads(1);torch.manual_seed(1000);rng=np.random.default_rng(1000);model=network();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    frames=[];reward_log=[];start=time.perf_counter();environments={};episode=0;t=0;env=None
    for k in range(steps):
        if env is None or t>=len(env['times']):
            seed=1000+episode%10
            if seed not in environments:environments[seed]=build_environment(Config(seed=seed,planes=8,per_plane=8,flows=16,orbits=1))
            env=environments[seed];energy=np.full(64,.8*env['battery'].capacity_j);t=0;episode+=1
        obs=observation(env,t,energy)
        with torch.no_grad():
            logits,value=model(torch.from_numpy(obs));dist=torch.distributions.Categorical(logits=logits);action=dist.sample();logp=dist.log_prob(action)
        frac=LEVELS[action.numpy()];p,meta=make_problem(env,t,energy,'PPO-shared');cat=env['catalogues'][t]
        admits=np.where(cat.sources>=0,frac[np.maximum(cat.sources,0)],0.);p=replace(p,demand=p.demand*admits)
        result=solve_distributed(p,max_iterations=200,tolerance=1e-3);x=result.allocation
        q=node_power(cat,env['terminal'],x,meta['idle_w']);record=step(energy,env['harvest'][t],55.,q,env['dt'],env['battery'])
        delivered=np.bincount(np.maximum(cat.sources,0),weights=x.sum(axis=1),minlength=64)
        reward=delivered-.01*q/100-10*(record.after_j<env['battery'].minimum_j-1e-6)
        frames.append((obs,action.numpy(),logp.numpy(),value.numpy(),reward.astype(np.float32),t==len(env['times'])-1));reward_log.append(float(reward.mean()));energy=record.after_j;t+=1
        if len(frames)==128 or k==steps-1:
            with torch.no_grad():_,next_value=model(torch.from_numpy(observation(env,min(t,len(env['times'])-1),energy)));last=next_value.numpy()
            advantages=np.zeros((len(frames),64),np.float32);gae=np.zeros(64,np.float32)
            for j in range(len(frames)-1,-1,-1):
                _,_,_,v,r,done=frames[j];continuation=0. if done else 1.
                delta=r+.99*last*continuation-v;gae=delta+.99*.95*continuation*gae;advantages[j]=gae;last=v
            observations=torch.from_numpy(np.concatenate([f[0] for f in frames]));actions=torch.from_numpy(np.concatenate([f[1] for f in frames]));oldlog=torch.from_numpy(np.concatenate([f[2] for f in frames]));values=np.stack([f[3] for f in frames]);returns=torch.from_numpy((advantages+values).ravel());adv=torch.from_numpy(advantages.ravel());adv=(adv-adv.mean())/(adv.std()+1e-8)
            for epoch in range(4):
                for index in np.array_split(rng.permutation(len(actions)),16):
                    logits,v=model(observations[index]);dist=torch.distributions.Categorical(logits=logits);ratio=torch.exp(dist.log_prob(actions[index])-oldlog[index]);surrogate=torch.minimum(ratio*adv[index],torch.clamp(ratio,.8,1.2)*adv[index]);loss=-surrogate.mean()+.5*((v-returns[index])**2).mean()-.01*dist.entropy().mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            frames=[]
            write_json(root/'training_status.json',dict(steps=k+1,budget=steps,elapsed_s=time.perf_counter()-start,recent_reward=float(np.mean(reward_log[-128:]))))
    torch.save(dict(state_dict=model.state_dict(),steps=steps,seed=1000),root/'model.pt');np.savez_compressed(root/'training_raw.npz',mean_step_reward=reward_log)
    metadata=dict(training_env_steps=steps,training_satellite_transitions=steps*64,seeds=list(range(1000,1010)),policy_seed=1000,nodes=64,network='4-32-32 shared tanh actor-critic; 5 admission fractions',optimizer='Adam lr=3e-4',ppo=dict(clip=.2,gamma=.99,gae_lambda=.95,epochs=4,rollout=128),runtime_s=time.perf_counter()-start,device='CPU',evaluation='expected action fraction; no evaluation-time learning',routing_safety_layer='Same V1 routing optimizer and hard budget after PPO per-source demand admission. PPO+optimizer hybrid, NOT an external MAAC/DeepISL reproduction.',source_sha256=source_hashes())
    write_json(root/'training.json',metadata);return metadata
