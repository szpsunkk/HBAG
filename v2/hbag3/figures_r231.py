"""Additive paper figures from recorded v2.3 data; fixed Appendix B choices."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .environment import ROOT
from .compact_r2 import read_raw,inputs_for
from .figures_r2 import style,COLORS,MAIN,selected_runs
from .analysis_r231 import interval
from .audit_r231 import identity,sha
from .runner import write_json

OUT=ROOT/'figures_r2/v231'


def save(fig,name,sources,**metadata):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),facecolor='white',dpi=300)
    write_json(OUT/(name+'.sources.json'),dict(sources=sources,width_in=3.5,min_font_pt=8,**metadata))
    plt.close(fig)


def select_satellite(raw,inputs):
    """First complete eclipse per satellite; lower median of recorded entry Z."""
    dark=inputs['harvest_w']==0;candidates=[]
    for sat in range(dark.shape[1]):
        entries=np.flatnonzero((~dark[:-1,sat])&dark[1:,sat])+1
        for t in entries:
            exits=np.flatnonzero(~dark[t:,sat])
            if not len(exits):continue
            candidates.append(dict(satellite=sat,entry_slot=int(t),exit_slot=int(t+exits[0]),
                                   discretionary_energy_j=float(raw['discretionary_energy_j'][t,sat])))
            break
    if not candidates:raise ValueError('No complete eclipse window for the declared selection rule')
    ordered=sorted(candidates,key=lambda r:(r['discretionary_energy_j'],r['satellite']))
    rank=(len(ordered)-1)//2
    return ordered[rank],ordered,rank


def trends():
    frozen=json.loads((ROOT/'results_r2/E11_validation/frozen.json').read_text());rhos=[.6,1.,1.2,1.4,1.6]
    fig,axes=plt.subplots(2,1,figsize=(3.5,5.7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    fig.subplots_adjust(left=.22,right=.97,bottom=.12,top=.77,hspace=.28)
    for j,method in enumerate(MAIN):
        rows=[interval([r['delivered_bits']/1e12 for r in selected_runs(rho,method,frozen)]) for rho in rhos]
        values=np.array([r['mean'] for r in rows]);lo=np.array([r['ci95_low'] for r in rows]);hi=np.array([r['ci95_high'] for r in rows])
        axes[0].errorbar(rhos,values,yerr=np.maximum([values-lo,hi-values],0),color=COLORS[method],
                         marker=['o','s','^','v','D','P'][j],ms=3,lw=1,capsize=2,label=method)
    contrasts=[]
    for rho in rhos:
        a=selected_runs(rho,'HBAG',frozen);b=selected_runs(rho,'V1',frozen)
        assert [r['seed'] for r in a]==[r['seed'] for r in b]
        contrasts.append(interval([(x['delivered_bits']-y['delivered_bits'])/1e12 for x,y in zip(a,b)]))
    mean=np.array([r['mean'] for r in contrasts]);lo=np.array([r['ci95_low'] for r in contrasts]);hi=np.array([r['ci95_high'] for r in contrasts])
    axes[1].axhline(0,color='#777777',lw=.8,ls='--')
    axes[1].errorbar(rhos,mean,yerr=np.maximum([mean-lo,hi-mean],0),color=COLORS['HBAG'],marker='o',ms=3,lw=1,capsize=2)
    axes[0].set(ylabel='Delivered traffic (Tbit)',title='(a) Total delivery')
    axes[1].set(xlabel=r'Energy tightness $\rho_E$',ylabel='Paired difference (Tbit)',title='(b) HBAG − V1',xticks=rhos)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,.985))
    save(fig,'E2_delivered_bits_vs_rho',['results_r2/E2/*/seed_4000–4019/*/summary.json'],
         paired_contrast='HBAG−V1 within seeds',bootstrap=10000,bootstrap_seed=2311729,paired_intervals=contrasts)


def pacing():
    cell=ROOT/'results_r2/E2/rho_1.4/seed_4000';inputs=inputs_for(cell/'HBAG');raw=read_raw(cell/'HBAG')
    chosen,ranking,rank=select_satellite(raw,inputs);sat=chosen['satellite']
    summary=json.loads((cell/'HBAG/summary.json').read_text());dt=summary['dt_s'];times=inputs['times_s'].astype(float)/60
    zero=raw['mu'][:,sat]==0
    transitions=np.diff(np.r_[False,zero,False].astype(int));spans=list(zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)))
    fig,axes=plt.subplots(3,1,figsize=(3.5,5.0),sharex=True)
    fig.subplots_adjust(left=.23,right=.97,bottom=.12,top=.83,hspace=.3)
    for ax,key,label in zip(axes,['mu','pace_w','terminal_w'],[r'Price $\mu$','Pace (W)','ISL power (W)']):
        values=raw[key][:,sat]
        for a,b in spans:ax.axvspan(times[a],times[b-1]+dt/60,color='#D0D0D0',alpha=.6,lw=0,zorder=0)
        ax.plot(times,values,color='#888888',lw=.9)
        ax.plot(times,np.where(zero,np.nan,values),color=COLORS['HBAG'],lw=1.05)
        ax.set_ylabel(label)
    axes[0].set_title('Seed 4000, median-entry satellite %d'%sat)
    axes[-1].set_xlabel('Time (min)')
    fig.legend(handles=[Line2D([],[],color=COLORS['HBAG'],lw=1,label='Positive price'),Patch(facecolor='#D0D0D0',label='Zero-price interval')],
               loc='upper center',ncol=1,frameon=False,bbox_to_anchor=(.53,.99))
    source=str((cell/'HBAG').relative_to(ROOT))
    save(fig,'E2_pacing_satellite',[source],satellite=sat,zero_price_rule='Recorded mu == 0, no threshold',zero_price_slots=int(zero.sum()))
    write_json(OUT/'representative.json',identity(state='PASS',seed=4000,rho_energy=1.4,method='HBAG',
        rule='First complete eclipse entry per satellite; sort by recorded Z=max(E−S,0), then satellite ID; choose lower median',
        source_key='discretionary_energy_j',median_zero_based_rank=rank,chosen=chosen,candidates=ranking))


def topology():
    cell=ROOT/'results_r2/E2/rho_1.4/seed_4000';inp=inputs_for(cell/'V1');t=24
    lo,hi=inp['link_offsets'][t:t+2];edges=inp['directed_edges'][lo:hi];count=len(edges)//2
    xy=np.column_stack([np.arange(576)%24,np.arange(576)//24]);dark=inp['harvest_w'][t]==0
    short=np.max(np.abs(xy[edges[:count,0]]-xy[edges[:count,1]]),axis=1)<=1
    for method in ['V1','HBAG']:
        raw=read_raw(cell/method);chosen=raw['link_on'][lo:hi][:count]&dark[edges[:count,0]]&dark[edges[:count,1]]
        fig=plt.figure(figsize=(3.5,4.1));grid=fig.add_gridspec(2,1,height_ratios=[.65,3.0],left=.17,right=.97,bottom=.14,top=.98,hspace=.15)
        legend_ax=fig.add_subplot(grid[0]);legend_ax.axis('off');ax=fig.add_subplot(grid[1])
        ax.add_collection(LineCollection(xy[edges[:count][chosen&short]],colors=COLORS[method],linewidths=.55))
        ax.scatter(xy[dark,0],xy[dark,1],s=3,color='gray')
        holes=raw['hole_node_slot'][t]&dark;ax.scatter(xy[holes,0],xy[holes,1],s=20,marker='x',color='#D55E00')
        ax.set(xlim=(-1,24),ylim=(-1,24),xlabel='Along-track index',ylabel='Orbital plane',title=method+', slot 24; wrap edges omitted')
        ax.set_aspect('equal')
        handles=[Line2D([],[],color=COLORS[method],lw=1,label='Active eclipse link'),
                 Line2D([],[],color='gray',marker='o',ls='none',ms=3,label='Eclipse satellite'),
                 Line2D([],[],color='#D55E00',marker='x',ls='none',ms=4,label='Access hole')]
        legend_ax.legend(handles=handles,loc='center',frameon=False,ncol=1)
        save(fig,'E2_eclipse_topology_'+method,[str((cell/method).relative_to(ROOT))],seed=4000,slot=t,
             topology_unchanged=True,legend='Dedicated axes outside data and title')


def main():
    style();trends();pacing();topology()
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json(ROOT/'results_r2/v231/audit/figures.json',identity(state='PASS',files=files,pdf_count=4))


if __name__=='__main__':main()
