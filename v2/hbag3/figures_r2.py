"""IEEE single-column vector figures from recorded independent R2 runs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .environment import ROOT
from .compact_r2 import read_raw,inputs_for
from .runner import write_json

COLORS={'V0':'#555555','Guard*':'#D55E00','V1':'#009E73','V2':'#CC79A7',
        'HBAG':'#0072B2','Util-Sleep':'#E69F00','Lyapunov':'#56B4E9',
        'HBAG-xi':'#882255','HBAG-Lyap':'#44AA99','HBAG-central':'#332288'}
MAIN=['V0','Guard*','Util-Sleep','Lyapunov','V1','HBAG']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})


def save(fig,name,sources):
    root=ROOT/'figures_r2';root.mkdir(parents=True,exist_ok=True)
    # Keep the PDF canvas exactly 3.5 inches wide; tight cropping changes print size.
    fig.savefig(root/(name+'.pdf'),facecolor='white')
    fig.savefig(root/(name+'.png'),facecolor='white',dpi=150)
    write_json(root/(name+'.sources.json'),dict(sources=sources,width_in=3.5,min_font_pt=8))
    plt.close(fig)


def guard_label(rho,frozen):
    chosen=frozen['selected_by_rho'][str(rho if rho>=1.2 else 1.2)]['Static+Guard']['parameters']['guard_soc']
    return 'Guard_'+str(chosen) if rho>=1.2 else 'Guard*'


def selected_runs(rho,method,frozen):
    label=guard_label(rho,frozen) if method=='Guard*' else method
    paths=[ROOT/'results_r2/E2'/('rho_'+str(float(rho)))/('seed_%d'%s)/label/'summary.json' for s in range(4000,4020)]
    return [json.loads(p.read_text()) for p in paths]


def interval(values):
    a=np.asarray(values,float);rng=np.random.default_rng(1729)
    boot=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return float(a.mean()),np.percentile(boot,[2.5,97.5])


def trends(frozen):
    for metric,scale,ylabel in [('esr',100.,'Energy safety ratio (%)'),('delivered_bits',1e-12,'Delivered traffic (Tbit)')]:
        fig,ax=plt.subplots(figsize=(3.5,3.6));fig.subplots_adjust(left=.19,right=.98,bottom=.18,top=.72)
        for j,method in enumerate(MAIN):
            values=[interval([r[metric]*scale for r in selected_runs(rho,method,frozen)]) for rho in [.6,1.,1.2,1.4,1.6]]
            mean=np.array([v[0] for v in values]);ci=np.array([v[1] for v in values])
            ax.errorbar([.6,1.,1.2,1.4,1.6],mean,yerr=np.maximum(np.array([mean-ci[:,0],ci[:,1]-mean]),0),
                        color=COLORS[method],label=method,marker=['o','s','^','v','D','P'][j],markersize=3,lw=1,capsize=2)
        ax.set(xlabel=r'Energy tightness $\rho_E$',ylabel=ylabel,xticks=[.6,1.,1.2,1.4,1.6])
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        save(fig,'E2_'+metric+'_vs_rho',['results_r2/E2/*/seed_*/*/summary.json','results_r2/E11_validation/frozen.json'])


def trajectories(frozen):
    rho=1.4;cell=ROOT/'results_r2/E2/rho_1.4/seed_4000'
    # First indexed satellite that has both illumination states; independent of outcomes.
    inp=inputs_for(cell/'V0');dark=inp['harvest_w']==0
    eligible=np.flatnonzero(dark.any(axis=0)&(~dark).any(axis=0));sat=int(eligible[0]) if len(eligible) else 0
    cap=frozen['capacity_by_rho']['1.4'];times=np.r_[inp['times_s'],inp['times_s'][-1]+json.loads((cell/'V0/summary.json').read_text())['dt_s']]/60
    for method in ['V0','Guard*','V1','HBAG']:
        label=guard_label(rho,frozen) if method=='Guard*' else method
        raw=read_raw(cell/label)
        fig,axes=plt.subplots(2,1,figsize=(3.5,4.1),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        fig.subplots_adjust(left=.18,right=.97,bottom=.13,top=.85,hspace=.18)
        axes[0].plot(times,raw['energy_j'][:,sat]/cap,color=COLORS[method],label=method,lw=1)
        axes[0].plot(times,inp['reserve_j'][:,sat]/cap,color='black',ls='--',label='Reserve',lw=.8)
        axes[0].axhline(.7,color='gray',ls=':',label='DoD floor',lw=.8)
        axes[0].set(ylabel='State of charge',title='Seed 4000, satellite %d'%sat)
        axes[1].step(times[:-1],raw['terminals_on'][:,sat].sum(axis=1),where='post',color=COLORS[method])
        axes[1].set(xlabel='Time (min)',ylabel='ON terminals',yticks=[0,2,4])
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        save(fig,'E2_SOC_'+method.replace('*','selected'),[str(cell.relative_to(ROOT)/label),'representative: first satellite with both sun and eclipse'])
        q=np.quantile(raw['energy_j']/cap,[.1,.25,.5,.75,.9],axis=1)
        fig,ax=plt.subplots(figsize=(3.5,2.8));fig.subplots_adjust(left=.19,right=.97,bottom=.2,top=.88)
        ax.fill_between(times,q[0],q[4],color=COLORS[method],alpha=.15,label='10–90%')
        ax.fill_between(times,q[1],q[3],color=COLORS[method],alpha=.3,label='25–75%')
        ax.plot(times,q[2],color=COLORS[method],lw=1,label='Median');ax.axhline(.7,color='gray',ls=':',lw=.8)
        ax.set(xlabel='Time (min)',ylabel='Population SOC',title=method+', seed 4000')
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        save(fig,'E2_SOC_quantiles_'+method.replace('*','selected'),[str(cell.relative_to(ROOT)/label)])
    raw=read_raw(cell/'HBAG')
    fig,axes=plt.subplots(3,1,figsize=(3.5,4.6),sharex=True)
    fig.subplots_adjust(left=.2,right=.97,bottom=.12,top=.94,hspace=.24)
    for ax,key,ylabel in zip(axes,['mu','pace_w','terminal_w'],[r'$\mu$','Pace (W)','ISL power (W)']):
        ax.plot(times[:-1],raw[key][:,sat],color=COLORS['HBAG'],lw=1);ax.set_ylabel(ylabel)
    axes[-1].set_xlabel('Time (min)')
    save(fig,'E2_pacing_satellite',[str(cell.relative_to(ROOT)/'HBAG'),'satellite %d'%sat])
    write_json(ROOT/'figures_r2/representative.json',dict(seed=4000,satellite=sat,rule='First indexed satellite with both sun and eclipse; no outcome ranking'))


def pareto(frozen):
    fig,ax=plt.subplots(figsize=(3.5,3.8));fig.subplots_adjust(left=.2,right=.98,bottom=.18,top=.68)
    for method in ['HBAG','V1','V2','Util-Sleep','Lyapunov']:
        rows=selected_runs(1.4,method,frozen)
        ax.scatter(np.mean([r['min_soc'] for r in rows]),np.mean([r['delivered_bits']/1e12 for r in rows]),label=method,color=COLORS[method],s=20)
    rows=[]
    for theta in [.75,.8,.9]:
        samples=selected_runs(1.4,'Guard_'+str(theta),frozen)
        rows.append([np.mean([r['min_soc'] for r in samples]),np.mean([r['delivered_bits']/1e12 for r in samples])])
    values=np.array(rows);ax.plot(values[:,0],values[:,1],'s-',color=COLORS['Guard*'],lw=1,ms=3,label='Guard 0.75/0.80/0.90')
    ax.set(xlabel='Mean minimum SOC',ylabel='Mean delivered traffic (Tbit)')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    save(fig,'E2_Pareto',['results_r2/E2/rho_1.4/seed_*/*/summary.json'])


def topology():
    cell=ROOT/'results_r2/E2/rho_1.4/seed_4000';inp=inputs_for(cell/'V1')
    # Fixed t=one orbital quarter, independent of either method's state.
    t=24;lo,hi=inp['link_offsets'][t:t+2];edges=inp['directed_edges'][lo:hi];count=len(edges)//2
    xy=np.column_stack([np.arange(576)%24,np.arange(576)//24])
    for method in ['V1','HBAG']:
        raw=read_raw(cell/method);on=raw['link_on'][lo:hi][:count];dark=inp['harvest_w'][t]==0
        chosen=on&dark[edges[:count,0]]&dark[edges[:count,1]]
        # Omit wraparound strokes only for readability; every node remains shown.
        short=np.max(np.abs(xy[edges[:count,0]]-xy[edges[:count,1]]),axis=1)<=1
        lines=xy[edges[:count][chosen&short]]
        fig,ax=plt.subplots(figsize=(3.5,3.5));fig.subplots_adjust(left=.16,right=.96,bottom=.15,top=.86)
        ax.add_collection(LineCollection(lines,colors=COLORS[method],linewidths=.55))
        ax.scatter(xy[dark,0],xy[dark,1],s=3,color='gray',label='Eclipse satellite')
        holes=raw['hole_node_slot'][t]&dark
        ax.scatter(xy[holes,0],xy[holes,1],s=20,marker='x',color='#D55E00',label='Access hole')
        ax.set(xlim=(-1,24),ylim=(-1,24),xlabel='Along-track index',ylabel='Orbital plane',title=method+', slot 24 (wrap edges omitted)')
        ax.set_aspect('equal');fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=1,frameon=False)
        save(fig,'E2_eclipse_topology_'+method,[str(cell.relative_to(ROOT)/method),'fixed slot 24'])


def mechanism(frozen):
    for rho in [1.2,1.4,1.6]:
        fig,ax=plt.subplots(figsize=(3.5,3.3));fig.subplots_adjust(left=.18,right=.98,bottom=.2,top=.74)
        for method in ['V0','Util-Sleep','V1','HBAG']:
            values=np.array([r['eclipse_mean_on_terminals_by_third'] for r in selected_runs(rho,method,frozen)],float)
            ax.plot([1,2,3],values.mean(axis=0),marker='o',ms=3,lw=1,color=COLORS[method],label=method)
        ax.set(xticks=[1,2,3],xticklabels=['Early','Middle','Late'],ylabel='Mean ON terminals',xlabel='Eclipse progress',ylim=(0,4.1))
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        save(fig,'E2_eclipse_progress_rho_'+str(rho),['results_r2/E2/rho_'+str(rho)])


def make_main_figures(frozen):
    style();trends(frozen);trajectories(frozen);pareto(frozen);topology();mechanism(frozen)


def convergence_figures():
    style();rows=json.loads((ROOT/'tables_r2/E9_convergence.json').read_text())
    names={'relative_objective_gap':'Relative objective gap','raw_violation':'Raw constraint violation',
           'recovered_objective':'Recovered objective','kkt_residual':'KKT residual'}
    for size in ['M64','M576']:
        for metric,ylabel in names.items():
            for xkey,xlabel in [('iteration','Iterations'),('messages','Logical scalar messages')]:
                fig,ax=plt.subplots(figsize=(3.5,3.1));fig.subplots_adjust(left=.22,right=.98,bottom=.22,top=.8)
                for method,color in [('HBAG','#0072B2'),('DualDecomp','#D55E00'),('ADMM','#009E73')]:
                    group=sorted([r for r in rows if r['size']==size and r['method']==method and r['metric']==metric],key=lambda r:r['iteration'])
                    message={r['iteration']:r['mean'] for r in rows if r['size']==size and r['method']==method and r['metric']=='logical_scalar_messages'}
                    x=np.array([r['iteration'] if xkey=='iteration' else message[r['iteration']] for r in group])
                    y=np.array([r['mean'] for r in group]);lo=np.array([r['ci95_low'] for r in group]);hi=np.array([r['ci95_high'] for r in group])
                    # Exact zeros stay zero; symlog displays them without invented epsilon values.
                    ax.plot(x,y,color=color,lw=1,label=method);ax.fill_between(x,lo,hi,color=color,alpha=.15)
                ax.set_xscale('log')
                if metric!='recovered_objective':ax.set_yscale('symlog',linthresh=1e-10)
                ax.set(xlabel=xlabel,ylabel=ylabel,title=size+', seed means with 95% CI')
                fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
                save(fig,'E9_'+size+'_'+metric+'_'+xkey,['tables_r2/E9_convergence.json'])
