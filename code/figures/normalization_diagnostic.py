from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def label(ax,letter,title):
    pos=ax.get_position();x=max(-.15,(.025-pos.x0)/pos.width)
    ax.text(x,1.08,letter,transform=ax.transAxes,fontsize=11,weight='bold')
    ax.set_title(title,loc='left',pad=8,fontsize=9)

def run(root, marks, raw, export_, save_, check_, colors, styles):
    global ROOT,export,save,check,C,STYLE
    ROOT,export,save,check,C,STYLE=root,export_,save_,check_,colors,styles
    collapse_diagnostic(marks,raw)

def collapse_diagnostic(marks,raw):
    n0=raw[(raw.branch=='natural')&np.isclose(raw.relative_time,0)].copy()
    check('Initial energy-time duplicated natural rows agree',
        n0.groupby('checkpoint')[['energy','dissipation_rate']].nunique().max().max()==1)
    n0=n0.drop_duplicates('checkpoint');n0['T_E']=n0.energy/n0.dissipation_rate
    check('Energy time is positive finite',np.isfinite(n0.T_E).all() and (n0.T_E>0).all())
    export(n0,'FigS6_initial_energy_times.csv')
    te=dict(zip(n0.checkpoint,n0.T_E));d=marks.copy();d['T_E']=d.checkpoint.map(te)
    d['t_over_T_E']=d.relative_time/d.T_E;export(d,'FigS6_all_scaled_trajectories.csv')
    nl=d[(d.branch=='NL')&(d.relative_time<=1+1e-12)]
    t_ref=np.median(list(te.values()));limit=min(t_ref/np.array(list(te.values())))
    grid=np.linspace(0,limit,101);native=[];scaled=[];comparison=[]
    for (cp,seed),q in nl.groupby(['checkpoint','phase_seed']):
        q=q.sort_values('relative_time')
        a=np.interp(grid,q.relative_time,q.R);b=np.interp(grid,q.relative_time*t_ref/te[cp],q.R)
        native.append(a);scaled.append(b)
        for j in range(len(grid)):
            comparison.append(dict(checkpoint=cp,phase_seed=seed,common_time_in_median_TE_units=grid[j],
                R_native_interpolated=a[j],R_TE_interpolated=b[j]))
    score_native=float(np.sqrt(np.mean(np.var(native,axis=0))))
    score_scaled=float(np.sqrt(np.mean(np.var(scaled,axis=0))))
    report={'candidate':'T_E=natural initial energy / natural initial dissipation_rate',
        'T_E_by_checkpoint':te,'no_fitted_parameters':True,'n_crossed_cases':9,
        'window':'0 <= t <= 1, selected NL flux scalar',
        'comparison_domain_end':float(limit),'median_TE':float(t_ref),
        'score_definition':'RMS across-case population spread over 101 shared comparison coordinates; descriptive only',
        'interpolation':'piecewise linear within recorded support, diagnostic score only',
        'RMS_spread_native':score_native,'RMS_spread_energy_scaled':score_scaled,
        'relative_spread_change':score_scaled/score_native-1,
        'conclusion':'No improved collapse for this candidate in this dataset; no universal-law inference.'}
    (ROOT/'collapse_diagnostic.json').write_text(json.dumps(report,indent=2))
    export(pd.DataFrame(comparison),'FigS6_spread_comparison_interpolated_not_observed.csv')
    fig=plt.figure(figsize=(7.1,4.8));gs=fig.add_gridspec(2,2,left=.1,right=.98,
        bottom=.14,top=.9,wspace=.30,hspace=.6)
    for j,(xcol,xlab,title) in enumerate([('relative_time','Elapsed time t','Native recorded time'),
        ('t_over_T_E',r'$t/T_E$,  $T_E=K(0)/\varepsilon(0)$','Energy-time normalization')]):
        ax=fig.add_subplot(gs[0,j]);label(ax,'ab'[j],title)
        for (cp,seed),q in nl.groupby(['checkpoint','phase_seed']):
            q=q.sort_values('relative_time');ax.plot(q[xcol],q.R,color=C['NL'],alpha=.58,marker='o',ms=2,lw=1)
        ax.set_xlabel(xlab);ax.set_ylabel(r'$R_\Pi$');ax.set_ylim(0,1.07)
        score=score_native if j==0 else score_scaled
        ax.text(.98,.1,f'RMS spread = {score:.4f}',transform=ax.transAxes,ha='right',fontsize=7)
    ax=fig.add_subplot(gs[1,:]);label(ax,'c','Full recorded interval: all three branches')
    for b in ['NL','LIN','FRZ']:
        for (cp,seed),q in d[d.branch==b].groupby(['checkpoint','phase_seed']):
            q=q.sort_values('relative_time');ax.plot(q.relative_time,q.R,color=C[b],ls=STYLE[b][0],alpha=.55,lw=1)
    ax.axhline(0,color='#888888',ls=':',lw=.7);ax.axhline(1,color='#888888',ls=':',lw=.7)
    ax.set_xlabel('Elapsed time t');ax.set_ylabel(r'$R_\Pi$');ax.set_xlim(0,3);ax.set_ylim(-.22,1.09)
    ax.legend(handles=[Line2D([0],[0],color=C[b],ls=STYLE[b][0],label=b) for b in ['NL','LIN','FRZ']],
        ncol=3,frameon=False,loc='center right')
    fig.text(.1,.025,'No fitted master curve. Energy scaling increases descriptive spread by '+
        f'{100*(score_scaled/score_native-1):.1f}%. Same nine crossed cases; no universality claim.',fontsize=7,color='#555555')
    save(fig,'FigS6_Normalization_Test')
