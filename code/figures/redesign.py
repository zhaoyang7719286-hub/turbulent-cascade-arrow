"""Data-only redesign. No invented fields, fitted master curves or new simulations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def setup(root, load_, export_, save_, check_, colors, styles):
    global ROOT,load,export,save,check,C,STYLE
    ROOT,load,export,save,check,C,STYLE=root,load_,export_,save_,check_,colors,styles
    C['FRZ']='#757575'

def label(ax, letter, title):
    ax.set_title(title,loc='left',pad=9,fontsize=9)
    pos=ax.get_position()
    x=max(-.15,(.025-pos.x0)/pos.width)
    ax.text(x,1.08,letter,transform=ax.transAxes,fontsize=11,weight='bold')

def figure1(main,seeds,qc):
    fig=plt.figure(figsize=(7.1,4.6))
    gs=fig.add_gridspec(2,2,left=.11,right=.98,top=.91,bottom=.19,
        hspace=.62,wspace=.42,height_ratios=[1,1])
    ax=fig.add_subplot(gs[0,0]);label(ax,'a','Resolved spectra: reference projection')
    d=load('spectral_energy/gate2_spectrum_diagnostics.csv')
    # A common DNS normalization; shell 0 is retained in the exported table.
    e0=d.dns_mean_spectrum.sum();d['E0_DNS_all_shells']=e0
    for col,co,ls,mk,txt in [('dns_mean_spectrum',C['DNS'],'-',None,'DNS'),
        ('star64_mean_spectrum','#BA6900','--','o',r'$u^*_{64}$'),
        ('star32_mean_spectrum','#8464A4',':','s',r'$u^*_{32}$')]:
        d[col+'_over_E0']=d[col]/e0
        q=d[(d.shell>0)&(d[col]>0)]
        ax.loglog(q.shell,q[col]/e0,color=co,ls=ls,marker=mk,markevery=5,
            mfc='none',ms=3.8,lw=1.1,label=txt)
    ax.set_xlabel('Shell index q (nonzero modes)');ax.set_ylabel(r'$E_q/E_0$')
    lo=d.loc[d.shell>0,['dns_mean_spectrum','star64_mean_spectrum','star32_mean_spectrum']].min().min()/e0
    ax.set_ylim(lo*.4,.12);ax.set_xlim(1,56)
    ax.legend(frameon=False,fontsize=7,ncol=3,loc='lower left',handlelength=1.6)
    export(d,'Fig1a_normalized_reference_spectra.csv')
    ax=fig.add_subplot(gs[0,1]);label(ax,'b','Phase intervention: measured fidelity')
    cols=['spectrum_error_max','energy_ratio_error_max','unit_modulus_error_max']
    out=[]
    for i,col in enumerate(cols):
        x=i+np.linspace(-.15,.15,len(qc));v=qc[col].to_numpy()
        ax.scatter(x,v,s=13,color=C['null'],alpha=.65,edgecolors='none')
        for j,(_,r) in enumerate(qc.iterrows()):
            out.append(dict(metric=col,display_x=x[j],error=r[col],seed=r.seed,
                source_file=r.source_file,source_row_1based=r.source_row_1based))
    ax.set_yscale('log');ax.set_ylim(1e-16,2e-15)
    ax.set_xticks(range(3),['Spectrum','Energy','Phase modulus']);ax.tick_params(axis='x',length=0)
    ax.set_ylabel('Recorded numerical error')
    export(pd.DataFrame(out),'Fig1b_all_mask_fidelity.csv')
    ax=fig.add_subplot(gs[1,:]);label(ax,'c','The signed arrow changes while quadratic information is retained')
    out=[]
    for i,(scale,metric,txt) in enumerate([(4,'normalized_mean_Pi','Mean, scale 4'),
        (8,'normalized_mean_Pi','Mean, scale 8'),(4,'sign_bias_Pi','Sign, scale 4'),
        (8,'sign_bias_Pi','Sign, scale 8')]):
        ref=main[(main.scale==scale)&(main.metric==metric)&(main.statistic_type=='instantaneous')].iloc[0]
        q=seeds[(seeds.scale==scale)&(seeds.metric==metric)&(seeds.statistic_type=='instantaneous')]
        check('Fig1 primary full-mask group '+str((scale,metric)),len(q)==32 and ref.dns_value!=0)
        x=i+np.linspace(-.15,.15,len(q));y=q.value.to_numpy()/ref.dns_value
        ax.scatter(x,y,s=14,color=C['null'],alpha=.55,edgecolors='none',zorder=2)
        ax.plot(i,1,'*',color=C['DNS'],ms=8,zorder=3)
        for j,(_,r) in enumerate(q.iterrows()):
            out.append(dict(metric=metric,scale=scale,seed=r.seed,original_value=r.value,
                reference_value=ref.dns_value,normalized_by_reference=y[j],display_x=x[j],
                source_file=r.source_file,source_row_1based=r.source_row_1based,
                reference_source_file=ref.source_file,reference_source_row_1based=ref.source_row_1based))
    ax.axhline(1,color=C['DNS'],ls=':',lw=.8);ax.axhline(0,color='#888888',ls=':',lw=.8)
    ax.set_xticks(range(4),['Mean\nscale 4','Mean\nscale 8','Sign bias\nscale 4','Sign bias\nscale 8'])
    ax.set_ylabel('Observable / DNS reference');ax.set_ylim(-1.18,1.22);ax.set_xlim(-.4,3.5)
    ax.text(.98,.9,'DNS',transform=ax.transAxes,ha='right',color=C['DNS'],fontsize=7)
    ax.text(.98,.1,'32 phase masks',transform=ax.transAxes,ha='right',color=C['null'],fontsize=7)
    export(pd.DataFrame(out),'Fig1c_reference_normalized_arrow.csv')
    fig.text(.11,.083,r'$\hat{u}^{\phi}_{\mathbf{k}}=e^{i\phi_{\mathbf{k}}}\hat{u}_{\mathbf{k}}$: same phase across components and times; Hermitian pairing.',fontsize=8)
    fig.text(.11,.018,'(a) DNS versus projected references, not surrogate spectra. (b,c) Primary phase-mask block.\nReference normalization enables comparison; it does not demonstrate universal collapse.',fontsize=6.8,color='#555555')
    save(fig,'Fig1_Intervention')

def figure3(marks,ep,raw,amp):
    tr=marks[marks.relative_time<=1+1e-12].copy()
    export(tr,'Fig3a_all_297_trajectory_marks.csv')
    fig=plt.figure(figsize=(7.1,4.5))
    gs=fig.add_gridspec(2,2,left=.10,right=.98,bottom=.19,top=.91,
        width_ratios=[1.5,1],height_ratios=[.72,1.28],wspace=.39,hspace=.58)
    ax=fig.add_subplot(gs[:,0]);label(ax,'a','Regeneration requires nonlinear evolution')
    ax.axhline(1,color='#888888',ls=':',lw=.8);ax.axhline(0,color='#888888',ls=':',lw=.6)
    for b in ['NL','LIN','FRZ']:
        for (cp,seed),d in tr[tr.branch==b].groupby(['checkpoint','phase_seed']):
            d=d.sort_values('relative_time');ls,mk=STYLE[b]
            ax.plot(d.relative_time,d.R,color=C[b],ls=ls,marker=mk,ms=2.3,
                lw=1.05,alpha=.6,mfc=C[b] if b=='NL' else 'white',mew=.6)
    ax.set_xlabel('Elapsed simulation time t');ax.set_ylabel(r'$R_\Pi(t)$')
    ax.set_xlim(0,1.06);ax.set_ylim(-.17,1.22)
    hs=[Line2D([0],[0],color=C[b],ls=STYLE[b][0],marker=STYLE[b][1],ms=4,
        mfc=C[b] if b=='NL' else 'white',label=b) for b in ['NL','LIN','FRZ']]
    ax.legend(handles=hs,frameon=False,ncol=3,loc='upper left',handlelength=1.4,columnspacing=.9)
    ax.text(.04,.86,'9 crossed cases per branch',transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(gs[0,1]);label(ax,'b','FRZ copies NL modal amplitudes')
    ds=[]
    for p in sorted((ROOT/'sources/audits').rglob('amplitude_match_audit.csv')):
        d=pd.read_csv(p);d['source_file']=str(p.relative_to(ROOT));d['source_row_1based']=np.arange(1,len(d)+1)
        ds.append(d)
    d=pd.concat(ds,ignore_index=True)
    # Source audit local_step is mapped to time using the recorded diagnostic table.
    mapping=amp[['local_step','relative_time']].drop_duplicates()
    check('FRZ step/time map unique',mapping.local_step.is_unique)
    if 'relative_time' in d:
        check('FRZ recorded audit time matches diagnostic map',np.allclose(
            d.relative_time,d.local_step.map(dict(zip(mapping.local_step,mapping.relative_time))),atol=1e-12))
    else:
        d=d.merge(mapping,on='local_step',how='left',validate='many_to_one')
    check('FRZ time mapping complete',d.relative_time.notna().all())
    ax.scatter(d.relative_time,d.modal_amplitude_relative_max_error*1e15,
        s=7,color=C['FRZ'],alpha=.48,edgecolors='none')
    ax.set_xlim(0,3);ax.set_ylim(0,max(.65,float(d.modal_amplitude_relative_max_error.max()*1e15)*1.2))
    ax.set_xlabel('Elapsed time t');ax.set_ylabel(r'Max. relative error ($10^{-15}$)')
    export(d,'Fig3b_modal_amplitude_audit.csv')
    ax=fig.add_subplot(gs[1,1]);label(ax,'c','Matched endpoints at t = 1')
    ax.axhline(1,color='#888888',ls=':',lw=.8);ax.axhline(0,color='#BBBBBB',ls=':',lw=.6)
    for i,cp in enumerate([1750,2250,2750]):
        for j,seed in enumerate([2026074101,2026074102,2026074103]):
            x=i*4+j;q=ep[(ep.checkpoint==cp)&(ep.phase_seed==seed)]
            ax.plot([x,x],[q.R.min(),q.R.max()],color='#D5D5D5',lw=.6,zorder=0)
            for b,off in [('NL',-.17),('LIN',0),('FRZ',.17)]:
                ax.plot(x+off,q[q.branch==b].R.iloc[0],marker=STYLE[b][1],color=C[b],
                    mfc=C[b] if b=='NL' else 'white',ms=4,ls='')
    ax.set_xticks([1,5,9],['1750','2250','2750']);ax.set_xlabel('Checkpoint (3 masks each)')
    ax.set_xlim(-.6,10.6);ax.set_ylim(-.17,1.1);ax.set_ylabel(r'$R_\Pi(1)$')
    fig.text(.10,.087,r'$R_\Pi=1-|X_b-X_{\rm nat}|/D_0,\quad X=\langle\Pi\rangle/\langle|\Pi|\rangle,\quad D_0=|X_{\rm nat}(0)-X_{\rm pert}(0)|$',fontsize=8)
    fig.text(.10,.018,'Generated forced periodic DNS; box filter 8 cells. NL: full dynamics; LIN: quadratic advection off.\nFRZ: NL vector-modal amplitudes + initial scrambled directions. Each line is a recorded case; no pooled CI.',fontsize=6.8,color='#555555')
    save(fig,'Fig3_Dynamic_Recovery')

def figure4(raw,mech,masks):
    rows=[]
    for (cp,seed),d in raw.groupby(['checkpoint','phase_seed']):
        n=d[d.branch=='natural'].sort_values('relative_time')
        start=d[d.branch=='nonlinear_recovery'].sort_values('relative_time')
        for br,b in [('nonlinear_recovery','NL'),('linear_control','LIN')]:
            q=d[d.branch==br].sort_values('relative_time')
            for col,metric in [('Pi_normalized_mean','Flux'),('M_normalized_mean','Geometry')]:
                D0=abs(n[col].iloc[0]-start[col].iloc[0])
                check('Geometry/flux initial distance nonzero '+str((cp,seed,metric)),D0>1e-10)
                v=1-np.abs(q[col].to_numpy()-n[col].to_numpy())/D0
                for j,(_,r) in enumerate(q.iterrows()):
                    rows.append(dict(checkpoint=cp,phase_seed=seed,branch=b,metric=metric,
                        relative_time=r.relative_time,value=r[col],natural_value=n[col].iloc[j],D0=D0,
                        recovery=v[j],source_file=r.source_file,source_row_1based=r.source_row_1based,
                        natural_source_file=n.source_file.iloc[j],natural_source_row_1based=n.source_row_1based.iloc[j],
                        initial_source_row_1based=start.source_row_1based.iloc[0],
                        natural_initial_source_row_1based=n.source_row_1based.iloc[0]))
    d=pd.DataFrame(rows);export(d,'Fig4a_joint_recovery_all_values.csv')
    shown=d[d.relative_time<=1+1e-12].copy();export(shown,'Fig4a_heatmap_396_cells.csv')
    check('Heatmap all 396 recorded cells retained',len(shown)==396)
    check('Heatmap range not clipped',shown.recovery.min()>=-.25 and shown.recovery.max()<=1.001)
    fig=plt.figure(figsize=(7.1,5.4))
    gs=fig.add_gridspec(3,3,left=.14,right=.90,bottom=.13,top=.91,
        width_ratios=[1,1,.05],height_ratios=[1,1,.85],wspace=.24,hspace=.72)
    cmap=LinearSegmentedColormap.from_list('signed_recovery',['#D57843','#FFFFFF','#183F66'])
    norm=TwoSlopeNorm(vmin=-.25,vcenter=0,vmax=1)
    cases=[(cp,seed) for cp in [1750,2250,2750] for seed in masks]
    for i,metric in enumerate(['Flux','Geometry']):
        for j,b in enumerate(['NL','LIN']):
            ax=fig.add_subplot(gs[i,j]);q=shown[(shown.branch==b)&(shown.metric==metric)]
            mat=q.pivot(index=['checkpoint','phase_seed'],columns='relative_time',values='recovery').reindex(cases)
            check('Heatmap populated '+b+metric,mat.shape==(9,11) and mat.notna().all().all())
            # Vector cells: every heatmap rectangle remains editable in SVG/PDF.
            im=ax.pcolormesh(np.arange(12)-.5,np.arange(10)-.5,mat.to_numpy(),
                cmap=cmap,norm=norm,shading='flat',rasterized=False)
            ax.set_ylim(8.5,-.5);ax.set_xlim(-.5,10.5)
            ax.set_xticks([0,5,10],['0','0.5','1']);ax.set_xlabel('Elapsed time t')
            if j==0:
                ax.set_yticks(range(9),[f'{cp} / {s}' for cp in [1750,2250,2750] for s in [1,2,3]])
                ax.set_ylabel('Checkpoint / mask')
            else:ax.set_yticks([])
            ax.set_title(b+' · '+(r'$R_\Pi$' if metric=='Flux' else r'$R_M$'),loc='left',pad=7,
                color=C[b],fontsize=9)
            for y in [2.5,5.5]:ax.axhline(y,color='#808080',lw=.8)
            ax.spines[['top','right','bottom','left']].set_visible(False)
    cax=fig.add_subplot(gs[:2,2]);cb=fig.colorbar(im,cax=cax,ticks=[-.25,0,.5,1],label='Scalar recovery R')
    cb.solids.set_rasterized(False)
    fig.text(.045,.955,'a',fontsize=11,weight='bold')
    fig.text(.14,.955,'Flux and geometry recover together across every recorded case',fontsize=9)
    ax=fig.add_subplot(gs[2,:2]);label(ax,'b','Local correlation persists even when the arrow is erased')
    out=[]
    for i,(metric,txt) in enumerate([('corr_Pi_M_pooled','M'),('corr_Pi_T_pooled','T'),
        ('corr_Pi_quarter_W_pooled','W/4')]):
        for scale,off,mk in [(4,-.14,'o'),(8,.14,'s')]:
            r=mech[(mech.metric==metric)&(mech.scale==scale)].iloc[0];x=i+off
            ax.plot([x,x],[r.surrogate_mean,r.dns_value],color='#CFCFCF',lw=1)
            ax.plot(x,r.dns_value,marker=mk,color=C['DNS'],ms=5,ls='')
            ax.errorbar(x,r.surrogate_mean,yerr=r.surrogate_std,fmt=mk,color=C['null'],
                mfc='white',ms=4,capsize=2,lw=.8)
            out.append(dict(metric=metric,scale=scale,display_x=x,DNS_correlation=r.dns_value,
                null_correlation_mean=r.surrogate_mean,null_correlation_sd=r.surrogate_std,
                source_file=r.source_file,source_row_1based=r.source_row_1based))
    ax.set_xticks([0,1,2],['M','T','W/4']);ax.set_xlim(-.5,2.5);ax.set_ylim(0,1.02)
    ax.set_ylabel(r'$\mathrm{corr}(\Pi,Q)$')
    ax.text(.99,.035,'DNS: filled; phase ensemble: open ± SD\nScale 4: circles; scale 8: squares',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=6.7)
    export(pd.DataFrame(out),'Fig4b_static_association.csv')
    fig.text(.14,.027,'(a) Generated DNS, scale 8; R uses a fixed initial distance for each scalar. FRZ M is unavailable.\n(b) Static JHTDB block. Local association is not a sufficient test of the global signed arrow.',fontsize=6.8,color='#555555')
    save(fig,'Fig4_Geometry_Association')

def collapse_diagnostic(marks,raw):
    n0=raw[(raw.branch=='natural')&np.isclose(raw.relative_time,0)].copy()
    check('Initial energy-time duplicated natural rows agree',
        n0.groupby('checkpoint')[['energy','dissipation_rate']].nunique().max().max()==1)
    n0=n0.drop_duplicates('checkpoint');n0['T_E']=n0.energy/n0.dissipation_rate
    check('Energy time is positive finite',np.isfinite(n0.T_E).all() and (n0.T_E>0).all())
    export(n0,'FigS1_initial_energy_times.csv')
    te=dict(zip(n0.checkpoint,n0.T_E));d=marks.copy();d['T_E']=d.checkpoint.map(te)
    d['t_over_T_E']=d.relative_time/d.T_E;export(d,'FigS1_all_scaled_trajectories.csv')
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
    export(pd.DataFrame(comparison),'FigS1_spread_comparison_interpolated_not_observed.csv')
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
    save(fig,'FigS1_Normalization_Test')
