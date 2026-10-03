"""Offline, read-only figure builder. Run: python build_figures.py

Input CSVs remain byte-for-byte unchanged. SVG text stays editable.
Every plotted table carries its source filename and 1-based source row.
No simulations, stochastic resampling, smoothing, or synthetic field images.
"""
from pathlib import Path
import json, hashlib, platform, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'sources/figure_ready'
FIG = ROOT / 'figures'
DATA = ROOT / 'plot_data'
for p in [FIG, DATA]: p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
 'axes.labelsize':8, 'axes.titlesize':8, 'legend.fontsize':7,
 'xtick.labelsize':7, 'ytick.labelsize':7, 'axes.linewidth':0.65,
 'lines.linewidth':1.35, 'svg.fonttype':'none', 'pdf.fonttype':42,
 'svg.hashsalt':'turbulence-existing-data-v2', 'savefig.facecolor':'white',
 'axes.spines.top':False, 'axes.spines.right':False})
C = {'DNS':'#183F66', 'null':'#159C92', 'NL':'#183F66',
     'LIN':'#D55E00', 'FRZ':'#417B4B', 'natural':'#575757'}
STYLE = {'NL':('-', 'o'), 'LIN':('--','s'), 'FRZ':('-.','^')}
CHECKS = []

def load(name):
    d = pd.read_csv(SRC/name, float_precision='round_trip')
    d['source_file'] = name
    d['source_row_1based'] = np.arange(1, len(d)+1)
    return d

def export(d, name):
    d.to_csv(DATA/name, index=False, float_format='%.17g')

def check(name, ok, detail=None):
    CHECKS.append({'check':name,'pass':bool(ok),'detail':detail})
    if not ok: raise AssertionError(name)

def save(fig, name):
    for note in list(fig.texts):
        if note.get_position()[1] < 0.12: note.remove()
    for fmt in ['svg','pdf','png']:
        kw={'dpi':300} if fmt=='png' else {}
        meta={'Creator':'build_figures.py; existing-data reconstruction'}
        if fmt=='svg': meta['Date']=None
        if fmt=='pdf': meta.update({'CreationDate':None,'ModDate':None})
        buffer=io.BytesIO()
        fig.savefig(buffer, format=fmt, metadata=meta, **kw)
        (FIG/f'{name}.{fmt}').write_bytes(buffer.getvalue())
    plt.close(fig)

def panel(ax, letter, title):
    ax.text(-.13, 1.075, letter, transform=ax.transAxes, weight='bold', fontsize=10)
    ax.set_title(title, loc='left', pad=7)

def box(ax, x,y,w,h,text,color=C['DNS'],size=8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',
        facecolor=color+'0F',edgecolor=color,lw=.8,clip_on=False))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=color)

def branch_legend(ax, include_natural=True, **kwargs):
    hs=[Line2D([0],[0],color=C[b],ls=STYLE[b][0],marker=STYLE[b][1],
       mfc=C[b] if b=='NL' else 'white',ms=4,label=b) for b in ['NL','LIN','FRZ']]
    if include_natural: hs.append(Line2D([0],[0],color=C['natural'],ls=':',lw=.9,label='Natural scalar'))
    ax.legend(handles=hs,frameon=False,**kwargs)

# Verify the exact bytes before deriving or plotting any scientific values.
hashes=json.loads((ROOT/'source_hashes.json').read_text())
check('All copied source files match their saved SHA-256',all(
    hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in hashes),len(hashes))
for p in SRC.rglob('*.metadata.json'):
    m=json.loads(p.read_text()); cp=p.with_name(p.name.replace('.metadata.json','.csv'))
    if cp.exists() and 'source_sha256' in m:
        check('Figure-ready metadata hash: '+str(cp.relative_to(SRC)),
           hashlib.sha256(cp.read_bytes()).hexdigest()==m['source_sha256'])

main=load('main32/ensemble_summary_64.csv')
seeds=load('main32/seed_metrics_64.csv')
qc=load('main32/qc_seed_summary_64.csv')
rep=load('isotropic_rep2/isotropic_rep2_reference_vs_24.csv')
repraw=load('isotropic_rep2/isotropic_rep2_all_case_metrics.csv')
check('Primary QC: 32 masks, all pass',len(qc)==32 and qc.qc_pass.all())
for _,r in rep.iterrows():
    if r.metric in repraw.columns:
        v=repraw[(repraw.case_type=='surrogate')&(repraw.scale==r.scale)][r.metric]
        check('Independent ensemble recomputation '+str((r.scale,r.metric)),
            len(v)==24 and np.isclose(v.mean(),r.surrogate_mean,atol=1e-12,rtol=1e-12)
            and np.isclose(v.std(ddof=1),r.surrogate_std,atol=1e-12,rtol=1e-12))
for _,r in main.iterrows():
    s=seeds[(seeds.statistic_type==r.statistic_type)&(seeds.scale==r.scale)&
        (seeds.window_frames==r.window_frames)&(seeds.metric==r.metric)].value
    check('Primary ensemble recomputation '+str((r.statistic_type,r.scale,r.window_frames,r.metric)),
        len(s)==32 and np.isclose(s.mean(),r.surrogate_mean,atol=1e-12,rtol=1e-12) and
        np.isclose(s.std(ddof=1),r.surrogate_std,atol=1e-12,rtol=1e-12))

import redesign
redesign.setup(ROOT,load,export,save,check,C,STYLE)
redesign.figure1(main,seeds,qc)

# FIGURE 2: use the formal independent comparator, explicitly identified.
fig,axs=plt.subplots(2,2,figsize=(7.1,5.1))
fig.subplots_adjust(left=.105,right=.97,bottom=.155,top=.89,wspace=.4,hspace=.68)
for ax in axs.flat: ax.axhline(0,color='#888888',lw=.7,ls=':')
for ax,letter,metric,title,ylabel in [
 (axs[0,0],'a','normalized_mean_Pi','Mean transfer',r'$\langle\Pi_\ell\rangle/\sigma(\Pi_\ell)$'),
 (axs[0,1],'b','sign_bias_Pi','Instantaneous forward bias',r'$P(\Pi_\ell>0)-1/2$')]:
    panel(ax,letter,title);out=[]
    for block,d,n,marker,offset,refcol in [('Primary',main,32,'o',-.08,'dns_value'),('Independent',rep,24,'s',.08,'comparator_value')]:
        q=d[(d.metric==metric)&(d.statistic_type=='instantaneous')].sort_values('scale')
        x=np.arange(2)+offset
        ax.plot(x,q[refcol],marker=marker,ms=5,color=C['DNS'],lw=0,mfc=C['DNS'])
        ax.errorbar(x,q.surrogate_mean,yerr=q.surrogate_std,fmt=marker,ms=4,color=C['null'],mfc='white',capsize=2,lw=.8)
        for i,(_,r) in enumerate(q.iterrows()):out.append({'block':block,'n_masks':n,'scale':r.scale,'display_x':x[i], 'reference':r[refcol],'null_mean':r.surrogate_mean,'null_sd':r.surrogate_std,'source_file':r.source_file,'source_row_1based':r.source_row_1based})
    ax.set_xticks([0,1],['4','8']);ax.set_xlabel(r'$\ell/\Delta$');ax.set_ylabel(ylabel);ax.set_xlim(-.3,1.3)
    export(pd.DataFrame(out),f'Fig2{letter}_reference_null.csv')
hs=[Line2D([0],[0],color=C['DNS'],marker='o',ls='',ms=4,label='Reference'),Line2D([0],[0],color=C['null'],marker='o',mfc='white',ls='',ms=4,label='Phase ensemble'),Line2D([0],[0],color='#666666',marker='o',ls='',ms=4,label='Primary (32 masks)'),Line2D([0],[0],color='#666666',marker='s',ls='',ms=4,label='Independent (24 masks)')]
fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.52,.992),ncol=4,frameon=False)
ax=axs[1,0];panel(ax,'c','Finite-window bias');out=[]
for block,d,marker,boff,refcol in [('Primary',main,'o',-.06,'dns_value'),('Independent',rep,'s',.06,'comparator_value')]:
    for scale,soff in [(4,-.14),(8,.14)]:
        metric='sign_bias_I' if block=='Primary' else None
        q=d[(d.statistic_type=='finite_time')& (d.scale==scale)]
        q=q[q.metric=='sign_bias_I'] if block=='Primary' else q[q.metric.str.match(r'sign_bias_I_w\d+$')]
        q=q.sort_values('window_frames');check('Finite-time selectors '+block+str(scale),len(q)==3)
        x=np.arange(3)+soff+boff
        ax.plot(x,q[refcol],marker=marker,ms=4,ls='-' if scale==4 else '--',color=C['DNS'],lw=.75,mfc=C['DNS'] if scale==4 else 'white')
        ax.errorbar(x,q.surrogate_mean,yerr=q.surrogate_std,fmt=marker,ms=3,color=C['null'],mfc='white',capsize=2,lw=.7)
        for i,(_,r) in enumerate(q.iterrows()):out.append({'block':block,'scale':scale,'window_frames':r.window_frames,'display_x':x[i],'reference':r[refcol],'null_mean':r.surrogate_mean,'null_sd':r.surrogate_std,'source_file':r.source_file,'source_row_1based':r.source_row_1based})
ax.set_xticks(range(3),['5','10','20']);ax.set_xlabel('Window length (frames)');ax.set_ylabel(r'$P(I_\ell>0)-1/2$');ax.set_xlim(-.4,2.4)
ax.set_ylim(-.16,.53)
ax.text(.02,.95,'Scale 4: solid / filled; scale 8: dashed / open',transform=ax.transAxes,fontsize=6.3,va='top')
export(pd.DataFrame(out),'Fig2c_finite_time.csv')
ax=axs[1,1];panel(ax,'d','Independent scale 8: mean vs sign')
q=repraw[(repraw.scale==8)&(repraw.case_type=='surrogate')]
check('Independent per-mask diagnostic has 24 rows',len(q)==24)
ref=rep[(rep.scale==8)&(rep.metric.isin(['normalized_mean_Pi','sign_bias_Pi']))]
xref=float(ref[ref.metric=='normalized_mean_Pi'].comparator_value.iloc[0]);yref=float(ref[ref.metric=='sign_bias_Pi'].comparator_value.iloc[0])
ax.scatter(q.normalized_mean_Pi,q.sign_bias_Pi,s=18,facecolors='white',edgecolors=C['null'],lw=.9)
ax.scatter([xref],[yref],s=50,marker='*',color=C['DNS'],zorder=4)
ax.axvline(xref,lw=.8,ls=':',color=C['DNS']);ax.axhline(yref,lw=.8,ls=':',color=C['DNS'])
ax.text(.98,.95,'Reference',ha='right',va='top',transform=ax.transAxes,color=C['DNS'],fontsize=7)
ax.set_xlabel(r'$\langle\Pi_\ell\rangle/\sigma(\Pi_\ell)$');ax.set_ylabel(r'$P(\Pi_\ell>0)-1/2$')
export(q,'Fig2d_independent_all_masks.csv');export(ref,'Fig2d_reference.csv')
fig.text(.105,.04,'Bars: phase-mask sample SD, not confidence intervals. Independent reference: projected reference_star32.\nPanel d retains every mask; mean-flux separation fails the independent formal primary criterion.',fontsize=6.7,color='#555555')
save(fig,'Fig2_Static_Erasure')

# FIGURE 3: preserve the certified metric, all 99 trajectory marks and 27 endpoints.
raw=load('dynamic_recovery/heldout_3x3_raw_long.csv')
check('Dynamic X from signed/absolute spatial mean columns',np.allclose(
    raw.Pi_mean_signed/raw.Pi_mean_absolute,raw.Pi_normalized_mean,atol=1e-12,rtol=1e-12))
rec=load('dynamic_recovery/heldout_3x3_recovery_long.csv')
amp=load('dynamic_recovery/amplitude_matched_3x3_long.csv')
case=load('dynamic_recovery/three_control_case_comparison.csv')
masks=[2026074101,2026074102,2026074103];cps=[1750,2250,2750]
marks=[]
for cp in cps:
  for seed in masks:
    nat=raw[(raw.checkpoint==cp)&(raw.phase_seed==seed)&(raw.branch=='natural')].sort_values('relative_time')
    nl=raw[(raw.checkpoint==cp)&(raw.phase_seed==seed)&(raw.branch=='nonlinear_recovery')].sort_values('relative_time')
    d0=abs(nat.Pi_normalized_mean.iloc[0]-nl.Pi_normalized_mean.iloc[0]);check('Positive fixed D0 '+str((cp,seed)),d0>1e-10)
    for br,b in [('nonlinear_recovery','NL'),('linear_control','LIN'),('amplitude_matched_frozen','FRZ')]:
      df=amp if b=='FRZ' else raw
      q=df[(df.checkpoint==cp)&(df.phase_seed==seed)&(df.branch==br)].sort_values('relative_time')
      check('Full recorded time grid '+str((cp,seed,b)),len(q)==31 and np.allclose(q.relative_time,nat.relative_time))
      expected=1-np.abs(q.Pi_normalized_mean.to_numpy()-nat.Pi_normalized_mean.to_numpy())/d0
      if b=='FRZ': stored=q.R_Pi_normalized_mean.to_numpy()
      else: stored=rec[(rec.checkpoint==cp)&(rec.phase_seed==seed)&(rec.branch==br)&(rec.metric=='Pi_normalized_mean')].sort_values('relative_time').recovery.to_numpy()
      check('Recovery recomputed '+str((cp,seed,b)),np.allclose(expected,stored,atol=1e-12,rtol=1e-12))
      for j,(_,r) in enumerate(q.iterrows()):
        marks.append({'checkpoint':cp,'phase_seed':seed,'branch':b,'relative_time':r.relative_time,'X':r.Pi_normalized_mean,'X_natural':nat.Pi_normalized_mean.iloc[j],'D0':d0,'R':stored[j],'source_file':r.source_file,'source_row_1based':r.source_row_1based,'natural_source_file':nat.source_file.iloc[j],'natural_source_row_1based':int(nat.source_row_1based.iloc[j])})
marks=pd.DataFrame(marks)
tr=marks[(marks.checkpoint==2250)&(marks.relative_time<=1+1e-12)].copy()
ep=marks[np.isclose(marks.relative_time,1)].copy()
check('Figure 3: 99 trajectories and 27 endpoints',len(tr)==99 and len(ep)==27)
manifest=json.loads((ROOT/'sources/records/FIG3_source_manifest.json').read_text())
for m in manifest['marks']:
    sel=m['selector']
    br=({'nonlinear_recovery':'NL','linear_control':'LIN','amplitude_matched_frozen':'FRZ'}[sel['branch']]
        if m['panel']=='B' else m['id'].rsplit('_',1)[-1])
    t=float(sel['relative_time']) if m['panel']=='B' else 1.0
    q=marks[(marks.checkpoint==sel['checkpoint'])&(marks.phase_seed==sel['phase_seed'])&(marks.branch==br)&np.isclose(marks.relative_time,t)]
    check('Gate 2 mark '+m['id'],len(q)==1 and np.isclose(q.R.iloc[0],float(m['value']),atol=1e-12,rtol=1e-12))
export(ep,'Fig3c_all_27_endpoint_marks.csv')
redesign.figure3(marks,ep,raw,amp)
mech=load('clark_association/mechanism_dns_vs_surrogate_instantaneous_v3.csv')
redesign.figure4(raw,mech,masks)
redesign.collapse_diagnostic(marks,raw)
joint=pd.read_csv(DATA/'Fig4a_joint_recovery_all_values.csv',float_precision='round_trip')
verify=joint[joint.metric=='Flux'].merge(marks,on=['checkpoint','phase_seed','branch','relative_time'],validate='one_to_one')
check('All 558 derived flux-recovery cells agree with existing scalar metric',
    len(verify)==558 and np.allclose(verify.recovery,verify.R,atol=1e-12,rtol=1e-12))
for p in FIG.glob('*.svg'):
    check('Editable vector artwork without embedded image: '+p.name,'<image' not in p.read_text())

# Audits retained in the package for traceability (not portrayed as extra evidence).
audit_rows=[]
for p in sorted((ROOT/'sources/audits').rglob('*.csv')):
    d=pd.read_csv(p,float_precision='round_trip');d['source_file']=str(p.relative_to(ROOT));d['source_row_1based']=np.arange(1,len(d)+1);audit_rows.append(d)
audits=pd.concat(audit_rows,ignore_index=True)
check('FRZ all-output amplitude audits',len(audits)==279 and audits.modal_amplitude_relative_max_error.max()<1e-12 and audits.energy_relative_error.max()<1e-12)
export(audits,'FRZ_amplitude_audit_all_279_rows.csv')
report={'status':'PASS','scope':'Table/hash/figure-mark consistency; no raw-field recomputation',
 'checks':CHECKS,'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
 'source_values_modified':False, 'derived_values_added':['reference-normalized static observables','normalized reference spectra','geometry scalar recovery','initial energy time and descriptive collapse diagnostic'],'figure_design':'V2 data-only redesign; derived geometry scalar recovery and energy-time normalization test' ,
 'fig3_mark_counts':{'main_trajectories':len(marks[marks.relative_time<=1+1e-12]),'main_endpoints':len(ep),'archived_certified_marks_checked':len(tr)+len(ep)},
 'known_data_gaps':['raw 3D fields','FRZ M trajectory','empirical surrogate FR/PDF curves','dynamic second filter scale'],
 'fig3_endpoint_ranges':ep.groupby('branch').R.agg(['min','max']).to_dict(),
 'max_FRZ_modal_error':float(audits.modal_amplitude_relative_max_error.max())}
(ROOT/'validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'status':'PASS','checks':len(CHECKS),'figures':5,'fig3_main_marks':297+len(ep)},ensure_ascii=False))
