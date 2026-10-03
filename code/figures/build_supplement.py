"""Rebuild supplemental numerical panels from existing tables only.
No field generation, fitted density, synthetic surrogate curve or new simulation.
"""
from pathlib import Path
import json, hashlib, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'sources/figure_ready';FIG=ROOT/'figures';DATA=ROOT/'plot_data'
FIG.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
C={'DNS':'#183F66','null':'#159C92','ind':'#8464A4','FRZ':'#757575','LIN':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.linewidth':.65,'lines.linewidth':1.1,'svg.fonttype':'none','pdf.fonttype':42,
 'svg.hashsalt':'turbulence-supplement-data-v1','axes.spines.top':False,'axes.spines.right':False})
CHECKS=[]
def check(name,condition):
 CHECKS.append({'check':name,'pass':bool(condition)})
 if not condition:raise AssertionError(name)
def load(rel,extra=False):
 path=(ROOT/'sources' if extra else SRC)/rel
 d=pd.read_csv(path,float_precision='round_trip');d['source_file']=str(path.relative_to(ROOT))
 d['source_row_1based']=np.arange(1,len(d)+1);return d
def export(d,name):d.to_csv(DATA/name,index=False,float_format='%.17g')
def save(fig,name):
 for note in list(fig.texts):
  if note.get_position()[1] < 0.07: note.remove()
 for ext in ['svg','pdf','png']:
  meta={'Creator':'build_supplement.py; existing-table reconstruction'}
  if ext=='svg':meta['Date']=None
  if ext=='pdf':meta.update(CreationDate=None,ModDate=None)
  b=io.BytesIO();fig.savefig(b,format=ext,dpi=300,metadata=meta,facecolor='white')
  (FIG/f'{name}.{ext}').write_bytes(b.getvalue())
 plt.close(fig)
def panel(ax,letter,title):
 pos=ax.get_position();x=max(-.15,(.025-pos.x0)/pos.width)
 ax.text(x,1.08,letter,transform=ax.transAxes,fontsize=11,weight='bold')
 ax.set_title(title,loc='left',pad=8)
def zero(ax):ax.axhline(0,color='#999999',ls=':',lw=.7)
def legend_handles():
 return [Line2D([0],[0],color=C['DNS'],marker='o',ls='',label='Reference'),
  Line2D([0],[0],color=C['null'],marker='o',mfc='white',ls='',label='Phase masks ± SD'),
  Line2D([0],[0],color='#555555',marker='o',ls='',label='Scale 4'),
  Line2D([0],[0],color='#555555',marker='s',ls='',label='Scale 8')]

hashes=json.loads((ROOT/'source_hashes.json').read_text())
check('All copied V2 source SHA256 values match',all(
 hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in hashes))
extra=json.loads((ROOT/'extra_source_hashes.json').read_text())
check('Additional audit and definition-source SHA256 values match',all(
 hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in extra))
for p in SRC.rglob('*.metadata.json'):
 m=json.loads(p.read_text());q=p.with_name(p.name.replace('.metadata.json','.csv'))
 if q.exists() and 'source_sha256' in m:
  check('Source metadata '+str(q.relative_to(SRC)),hashlib.sha256(q.read_bytes()).hexdigest()==m['source_sha256'])
main=load('main32/ensemble_summary_64.csv');seeds=load('main32/seed_metrics_64.csv')
qc=load('main32/qc_seed_summary_64.csv');repqc=load('isotropic_rep2/isotropic_rep2_seed_qc.csv')
rep=load('isotropic_rep2/isotropic_rep2_reference_vs_24.csv');repraw=load('isotropic_rep2/isotropic_rep2_all_case_metrics.csv')
mech=load('clark_association/mechanism_dns_vs_surrogate_instantaneous_v3.csv')
finite=load('clark_association/mechanism_dns_vs_surrogate_finite_time_v3.csv')
check('Static QC: all 32+24 masks pass',len(qc)==32 and len(repqc)==24 and qc.qc_pass.all() and repqc.qc_pass.all())

# S1: summary evidence only; no raw flux decomposition scatter exists.
fig,axs=plt.subplots(1,3,figsize=(7.1,3.2));fig.subplots_adjust(left=.10,right=.98,top=.85,bottom=.29,wspace=.52)
out=[]
for ax,letter,metric,title,deficit in [(axs[0],'a','corr_Pi_Pi_dev_pooled','Deviatoric consistency',True),
 (axs[1],'b','corr_Pi_Pi_vol_pooled','Volumetric association',False)]:
 panel(ax,letter,title);q=mech[mech.metric==metric].sort_values('scale')
 check('Both decomposition scales '+metric,len(q)==2)
 for i,(_,r) in enumerate(q.iterrows()):
  ref=1-r.dns_value if deficit else r.dns_value
  null=1-r.surrogate_mean if deficit else r.surrogate_mean
  ax.plot(i-.055,ref,'o',ms=5,color=C['DNS'])
  ax.errorbar(i+.055,null,yerr=r.surrogate_std,fmt='o',ms=4,color=C['null'],mfc='white',capsize=2,lw=.8)
  out.append(dict(panel=letter,metric=metric,scale=r.scale,reference_correlation=r.dns_value,
   null_mean_correlation=r.surrogate_mean,null_sample_sd=r.surrogate_std,
   plotted_reference=ref,plotted_null_mean=null,source_file=r.source_file,source_row_1based=r.source_row_1based))
 ax.set_xticks([0,1],['4','8']);ax.set_xlabel(r'$\ell/\Delta$');ax.set_xlim(-.3,1.3)
 if deficit:
  ax.set_ylabel(r'$1-\mathrm{corr}(\Pi,\Pi_{dev})$');ax.set_yscale('symlog',linthresh=1e-14)
  ax.set_yticks([0,1e-12,1e-8,1e-4]);ax.set_ylim(-2e-14,1.5e-4)
 else:ax.set_ylabel(r'$\mathrm{corr}(\Pi,\Pi_{vol})$');zero(ax);ax.set_ylim(-.015,.055)
export(pd.DataFrame(out),'FigS1ab_decomposition_association.csv')
ax=axs[2];panel(ax,'c','Surrogate volumetric ratio')
rows=[]
for i,(block,d,co) in enumerate([('Primary',qc,C['null']),('Independent',repqc,C['ind'])]):
 x=i+np.linspace(-.13,.13,len(d));ax.scatter(x,d.surrogate_volumetric_ratio_max,s=12,color=co,alpha=.65,edgecolors='none')
 q=d.copy();q['block']=block;q['display_x']=x;rows.append(q)
ax.set_yscale('log');ax.set_ylabel('Recorded maximum ratio');ax.set_xticks([0,1],['Primary\n32 masks','Independent\n24 masks'])
export(pd.concat(rows,ignore_index=True),'FigS1c_volumetric_ratio_audit.csv')
fig.legend(handles=legend_handles()[:2],ncol=2,loc='upper center',bbox_to_anchor=(.52,.995),frameon=False)
fig.text(.10,.055,'(a,b) Static primary block; errors are mask sample SD. Symlog in (a) retains zero and roundoff signs.\n(c) Source-defined per-mask maximum ratios; no spatial samples or per-scale ratio arrays were supplied.',fontsize=6.7,color='#555555')
save(fig,'FigS1_Flux_Decomposition')

# S2: available 64-cube and generated audit data; not a 128-cube replication.
dyn=load('dynamic_audits/amplitude_matched_3x3_numerical_audit.csv',extra=True)
init=load('dynamic_audits/heldout_3x3_numerical_audit.csv',extra=True)
check('Nine dynamic initial and FRZ audits pass',len(dyn)==9 and len(init)==9 and dyn['pass'].all() and init.numerical_pass.all())
export(init,'FigS2_dynamic_initial_audit.csv')
fig,axs=plt.subplots(2,2,figsize=(7.1,4.9));fig.subplots_adjust(left=.115,right=.98,top=.89,bottom=.18,hspace=.66,wspace=.44)
ax=axs[0,0];panel(ax,'a','Static mask gates: all records');out=[]
for i,col in enumerate(['hermitian_error_max','inverse_imaginary_ratio','spectrum_error_max','energy_ratio_error_max']):
 for block,d,off,co,mk in [('Primary',qc,-.09,C['null'],'o'),('Independent',repqc,.09,C['ind'],'s')]:
  x=i+off+np.linspace(-.06,.06,len(d));ax.scatter(x,d[col],s=10,marker=mk,color=co,alpha=.6,edgecolors='none')
  for j,(_,r) in enumerate(d.iterrows()):out.append(dict(metric=col,block=block,seed=r.seed,value=r[col],display_x=x[j],source_file=r.source_file,source_row_1based=r.source_row_1based))
ax.set_yscale('symlog',linthresh=1e-17);ax.set_ylabel('Recorded gate value');ax.set_xticks(range(4),['Hermitian','Inverse\nimaginary','Spectrum','Energy']);ax.tick_params(axis='x',length=0)
export(pd.DataFrame(out),'FigS2a_static_gate_marks.csv')
ax=axs[0,1];panel(ax,'b','Dynamic FRZ gates: all 9 cases');out=[]
for i,col in enumerate(['modal_amplitude_error','energy_relative_error','divergence_relative_error','nonlinear_replay_error']):
 x=i+np.linspace(-.12,.12,len(dyn));ax.scatter(x,dyn[col],s=17,color=C['FRZ'],edgecolors='none')
 for j,(_,r) in enumerate(dyn.iterrows()):out.append(dict(metric=col,checkpoint=r.checkpoint,phase_seed=r.phase_seed,value=r[col],display_x=x[j],source_file=r.source_file,source_row_1based=r.source_row_1based))
ax.set_yscale('symlog',linthresh=1e-17);ax.set_ylabel('Recorded gate value');ax.set_xticks(range(4),['Modal\namplitude','Energy','Divergence','NL replay']);ax.tick_params(axis='x',length=0)
export(pd.DataFrame(out),'FigS2b_dynamic_FRZ_gate_marks.csv')
tri=load('spectral_energy/gate3_exact_bispectrum_16cube/gate3_case_results.csv')
check('All 96 physical/bispectral identity rows',len(tri)==96 and np.allclose(tri.A_physical,tri.A_bispectral,atol=5e-17,rtol=1e-12))
check('Triad closure residual matches recorded values',np.allclose(abs(tri.A_physical-tri.A_bispectral),tri.A_closure_abs_error,atol=1e-17,rtol=1e-12))
ax=axs[1,0];panel(ax,'c','Triad closure residuals');out=[]
for i,col in enumerate(['A_closure_abs_error','delta_response_abs_error','T_invariance_rel_error']):
 x=i+np.linspace(-.16,.16,len(tri));ax.scatter(x,tri[col],s=8,color=C['DNS'],alpha=.6,edgecolors='none')
 for j,(_,r) in enumerate(tri.iterrows()):out.append(dict(metric=col,value=r[col],display_x=x[j],source_file=r.source_file,source_row_1based=r.source_row_1based))
ax.set_yscale('symlog',linthresh=1e-18);ax.set_ylabel('Recorded error (source units)');ax.set_xticks(range(3),['Flux\nabsolute','Response\nabsolute','T invariance\nrelative']);ax.tick_params(axis='x',length=0)
export(pd.DataFrame(out),'FigS2c_triad_error_marks.csv')
ax=axs[1,1];panel(ax,'d','Physical vs Fourier mean flux')
for audit,co,mk in [('A_bandlimited',C['DNS'],'o'),('B_nonbandlimited_2x',C['ind'],'s')]:
 q=tri[tri.audit==audit];ax.scatter(q.A_physical,q.A_bispectral,s=12,color=co,marker=mk,alpha=.65,label='Bandlimited' if mk=='o' else 'Nonbandlimited, 2×')
lo=tri[['A_physical','A_bispectral']].min().min();hi=tri[['A_physical','A_bispectral']].max().max()
ax.plot([lo,hi],[lo,hi],color='#888888',ls=':',lw=.8);ax.set_xlabel('Physical mean flux');ax.set_ylabel('Bispectral mean flux');ax.legend(frameon=False,fontsize=6.4)
ax.set_xticks([-.005,0,.005,.01])
export(tri,'FigS2d_all_96_identity_pairs.csv')
fig.text(.115,.975,'Static primary: teal circles; independent: purple squares',fontsize=7,color='#555555')
fig.text(.115,.03,'Symlog axes retain exact zeros. (a) 64-cube static blocks; (b) generated 64-cube dynamic experiment.\n(c,d) Generated 16-cube identity audits, spectral/D4 derivatives, filters 2/4; not turbulence replication at 128³.',fontsize=6.7,color='#555555')
save(fig,'FigS2_Numerical_Admissibility')

# S3: actual DNS counts; antithetic curve is derived exactly by swapping counts.
fr=load('main32/dns_finite_time_FR_curves_v3.csv');summary=load('main32/dns_finite_time_arrow_summary_v3.csv')
check('All 202 FR values equal log recorded tail-bin count ratio',len(fr)==202 and np.allclose(fr.R,np.log(fr.count_pos/fr.count_neg),atol=1e-12,rtol=1e-12))
centered=fr[fr['mode']=='centered'].copy();centered['R_antithetic_derived']=-centered.R
centered['antithetic_definition']='Exact algebraic sign inversion; swap count_pos/count_neg; not measured surrogate'
check('All centered FR bins meet source minimum count',len(centered)==117 and (centered.count_pos>=50).all() and (centered.count_neg>=50).all())
export(centered,'FigS3_all_117_centered_FR_bins.csv');export(fr,'FigS3_all_202_source_FR_bins.csv');export(summary,'FigS3_source_window_summary.csv')
fig,axs=plt.subplots(2,3,figsize=(7.1,4.9));fig.subplots_adjust(left=.095,right=.98,top=.86,bottom=.17,hspace=.56,wspace=.29)
for i,scale in enumerate([4,8]):
 for j,w in enumerate([5,10,20]):
  ax=axs[i,j];q=centered[(centered.scale==scale)&(centered.window_frames==w)].sort_values('z')
  check('FR selected bin count '+str((scale,w)),len(q)>0)
  panel(ax,'abcdef'[i*3+j],f'Scale {scale}, {w} frames')
  ax.plot(q.z,q.R,color=C['DNS'],marker='o',ms=2.8,lw=1)
  ax.plot(q.z,q.R_antithetic_derived,color=C['FRZ'],ls='--',lw=1)
  zero(ax);ax.set_ylim(-3.5,3.5);ax.set_xlim(0,3.3)
  if j==0:ax.set_ylabel(r'$\log[N(+s)/N(-s)]$')
  else:ax.tick_params(labelleft=False)
fig.text(.5,.09,r'$s=(I-\langle I\rangle)/\sigma(I)>0$',ha='center',fontsize=8)
fig.legend(handles=[Line2D([0],[0],color=C['DNS'],marker='o',ms=4,label='Recorded DNS'),
 Line2D([0],[0],color=C['FRZ'],ls='--',label='Derived sign inversion: -R')],ncol=2,loc='upper center',bbox_to_anchor=(.52,.99),frameon=False)
fig.text(.095,.027,'One fixed mask per sequence. Curves stop at the supplied supported bins; no extrapolation or linear fit.\nDashed curves are exact count-swapping consequences, not empirical surrogate curves. Surrogate FR arrays are absent.',fontsize=6.7,color='#555555')
save(fig,'FigS3_Window_Robustness')

# S4: primary and independent 64 blocks; show every instantaneous mask.
fig,axs=plt.subplots(2,2,figsize=(7.1,5.2));fig.subplots_adjust(left=.105,right=.98,top=.88,bottom=.18,hspace=.65,wspace=.36)
for ax,letter,metric,title,ylabel in [(axs[0,0],'a','normalized_mean_Pi','Every mask: normalized mean',r'$\langle\Pi\rangle/\sigma(\Pi)$'),
 (axs[0,1],'b','sign_bias_Pi','Every mask: instantaneous sign',r'$P(\Pi>0)-1/2$')]:
 panel(ax,letter,title);out=[]
 for i,(block,scale) in enumerate([('Primary',4),('Independent',4),('Primary',8),('Independent',8)]):
  if block=='Primary':
   q=seeds[(seeds.metric==metric)&(seeds.scale==scale)&(seeds.statistic_type=='instantaneous')]
   val=q.value.to_numpy();r=main[(main.metric==metric)&(main.scale==scale)&(main.statistic_type=='instantaneous')].iloc[0];ref=r.dns_value
  else:
   q=repraw[(repraw.case_type=='surrogate')&(repraw.scale==scale)];val=q[metric].to_numpy()
   r=rep[(rep.metric==metric)&(rep.scale==scale)&(rep.statistic_type=='instantaneous')].iloc[0];ref=r.comparator_value
  check('S4 every mask group '+str((block,scale,metric)),len(q)==(32 if block=='Primary' else 24))
  x=i+np.linspace(-.16,.16,len(q));ax.scatter(x,val,s=12,color=C['null'] if block=='Primary' else C['ind'],alpha=.6,edgecolors='none')
  ax.plot(i,ref,'*',ms=8,color=C['DNS'],zorder=3)
  for j,(_,s) in enumerate(q.iterrows()):out.append(dict(block=block,scale=scale,metric=metric,value=val[j],reference=ref,display_x=x[j],source_file=s.source_file,source_row_1based=s.source_row_1based,reference_source_file=r.source_file,reference_source_row_1based=r.source_row_1based))
 zero(ax);ax.set_ylabel(ylabel);ax.set_xticks(range(4),['P / 4','I / 4','P / 8','I / 8']);ax.set_xlabel('Block / filter scale');ax.set_xlim(-.4,3.4)
 export(pd.DataFrame(out),f'FigS4{letter}_all_112_masks.csv')
for ax,letter,metric,title,ylabel in [(axs[1,0],'c','normalized_mean_I','Finite-window mean transfer',r'$\langle I\rangle/\sigma(I)$'),
 (axs[1,1],'d','sign_bias_I','Finite-window sign bias',r'$P(I>0)-1/2$')]:
 panel(ax,letter,title);out=[]
 for block,d,n,mk,boff,refcol in [('Primary',main,32,'o',-.055,'dns_value'),('Independent',rep,24,'s',.055,'comparator_value')]:
  for scale,soff in [(4,-.13),(8,.13)]:
   q=d[(d.statistic_type=='finite_time')&(d.scale==scale)]
   q=q[q.metric==metric] if block=='Primary' else q[q.metric.str.match(metric+r'_w\d+$')]
   q=q.sort_values('window_frames');check('S4 finite-time group '+str((block,scale,metric)),len(q)==3)
   x=np.arange(3)+soff+boff
   ax.plot(x,q[refcol],color=C['DNS'],ls='-' if scale==4 else '--',marker=mk,ms=4,mfc=C['DNS'] if scale==4 else 'white')
   ax.errorbar(x,q.surrogate_mean,yerr=q.surrogate_std,fmt=mk,color=C['null'] if block=='Primary' else C['ind'],mfc='white',ms=3,capsize=2,lw=.7)
   for j,(_,r) in enumerate(q.iterrows()):out.append(dict(block=block,scale=scale,metric=metric,window_frames=r.window_frames,display_x=x[j],reference=r[refcol],null_mean=r.surrogate_mean,null_sd=r.surrogate_std,source_file=r.source_file,source_row_1based=r.source_row_1based))
 zero(ax);ax.set_ylabel(ylabel);ax.set_xticks(range(3),['5','10','20']);ax.set_xlabel('Window length (frames)');ax.set_xlim(-.4,2.4)
 export(pd.DataFrame(out),f'FigS4{letter}_finite_time.csv')
fig.text(.105,.965,'P: primary, 32 masks; I: independent, 24 masks. Stars/filled blue: reference.',fontsize=7)
fig.text(.105,.025,'Both available static blocks are 64³; no 128³ values were supplied. Independent reference: reference_star32.\nBars: mask sample SD. Independent scale-8 mean primary criterion fails and is retained. Scale 4: solid; scale 8: dashed.',fontsize=6.7,color='#555555')
save(fig,'FigS4_Extended_Static')

# S5: signed cubic variables only; unsigned controls were not supplied.
fig,axs=plt.subplots(2,3,figsize=(7.1,5.1));fig.subplots_adjust(left=.105,right=.98,top=.87,bottom=.18,wspace=.42,hspace=.67)
for ax,letter,suffix,title,ylabel in [(axs[0,0],'a','normalized_mean','Signed cubic mean',r'$\langle Q\rangle/\sigma(Q)$'),
 (axs[0,1],'b','sign_bias','Instantaneous sign bias',r'$P(Q>0)-1/2$'),
 (axs[0,2],'c','corr','Local flux association',r'$\mathrm{corr}(\Pi,Q)$')]:
 panel(ax,letter,title);out=[]
 for i,(var,txt) in enumerate([('M','M'),('T','T'),('quarter_W','W/4')]):
  for scale,off,mk in [(4,-.12,'o'),(8,.12,'s')]:
   metric='corr_Pi_'+var+'_pooled' if suffix=='corr' else var+'_'+suffix
   r=mech[(mech.metric==metric)&(mech.scale==scale)].iloc[0];x=i+off
   ax.plot(x,r.dns_value,marker=mk,color=C['DNS'],ms=4,ls='')
   ax.errorbar(x,r.surrogate_mean,yerr=r.surrogate_std,fmt=mk,color=C['null'],mfc='white',ms=3,capsize=2,lw=.7)
   out.append(dict(metric=metric,scale=scale,display_x=x,reference=r.dns_value,null_mean=r.surrogate_mean,null_sd=r.surrogate_std,source_file=r.source_file,source_row_1based=r.source_row_1based))
 zero(ax);ax.set_xticks(range(3),['M','T','W/4']);ax.set_ylabel(ylabel);ax.set_xlim(-.5,2.5)
 export(pd.DataFrame(out),f'FigS5{letter}_signed_cubic.csv')
for i,var in enumerate(['M','T','quarter_W']):
 ax=axs[1,i];panel(ax,'def'[i],('W/4' if var=='quarter_W' else var)+': finite-time sign');out=[]
 for scale,off,mk in [(4,-.055,'o'),(8,.055,'s')]:
  q=finite[(finite.metric=='I_'+var+'_sign_bias')&(finite.scale==scale)].sort_values('window_frames')
  check('S5 all three windows '+str((var,scale)),len(q)==3)
  x=np.arange(3)+off
  ax.plot(x,q.dns_value,color=C['DNS'],ls='-' if scale==4 else '--',marker=mk,ms=4,mfc=C['DNS'] if scale==4 else 'white')
  ax.errorbar(x,q.surrogate_mean,yerr=q.surrogate_std,fmt=mk,color=C['null'],mfc='white',ms=3,capsize=2,lw=.7)
  for j,(_,r) in enumerate(q.iterrows()):out.append(dict(metric=r.metric,scale=scale,window_frames=r.window_frames,reference=r.dns_value,null_mean=r.surrogate_mean,null_sd=r.surrogate_std,display_x=x[j],source_file=r.source_file,source_row_1based=r.source_row_1based))
 zero(ax);ax.set_xticks(range(3),['5','10','20']);ax.set_xlabel('Window length (frames)');ax.set_ylabel(r'$P(I_Q>0)-1/2$');ax.set_xlim(-.3,2.3)
 export(pd.DataFrame(out),f'FigS5{"def"[i]}_finite_cubic.csv')
fig.legend(handles=legend_handles(),ncol=4,loc='upper center',bbox_to_anchor=(.52,.995),frameon=False)
fig.text(.105,.027,'Static primary block. Bars: mask sample SD; finite-time scale 4 solid, scale 8 dashed.\nOnly supplied signed cubic diagnostics are shown. S² and ω² tables are absent; local association is not geometric causation.',fontsize=6.7,color='#555555')
save(fig,'FigS5_Local_Gradient')

# S6: regenerate the previously tested normalization diagnostic from its source tables.
raw=load('dynamic_recovery/heldout_3x3_raw_long.csv')
amp=load('dynamic_recovery/amplitude_matched_3x3_long.csv')
rec=load('dynamic_recovery/heldout_3x3_recovery_long.csv')
marks=[]
for (cp,seed),allrows in raw.groupby(['checkpoint','phase_seed']):
 n=allrows[allrows.branch=='natural'].sort_values('relative_time')
 start=allrows[allrows.branch=='nonlinear_recovery'].sort_values('relative_time')
 D0=abs(n.Pi_normalized_mean.iloc[0]-start.Pi_normalized_mean.iloc[0])
 for br,b in [('nonlinear_recovery','NL'),('linear_control','LIN'),('amplitude_matched_frozen','FRZ')]:
  q=(amp if b=='FRZ' else raw)
  q=q[(q.checkpoint==cp)&(q.phase_seed==seed)&(q.branch==br)].sort_values('relative_time')
  check('S6 matched recorded times '+str((cp,seed,b)),len(q)==31 and np.allclose(q.relative_time,n.relative_time))
  expected=1-abs(q.Pi_normalized_mean.to_numpy()-n.Pi_normalized_mean.to_numpy())/D0
  stored=q.R_Pi_normalized_mean.to_numpy() if b=='FRZ' else rec[(rec.checkpoint==cp)&(rec.phase_seed==seed)&(rec.branch==br)&(rec.metric=='Pi_normalized_mean')].sort_values('relative_time').recovery.to_numpy()
  check('S6 exact existing recovery '+str((cp,seed,b)),np.allclose(expected,stored,atol=1e-12,rtol=1e-12))
  for j,(_,r) in enumerate(q.iterrows()):
   marks.append(dict(checkpoint=cp,phase_seed=seed,branch=b,relative_time=r.relative_time,X=r.Pi_normalized_mean,X_natural=n.Pi_normalized_mean.iloc[j],D0=D0,R=stored[j],source_file=r.source_file,source_row_1based=r.source_row_1based,natural_source_file=n.source_file.iloc[j],natural_source_row_1based=n.source_row_1based.iloc[j]))
import normalization_diagnostic
C['NL']=C['DNS'];STYLE={'NL':('-','o'),'LIN':('--','s'),'FRZ':('-.','^')}
normalization_diagnostic.run(ROOT,pd.DataFrame(marks),raw,export,save,check,C,STYLE)

for p in FIG.glob('*.svg'):check('SVG is vector only '+p.name,'<image' not in p.read_text())
check('Six final supplemental PDFs',len(list(FIG.glob('FigS*.pdf')))==6)
coverage={'S1':'Summary correlations and surrogate QC ratios; original spatial scatter unavailable',
 'S2':'Available 64³ gates and generated 16³ identity audit; original 128³ audits unavailable',
 'S3':'117 DNS centered bins with derived exact sign inversion; empirical surrogate FR curves unavailable',
 'S4':'Two 64³ blocks; 128³ extended comparisons unavailable; scale-8 failure retained',
 'S5':'Signed cubic variables and finite-time diagnostics; unsigned S²/ω² tables unavailable',
 'S6':'V2 normalization diagnostic moved from temporary FigS1; numerical content unchanged'}
(ROOT/'validation_report.json').write_text(json.dumps({'status':'PASS','checks':CHECKS,
 'scope':'Copied tables and derived count-ratio consistency, not raw-field recomputation',
 'original_panel_coverage':coverage,'source_values_modified':False},ensure_ascii=False,indent=2))
print(json.dumps({'status':'PASS','checks':len(CHECKS),'supplemental_figures':6,'centered_FR_bins':len(centered)},ensure_ascii=False))
