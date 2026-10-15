#!/usr/bin/env python3
"""Analyse time series statistics.json with BEAST. 

Usage: 
python ./subsystems/map/scripts/analyse_temporal_decomposition.py \
    ./tsf_experiments/slope_monitoring_cadia/results/eda/statistics.json \
    --out ./tsf_experiments/slope_monitoring_cadia/results/temporal_decomposition

Retrospective analysis, not a causal early-warning algorithm. Input values are LOS displacement in millimetres. Spatial std/percentiles are NOT measurement
uncertainty. A fixed spatial mask is needed to interpret aggregate changes.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_series(path):
    raw = json.loads(Path(path).read_text())['per_acquisition']
    df = pd.DataFrame({date: {**{k:v for k,v in row.items() if k != 'percentiles'},
                               **row.get('percentiles', {})}
                       for date,row in raw.items()}).T
    df.index = pd.to_datetime(df.index, errors='raise')
    df = df.sort_index().apply(pd.to_numeric, errors='coerce')
    if df.index.has_duplicates or len(df) < 20:
        raise ValueError('Need >=20 unique acquisition dates.')
    offsets = ((df.index-df.index[0])/pd.Timedelta(days=1)).to_numpy()
    if not np.allclose(offsets, np.round(offsets)):
        raise ValueError('This script expects whole-day acquisition dates.')
    step = int(np.gcd.reduce(np.diff(offsets.astype(int))))
    if step < 1:
        raise ValueError('Invalid acquisition spacing.')
    grid = pd.date_range(df.index[0], df.index[-1], freq=f'{step}D')
    return df, df.reindex(grid), step


def vector(component, key, n):
    value = getattr(component, key, None)
    if value is None:
        return np.full(n, np.nan)
    arr = np.asarray(value).reshape(-1)
    if len(arr) != n:
        raise ValueError(f'Unexpected {key} output shape: {np.shape(value)}')
    return arr


def analyse(df, regular, step, stat, args):
    import Rbeast as rb
    y = regular[stat].to_numpy(dtype=float)
    if np.mean(~np.isfinite(y)) >= .75:
        raise ValueError('Too many missing grid values; use beast_irreg with explicit aggregation.')
    # Numeric elapsed DAYS avoid decimal-calendar-year distortions.
    # All original dates land exactly on this grid; missing slots remain NaN.
    model = rb.beast(
        y, start=0, deltat=step, period=365.25, season='harmonic',
        sorder_minmax=[0,args.harmonics], scp_minmax=[0,args.season_breaks],
        sseg_minlength=math.ceil(365.25/step),
        sseg_leftmargin=math.ceil(365.25/step),
        sseg_rightmargin=math.ceil(365.25/step),
        tcp_minmax=[0,args.trend_breaks], torder_minmax=[1,1],
        tseg_minlength=math.ceil(180/step),
        tseg_leftmargin=math.ceil(180/step), tseg_rightmargin=math.ceil(180/step),
        mcmc_seed=args.seed, mcmc_samples=8000, mcmc_chains=3,
        mcmc_burbin=1000, dump_ci=True, print_progress=False, print_param=False,
    )
    out = args.out/stat
    out.mkdir(parents=True,exist_ok=True)
    n = len(y)
    frame = pd.DataFrame({'date':regular.index,'observed':y})
    for name in ['trend','season']:
        comp = getattr(model,name)
        for key in ['Y','SD','cpOccPr','order']:
            frame[f'{name}_{key}'] = vector(comp,key,n)
        ci = np.asarray(comp.CI)
        if ci.shape == (2,n): ci = ci.T
        if ci.shape != (n,2): raise ValueError(f'Unexpected CI shape {ci.shape}')
        frame[f'{name}_lower'], frame[f'{name}_upper'] = ci[:,0],ci[:,1]
    frame['fitted'] = frame.trend_Y + frame.season_Y
    if not np.isfinite(frame.fitted).all(): raise RuntimeError('BEAST did not return a finite fit.')
    frame['residual'] = y-frame.fitted
    # Report fitted components only at actual acquisition dates.
    # Missing slots define timing, not additional observations or reliable predictions.
    observed = np.isfinite(y)
    for col in frame.columns:
        if col not in ['date', 'observed'] and not col.endswith('cpOccPr'):
            frame.loc[~observed, col] = np.nan
    frame.to_csv(out/'components.csv',index=False)
    origin = regular.index[0]
    def date(days):
        return (origin+pd.to_timedelta(float(days),unit='D')).isoformat() if np.isfinite(days) else ''
    rows=[]
    for name in ['trend','season']:
        comp=getattr(model,name)
        cps=np.asarray(comp.cp).reshape(-1)
        probs=np.asarray(comp.cpPr).reshape(-1)
        cis=np.asarray(comp.cpCI).reshape(-1,2)
        for cp,p,ci in zip(cps,probs,cis):
            if np.isfinite(cp) and np.isfinite(p):
                rows.append(dict(component=name,date=date(cp),probability=float(p),
                                 interval_start=date(ci[0]),interval_end=date(ci[1]),
                                 selected_for_plot=bool(p>=args.threshold)))
    columns=['component','date','probability','interval_start','interval_end','selected_for_plot']
    breaks=pd.DataFrame(rows,columns=columns)
    breaks.to_csv(out/'breakpoints.csv',index=False)
    summary={'statistic':stat,'grid_days':step,'observations':int(np.isfinite(y).sum()),
             'grid_points':n,'seed':args.seed,'arguments':{k:str(v) for k,v in vars(args).items()},
             'Rbeast_version':__import__('importlib.metadata',fromlist=['version']).version('Rbeast')}
    for name in ['trend','season']:
        comp=getattr(model,name)
        summary[name]={'expected_breaks':float(np.asarray(comp.ncp).squeeze()),
                       'break_count_probabilities':np.asarray(comp.ncpPr).reshape(-1).tolist()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

    # Descriptive Fourier projection of BEAST seasonal posterior mean at observed
    # dates, within selected seasonal segments. NOT posterior harmonic coefficients
    # or independent evidence of seasonality. No amplitude significance test.
    t=((regular.index-origin)/pd.Timedelta(days=1)).to_numpy()
    boundaries=[0.0]+sorted(float((pd.Timestamp(r['date'])-origin)/pd.Timedelta(days=1))
                           for r in rows if r['component']=='season' and r['selected_for_plot'])+[t[-1]+step]
    harmonic_rows=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        mask=(t>=left)&(t<right)&np.isfinite(y)
        tt=t[mask]
        if len(tt)<2*args.harmonics+2: continue
        X=np.column_stack([np.ones(len(tt))]+[f(2*np.pi*k*tt/365.25)
                          for k in range(1,args.harmonics+1) for f in (np.cos,np.sin)])
        coef=np.linalg.lstsq(X,frame.season_Y.to_numpy()[mask],rcond=None)[0]
        for k in range(1,args.harmonics+1):
            a,b=coef[2*k-1:2*k+1]
            harmonic_rows.append(dict(segment_start=date(left),segment_end_exclusive=date(right),
                harmonic=k,period_days=365.25/k,cos_coefficient=a,sin_coefficient=b,
                amplitude=np.hypot(a,b),phase_radians=np.arctan2(b,a),
                phase_reference=origin.isoformat(),observed_points=int(mask.sum()),
                interpretation='Descriptive projection of posterior mean; no uncertainty estimate'))
    pd.DataFrame(harmonic_rows).to_csv(out/'harmonics_descriptive.csv',index=False)
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True,constrained_layout=True)
    d=regular.index[observed]
    plotted=frame.loc[observed]
    axs[0].plot(df.index,df[stat],'.',color='black',label=f'Observed {stat}')
    axs[0].plot(d,plotted.fitted,color='#087f8c',label='Trend + seasonal')
    axs[0].legend();axs[0].set_ylabel(f'{stat.capitalize()} LOS displacement [mm]')
    for ax,name,color in [(axs[1],'trend','#265a96'),(axs[2],'season','#b96d20')]:
        ax.plot(d,plotted[f'{name}_Y'],color=color,label=name)
        ax.fill_between(d,plotted[f'{name}_lower'],plotted[f'{name}_upper'],color=color,alpha=.2,label='credible interval')
        ax.legend();ax.set_ylabel(f'{name.capitalize()} LOS displacement [mm]')
    for r in rows:
        if r['selected_for_plot']:
            ax=axs[1] if r['component']=='trend' else axs[2]
            ax.axvline(pd.Timestamp(r['date']),color='crimson',ls='--',alpha=.7)
    for ax in axs: ax.grid(alpha=.2)
    fig.suptitle('LOS displacement seasonal decomposition with breaks')
    axs[-1].set_xlabel('Date')
    fig.savefig(out/'decomposition.png',dpi=160);plt.close(fig)
    print(stat,':',summary['trend'],summary['season'])
    print(breaks.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,default=Path('beast_results'))
    p.add_argument('--stats',nargs='+',default=['median','mean'])
    p.add_argument('--harmonics',type=int,choices=[1,2,3],default=2)
    p.add_argument('--trend-breaks',type=int,default=3)
    p.add_argument('--season-breaks',type=int,default=1)
    p.add_argument('--threshold',type=float,default=.5,help='Plot selection only; not a significance threshold')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    df,regular,step=load_series(args.input)
    args.out.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out/'input_acquisitions.csv',index_label='date')
    print(f'{len(df)} dates: {df.index[0].date()} to {df.index[-1].date()}; grid {step} days')
    for stat in args.stats:
        if stat not in df: raise ValueError(f'Unknown statistic: {stat}')
        analyse(df,regular,step,stat,args)

if __name__=='__main__': main()
