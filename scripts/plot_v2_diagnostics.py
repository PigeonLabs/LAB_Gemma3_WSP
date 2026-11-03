"""Standalone, deterministic Matplotlib diagnostic figure from measured results."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/wsp-matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def build():
    data=json.loads((ROOT/'artifacts/study_v2/diagnostic_summary.json').read_text())
    keys=list(data); y=np.arange(len(keys))
    plt.rcParams.update({'svg.hashsalt':'wsp-v2-diagnosis','font.size':10,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(11,7))
    fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.13)
    fig.set_facecolor('#f7f9fc');ax.set_facecolor('#f7f9fc')
    a=[100*data[k]['accuracy'] for k in keys]
    ax.barh(y-.15,a,height=.27,color='#167d9a',label='Correct answer')
    for i,k in enumerate(keys):
        ax.text(a[i]+1,i-.15,f'{a[i]:.1f}%',va='center',fontsize=9)
        if k.startswith('D0'):
            ax.text(1,i+.16,'numeric output not requested',va='center',color='#687483',fontsize=8)
        else:
            v=100*data[k]['joint_accuracy'];ax.barh(i+.15,v,height=.27,color='#b07819',label='Answer + two numeric premises' if k=='D1-N16' else None)
            ax.text(v+1,i+.15,f'{v:.1f}%',va='center',fontsize=9,color='#8c611a')
    ax.set_yticks(y,keys);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel('Percent of the 64 planned episodes per row')
    fig.text(.12,.96,'Gemma 3 diagnostic results: 16 reused circuits, four frames',fontweight='bold',fontsize=14,va='top')
    ax.axvline(80,ls='--',lw=1,color='#8a939d',label='80% reference (not an independent calibration)')
    ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.legend(loc='lower right',fontsize=8,frameon=False)
    fig.text(.12,.905,'D0: immediate choice   D1: generated calculation   D2: primitive tools   D3: exact F values supplied',fontsize=9)
    fig.text(.12,.025,'Original scores retained. Some N8/N4 cuts are tied; see the audit for the non-tied sensitivity analysis.',fontsize=8,color='#596370')
    fig.savefig(ROOT/'docs/figures/v2_diagnostics_measured.svg',metadata={'Date':None})
    fig.savefig('/tmp/wsp-v2-diagnostics.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':build()
