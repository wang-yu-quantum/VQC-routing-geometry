"""Compute Bell capacity and horizontal state-gradient diagnostics."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE
FIGURES=HERE/'figures'
FIGURES.mkdir(exist_ok=True)
I=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1.]); H=(X+Z)/np.sqrt(2)
def gate(v,p,q,n):
 t=v.reshape([2]*n); return np.moveaxis(np.tensordot(p,t,axes=(1,q)),0,q).reshape(-1)
def cx(v,c,t,n):
 j=np.arange(2**n); destination=j^(((j>>(n-1-c))&1)<<(n-1-t)); out=np.empty_like(v); out[destination]=v; return out
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--from-frozen',action='store_true',help='Plot the released state-metric CSV without recomputing it.')
args=parser.parse_args()
rows=[]
if args.from_frozen:
 with (OUT/'bell_floor_state_metric.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   rows.append({key: (int(value) if key in ('m','n','k','state_schmidt_rank') else float(value))
                if value else '' for key,value in row.items()})
else:
 for m in range(2,6):
  n=2*m; d=2**n; initial=np.zeros(d,complex); initial[0]=1
  # k Hadamards followed by k independent cross-cut CNOTs.
  for k in range(m+1):
   before=initial.copy()
   for q in range(k): before=gate(before,H,q,n)
   def route(v):
    for q in range(k): v=cx(v,q,q+m,n)
    return v
   psi=route(before); target=np.zeros(d,complex)
   for z in range(2**m): target[z*2**m+z]=2**(-m/2)
   amplitude=np.vdot(target,psi); F=float(abs(amplitude)**2); loss=1-F
   G=-2*amplitude*target+2*F*psi
   tangent=[]
   for q in range(n):
    for p in (X,Y,Z):
     tangent.append(-.5j*gate(psi,p,q,n))
     tangent.append(-.5j*route(gate(before,p,q,n)))
   J=np.column_stack(tangent); V=J-np.outer(psi,psi.conj()@J)
   VR=np.concatenate([V.real,V.imag]); GR=np.r_[G.real,G.imag]
   M=VR.T@VR; g=VR.T@GR
   projection2=max(0.,float(g@np.linalg.pinv(M,rcond=1e-10,hermitian=True)@g))
   w,s,_=np.linalg.svd(VR,full_matrices=False); w=w[:,s>s[0]*1e-10]
   direct=float(np.linalg.norm(w.T@GR)); ambient=float(np.linalg.norm(G))
   assert abs(projection2-direct**2)<1e-24
   assert abs(ambient**2-4*F*(1-F))<1e-13
   assert abs(loss-(1-2**(k-m)))<1e-13
   rank=int(np.linalg.matrix_rank(psi.reshape(2**m,2**m),tol=1e-10)); assert rank==2**k
   assert direct<1e-12
   if k<m: assert ambient>0
   rows.append({'m':m,'n':n,'k':k,'theory_loss':1-2**(k-m),'constructor_loss':loss,'state_projected_gradient_norm':projection2**.5,'ambient_state_gradient_norm':ambient,'svd_state_projected_gradient_norm':direct,'state_schmidt_rank':rank,'coverage_state':projection2**.5/ambient if ambient>1e-12 else ''})
 with (OUT/'bell_floor_state_metric.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
colors={2:'#3C6EAF',3:'#D6675B',4:'#2A9D8F',5:'#A06DA8'}
offsets={2:-.12,3:-.04,4:.04,5:.12}
fig,axes=plt.subplots(1,2,figsize=(7.2,3.05),constrained_layout=True,
                      gridspec_kw={'wspace':.08})

# All tight Bell floors depend only on the channel deficit r=m-k.
rgrid=np.arange(0,6)
axes[0].plot(rgrid,1-2.0**(-rgrid),color='#30343B',lw=1.65,
             label=r'Theory $1-2^{-r}$',zorder=2)
for m in range(2,6):
 subset=[r for r in rows if r['m']==m]
 deficit=np.array([m-r['k'] for r in subset],float)+offsets[m]
 axes[0].scatter(deficit,[r['constructor_loss'] for r in subset],
                 color=colors[m],edgecolor='white',linewidth=.55,s=38,
                 label=rf'$m={m}$',zorder=3)
axes[0].set(xlabel=r'Channel deficit $r=m-k$',
            ylabel=r'Minimum infidelity $\mathcal{L}^\star$',
            xlim=(-.28,5.28),ylim=(-.035,1.02),xticks=range(6))
axes[0].legend(frameon=False,ncol=2,fontsize=7.2,loc='lower right',
               handletextpad=.45,columnspacing=.85)
axes[0].set_title('Tight capacity floor',
                  loc='left',fontsize=8.8,fontweight='bold',pad=6,
                  color='#30343B')

for m in range(2,6):
 subset=[r for r in rows if r['m']==m and r['k']<m]
 deficit=np.array([m-r['k'] for r in subset],float)+offsets[m]
 ambient=np.array([r['ambient_state_gradient_norm'] for r in subset])
 projected=np.array([max(r['state_projected_gradient_norm'],1e-16) for r in subset])
 for x,y0,y1 in zip(deficit,projected,ambient):
  axes[1].plot([x,x],[y0,y1],color=colors[m],alpha=.16,lw=.8,zorder=1)
 axes[1].scatter(deficit,ambient,color=colors[m],edgecolor='white',linewidth=.4,
                 marker='o',s=34,zorder=3)
 axes[1].scatter(deficit,projected,facecolor='white',edgecolor=colors[m],
                 marker='D',s=31,linewidth=1.0,zorder=3)
axes[1].scatter([],[],color='#4A4A4A',marker='o',s=30,
                label=r'Ambient $\|G_{\rm st}\|$')
axes[1].scatter([],[],facecolor='white',edgecolor='#4A4A4A',marker='D',s=28,
                label=r'Visible $\|\Pi_{\mathcal{T}^{\rm st}}G_{\rm st}\|$')
axes[1].set(yscale='log',ylim=(3e-17,2.2),
            yticks=[1e-16,1e-12,1e-8,1e-4,1],
            xlim=(.58,5.42),xticks=[1,2,3,4,5],
            xlabel=r'Channel deficit $r=m-k$',ylabel='State-gradient norm')
axes[1].legend(frameon=False,fontsize=7.2,loc='center right',
               handletextpad=.55)
axes[1].text(.035,.055,r'visible values $<10^{-16}$ shown at $10^{-16}$',
             transform=axes[1].transAxes,fontsize=6.7,color='#555B63')
axes[1].set_title('Ambient slope remains hidden',
                  loc='left',fontsize=8.8,fontweight='bold',pad=6,
                  color='#30343B')

for ax,label in zip(axes,'ab'):
 ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',alpha=.18,lw=.65,color='#7A7F87')
 ax.tick_params(labelsize=8.2)
 ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
 ax.text(-.13,1.10,label,transform=ax.transAxes,fontsize=10.5,
         fontweight='bold',va='top')
pdf=FIGURES/'fig_bell_floor.pdf'; fig.savefig(pdf,bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
fig.savefig(OUT/'bell_floor_state_metric.png',dpi=180,bbox_inches='tight'); plt.close(fig)

manifest={'metric': 'pure-state horizontal real metric for BOTH numerator and denominator', 'observable': 'minus Bell-target projector', 'ambient_squared': '4 F (1-F)', 'projected_squared': 'g.T @ pinv(M_state) @ g', 'points': len(rows), 'max_projected_norm': max((r['state_projected_gradient_norm'] for r in rows)), 'max_loss_error': max((abs(r['constructor_loss'] - r['theory_loss']) for r in rows)), 'figure_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if args.from_frozen:
 manifest['mode'] = 'from-frozen'
 manifest['source_csv_sha256']=hashlib.sha256((OUT/'bell_floor_state_metric.csv').read_bytes()).hexdigest()
(OUT/'bell_metric_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
