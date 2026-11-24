import math
def wilson(k,n,z=1.959964):
    p=k/n
    d=1+z*z/n
    c=p+z*z/(2*n)
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d*100,(c+h)/d*100)
print("=== KWS Table IV, per-suite (assume n=500 per suite) ===")
rows={"Fixed-2":(97.5,96.5,93.0,85.0),"Fixed-4":(97.5,97.0,94.5,89.5),
      "Fixed-6":(98.0,97.0,94.5,90.0),"Fixed-8":(97.0,97.5,92.5,89.5),
      "Fixed-10":(97.0,97.0,95.5,89.5),"Ours":(99.5,98.5,96.5,94.5)}
suites=["Spatial","Object","Goal","Long"]
for name,v in rows.items():
    print(f"\n{name}  avg={sum(v)/4:.2f}")
    for s,p in zip(suites,v):
        lo,hi=wilson(round(p*5),500)
        print(f"   {s:8s} {p:5.1f}%  CI[{lo:5.1f},{hi:5.1f}]")
print("\n=== Avg over 4 suites, treat as n=2000 (upper bound on precision) ===")
for name,v in rows.items():
    k=sum(round(p*5) for p in v); n=2000
    lo,hi=wilson(k,n)
    print(f"  {name:9s} avg={k/n*100:5.2f}%  CI[{lo:5.1f},{hi:5.1f}]")
print("\n=== pairwise: Fixed-6 vs Fixed-2 / Fixed-8 / Fixed-10 (avg, n=2000) ===")
base=sum(round(p*5) for p in rows["Fixed-6"])
for other in ("Fixed-2","Fixed-8","Fixed-10"):
    k=sum(round(p*5) for p in rows[other])
    diff=(base-k)/2000*100
    print(f"  Fixed-6 - {other:9s} = {diff:+.2f} pp")
