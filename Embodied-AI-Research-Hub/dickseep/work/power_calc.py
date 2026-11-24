import math
z_a2, z_a1, z_b = 1.959964, 1.644854, 0.841621

def ci_hw(n, p):
    return 1.96*math.sqrt(p*(1-p)/n)

print("== 95% CI half-width on a single success rate ==")
for n in (30, 100, 500, 1000):
    for p in (0.5, 0.8, 0.9):
        print(f"  n={n:5d} p={p:.1f} -> +/-{ci_hw(n,p)*100:5.1f} pp")

def n_unpaired(p, delta, power_z=z_b, alpha_z=z_a2):
    return (alpha_z+power_z)**2 * (p*(1-p)+p*(1-p)) / delta**2

print("\n== two-proportion test, 80% power, alpha=0.05 two-sided, p0=p1=p ==")
for delta in (0.03, 0.05, 0.10, 0.15, 0.20):
    print(f"  delta={delta*100:4.0f} pp @ p=0.5 -> n/group = {math.ceil(n_unpaired(0.5, delta)):,}")
    print(f"  delta={delta*100:4.0f} pp @ p=0.9 -> n/group = {math.ceil(n_unpaired(0.9, delta)):,}")

def n_paired(discord, delta, alpha_z=z_a2, power_z=z_b):
    # McNemar, normal approximation
    num = alpha_z*math.sqrt(discord) + power_z*math.sqrt(max(discord-delta**2, 1e-9))
    return num**2/delta**2

print("\n== paired (McNemar) design: total discordant-needed depends on discordance rate ==")
for d in (0.05, 0.10, 0.20):
    for delta in (0.05, 0.10):
        print(f"  discordance={d:.2f} delta={delta*100:4.0f} pp -> n_pairs = {math.ceil(n_paired(d, delta)):,}")

print("\n== non-inferiority, margin=5pp, one-sided alpha=0.05, 80% power (unpaired) ==")
for p in (0.5, 0.8, 0.9):
    n = (z_a1+z_b)**2*(p*(1-p)+p*(1-p))/0.05**2
    print(f"  p={p:.1f} -> n/group = {math.ceil(n):,}")

print("\n== LIBERO-style budget ==")
print("  500 episodes/suite = 10 tasks x 50 trials")
print("  4 suites x 500 = 2000 episodes per config")
