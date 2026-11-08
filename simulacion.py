import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Uso: python3 simulacion.py carpeta_salida")
    sys.exit(1)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)

rng = np.random.default_rng(14)

THETA = 0.05
R = 20000
NS = [5, 10, 20, 30, 50, 100, 200, 500]

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# 1. Monte Carlo: varianza del MLE e insesgado frente a la cota

rows = []

for n in NS:
    x = rng.exponential(1 / THETA, size=(R, n))
    s = x.sum(axis=1)

    mle = n / s
    unb = (n - 1) / s
    crlb = THETA**2 / n

    def se_var(a):
        m = a.mean()
        mu4 = ((a - m)**4).mean()
        v = a.var(ddof=1)
        return np.sqrt(max(
            (mu4 - (R - 3) / (R - 1) * v**2) / R, 0
        ))

    rows.append(dict(
        n=n,
        crlb=crlb,
        var_mle=float(mle.var(ddof=1)),
        se_var_mle=float(se_var(mle)),
        var_unb=float(unb.var(ddof=1)),
        se_var_unb=float(se_var(unb)),
        bias_mle=float(mle.mean() - THETA),
        se_bias=float(mle.std(ddof=1) / np.sqrt(R)),
        var_unb_teo=float(THETA**2 / (n - 2)) if n > 2 else None,
        var_mle_teo=float(
            THETA**2 * n**2 / ((n - 1)**2 * (n - 2))
        ) if n > 2 else None,
        eff_unb=float(crlb / unb.var(ddof=1))
    ))

# 2. Información de Fisher empírica

fish = []
n_f = 200000

x1 = rng.exponential(1 / THETA, size=n_f)
score = 1 / THETA - x1

fish.append(dict(
    I_teo=1 / THETA**2,
    I_score2=float((score**2).mean()),
    I_score2_se=float(
        (score**2).std(ddof=1) / np.sqrt(n_f)
    ),
    I_hess=1 / THETA**2
))

# 3. Número mínimo de pruebas según el umbral

eps = np.logspace(-6, -3, 200)
n_min_abs = THETA**2 / eps

rel = np.linspace(0.02, 0.5, 200)
n_min_rel = 1 / rel**2

umbrales = [1e-3, 1e-4, 1e-5, 1e-6]

tabla_n = [
    dict(eps=e, n_min=int(np.floor(THETA**2 / e) + 1))
    for e in umbrales
]

rel_tab = [
    dict(rel=r, n_min=int(np.ceil(1 / r**2)))
    for r in [0.5, 0.2, 0.1, 0.05]
]

# 4. Verificación del número mínimo por Monte Carlo

eps0 = 1e-4
n_star = int(np.floor(THETA**2 / eps0) + 1)

chk = []

for n in [n_star - 5, n_star, n_star + 2,
          n_star + 3, n_star + 20, 100]:

    x = rng.exponential(1 / THETA, size=(R, n))
    s = x.sum(axis=1)

    chk.append(dict(
        n=n,
        crlb=THETA**2 / n,
        var_unb=float(((n - 1) / s).var(ddof=1)),
        var_mle=float((n / s).var(ddof=1))
    ))

# 5. Sensibilidad a la tasa

mtbf_rows = []

for n in [10, 30, 100]:
    x = rng.exponential(1 / THETA, size=(R, n))
    mtbf_hat = x.mean(axis=1)

    mtbf_rows.append(dict(
        n=n,
        sd_mc=float(mtbf_hat.std(ddof=1)),
        sd_lin=float(1 / (THETA * np.sqrt(n)))
    ))

# FIGURA 1: Varianza frente a cota de Cramer-Rao

n_arr = np.array([r["n"] for r in rows])
crlb_arr = np.array([r["crlb"] for r in rows])

fig, ax = plt.subplots(figsize=(6.4, 4.2))

nn = np.logspace(np.log10(5), np.log10(500), 200)

ax.plot(
    nn, THETA**2 / nn,
    "k-", lw=2,
    label=r"Cota de Cramer-Rao $\theta^2/n$"
)

ax.errorbar(
    n_arr,
    [r["var_mle"] for r in rows],
    yerr=[1.96 * r["se_var_mle"] for r in rows],
    fmt="o", color="#c0392b", capsize=3,
    label="MLE (MC)"
)

ax.errorbar(
    n_arr * 1.04,
    [r["var_unb"] for r in rows],
    yerr=[1.96 * r["se_var_unb"] for r in rows],
    fmt="s", color="#1f77b4", capsize=3,
    label="Insesgado (MC)"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Numero de fallos observados n")
ax.set_ylabel(r"Varianza de $\hat{\theta}$ (h$^{-2}$)")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(f"{out}/fig_varianza_vs_cota.png", dpi=200)
plt.close(fig)

# FIGURA 2: Sesgo

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.errorbar(
    n_arr,
    [r["bias_mle"] for r in rows],
    yerr=[1.96 * r["se_bias"] for r in rows],
    fmt="o-", color="#c0392b", capsize=3,
    label="Sesgo MC del MLE"
)

ax.plot(
    nn, THETA / (nn - 1),
    "k--",
    label=r"Sesgo teorico $\theta/(n-1)$"
)

ax.set_xscale("log")
ax.set_xlabel("Numero de fallos observados n")
ax.set_ylabel(r"Sesgo de $\hat{\theta}$ (h$^{-1}$)")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(f"{out}/fig_sesgo.png", dpi=200)
plt.close(fig)

# Numero minimo con varianza exacta

def n_exacto(e, kind):
    n = 3

    while True:
        if kind == "unb":
            v = THETA**2 / (n - 2)
        else:
            v = (
                THETA**2 * n**2 /
                ((n - 1)**2 * (n - 2))
            )

        if v < e:
            return n

        n += 1


tabla_prac = [
    dict(
        eps=e,
        n_cr=int(np.floor(THETA**2 / e) + 1),
        n_unb=n_exacto(e, "unb"),
        n_mle=n_exacto(e, "mle")
    )
    for e in umbrales
]

# FIGURA 3: Numero minimo absoluto

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.loglog(
    eps, n_min_abs,
    "k-", lw=2,
    label=r"Cota Cramer-Rao: $n>\theta^2/\varepsilon$"
)

ax.loglog(
    eps, THETA**2 / eps + 2,
    "--", color="#1f77b4",
    label=r"Insesgado: $n>\theta^2/\varepsilon+2$"
)

for t in tabla_prac:
    ax.plot(t["eps"], t["n_cr"], "o", color="#c0392b")
    ax.annotate(
        f"n={t['n_cr']}",
        (t["eps"], t["n_cr"]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=9
    )

ax.set_ylim(1.5, 8000)
ax.set_yticks([1, 10, 100, 1000])
ax.set_yticklabels(["1", "10", "100", "1000"])
ax.set_xlabel(r"Umbral de varianza $\varepsilon$ (h$^{-2}$)")
ax.set_ylabel(r"$n$ minimo ($\theta=0.05$ h$^{-1}$)")
ax.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
fig.savefig(f"{out}/fig_nmin_umbral.png", dpi=200)
plt.close(fig)

# FIGURA 4: Numero minimo relativo

fig, ax = plt.subplots(figsize=(6.4, 4.2))

ax.plot(rel * 100, n_min_rel, "k-", lw=2)

for r in [0.5, 0.2, 0.1, 0.05]:
    ax.plot(r * 100, 1 / r**2, "o", color="#c0392b")
    ax.annotate(
        f"n={int(np.ceil(1 / r**2))}",
        (r * 100, 1 / r**2),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=9
    )

ax.set_yscale("log")
ax.set_xlabel("Desviacion estandar relativa objetivo (%)")
ax.set_ylabel("n minimo")
fig.tight_layout()
fig.savefig(f"{out}/fig_nmin_relativo.png", dpi=200)
plt.close(fig)

# FIGURA 5: Log-verosimilitud y curvatura

fig, ax = plt.subplots(figsize=(6.4, 4.2))

x_obs = rng.exponential(1 / THETA, size=30)
th = np.linspace(0.02, 0.1, 300)

ll = len(x_obs) * np.log(th) - th * x_obs.sum()

ax.plot(
    th, ll - ll.max(),
    "k-", lw=2,
    label=r"$\ell(\theta)$ (n=30)"
)

mle0 = len(x_obs) / x_obs.sum()

curv = -len(x_obs) / th**2

ax.axvline(
    mle0, color="#c0392b", ls="--",
    label=fr"$\hat{{\theta}}={mle0:.4f}$"
)

ax.set_xlabel(r"$\theta$ (h$^{-1}$)")
ax.set_ylabel(r"$\ell(\theta)-\ell(\hat{\theta})$")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(f"{out}/fig_loglik.png", dpi=200)
plt.close(fig)

# Guardar resultados

res = dict(
    theta=THETA,
    R=R,
    rows=rows,
    fisher=fish,
    tabla_n=tabla_n,
    rel_tab=rel_tab,
    tabla_prac=tabla_prac,
    n_star=n_star,
    check=chk,
    mtbf=mtbf_rows,
    mle_ejemplo=float(mle0)
)

with open(f"{out}/resultados.json", "w") as f:
    json.dump(res, f, indent=1)

print(json.dumps(res, indent=1))
print("\nSimulacion finalizada correctamente.")
