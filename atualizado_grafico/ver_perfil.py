# -*- coding: utf-8 -*-
# Visualiza o perfil extraido - so para diagnostico

import csv
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ARQ = SCRIPT_DIR / "perfil_rio_2km.csv"

with open(ARQ, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    dados = [(float(r[0]), float(r[1])) for r in reader]

d = np.array([x[0] for x in dados])
z = np.array([x[1] for x in dados])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(d, z, 'o-', color='#1E88E5', lw=1.5, markersize=4,
        label="Perfil bruto (MDE)")
ax.axhline(np.median(z), color='#8B2635', lw=1.5, ls='--',
           label="Mediana = " + str(round(np.median(z), 2)) + " m")
ax.fill_between(d, np.median(z) - 0.5, np.median(z) + 0.5,
                 color='#E8F5E9', alpha=0.4,
                 label="Faixa esperada (+/-0.5 m)")
ax.set_xlabel("Distancia a partir de P01 (m)")
ax.set_ylabel("Cota (m)")
ax.set_title("Diagnostico do Perfil Extraido - Rio Aracatuba (2 km)")
ax.legend()
ax.grid(True, alpha=0.3)

saida = SCRIPT_DIR / "diagnostico_perfil.png"
fig.savefig(str(saida), dpi=150, bbox_inches='tight')
print("[OK] Salvo: " + str(saida))