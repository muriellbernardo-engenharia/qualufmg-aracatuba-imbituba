# -*- coding: utf-8 -*-
# Limpa outliers do perfil e gera CSV final

import csv
import numpy as np
from pathlib import Path
from scipy.signal import medfilt

SCRIPT_DIR = Path(__file__).resolve().parent
ARQ = SCRIPT_DIR / "perfil_rio_2km.csv"
ARQ_OUT = SCRIPT_DIR / "perfil_rio_2km_limpo.csv"

# Ler
with open(ARQ, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    dados = [(float(r[0]), float(r[1])) for r in reader]

d = np.array([x[0] for x in dados])
z = np.array([x[1] for x in dados])
print("[INFO] " + str(len(d)) + " pontos lidos")

# Filtro de mediana (janela 5) para remover picos
z_med = medfilt(z, kernel_size=5)
print("[INFO] Apos mediana: min=" + str(round(z_med.min(), 2)) +
      " max=" + str(round(z_med.max(), 2)))

# Remover pontos onde desvio > 1.5 m da mediana
desvio = np.abs(z - z_med)
mask = desvio < 1.5
print("[INFO] Pontos removidos: " + str(int((~mask).sum())))

d_limpo = d[mask]
z_limpo = z_med[mask]
print("[INFO] Pontos restantes: " + str(len(d_limpo)))

# Salvar
with open(ARQ_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['distance_m', 'elevation_m'])
    for di, zi in zip(d_limpo, z_limpo):
        w.writerow([str(round(di, 2)), str(round(zi, 3))])

print("")
print("[OK] Salvo: " + str(ARQ_OUT))
print("[SUCESSO] Pronto para o modelo!")