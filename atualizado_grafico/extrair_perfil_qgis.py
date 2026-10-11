# -*- coding: utf-8 -*-
# EXTRAI PERFIL DO RIO - 2 KM A PARTIR DE P01
# Versao 2 - le MDE direto, sem processing

from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature,
                        QgsGeometry, QgsPointXY, QgsField)
from PyQt5.QtCore import QVariant
import numpy as np
from pathlib import Path
import csv

# CONFIGURACAO
NOME_RIO = "rio aracatuba"
NOME_MDE = "mde_reprojetado"
PASTA_SAIDA = Path(r"C:\Users\murie\Downloads\orcamento cooltive garopaba\qualufmg-aracatuba-imbituba\atualizado_grafico")

P01_E = 726290.4763318696059287
P01_N = 6888447.66134350001811981

DISTANCIA = 2000.0
PASSO = 25.0

print("=" * 60)
print("EXTRACAO DE PERFIL - RIO ARACATUBA (2 KM) - v2")
print("=" * 60)

# 1) Pegar camada do rio
rio = QgsProject.instance().mapLayersByName(NOME_RIO)[0]
print("[OK] Rio: " + rio.name())

# 2) Pegar a geometria
geom = list(rio.getFeatures())[0].geometry()
if geom.isMultipart():
    linhas = geom.asMultiPolyline()
    linha = max(linhas, key=lambda l: sum(
        np.hypot(l[i+1].x()-l[i].x(), l[i+1].y()-l[i].y())
        for i in range(len(l)-1)))
else:
    linha = geom.asPolyline()

pts = np.array([(p.x(), p.y()) for p in linha])
print("[OK] " + str(len(pts)) + " vertices no rio")

# 3) Localizar P01
d = np.sqrt((pts[:, 0] - P01_E)**2 + (pts[:, 1] - P01_N)**2)
idx_p01 = int(np.argmin(d))
print("[OK] P01 no vertice " + str(idx_p01) + " (dist=" + str(round(d[idx_p01], 2)) + " m)")

# 4) Direcao do fluxo
direction = +1 if pts[-1, 1] < pts[0, 1] else -1

# 5) Extrair trecho
idx = idx_p01
total = 0.0
trecho = [pts[idx].copy()]
while total < DISTANCIA:
    i_next = idx + direction
    if i_next < 0 or i_next >= len(pts):
        print("[AVISO] Fim do rio antes de 2000 m")
        break
    seg = np.linalg.norm(pts[i_next] - pts[idx])
    total += seg
    trecho.append(pts[i_next].copy())
    idx = i_next
trecho = np.array(trecho)
print("[OK] Trecho: " + str(len(trecho)) + " vertices, " + str(round(total, 1)) + " m")

# 6) Densificar
seg_len = np.linalg.norm(np.diff(trecho, axis=0), axis=1)
dist_acum = np.concatenate([[0], np.cumsum(seg_len)])
d_novo = np.arange(0, dist_acum[-1] + 0.1, PASSO)
x_novo = np.interp(d_novo, dist_acum, trecho[:, 0])
y_novo = np.interp(d_novo, dist_acum, trecho[:, 1])
print("[OK] Densificado: " + str(len(d_novo)) + " pontos")

# 7) Amostrar MDE DIRETO (sem processing)
print("[INFO] Lendo valores do MDE...")
mde = QgsProject.instance().mapLayersByName(NOME_MDE)[0]
provider = mde.dataProvider()

elevs = []
for xn, yn in zip(x_novo, y_novo):
    val, ok = provider.sample(QgsPointXY(xn, yn), 1)
    if ok:
        elevs.append(float(val))
    else:
        elevs.append(0.0)

elevs = np.array(elevs)
print("[OK] MDE amostrado: min=" + str(round(elevs.min(), 2)) +
      " m, max=" + str(round(elevs.max(), 2)) + " m")

# 8) Salvar CSV
csv_path = PASTA_SAIDA / "perfil_rio_2km.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['distance_m', 'elevation_m'])
    for dn, el in zip(d_novo, elevs):
        w.writerow([str(round(dn, 2)), str(round(el, 3))])

print("")
print("[OK] CSV salvo: " + str(csv_path))

# 9) Resumo
with open(csv_path, 'r') as f:
    linhas = f.readlines()
print("")
print("=" * 60)
print("PRIMEIRAS 5 LINHAS DO CSV:")
for l in linhas[:5]:
    print("   " + l.strip())
print("ULTIMAS 3 LINHAS DO CSV:")
for l in linhas[-3:]:
    print("   " + l.strip())
print("=" * 60)
print("")
print("[SUCESSO] Arquivo salvo em:")
print("   " + str(csv_path))