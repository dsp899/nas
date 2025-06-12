# importancia_relativa.py
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from collections import defaultdict
import matplotlib
# --- Configuración ---
archivo_json = "../nas_log_20250525_153457.json"
directorio_salida = "figuras"
os.makedirs(directorio_salida, exist_ok=True)

# Estilo
matplotlib.rcParams['font.style'] = 'italic'
sns.set_style("whitegrid")
sns.set_palette("gray")

# --- Vocabulario ---
vocabulario = {
    "layers": [str(v) for v in [1, 2, 3]],
    "rnn": ['lstm', 'gru'],
    "units_0": [str(v) for v in [8, 16, 32, 64, 128, 256, 300, 512, 700, 900, 1024]],
    "units_1": [str(v) for v in [0, 8, 16, 32, 64, 128, 256, 300, 512, 700]],
    "units_2": [str(v) for v in [0, 8, 16, 32, 64, 128, 256, 300, 512, 700]],
    "direction": ['unidirectional', 'bidirectional'],
    "state": ['stateless', 'statefull'],
    "seq": [str(v) for v in [3, 6, 9, 12]],
    "cnn": ['vgg16', 'resnet50', 'inceptionV3']
}
n_opciones = {campo: len(valores) for campo, valores in vocabulario.items()}

# --- Leer datos ---
with open(archivo_json, "r") as f:
    data = json.load(f)
architectures = data["architectures"]
total_architectures = len(architectures)

# --- Conteo de tokens ---
token_counts = defaultdict(int)
for entry in architectures:
    seq = entry["decoded"]
    for campo, valor in zip(vocabulario.keys(), seq):
        token = f"{campo}_{valor}"
        token_counts[token] += 1

# --- Importancia relativa ---
results = []
for token, count in token_counts.items():
    parts = token.split('_')
    campo = '_'.join(parts[:2]) if 'units' in token else parts[0]
    valor = '_'.join(parts[2:]) if 'units' in token else '_'.join(parts[1:])
    if campo not in n_opciones:
        continue
    p_post = count / total_architectures
    p_prior = 1 / n_opciones[campo]
    importancia = p_post / p_prior

    rewards = [e["metrics"]["normalized_reward"] for e in architectures if f"{campo}_{valor}" in [f"{c}_{v}" for c, v in zip(vocabulario.keys(), e["decoded"])]]
    mean_reward = np.mean(rewards) if rewards else 0

    results.append({
        'token': token,
        'campo': campo,
        'valor': valor,
        'count': count,
        'importancia': importancia,
        'mean_reward': mean_reward
    })

df = pd.DataFrame(results)
top_tokens = df.sort_values('importancia', ascending=False).head(20).sort_values('importancia', ascending=True)

# --- Gráfica ---
unique_campos_top = top_tokens['campo'].unique()
palette = sns.color_palette("gray", len(unique_campos_top))
color_map_top = {campo: palette[i] for i, campo in enumerate(unique_campos_top)}

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(
    top_tokens['token'],
    top_tokens['importancia'],
    color=[color_map_top[campo] for campo in top_tokens['campo']],
    height=0.65,
    edgecolor='k',
    linewidth=0.3
)
ax.axvline(x=1, color='black', linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.set_xticks([0.5, 1, 2, 4, 8])  # Ajusta según tu distribución
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


legend_handles = [
    mpatches.Patch(color=color_map_top[c], label=c.replace('_', ' ').capitalize())
    for c in unique_campos_top
]
ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=9, frameon=False, fontsize=9)

ax.set_xlabel('Relative Importance', fontsize=12)
#ax.set_ylabel('Token', fontsize=12)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=12)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

nombre_importancia = f"{directorio_salida}/top20_importancia_relativa"
for ext in ["png", "svg", "pdf"]:
    plt.savefig(f"{nombre_importancia}.{ext}", bbox_inches='tight', pad_inches=0.1, dpi=300 if ext == "png" else None)
plt.close()
print(f"[✔] Gráfica exportada: {nombre_importancia}.[png/svg/pdf]")
