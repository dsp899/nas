# token_vs_metrica.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import defaultdict
import matplotlib.patches as mpatches

# --- Configuración general ---
archivo_json = "../nas_log_20250525_153457.json"
directorio_salida = "figuras"
os.makedirs(directorio_salida, exist_ok=True)

# 📌 Parámetro para cambiar la métrica del eje Y
metrica_y = "accuracy"  # Cambia a: "accuracy", "raw_reward", "normalized_reward"

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

# --- Leer datos ---
with open(archivo_json, "r") as f:
    data = json.load(f)
architectures = data["architectures"]

# --- Token stats ---
xlim_min, xlim_max = 0.75, 0.95
ylim_min, ylim_max = 0.5, 0.77  # Se adapta luego
tokens_a_excluir = {"units_1_0", "units_2_0"}

token_stats = defaultdict(lambda: {
    "count": 0, "accuracy": [], "raw_reward": [],
    "normalized_reward": [], "campo": ""
})

for arch in architectures:
    seq = arch["decoded"]
    metrics = arch["metrics"]
    for campo, valor in zip(vocabulario.keys(), seq):
        token = f"{campo}_{valor}"
        token_stats[token]["count"] += 1
        token_stats[token]["accuracy"].append(metrics["accuracy"])
        token_stats[token]["raw_reward"].append(metrics["raw_reward"])
        token_stats[token]["normalized_reward"].append(metrics["normalized_reward"])
        token_stats[token]["campo"] = campo

# --- Normalización y filtrado ---
conteos_crudos = {t: s["count"] for t, s in token_stats.items()}
valores_log = np.log1p(list(conteos_crudos.values()))
max_log = np.max(valores_log) or 1
conteos_norm = {t: np.log1p(c) / max_log for t, c in conteos_crudos.items()}

datos_finales = []
for token, stats in token_stats.items():
    if token in tokens_a_excluir:
        continue
    mean_value = np.mean(stats[metrica_y])
    norm_count = conteos_norm[token]
    if xlim_min <= norm_count <= xlim_max:
        datos_finales.append({
            "token": token,
            "campo": stats["campo"],
            "valor_y": mean_value,
            "normalized_count": norm_count,
            "raw_count": stats["count"]
        })

# --- Gráfica ---
unique_campos = sorted(set(d["campo"] for d in datos_finales))
grayscale_values = np.linspace(0.2, 0.8, len(unique_campos))
color_map = {campo: str(gray) for campo, gray in zip(unique_campos, grayscale_values)}

x = [d["normalized_count"] for d in datos_finales]
y = [d["valor_y"] for d in datos_finales]
colors = [color_map[d["campo"]] for d in datos_finales]
sizes = [30 + d["raw_count"] * 1.5 for d in datos_finales]

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x, y, s=sizes, c=colors, alpha=0.8, edgecolors='k', linewidth=0.3)

# Líneas guía + etiquetas
for i in range(len(y)):
    ax.vlines(x[i], ymin=0, ymax=y[i], colors='#EAEAEA', linestyles='dashed', linewidth=0.5)
    ax.hlines(y[i], xmin=0.75, xmax=x[i], colors='#EAEAEA', linestyles='dashed', linewidth=0.5)

arriba = True
for d in datos_finales:
    offset = 0.005 if arriba else -0.01
    va = 'bottom' if arriba else 'top'
    ax.text(d["normalized_count"], d["valor_y"] + offset, d["token"].capitalize(),
            ha='center', va=va, fontsize=7, fontweight='bold', rotation=90)
    arriba = not arriba

# Leyenda
legend_handles = [mpatches.Patch(color=color_map[c], label=c.replace('_', ' ').capitalize()) for c in unique_campos]
ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=9, frameon=False, fontsize=9)

ax.set_xlabel("Normalized Frequency", fontsize=12)
ax.set_ylabel(f"Mean {metrica_y.replace('_', ' ').capitalize()}", fontsize=12)
ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(ylim_min, ylim_max)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, linestyle='--', linewidth=0.5)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()

nombre_base = f"{directorio_salida}/token_freq_vs_{metrica_y}_filtered_x{xlim_min}-{xlim_max}_y{ylim_min}-{ylim_max}"
for ext in ["png", "svg", "pdf"]:
    plt.savefig(f"{nombre_base}.{ext}", bbox_inches='tight', pad_inches=0.1, dpi=300 if ext == "png" else None)
plt.close()
print(f"[✔] Gráfica exportada: {nombre_base}.[png/svg/pdf]")
