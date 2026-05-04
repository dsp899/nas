# benchlib

Repositorio de bench libraries para tres niveles:

- `rnn_benchlib`: generación, benchmark y export de datasets para GNN de modelos RNN
- `cnn_benchlib`: generación, conversión y benchmark de CNNs
- `hybrid_benchlib`: composición y benchmark end-to-end de pipelines CNN-RNN

## Organización recomendada

- `artifacts/`: persistencia por defecto de todas las ramas
- `configs/`: directorio común para ficheros de configuración de las distintas ramas
  - por convención, los ficheros de la rama RNN deben diferenciarse en el nombre, por ejemplo `rnn_config_example.json`

La versión del repositorio se publica en `version.txt`.
