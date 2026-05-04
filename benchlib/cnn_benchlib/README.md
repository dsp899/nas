# cnn_benchlib

Benchlib específico para CNN. Gestiona artefactos por etapas:

- modelo float (`keras`)
- export `tflite`
- cuantización Xilinx/Vitis AI
- compilación a `xmodel`
- benchmark de componente CNN en host y, cuando el entorno lo permita, en ZCU102.

A diferencia de `rnn_benchlib`, la filosofía es más individualizada: los CLIs operan sobre un modelo o artefacto concreto, no sobre grandes pools por defecto.
