# hybrid_benchlib

Runtime híbrido **real** para pipelines CNN-RNN que reutiliza artefactos ya generados por:

- `cnn_benchlib`
- `rnn_benchlib`

## Idea principal

`hybrid_benchlib` ya no modela un pipeline con latencias sintéticas. Ejecuta un **pipeline concurrente real** con:

- workers CNN reales
- colas reales
- ensamblado real de features por orden temporal
- construcción real de clips
- consumo real por la RNN
- agregación real de decisión de vídeo

La única diferencia práctica entre host y target es la **disponibilidad de backends**. El runtime es el mismo.

## Presets de runtime

- `float_all`
  - CNN float CPU
  - RNN float TensorFlow
- `tflite_all`
  - CNN TFLite CPU
  - RNN TFLite/TensorFlow
- `xmodel_tflite`
  - CNN XModel/DPU
  - RNN TFLite/TensorFlow

En host normalmente `xmodel_tflite` solo funcionará si el entorno dispone de `xir`/`vart`. En target debería ser el preset natural.

## Modos de pipeline

### `cnn_rnn_overlap`

- paralelismo real entre workers CNN
- solape real CNN↔RNN
- si la RNN se retrasa, aparece cola real delante de la etapa RNN

### `cnn_rnn_serialized`

- mantiene paralelismo real entre workers CNN
- elimina el solape CNN↔RNN
- la fase RNN arranca cuando la fase CNN ya dejó preparados los clips del vídeo

## Qué mide

Por vídeo:

- `t_first_ms`
- `t_update_mean_ms`
- `t_update_min_ms`
- `t_update_max_ms`
- `cnn_total_ms`
- `rnn_total_ms`
- `queue_wait_total_ms`
- `video_total_ms`

Por evento de feature:

- `produced_ms`
- `preprocess_ms`
- `infer_ms`
- `postprocess_ms`
- `total_ms`

Por clip:

- `clip_ready_ms`
- `rnn_queue_wait_ms`
- `rnn_start_ms`
- `encoder_ms`
- `clip_head_ms`
- `aggregation_ms`
- `video_head_ms`
- `decision_ready_ms`

## CLI

Host o target usando el mismo runtime:

```bash
python -m hybrid_benchlib.cli.benchmark_host \
  --output-root ./artifacts \
  --hybrid-model-id <hybrid_model_id> \
  --runtime-preset tflite_all \
  --overlap-mode cnn_rnn_overlap \
  --cnn-workers 3 \
  --hop 1 \
  --sample-stride-frames 1 \
  --video-fps 30 \
  --num-videos 8 \
  --frames-per-video 64
```

Alias equivalente:

```bash
python -m hybrid_benchlib.cli.benchmark_pipeline ...
```

En target puedes usar también:

```bash
python -m hybrid_benchlib.cli.benchmark_target ...
```
