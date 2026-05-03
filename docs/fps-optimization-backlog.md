# Backlog de optimización FPS (swap / restore / pipeline)

Documento de **consulta interna** para priorizar e implementar mejoras de rendimiento de forma incremental. Cada entrada tiene ID estable: úsalo al pedir implementación o pruebas (“implementa **PERF-012**”).

**Documentos relacionados**

- Ajustes ya expuestos en UI: [`performance-settings-guide.md`](performance-settings-guide.md)
- Arquitectura, colas, variables `VISIOMASTER_*`: [`agent-architecture.md`](agent-architecture.md)

---

## Cómo usar este backlog

1. **Medir antes**: fija un vídeo de referencia (resolución, nº caras, modelo de swap/restaurador) y anota FPS o ms/frame con `VISIOMASTER_PERF_BUNDLE=1` y, si hace falta, `VISIOMASTER_PERF_STAGES` / `VISIOMASTER_PIPELINE_METRICS`.
2. **Elegir por fase**: la tabla [Orden sugerido global](#orden-sugerido-global-fases) minimiza riesgo; dentro de una fase, prioriza por **impacto** y por el cuello de botella que muestre el perfil.
3. **Registrar resultados**: al cerrar una idea, actualiza la [tabla de estado](#estado-del-backlog-edítalo-en-git) (commit en git) y, si quieres, el mensaje de commit o PR con el ID **PERF-xxx**.

### Estado del backlog (edítalo en git)

Valores sugeridos en **Estado**: `Pendiente` | `En curso` | `Hecho` | `Descartado` | `En pausa`. La columna **Notas** puede enlazar PR, commit o breve resultado de medición.

| ID | Estado | Notas |
|----|--------|-------|
| PERF-001 | Pendiente | |
| PERF-002 | Pendiente | |
| PERF-003 | Hecho | FaceParser ORT en CUDA: IOBinding directo (sin ``img.cpu().numpy()`` + ``session.run``). |
| PERF-004 | Hecho | ``rgb_hwc_uint8_numpy_to_torch_chw`` en ``miscellaneous.py`` (pinned + ``non_blocking``); usado en detección secuencial, issue scan y ``FrameWorker`` cuando no hay handoff CHW. ``VISIOMASTER_DISABLE_PINNED_H2D=1`` para desactivar. |
| PERF-005 | Pendiente | |
| PERF-006 | Pendiente | |
| PERF-007 | Pendiente | |
| PERF-008 | Pendiente | |
| PERF-009 | Pendiente | |
| PERF-010 | Pendiente | |
| PERF-011 | Pendiente | |
| PERF-012 | Pendiente | |
| PERF-013 | Pendiente | |
| PERF-014 | Pendiente | |
| PERF-015 | Pendiente | |
| PERF-016 | Pendiente | |
| PERF-017 | Pendiente | |
| PERF-018 | Hecho | Implementado: `RecordOutputDecoupleResizeToggle` + `RecordOutputResizeSizeSelection` en ajustes; `VideoProcessor.apply_record_output_resize_decouple_to_dims` + resize antes de escribir a FFmpeg. |
| PERF-019 | Pendiente | |

### Leyenda de impacto estimado (rendimiento)

| Etiqueta | Significado orientativo |
|----------|-------------------------|
| **Alto** | Suele recortar una fracción grande del tiempo de frame en escenarios típicos (p. ej. −20% ms o más) cuando el subsistema afectado es el cuello de botella. |
| **Medio** | Ganancia clara en subconjuntos (multi-cara, TRT, muchas sincronizaciones, etc.). |
| **Bajo** | Mejora marginal, solo en hardware/caso concreto, o principalmente latencia/arranque. |

*Los porcentajes son **estimaciones**; el impacto real depende del GPU, EP, resolución y qué etapa domina el perfil.*

### Leyenda de esfuerzo / riesgo

| Esfuerzo | Significado |
|----------|----------------|
| S | Horas–1 día; cambios localizados. |
| M | Varios días; toca varios módulos o contratos de datos. |
| L | Semanas; concurrencia, nuevos modos UI o pipelines paralelos. |

| Riesgo | Significado |
|--------|-------------|
| Bajo | Regresiones fáciles de cubrir con tests o comparación visual. |
| Medio | Puede afectar tracking, identidad o calidad en bordes. |
| Alto | Concurrencia, TRT shapes, o trade-offs fuertes calidad/FPS. |

---

## Orden sugerido global (fases)

| Fase | Objetivo | IDs principales |
|------|----------|-----------------|
| **0** | Baseline y decisiones basadas en datos | PERF-001 |
| **1** | Infra segura: menos sync, mejor transferencia | PERF-002, PERF-003, PERF-004 |
| **2** | Afinar ORT/TRT y batch ya alineado con el diseño | PERF-005, PERF-006, PERF-007 |
| **3** | Batch de dominio (más caras, mismos modelos) | PERF-008 |
| **4** | Estrategias temporales (FPS a costa de latencia/consistencia) | PERF-009, PERF-010, PERF-011 |
| **5** | Presupuesto de tiempo y modos de calidad | PERF-012 |
| **6** | Arquitectura “doble velocidad” (preview vs export, modelos distintos) | PERF-013, PERF-014, PERF-015 |

Implementar **en orden de fase** salvo que el perfil muestre claramente otro cuello (p. ej. si el restaurador no aparece en el top de etapas, posponer PERF-009).

---

## PERF-001 — Baseline con telemetría integrada

**Resumen**  
Uso sistemático de variables de rendimiento ya soportadas para etiquetar el cuello de botella antes de codificar.

**Impacto rendimiento**  
Bajo *directo* (solo diagnóstico); **Alto** *indirecto* (evita optimizar la rama equivocada).

**Esfuerzo**  
S | **Riesgo** Bajo

**Implementación**  
No requiere cambio de código si solo se documenta el protocolo de medición; opcional: script o plantilla de “benchmark run” (env + preset + clip).

**Criterios de decisión**  
- Hacer **siempre** antes de PERF-002 en adelante.  
- Si `prep_scaling`, `detect`, `swap`, `restore`, `mask` o colas dominan, la tabla de prioridades interna de cada idea indica qué implementar después.

**Referencias**  
`VISIOMASTER_PERF_BUNDLE`, `VISIOMASTER_PERF_STAGES`, `VISIOMASTER_PIPELINE_METRICS`, `VISIOMASTER_PERF_LOG`, `VISIOMASTER_PERF_SWAP_CORE`; `FrameWorker` (`app/processors/workers/frame_worker.py`); `docs/agent-architecture.md`.

---

## PERF-002 — Auditoría de `cuda synchronize` / sync en caliente

**Resumen**  
Inventariar y eliminar o aislar sincronizaciones GPU innecesarias en el camino feeder → detección → worker (salvo telemetría explícita o depuración).

**Impacto rendimiento**  
Medio–Alto cuando hoy hay sync por frame o por etapa en hardware rápido.

**Esfuerzo**  
M | **Riesgo** Medio (hay que no romper métricas ni modos “GPU sync” del overlay)

**Archivos probables**  
`app/processors/video_processor.py`, `app/processors/workers/frame_worker.py`, `models_processor.py` (callbacks ORT).

**Criterios de decisión**  
- Priorizar si `VISIOMASTER_PERF_STAGES` muestra huecos grandes entre marcas o GPU subutilizada con cola llena.  
- Posponer si el cuello es 100% inferencia ORT sin sync relevante.

---

## PERF-003 — Extender IOBinding / buffers reutilizados

**Resumen**  
Asegurar que los modelos más calientes usan IOBinding y tensores reutilizados donde el código aún haga alloc o rutas menos eficientes.

**Impacto rendimiento**  
Medio (menos alloc y copias); mayor en vídeos largos y muchas caras.

**Esfuerzo**  
M–L según cobertura | **Riesgo** Medio (TRT shapes, dtypes)

**Referencias**  
`VISIOMASTER_ORT_IOBINDING_POST_SYNC`; `app/processors/models_processor.py` y módulos `face_*.py`.

**Criterios de decisión**  
- Fuerte si hay picos de VRAM o tiempo en “bind/copy” entre forwards.  
- Coordinar con PERF-005 (perfiles TRT) al cambiar shapes de entrada.

**Implementación (Fusion)**  
- `FaceMasks.run_faceparser`: ORT + CUDA + entrada en GPU → IOBinding (sin `cpu().numpy()` + `session.run`).  
- `run_rvm_portrait_alpha` / `run_u2netp_salient_alpha`: rama sin CUDA EP → IOBinding con tensores en el dispositivo de enlace (`cpu` / `cuda` según `get_ort_bind_device_type`).  
- `FrameEnhancers` (RIFE preview): rama no `uses_cuda_ep_for_thread` → `run_onnx_io_binding` en lugar de `session.run`.  
- `mouth_action_detector`: modelo **TensorFlow** (no ORT); no aplica IOBinding ORT.

---

## PERF-004 — H2D: pinned host + `non_blocking` y handoff CHW

**Resumen**  
Generalizar el patrón ya existente de evitar doble subida del frame (feeder → detección → worker) y usar memoria fijada y copias asíncronas donde falte.

**Impacto rendimiento**  
Medio en CPUs débiles o resoluciones altas; Bajo si el cuello es solo inferencia.

**Esfuerzo**  
M | **Riesgo** Medio (condiciones de carrera stream/stream)

**Archivos probables**  
`video_processor.py`, `frame_worker.py`, feeders.

**Criterios de decisión**  
- Subir si el perfil marca mucho tiempo en preparación / numpy→torch.  
- Tras PERF-002 (streams coherentes).

**Implementación (Fusion)**  
`app/helpers/miscellaneous.rgb_hwc_uint8_numpy_to_torch_chw`: staging `pin_memory` + `.to(..., non_blocking=True)` en CUDA. Usado en `_run_sequential_detection` (frame completo y ROI), issue-scan tensor, y `FrameWorker` sin `_feeder_chw_tensor`. Variable `VISIOMASTER_DISABLE_PINNED_H2D=1` desactiva el pin. Tests: `tests/unit/helpers/test_miscellaneous_h2d.py`.

---

## PERF-005 — Afinar perfiles TensorRT de batch (`VISIOMASTER_TRT_*`)

**Resumen**  
Ajustar `max/opt` de batch para Inswapper batched, ArcFace batched, LivePortrait motion/stitch, etc., al histograma real de caras y al modo de uso.

**Impacto rendimiento**  
Medio–Alto en multi-cara y LP; Bajo en siempre 1 cara con engines batch=1.

**Esfuerzo**  
S (solo env + medición) a M (si hace falta lógica de selección de perfil) | **Riesgo** Medio (recompilar engines)

**Criterios de decisión**  
- Hacer después de PERF-001.  
- Si el usuario casi nunca supera 2 caras, capar `opt` puede acelerar builds y runtime.

---

## PERF-006 — Claridad UI/docs: Inswapper ORT batch vs TensorRT-Engine

**Resumen**  
El path `run_inswapper_ort_batched` puede desactivarse con engines nativos batch 1; documentar o exponer en UI cuándo el batch ORT ayuda y cuándo no.

**Impacto rendimiento**  
Bajo directo; Medio en evitar configuraciones “muertas” que el usuario cree óptimas.

**Esfuerzo**  
S | **Riesgo** Bajo

**Criterios de decisión**  
Rápido de hacer; mejora soporte y reduce falsos positivos en pruebas A/B.

---

## PERF-007 — Variable `VISIOMASTER_INSWAPPER_ORT_BATCH` y pruebas de regresión

**Resumen**  
Ensayos automatizados o checklist manual con EP CUDA vs TRT-Engine forzando batch on/off para asegurar fallback correcto.

**Impacto rendimiento**  
Bajo; **Medio** en confianza para activar batch en más escenarios.

**Esfuerzo**  
S–M | **Riesgo** Bajo

**Referencias**  
`app/processors/face_swappers.py` (`run_inswapper_ort_batched`).

---

## PERF-008 — Batch de restaurador (misma red, varias caras / teselas)

**Resumen**  
Agrupar forwards del **mismo** modelo restaurador en un único `run` ORT cuando las entradas compartan tamaño y dtype (análogo conceptual al batch de Inswapper/ArcFace).

**Impacto rendimiento**  
Alto cuando 2+ caras usan restaurador caro (GFPGAN, CodeFormer, etc.); Bajo si restaurador off o 1 cara.

**Esfuerzo**  
L | **Riesgo** Medio–Alto (padding, VRAM pico, TRT perfil)

**Dependencias**  
PERF-001, PERF-005, PERF-003 recomendables antes.

**Criterios de decisión**  
- Implementar si el perfil muestra restaurador > swap en multi-cara.  
- Posponer si casi todos los usuarios usan 1 cara o restaurador ligero.

---

## PERF-009 — Restaurador subsampled (cada N frames) + blend estable

**Resumen**  
Ejecutar restaurador pesado cada 2–3 frames; en frames intermedios reutilizar o interpolar (mapa de mejora, alpha de blend, o salida anterior con umbral de movimiento).

**Impacto rendimiento**  
Alto si el restaurador es dominante; calidad variable en boca/ojos rápidos.

**Esfuerzo**  
L | **Riesgo** Alto (artefactos, desync con swap)

**Criterios de decisión**  
- Ideal para preview en vivo o GPUs medias.  
- Peor para close-up de labios; requiere toggle “calidad máxima” vs “FPS”.

**Dependencias**  
PERF-001 imprescindible; PERF-002 ayuda a no mezclar latencias mal con sync.

---

## PERF-010 — Detección asíncrona con desfase K frames + predicción de bbox

**Resumen**  
El worker consume geometría del frame N−K mientras otro hilo ejecuta detector en N; predicción corta vía velocidad del track para alinear.

**Impacto rendimiento**  
Alto en coste de detector dominante; introduce **latencia** y riesgo en entradas/salidas de cara.

**Esfuerzo**  
L | **Riesgo** Alto (arquitectura de colas, consistencia con ByteTrack)

**Criterios de decisión**  
Solo si el producto acepta latencia adicional (p. ej. streaming no interactivo).  
No como primera opción para edición frame-a-frame.

---

## PERF-011 — ROI adaptativo por varianza en máscara / motion score

**Resumen**  
Además del track-guided ROI en frames skip, encoger ROI cuando la región facial es estable y expandir cuando sube varianza de píxeles o movimiento.

**Impacto rendimiento**  
Medio con intervalo de detección > 1; Bajo si detección full-frame cada frame.

**Esfuerzo**  
M | **Riesgo** Medio (pérdida de cara en gestos grandes si el umbral es agresivo)

---

## PERF-012 — Presupuesto de ms por frame (“deadline scheduler”)

**Resumen**  
Antes del frame, estimar o medir EMA de costes; si el presupuesto se agota, omitir pasos opcionales (restaurador secundario, parsing fino de máscara, etc.).

**Impacto rendimiento**  
Medio–Alto en escenas mixtas; complejidad de UX (modo “estable FPS”).

**Esfuerzo**  
L | **Riesgo** Medio (predecibilidad para el usuario)

**Criterios de decisión**  
Mejor tras tener PERF-001 y varias optimizaciones ya acumuladas para conocer costes típicos.

---

## PERF-013 — “Auto-restaurador” por tamaño de cara / motion (extensión de auto-res)

**Resumen**  
Jerarquía automática off → modelo ligero → pesado según área de bbox en pantalla y/o score de movimiento (similar filosofía a swapper auto-res ya descrita en guía de settings).

**Impacto rendimiento**  
Medio–Alto en vídeos con mezcla de planos generales y primeros planos.

**Esfuerzo**  
M–L | **Riesgo** Medio (cambios visibles en cortes)

**Referencias**  
Controles de swapper auto-res / histéresis en `performance-settings-guide.md`.

---

## PERF-014 — Pipeline dual: modelo “student” / ligero en preview, pesado en export

**Resumen**  
Preview en vivo con swapper+restore baratos; job de export usa checkpoint pesado (o misma arquitectura con más pasadas).

**Impacto rendimiento**  
Alto percibido en uso interactivo; el export sigue lento.

**Esfuerzo**  
L (UX, jobs, posible entrenamiento/distillation externa) | **Riesgo** Medio

**Criterios de decisión**  
Requiere modelo ligero aceptable o inversión en assets; no es solo ingeniería.

---

## PERF-015 — Export: procesar a FPS menor + interpolación (RIFE / etc.)

**Resumen**  
Pipeline pesado a 15–20 FPS; interpolación a 30/60 solo en codificación final.

**Impacto rendimiento**  
Alto en **tiempo total de export**; no aumenta FPS en vivo.

**Esfuerzo**  
L | **Riesgo** Medio (artefactos de interpolación, dependencia extra)

---

## PERF-016 — Extrapolación de bbox con flujo óptico (p. ej. API hardware)

**Resumen**  
En frames sin detector full, usar flujo bloque u óptico para desplazar bbox sin red.

**Impacto rendimiento**  
Medio si el detector se salta a menudo; depende de HW/API.

**Esfuerzo**  
M–L | **Riesgo** Medio

---

## PERF-017 — `torch.compile` en subgrafos PyTorch estables

**Resumen**  
Activar compilación solo en bloques deterministas (p. ej. warps/color) con fallback documentado (`VISIOMASTER_TORCH_COMPILE`).

**Impacto rendimiento**  
Variable (PyTorch/inductor); puede ser Medio o nulo según versión.

**Esfuerzo**  
M | **Riesgo** Medio (arranque, compatibilidad)

---

## PERF-018 — Resolución de preview vs resolución de grabación

**Resumen**  
Desacoplar “resize de trabajo” en preview de la resolución final al grabar (hoy el resize global ya es el principal mando; esto sería un segundo eje explícito).

**Impacto rendimiento**  
Alto en preview; requiere upscale al final de grabación (coste en export).

**Esfuerzo**  
L | **Riesgo** Medio (consistencia máscaras/swap)

**Implementación (Fusion)**  
Ajustes → *Resize Input Source*: `RecordOutputDecoupleResizeToggle` + `RecordOutputResizeSizeSelection`; código en `VideoProcessor.apply_record_output_resize_decouple_to_dims`, `resize_numpy_bgr_for_recording_stdin`, `create_ffmpeg_subprocess`, `display_next_frame`. Tests: `tests/unit/processors/test_video_processor_record_decouple.py`.

---

## PERF-019 — Micro-batch entre workers (fusionar dos frames en batch 2)

**Resumen**  
Dos workers con 1 cara cada uno podrían teóricamente unir inferencias; implica sincronización fuerte y puede **empeorar latencia**.

**Impacto rendimiento**  
Teórico Medio–Alto; en la práctica muy sensible a scheduling.

**Esfuerzo**  
L | **Riesgo** Alto

**Criterios de decisión**  
Solo investigación después de PERF-008 y PERF-005; probablemente **no** priorizar frente a batch intra-frame.

---

## Tabla resumen (decisión rápida)

El **Estado** vigente vive en la [tabla de estado](#estado-del-backlog-edítalo-en-git); aquí no se duplica para evitar dos fuentes de verdad.

| ID | Idea | Impacto | Esfuerzo | Riesgo | Cuándo priorizar |
|----|------|---------|-----------|--------|------------------|
| PERF-001 | Telemetría / baseline | Indirecto alto | S | Bajo | Siempre primero |
| PERF-002 | Quitar sync caliente | Med–Alto | M | Medio | GPU ociosa, cola llena |
| PERF-003 | IOBinding extendido | Medio | M–L | Medio | Muchos forwards / alloc |
| PERF-004 | Pinned + non_blocking CHW | Medio | M | Medio | H2D visible en perfil |
| PERF-005 | TRT batch profiles | Med–Alto | S–M | Medio | Multi-cara / LP |
| PERF-006 | Doc UI Inswapper batch | Bajo–Med | S | Bajo | Mejora soporte |
| PERF-007 | Tests/checklist batch | Bajo–Med | S–M | Bajo | Tras tocar swap batch |
| PERF-008 | Batch restaurador | Alto (multi) | L | Med–Alto | Restore > swap en perfil |
| PERF-009 | Restaurador cada N frames | Alto | L | Alto | Preview, restore caro |
| PERF-010 | Detección async desfasada | Alto | L | Alto | Latencia aceptable |
| PERF-011 | ROI adaptativo varianza | Medio | M | Medio | Intervalo det > 1 |
| PERF-012 | Deadline ms / modos FPS | Med–Alto | L | Medio | UX “stable FPS” |
| PERF-013 | Auto-restaurador | Med–Alto | M–L | Medio | Mezcla planos / close-up |
| PERF-014 | Dual pipeline student | Alto UX | L | Medio | Hay modelo ligero |
| PERF-015 | Export + interpolación | Alto export | L | Medio | Solo batch export |
| PERF-016 | Bbox + flujo óptico | Medio | M–L | Medio | Skip detect frecuente |
| PERF-017 | torch.compile parcial | Variable | M | Medio | Tras baseline PyTorch |
| PERF-018 | Preview res ≠ record res | Alto preview | L | Medio | Separar calidad/live |
| PERF-019 | Micro-batch entre workers | ? | L | Alto | Investigación tardía |

---

## Notas para el asistente / implementación futura

- Al implementar cualquier ID, citar este archivo y el ID en el mensaje de commit o PR para trazabilidad.  
- Tras cada implementación: repetir **PERF-001** (misma máquina, mismo clip) y guardar números antes/después.  
- No combinar PERF-009 + PERF-010 en el mismo PR sin pruebas aisladas: ambos alteran temporalidad y multiplican escenarios de fallo.
