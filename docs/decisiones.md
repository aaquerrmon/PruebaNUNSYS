# Decisiones técnicas (ADR resumidos)

Relaciona cada decisión con su coste y su alternativa descartada. La motivación
funcional completa está en el [README](../README.md#2-decisiones-técnicas).

---

## ADR-001 · Inferencia exclusivamente local con vLLM en Compose

**Contexto**: la prueba exige un sistema autocontenido; no debe existir ningún
camino hacia un proveedor remoto.

**Decisión**: dos servidores vLLM en Docker Compose (LLM + embeddings),
compartiendo la GPU:

| Servicio | Modelo | Reserva GPU | Puerto |
|---|---|---|---|
| `llm` | `Qwen/Qwen3-4B-AWQ` (≈2.6 GB, AWQ 4-bit) | 0.62 | 127.0.0.1:8000 |
| `embeddings` | `intfloat/multilingual-e5-small` (≈0.5 GB) | 0.22 | 127.0.0.1:8001 |

Combinación con `--max-model-len 4096`, `--max-num-seqs 2` → ≈6.7 GB de los
8 GB disponibles (holgura ≈1.3 GB).

**Consecuencias**:

- Guardia en arranque: `Settings.assert_local_inference()` aborta si
  `LLM_BASE_URL`/`EMBEDDINGS_BASE_URL` no resuelve a `LOCAL_INFERENCE_HOSTS`.
- Cero *fallback*: si el servidor local falla, el error se eleva y aparece en
  `warnings`; no hay cadena hacia OpenAI/Anthropic.
- Los prefijos `passage:`/`query:` de e5 los aplica nuestro adaptador
  (`services/embeddings.py`), no el servidor.

**Alternativa descartada**: API remota (OpenAI/Gemini) — incumple la premisa
de aislamiento y añade credenciales.

---

## ADR-002 · Un solo agente Deep Agents (agentic workflow) sin subagentes

**Contexto**: Deep Agents permite subagentes, herramientas de sistema de
archivos y ejecución de código.

**Decisión**: registrar un `HarnessProfile` propio que fija el prompt de
sistema y excluye `ls, read_file, write_file, edit_file, delete, glob, grep,
execute, task`; además
`GeneralPurposeSubagentProfile(enabled=False)`.

**Consecuencias**:

- Preserva el contexto (4096 tokens) para las herramientas de negocio.
- Menos saltos de planificación: el modelo decide y ejecuta en el mismo hilo.
- El coste de orquestación (memoria, trazabilidad, compositor) vive **fuera**
  del bucle del modelo, en `orchestrator.py`.

**Alternativa descartada**: jerarquía de subagentes (ERP / producción /
documentos) — con un modelo de 4B y presupuesto de 4096 tokens el *overhead*
de handoff se come la ventana y degrada las respuestas.

---

## ADR-003 · Compositor determinista para la trazabilidad

**Contexto**: el enunciado califica la explicabilidad de "muy importante".

**Decisión**: el modelo **solo** redacta `answer`. Los campos `sources`,
`reasoning`, `data`, `citations`, `warnings` y `execution_trace` se derivan de
las `ToolObservation` y evidencias registradas por las herramientas en un
`EvidenceCollector` *thread-safe*. Si el borrador del LLM llega vacío, un
*fallback* construye la respuesta desde `data`.

**Consecuencias**:

- La trazabilidad es reproducible y testeable sin LLM (tests unitarios).
- `sources` solo contiene fuentes consultadas **con éxito**; los fallos
  terminan en `warnings` (nunca se afirma haber leído algo que falló).
- El tono/estilo de `answer` depende del modelo; los **hechos** no.

**Alternativa descartada**: pedir al LLM que enumere sus fuentes/pasos —
alucinable y no verificable.

---

## ADR-004 · ERP de SQLite en solo lectura *real*

**Decisión**: `make_readonly_engine` combina la URI `file:...?mode=ro&uri=true`
con `PRAGMA query_only = ON`. El seed usa un motor de escritura separado,
exclusivamente en `scripts/`.

**Consecuencias**: ninguna consulta del agente puede modificar el ERP, y el
comportamiento está verificado con un test que espera `OperationalError`.

**Alternativa descartada**: depende de la buena voluntad de las tools (un error
de prompt lo rompería).

---

## ADR-005 · Importes con `Decimal`, nunca `float`

**Decisión**: `line_total` y `order_totals` operan con
`decimal.Decimal(...).quantize(0.01, ROUND_HALF_UP)`.

**Consecuencias**: resultado exacto y verificable (pedido 10643 → 266.00),
comparaciones directas en tests.

---

## ADR-006 · Clasificación de dominio por prioridad exclusiva

**Decisión**: `completado > bloqueado > retrasado > pendiente`, con causa
legible asociada (`"producción: falta de pieza"`, `"fecha requerida vencida
el ..."`).

**Consecuencias**: un pedido enviado **nunca** es retrasado; un pedido
pendiente **nunca** es retrasado solo por estarlo; la API de producción puede
enriquecer la causa sin reescribir la regla.

---

## ADR-007 · La UI no habla con el modelo

**Decisión**: Chainlit solo consume la API (`/api/query`, SSE, historial y
salud). La memoria conversacional vive en la API (`data/app.db`).

**Consecuencias**: una sola implementación de la lógica y de la trazabilidad;
la UI se puede sustituir (otro cliente HTTP) sin tocar el agente.
