# Contrato de la API

Base (local): `http://127.0.0.1:8080` · En Compose: `http://127.0.0.1:8080`.

Todos los cuerpos son JSON salvo `POST /api/documents` (multipart).

---

## `POST /api/query`

**Request**

```jsonc
{
  "question": "¿Qué pedidos pendientes tiene ALFKI?",   // 3..2000 caracteres
  "conversation_id": "demo-1"                            // opcional: activa memoria
}
```

**Response `200`**

| Campo | Tipo | Descripción |
|---|---|---|
| `answer` | `string` | Respuesta redactada (único campo generado por el LLM) |
| `sources` | `string[]` | Fuentes consultadas **con éxito**: `ERP`, `Producción`, `Documentos` |
| `reasoning` | `string[]` | Pasos simplificados derivados de las herramientas ejecutadas |
| `data` | `object` | Datos estructurados (`orders`, `amounts`, `summaries`, `documents`) |
| `citations` | `Citation[]` | `doc_id`, `source`, `page`, `snippet`, `score` |
| `warnings` | `string[]` | Fuentes fallidas u observaciones no críticas |
| `execution_trace` | `ExecutionStep[]` | `tool`, `source`, `args_summary`, `record_count`, `duration_ms`, `ok`, `error` |
| `trace_id` | `string` | Identificador de la traza |
| `duration_ms` | `number` | Duración total |
| `model` | `string` | Modelo local usado |
| `used_memory` | `boolean` | `true` si se inyectó historial previo |

`Citation`: `{doc_id, source, page|null, snippet, score}`.
`ExecutionStep`: `{tool, source|null, args_summary, record_count|null, duration_ms, ok, error|null}`.

**Errores**

- `422` — pregunta ausente/<=2 caracteres o `conversation_id` inválido.
- `500` — fallo interno (el cuerpo incluye el detalle).

---

## `POST /api/query/stream` (SSE)

Mismo *request* que `/api/query`. Emite `data:` con eventos:

```jsonc
{"event":"start","data":{"question":"..."}}
{"event":"answer","data":{ ... QueryResponse completo ... }}
{"event":"done","data":{"trace_id":"..."}}
{"event":"error","data":{"error":"..."}}
```

---

## `GET /api/conversations/{conversation_id}/history`

```jsonc
{
  "conversation_id": "demo-1",
  "messages": [
    {"role": "human", "content": "...", "ts": 0.0},
    {"role": "assistant", "content": "...", "ts": 0.0}
  ]
}
```

Historial vacío → `messages: []`. El número de turnos reenviados en cada
petición lo limita `MEMORY_MAX_TURNS` (por defecto 5).

---

## `POST /api/documents`

`multipart/form-data` con `file` (PDF, hasta `MAX_PDF_BYTES`).

- `415` — no es PDF.
- `413` — supera el tamaño máximo.
- `422` — PDF ilegible / sin texto extraíble.

`201`:

```json
{"doc_id": "mi-contrato", "source": "mi_contrato.pdf", "pages": 4, "chunks": 9, "skipped_pages": 0}
```

La re-ingesta es idempotente: se reemplazan los fragmentos de ese `doc_id`.

## `GET /api/documents`

```json
[{"doc_id": "contrato-marco-suministro", "source": "contrato_marco_suministro.pdf", "chunks": 12, "pages": 5}]
```

---

## `GET /health`

```json
{
  "status": "ok",
  "llm": "ok",
  "embeddings": "ok",
  "production_api": "ok",
  "erp": "ok",
  "documents": 3
}
```

`status` es `degraded` si falla `llm`, `erp` o `production_api`.

---

## API de producción simulada (servicio aparte, `:8090`)

| Método | Ruta | Parámetros |
|---|---|---|
| `GET` | `/health` | — |
| `GET` | `/api/v1/production-orders` | `erp_order_ids` (CSV), `status`, `limit` |
| `GET` | `/api/v1/production-orders/{id}` | — (404 si no existe) |
| `GET` | `/api/v1/summary` | — (`total`, `by_status`) |

`status` admitido: `PENDING`, `IN_PROGRESS`, `BLOCKED`, `DELAYED`, `COMPLETED`
(valor distinto → `422`).

Ejemplo:

```bash
curl -s "http://127.0.0.1:8090/api/v1/production-orders?status=BLOCKED&limit=500"
curl -s "http://127.0.0.1:8090/api/v1/summary"
```
