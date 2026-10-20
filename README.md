# POC Agéntica — consultas de negocio multi-fuente (ERP + Producción + RAG)

## Video explicativo

<p align="center">
  <a href="https://www.youtube.com/watch?v=ctSBqmnsgwY" target="_blank">
    <img src="https://img.youtube.com/vi/ctSBqmnsgwY/hqdefault.jpg" alt="Ver vídeo en YouTube" width="600" />
  </a>
</p>




Aplicación de referencia que responde preguntas de negocio en lenguaje natural
combinando tres fuentes reales:

1. **ERP Northwind** (SQLite, solo lectura)
2. **API de producción simulada** (servicio REST independiente)
3. **Documentación PDF** (RAG sobre ChromaDB)

Orquestada con **Deep Agents (LangChain)**, expuesta por **FastAPI** e
interactuada desde **Chainlit**, con inferencia **100 % local** en **vLLM**
(`Qwen/Qwen3-4B-AWQ` + `intfloat/multilingual-e5-small`) dentro del límite de
**8 GB de VRAM**.

---

## 1. Arquitectura

```
  navegador -- :8585 --> Chainlit (src/agentic_poc/ui)
                            | HTTP (solo /api/*)
                            v
                       FastAPI (src/agentic_poc/api)
                       /api/query · /api/query/stream · /health
                       /api/conversations/{id}/history · /api/documents
                            |
                            v
              AgentOrchestrator (src/agentic_poc/agents)
              Deep Agents -- 8 tools (src/agentic_poc/tools) -- composer
                    |                 |                 |
                    v                 v                 v
             ErpRepository     ProductionClient      RagService
             SQLite (mode=ro + httpx -> API :8090    Chroma + e5
             PRAGMA query_only) (sistema externo)    (citas por pagina)
```

Capas separadas (responsabilidad única):

| Capa | Paquete | Rol |
|---|---|---|
| Presentación | `ui/` | Chainlit; solo presentador, no habla con el LLM |
| API | `api/` | Endpoints, validación, memoria conversacional |
| Orquestación | `agents/` | Deep Agents, prompts, compositor determinista |
| Lógica de negocio | `services/business.py` | Clasificación, importes, cruces multi-fuente |
| Herramientas | `tools/` | 8 tools tipadas (una por capacidad) |
| Acceso a datos | `repositories/`, `infrastructure/` | Repositorio ERP + SQLite de solo lectura |
| Integraciones | `services/{llm,embeddings,rag,production_client}.py` | Adaptadores reemplazables a vLLM/REST |
| Infraestructura local | `production_api/` | Sistema de producción simulado (servicio aparte) |

### Flujo de una petición

1. **UI -> API**: Chainlit envía `POST /api/query` con `conversation_id`.
2. **API**: recupera las últimas 5 interacciones de `app.db` (memoria opcional).
3. **Agente** (`orchestrator.answer`): un único agente Deep Agents con 8 tools
   tipadas; el modelo decide el plan y las invoca.
4. **Herramientas**: cada invocación registra una `ToolObservation` (fuente,
   nº de registros, duración, éxito/error) en un `EvidenceCollector`
   *thread-safe*.
5. **Compositor determinista**: `sources`, `reasoning`, `data`, `citations`,
   `warnings` y `execution_trace` se calculan **desde las evidencias**, no del
   texto del modelo. El LLM solo redacta el campo `answer`; si su borrador
   llega vacío, un *fallback* determinista construye la respuesta desde `data`.
6. **Respuesta**: JSON con `answer`, `sources`, `reasoning` + trazabilidad
   completa (`execution_trace`, `trace_id`, `citations`).

> **Clave de diseño**: la trazabilidad no depende de la fidelidad del modelo.
> El compositor garantiza que la respuesta sea verificable aunque el LLM
> alucine o responda de forma irregular.

---

## 2. Decisiones técnicas

### 2.1 Inferencia 100 % local (sin proveedores remotos)

- **`Qwen/Qwen3-4B-AWQ`** (AWQ 4-bit, ≈2.6 GB) servido por vLLM en `:8000` con
  `--tool-call-parser hermes`, `--enable-auto-tool-choice` y
  `--reasoning-parser qwen3`.
- **`intfloat/multilingual-e5-small`** (384 d, ≈0.5 GB) en `:8001`. Los prefijos
  `passage:`/`query:` los aplica **nuestro** cliente (`services/embeddings.py`),
  porque vLLM no los conoce.
- **Presupuesto de VRAM (8 GB)**: `llm` reserva 0.62 y `embeddings` 0.22
  (0.84 de 8 GB ≈ 6.7 GB), con `--max-model-len 4096` y `--max-num-seqs 2`.
- **Guardia anti-remoto**: `Settings.assert_local_inference()` se ejecuta al
  arrancar la API y la UI; si `LLM_BASE_URL`/`EMBEDDINGS_BASE_URL` apunta a un
  host que no está en `LOCAL_INFERENCE_HOSTS`, la aplicación **no arranca**
  (salvo `ALLOW_REMOTE_INFERENCE=true`, explícito y documentado).
- No existe ningún *fallback* hacia OpenAI/Anthropic: si el servidor local no
  responde, el error se propaga y aparece en `warnings`.

### 2.2 Un agente, no un multi-agente jerárquico

Deep Agents es el framework principal, pero el perfil de ejecución es
deliberadamente **unitario** (agentic workflow):

- `HarnessProfile` registrado con `base_system_prompt` propio y
  `excluded_tools` que retira `ls/read_file/write_file/edit_file/delete/glob/
  grep/execute/task`.
- `GeneralPurposeSubagentProfile(enabled=False)`: se desactiva el subagente
  general, porque el tool `task` multiplique el consumo de contexto.

**Razón**: con un modelo de 4B y `max_model_len=4096`, un FS virtual y
subagentes consumirían el presupuesto de tokens sin aportar a estas
consultas. El orquestador central (`orchestrator.py`) añade encima la memoria
por petición, la trazabilidad y el compositor determinista.

### 2.3 Trazabilidad determinista (requisito "muy importante" del enunciado)

| Campo | Origen |
|---|---|
| `sources` | solo fuentes **consultadas con éxito** (fallo -> `warnings`) |
| `reasoning` | construido desde `ToolObservation` ejecutadas |
| `data` | evidencias tipadas (`order_insights`, `amounts`, `document_list`...) |
| `citations` | chunks del RAG (doc, página, snippet, score), deduplicados |
| `execution_trace` | tool, fuente, nº de registros, duración, `ok`/`error` |
| `answer` | **único** campo redactado por el LLM (+ fallback determinista) |

### 2.4 Clasificación de dominio (mutuamente excluyente)

Prioridad: `completado` (shipped) > `bloqueado` (producción `BLOCKED`) >
`retrasado` (producción `DELAYED`/`delay_flag`, o `required_date` vencida) >
`pendiente`. Un pedido **nunca** se considera retrasado solo por estar
pendiente. La causa legible (`"producción: falta de pieza"`, `"fecha
requerida vencida el ..."`) viaja en la respuesta.

### 2.5 Sin coma flotante en dinero

Importes con `Decimal` y `ROUND_HALF_UP` a 2 decimales (`line_total`,
`order_totals`), verificados con tests (pedido 10643 -> total 266.00).

### 2.6 ERP de solo lectura real

`make_readonly_engine` combina la URI `mode=ro` de SQLite con
`PRAGMA query_only = ON`: cualquier escritura falla con
`OperationalError`. El seed usa un motor de escritura **solo en scripts**.

### 2.7 Memoria conversacional (opcional, implementada)

`POST /api/query` acepta `conversation_id`; se guardan los turnos en
`data/app.db` y se reenvían las últimas interacciones (configurable con
`MEMORY_MAX_TURNS=5`). Soporta preguntas encadenadas
("¿Y cuáles de esos están bloqueados?").

---

## 3. Dependencias

Gestionadas con **uv** (`uv.lock`). Python **3.12** (fijado en
`.python-version`; chainlit 2.12 exige `<3.14`).

| Paquete | Versión | Uso |
|---|---|---|
| `deepagents` | 0.7.x | Framework agéntico principal |
| `langchain` / `langchain-core` | 1.4.x / 1.6.x | Cadena, mensajes, tools |
| `langchain-openai` | 1.7.x | Cliente ChatOpenAI contra vLLM local |
| `langchain-chroma` / `chromadb` | 1.1.x / 1.5.x | Vector store del RAG |
| `fastapi` / `uvicorn` | 0.143.x / 0.54.x | API HTTP + SSE |
| `chainlit` | 2.12.x | Interfaz gráfica |
| `sqlalchemy` | 2.1.x | Acceso a SQLite (ERP y app) |
| `httpx` | 0.28.x | Cliente de la API de producción |
| `pypdf` / `python-multipart` | 6.x / 0.0.20 | Lectura de PDF y subida |
| `pytest` / `ruff` / `mypy` (dev) | 9.x / 0.17 / 2.x | Tests y gates de calidad |

Externos (Docker): `vllm/vllm-openai:v0.31.0`.

---

## 4. Configuración y arranque

### 4.1 Requisitos

- GPU NVIDIA con **≥ 8 GB** de VRAM y drivers recientes.
- Docker + Docker Compose (con soporte NVIDIA, `nvidia-container-toolkit`).
- `uv` para el desarrollo local.
- Copia de `.env.example` a `.env` (ambos solo afectan a local/Compose).

### 4.2 Arranque completo (recomendado)

```bash
cp .env.example .env
docker compose up -d --build
```

Servicios (todos publicados **solo** en `127.0.0.1`):

| Servicio | Puerto | Descripción |
|---|---|---|
| `chainlit` | 8585 | UI de chat |
| `api` | 8080 | API FastAPI (agente) |
| `production-api` | 8090 | Sistema de producción simulado |
| `llm` | 8000 | vLLM con Qwen3-4B-AWQ |
| `embeddings` | 8001 | vLLM con e5-small |
| `bootstrap` | — | Seed del ERP + ingesta de PDFs (una vez) |

La primera ejecución descarga los modelos a un volumen (`model-loader`) y
puede tardar varios minutos; el health de `llm`/`embeddings` tiene
`start_period: 240s`.

Estado y logs:

```bash
docker compose ps
docker compose logs -f api chainlit
docker compose down            # parar
docker compose down -v         # parar y borrar volúmenes (reset total)
```

### 4.3 Arranque local (desarrollo, sin la UI en Docker)

```bash
uv sync                                  # crea .venv e instala dependencias
uv run python scripts/bootstrap.py       # seed ERP + PDFs + ingest RAG
uv run uvicorn agentic_poc.production_api.app:app --port 8090
uv run uvicorn agentic_poc.api.app:app --port 8080
uv run chainlit run src/agentic_poc/ui/app.py --port 8585
```

En local los modelos deben correr en otra terminal (fuera de este repo):

```bash
vllm serve Qwen/Qwen3-4B-AWQ --served-model-name local-llm \
  --gpu-memory-utilization 0.62 --max-model-len 4096 --max-num-seqs 2 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3

vllm serve intfloat/multilingual-e5-small --served-model-name local-embeddings \
  --port 8001 --gpu-memory-utilization 0.22 --max-model-len 512
```

> Nota: `LLM_BASE_URL`/`EMBEDDINGS_BASE_URL` por defecto apuntan a
> `127.0.0.1`; en Compose los sobreescribe el `docker-compose.yml`
> (`http://llm:8000/v1`, `http://embeddings:8001/v1`).

---

## 5. Ejemplos de uso

### 5.1 Desde la UI

Abrir `http://127.0.0.1:8585` y probar los *starters* incluidos:

- "Resume el estado del pedido 10643"
- "¿Qué pedidos de ALFKI están pendientes?"
- "¿Cuánto factura la orden 10692?"
- "¿Qué dice el contrato sobre los plazos de entrega?"
- "¿Cuál es el motivo de retraso de los pedidos retenidos en producción?"

La respuesta muestra el mensaje del asistente, más paneles desplegables de
**Razonamiento** y **Traza de ejecución**, y el pie con fuentes, advertencias,
`trace_id` y duración.

### 5.2 Desde la API

```bash
curl -s http://127.0.0.1:8080/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "¿Qué pedidos pendientes tiene ALFKI?"}' | jq .
```

Respuesta (extracto):

```json
{
  "answer": "ALFKI tiene 3 pedidos pendientes: 10643, 10692 y 10702.",
  "sources": ["ERP", "Producción"],
  "reasoning": [
    "find_customers: 1 registro (fuente ERP)",
    "get_customer_orders: 3 registros (fuente ERP)",
    "fetch_records: 3 registros (fuente Producción)"
  ],
  "data": {"orders": [ ... ]},
  "citations": [],
  "warnings": [],
  "execution_trace": [
    {"tool": "find_customers", "source": "ERP", "record_count": 1, "duration_ms": 4.1, "ok": true}
  ],
  "trace_id": "...",
  "duration_ms": 1830.0,
  "model": "local-llm",
  "used_memory": false
}
```

Con memoria (preguntas encadenadas):

```bash
curl -s http://127.0.0.1:8080/api/query -H 'Content-Type: application/json' \
  -d '{"question": "¿Qué pedidos tiene ALFKI?", "conversation_id": "demo-1"}'
curl -s http://127.0.0.1:8080/api/query -H 'Content-Type: application/json' \
  -d '{"question": "¿Y cuáles de esos están bloqueados?", "conversation_id": "demo-1"}'
curl -s http://127.0.0.1:8080/api/conversations/demo-1/history | jq .
```

Streaming (SSE):

```bash
curl -N http://127.0.0.1:8080/api/query/stream -H 'Content-Type: application/json' \
  -d '{"question": "Resume el pedido 10643"}'
# data: {"event":"start",...}
# data: {"event":"answer",...}
# data: {"event":"done",...}
```

Subida de documentos (PDF):

```bash
curl -s http://127.0.0.1:8080/api/documents -F "file=@mi_contrato.pdf"
curl -s http://127.0.0.1:8080/api/documents
```

Salud del sistema:

```bash
curl -s http://127.0.0.1:8080/health | jq .
# {"status":"ok","llm":"ok","embeddings":"ok","production_api":"ok","erp":"ok","documents":3}
```

### 5.3 Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/query` | Pregunta + respuesta estructurada con trazabilidad |
| `POST` | `/api/query/stream` | Mismos eventos en SSE (`start`, `answer`, `done`, `error`) |
| `GET` | `/api/conversations/{id}/history` | Historial de la conversación |
| `POST` | `/api/documents` | Subida de PDF + ingesta al vector store |
| `GET` | `/api/documents` | Listado de documentos indexados |
| `GET` | `/health` | Estado de llm/embeddings/producción/ERP/documentos |

---

## 6. Tests y calidad

```bash
uv run pytest -q                 # 52 tests (unit + integración con fakes), 6 saltos
uv run pytest -m local_llm       # e2e contra vLLM local (si está levantado)
uv run pytest -m gpu             # presupuesto de VRAM y modelos servidos
uv run ruff check src/ scripts/ tests/
uv run ruff format --check src/ scripts/ tests/
uv run mypy src/
```

Los tests que necesitan inferencia real se marcan `local_llm`/`gpu` y se
**saltan automáticamente** si los servidores no responden: la suite normal no
depende de red ni de credenciales (usan fakes/transportes ASGI).

Cobertura de lo verificado:

- Dominio: prioridad de clasificación, causas de retraso, `Decimal`.
- ERP: seed idempotente, consultas, totales, **solo lectura real**.
- Producción: filtros, `summary`, 404, 422 (transporte ASGI, sin puerto).
- RAG: limpieza de boilerplate, IDs deterministas, re-ingesta idempotente.
- Compositor: `sources` solo con éxito, fallback determinista, hilos.
- API: contrato completo, memoria multi-turno, SSE, validaciones, `/health`.
- Configuración: rechazo de URLs remotas (guardia anti-proveedor externo).

---

## 7. Estructura del repositorio

```
.
├── docker-compose.yml          # stack completo (vLLM + app)
├── Dockerfile                  # imagen única de la aplicación
├── .dockerignore
├── pyproject.toml              # deps, ruff, mypy, pytest
├── uv.lock
├── scripts/
│   ├── bootstrap.py            # seed ERP + PDFs + ingest RAG (bootstrap)
│   ├── download_models.py      # precarga de modelos al volumen
│   └── generate_sample_docs.py # genera los 3 PDFs de demostración
├── src/agentic_poc/
│   ├── api/                    # FastAPI: endpoints, memoria, dependencias
│   ├── agents/                 # orchestrator, prompts, compositor, factory
│   ├── infrastructure/         # SQLite (ro/rw), seed de Northwind, logging
│   ├── models/                 # dominio (clasificación, importes) + respuestas
│   ├── production_api/         # API de producción simulada
│   ├── repositories/           # ErpRepository
│   ├── services/               # business, llm, embeddings, rag, production_client
│   ├── tools/                  # 8 herramientas tipadas + contexto/evidencias
│   └── ui/                     # Chainlit
└── tests/                      # pytest (fakes; local_llm/gpu opcionales)
```

---

## 8. Decisiones de despliegue

- **Puertos solo en 127.0.0.1**: ninguna interfaz se expone a la red.
- **Volumen `model-cache`**: descarga de modelos compartida y persistente.
- **`bootstrap` como servicio**: seed e ingesta idempotentes; `api`, `chainlit`
  y `production-api` dependen de su finalización correcta.
- **Healthchecks**: vLLM con `start_period` largo (carga de modelo), la app
  con corto; Compose orquesta el orden sin carreras.

## 9. Limitaciones conocidas

- Modelo de 4B: respuestas útiles pero no equivalentes a un LLM grande; por
  eso la veracidad de los datos descansa en el compositor determinista, no en
  el texto generado.
- `max_model_len=4096` limita el tamaño de contexto; se excluyeron FS virtual,
  `execute` y subagentes para preservarlo.
- El ERP es un *fixture* de Northwind (10 clientes, 20 pedidos), no una
  extracción real.
- La API de producción es simulada en memoria (fixtures con estados
  `BLOCKED`/`DELAYED` para ejercitar la clasificación).
- Demo (vídeo) no incluida: queda como paso manual.
