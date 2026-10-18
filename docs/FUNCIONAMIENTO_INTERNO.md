# Funcionamiento Interno de la Aplicación

Documento técnico completo que describe la arquitectura, el flujo de ejecución,
la memoria conversacional, el vector store (RAG) con filtrado por documento, la
gestión de documentos en el frontend y una simulación paso a paso de una llamada
real.

---

## 1. Arquitectura General

La aplicación es un asistente agéntico de consultas de negocio que combina tres
fuentes de datos:

| Fuente | Tecnología | Rol |
|--------|-----------|-----|
| **ERP Northwind** | SQLite (solo lectura real) | Clientes, pedidos, importes |
| **API de Producción** | FastAPI simulada (puerto 8090) | Estado de fabricación (BLOCKED, DELAYED, etc.) |
| **Documentos PDF** | ChromaDB + embeddings e5 | RAG sobre contratos, procedimientos, SLA |

**Orquestación**: Deep Agents (LangChain) con un único agente que tiene 8
herramientas tipadas.

**Capas**:

```
Navegador (:8585) --> Chainlit (ui/)
                              | HTTP (/api/*)
                              v
                         FastAPI (api/)
                         /api/query · /api/query/stream
                         /api/conversations/{id}/history
                         /api/documents (POST + GET)
                              |
                              v
              AgentOrchestrator (agents/orchestrator.py)
              Deep Agents -- 8 tools (tools/domain_tools.py) -- composer
                    |                 |                 |
              ErpRepository   ProductionClient   RagService
              SQLite ro       API REST :8090       Chroma + e5
```

**Inferencia 100 % local**: vLLM con `Qwen/Qwen3-4B-AWQ` (puerto 8000) e
`intfloat/multilingual-e5-small` (puerto 8001). No hay fallback a proveedores
remotos (`Settings.assert_local_inference()` bloquea URLs no locales al
arrancar).

---

## 2. Memoria Conversacional (guardada en `app.db`)

### Archivos implicados

| Fichero | Rol |
|---------|-----|
| `src/agentic_poc/api/app.py` | Endpoints `/api/query`, `/api/query/stream`, `/api/conversations/{id}/history`; `lifespan` crea tablas al arrancar |
| `src/agentic_poc/api/memory.py` | `ConversationStore`: CRUD de turnos en SQLite |
| `src/agentic_poc/api/dependencies.py` | Singleton perezoso de `ConversationStore` |
| `src/agentic_poc/ui/app.py` | Envía `conversation_id` en cada petición al chatbot |

### Flujo de memoria

1. Al iniciar el chat (`on_chat_start`), Chainlit genera un `conversation_id`
   (UUID) y lo guarda en la sesión de usuario.
2. El UI envía `POST /api/query` con `conversation_id` en el cuerpo.
3. La API llama a `store.get_history(conversation_id, limit=10)` para recuperar
   los últimos 5 turnos (10 mensajes: 5 humanos + 5 asistentes).
4. El historial se prepende a los mensajes enviados al agente Deep Agents.
5. Tras obtener la respuesta, `store.save_turn(conversation_id, question, answer)`
   persiste ambos turnos en la tabla `conversation_turns` de `data/app.db`.
6. `GET /api/conversations/{id}/history` devuelve el historial completo.

El campo `used_memory` en la respuesta indica si se inyectó historial previo.

### Corrección aplicada (por qué antes no se guardaba)

Antes de esta corrección, el `lifespan` de FastAPI no inicializaba las tablas de
memoria, y `ConversationStore` no era singleton. Ahora:

- `_get_conversation_store()` es un singleton perezoso (variable de módulo).
- El `lifespan` llama a `_get_conversation_store()` al arrancar, ejecutando
  `metadata.create_all()` y creando la tabla `conversation_turns` en
  `data/app.db` antes de recibir cualquier petición.
- El timestamp del asistente usa `now + ".5"` (antes era `now + "1"`, lo que
  corrompía el orden lexicográfico).

Para verificar que los datos se guardan, se puede consultar directamente:

```sql
sqlite3 data/app.db "SELECT * FROM conversation_turns ORDER BY ts;"
```

---

## 3. Vector Store y RAG con Filtrado por Documento

### Archivos implicados

| Fichero | Rol |
|---------|-----|
| `src/agentic_poc/services/rag.py` | `RagService`: ingesta PDF, búsqueda vectorial con filtro `doc_ids` |
| `src/agentic_poc/tools/schemas.py` | `SearchDocumentsArgs`: acepta `doc_ids` opcional |
| `src/agentic_poc/tools/domain_tools.py` | Tool `search_documents`: pasa `doc_ids` al RAG |
| `src/agentic_poc/tools/context.py` | `ToolContext`: almacena `doc_ids` de la petición |
| `src/agentic_poc/agents/orchestrator.py` | `answer()` acepta `doc_ids` y los inyecta en el contexto |
| `src/agentic_poc/api/app.py` | Endpoint `/api/query` recibe `doc_ids` del request |
| `src/agentic_poc/ui/app.py` | Panel de documentos con selección; envía `doc_ids` en cada query |

### Cómo funciona el RAG

**Ingesta** (`services/rag.py:add_pdf`):
1. Lee el PDF con `pypdf.PdfReader`, extrayendo texto página por página.
2. Limpia boilerplate (encabezados/pies repetidos en la mayoría de páginas)
   y números de página.
3. Divide el texto en chunks con `RecursiveCharacterTextSplitter`
   (chunk_size=800, overlap=120).
4. Calcula embeddings con prefijo `passage:` (convención e5).
5. Almacena en Chroma con IDs deterministas: `{doc_id}:p{page}:c{chunk_index}`.
6. Metadatos por chunk: `doc_id`, `source` (nombre de archivo), `page`,
   `chunk_index`.
7. Re-ingesta idempotente: `delete_document(doc_id)` elimina chunks previos
   antes de insertar los nuevos.

**Búsqueda con filtro por documento** (`services/rag.py:search`):
1. Aplica prefijo `query:` a la pregunta del usuario.
2. Si `doc_ids` se proporciona, construye filtro Chroma:
   `where = {"doc_id": {"$in": doc_ids}}` y lo pasa como `filter` a
   `similarity_search_with_score`.
3. `score = 1.0 - distance` (similitud coseno normalizada).
4. Filtra por `score_threshold` (default 0.35).
5. Devuelve `RetrievedChunk` con `doc_id`, `source`, `page`, `chunk_index`,
   `text`, `score`.

**Compositor** (`agents/composer.py:_collect_citations`):
- Recopila citas de la evidencia `document_chunks`.
- Deduplica por `(doc_id, page, chunk_index)`, queda con el score más alto.
- Ordena por score descendente, máximo 8 citas.

### Corrección del vector store (demasiadas tablas)

Antes, `get_rag_service` no estaba en caché: cada petición creaba una nueva
`RagService`, que a su vez instanciaba un nuevo objeto `Chroma` sobre el mismo
`persist_directory` (`data/vector_store`). Esto generaba conflictos en la
SQLite subyacente de ChromaDB y colecciones duplicadas.

Ahora `_get_embeddings()` y `_get_rag_service()` son singletons perezosos
(variables de módulo `_EMBEDDINGS` y `_RAG_SERVICE`). Se crean una sola vez
por proceso, garantizando una única instancia de Chroma sobre el mismo
directorio de persistencia.

---

## 4. Gestión de Documentos en el Frontend (Chainlit)

### Diseño visual (aproximación 1:3)

Chainlit no soporta paneles laterales tipo split-pane nativo. La aproximación
implementada es:

- **Panel superior colapsable** (`cl.Step` tipo "tool"): "Documentos indexados
  (RAG)" con la lista de PDFs, su estado de selección y conteo de fragmentos.
- **Acciones de selección** (`cl.Action`): una por documento, para marcar/
  desmarcar cuáles se consultan en el RAG.
- **Área de chat principal**: mensajes del usuario y asistente, razonamiento,
  traza de ejecución, citas RAG y metadatos.

```
┌─────────────────────────────────────────────────────────────────┐
│  📄 Documentos indexados (RAG)   [paso colapsable]              │
│  ✅ contrato-marco-suministro (12 frags, 5 págs)               │
│  ⬜ procedimiento-produccion (8 frags, 3 págs)                  │
│  ⬜ acuerdo-niveles-servicio (6 frags, 2 págs)                  │
│                                                                  │
│  [✅ contrato-marco-suministro] [⬜ procedimiento-produccion]   │
│  [⬜ acuerdo-niveles-servicio]                                  │
├─────────────────────────────────────────────────────────────────┤
│  Chat principal                                                  │
│  Usuario: "¿Cuál es el plazo de entrega?"                       │
│  Asistente: ...                                                 │
│  [Razonamiento] [Traza] [Citas (RAG)]                           │
└─────────────────────────────────────────────────────────────────┘
```

### Flujo de uso

1. **Subir PDF**: Arrastra un PDF al chat o úsalo como adjunto.
   - `@cl.on_file_upload` lo lee y envía `POST /api/documents`.
   - Chroma lo indexa y devuelve `doc_id`, fragmentos y páginas.
   - Se refresca automáticamente el panel de documentos.

2. **Seleccionar documentos**: Haz clic en las acciones de selección.
   - Cada documento tiene una acción `⬜` (no seleccionado) o `✅` (seleccionado).
   - El estado se guarda en `cl.user_session["selected_doc_ids"]`.

3. **Preguntar al chatbot**: Escribe tu pregunta.
   - Si hay documentos seleccionados, se envían sus `doc_ids` en la petición.
   - El RAG busca solo en esos documentos.
   - Si ninguno está seleccionado, se busca en todos los indexados.

4. **Ver documentos**: Escribe `/documentos` para listar todos los indexados.

---

## 5. Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado de llm, embeddings, producción, ERP, documentos |
| `POST` | `/api/query` | Pregunta + respuesta estructurada con trazabilidad |
| `POST` | `/api/query/stream` | Eventos SSE (start, answer, done, error) |
| `GET` | `/api/conversations/{id}/history` | Historial de conversación |
| `POST` | `/api/documents` | Subida de PDF + ingesta al vector store |
| `GET` | `/api/documents` | Listado de documentos indexados |

### Esquema `POST /api/query`

```json
{
  "question": "¿Qué dice el contrato sobre los plazos de entrega?",
  "conversation_id": "demo-123",
  "doc_ids": ["contrato-marco-suministro"]
}
```

`doc_ids` es opcional. Si se omite o es null, el RAG busca en todos los
documentos indexados.

---

## 6. Simulación de una Llamada Completa

### Escenario

Usuario quiere saber el plazo de entrega, filtrando solo por el documento
"contrato marco de suministro".

### 6.1. Paso 1: Subir el PDF

El usuario arrastra `contrato_marco_suministro.pdf` al chat.

**UI → API**:
```
POST http://127.0.0.1:8080/api/documents
Content-Type: multipart/form-data
file: <bytes del PDF>
```

**API → RAG** (`services/rag.py:add_pdf`):
1. `PdfReader` extrae 5 páginas.
2. Limpieza de boilerplate.
3. `RecursiveCharacterTextSplitter` genera 12 chunks.
4. Embeddings con prefijo `passage:`.
5. Chroma almacena con IDs `contrato-marco-suministro:p1:c0`, etc.
6. Resultado: `{"doc_id": "contrato-marco-suministro", "chunks": 12, "pages": 5}`.

**API → UI**: Respuesta 201 con los datos de ingestión.

**UI**: Muestra "✅ contrato_marco_suministro.pdf indexado (12 fragmentos, 5 páginas)" y refresca el panel.

### 6.2. Paso 2: Seleccionar el documento

El usuario hace clic en la acción `⬜ contrato-marco-suministro`.

**UI**:
- Cambia el estado de `selected_doc_ids` de `[]` a `["contrato-marco-suministro"]`.
- La acción cambia a `✅ contrato-marco-suministro`.
- Se refresca el panel de documentos.

### 6.3. Paso 3: Pregunta al chatbot

El usuario escribe: "¿Cuál es el plazo de entrega estándar?"

**UI → API**:
```
POST http://127.0.0.1:8080/api/query/stream
Content-Type: application/json

{
  "question": "¿Cuál es el plazo de entrega estándar?",
  "conversation_id": "abc123",
  "doc_ids": ["contrato-marco-suministro"]
}
```

### 6.4. Trace de funciones y ficheros

| Paso | Función | Fichero | Datos de entrada | Datos de salida |
|------|---------|---------|------------------|-----------------|
| 1 | `query_stream()` | `api/app.py:167` | `QueryRequest(question=..., conversation_id="abc123", doc_ids=["contrato-marco-suministro"])` | SSE events |
| 2 | `store.get_history()` | `api/memory.py:38` | `conversation_id="abc123", limit=10` | `[]` o historial previo |
| 3 | `AgentOrchestrator.answer()` | `agents/orchestrator.py:91` | `question, history, doc_ids=["contrato-marco-suministro"]` | `QueryResponse` |
| 4 | `create_chat_model()` | `services/llm.py:23` | `Settings` | `ChatOpenAI` (vLLM :8000) |
| 5 | `build_agent()` | `agents/agent_factory.py:51` | `model, tools=[8 tools]` | `CompiledStateGraph` |
| 6 | `agent.ainvoke()` | `agents/orchestrator.py:118` | `{"messages": [HumanMessage(question)]}` | Resultado del agente |
| 7 | **Tool: `search_documents`** | `tools/domain_tools.py:267` | `query, k=5, doc_ids=["contrato-marco-suministro"]` | JSON con chunks |
| 7a | `context.rag.search_relevant()` | `services/rag.py:272` | `query, doc_ids=["contrato-marco-suministro"]` | `[RetrievedChunk(...)]` |
| 7b | Chroma `similarity_search_with_score` | `langchain_chroma` | `query, filter={"doc_id": {"$in": [...]}}` | Chunks solo de ese doc |
| 8 | `record_documents_step()` | `tools/recording.py:105` | `evidence_kind="document_chunks"` | Evidencia registrada |
| 9 | `compose_response()` | `agents/composer.py:173` | `draft, collector, doc_ids` | `QueryResponse` estructurado |
| 10 | `store.save_turn()` | `api/memory.py:56` | `conversation_id, question, answer` | Filas en SQLite `app.db` |
| 11 | SSE `answer` event | `api/app.py:182` | `QueryResponse` completo | JSON al cliente |

### 6.5. Detalle de la búsqueda RAG filtrada

**Llamada a Chroma** (`services/rag.py:251`):
```python
results = self._store.similarity_search_with_score(
    query="query:¿Cuál es el plazo de entrega estándar?",
    k=5,
    filter={"doc_id": {"$in": ["contrato-marco-suministro"]}}
)
```

**Resultado**: Solo chunks con `doc_id = "contrato-marco-suministro"`.
Chunks de otros documentos (procedimiento, SLA) quedan excluidos.

### 6.6. Respuesta JSON devuelta por la API

```json
{
  "answer": "Según el contrato marco de suministro, el plazo de entrega estándar es de 15 días laborables desde la confirmación del pedido en firme.",
  "sources": ["Documentos"],
  "reasoning": [
    "Busqueda en documentos: 2 registros (fuente Documentos)"
  ],
  "data": {
    "documents": [
      {
        "doc_id": "contrato-marco-suministro",
        "source": "contrato_marco_suministro.pdf",
        "page": 1,
        "chunk_index": 0,
        "score": 0.82,
        "snippet": "El plazo de entrega estándar es de 15 días laborables..."
      }
    ]
  },
  "citations": [
    {
      "doc_id": "contrato-marco-suministro",
      "source": "contrato_marco_suministro.pdf",
      "page": 1,
      "snippet": "El plazo de entrega estándar es de 15 días laborables desde la confirmación del pedido en firme.",
      "score": 0.82
    }
  ],
  "warnings": [],
  "execution_trace": [
    {
      "tool": "search_documents",
      "source": "Documentos",
      "args_summary": "query='...', k=5, doc_ids=['contrato-marco-suministro']",
      "record_count": 2,
      "duration_ms": 45.2,
      "ok": true,
      "error": null
    }
  ],
  "trace_id": "a3f7c2e1...",
  "duration_ms": 2140.0,
  "model": "local-llm",
  "used_memory": false,
  "doc_ids": ["contrato-marco-suministro"]
}
```

### 6.7. Segunda llamada con memoria (preguntas encadenadas)

```
POST /api/query
{
  "question": "¿Y cuál es la penalización por retraso?",
  "conversation_id": "abc123",
  "doc_ids": ["contrato-marco-suministro"]
}
```

1. `store.get_history("abc123", limit=10)` devuelve los mensajes previos.
2. `used_memory = true` en la respuesta.
3. El agente recibe el historial como contexto.

---

## 7. Stack Tecnológico

| Componente | Versión | Propósito |
|------------|---------|-----------|
| Python | 3.12 | Lenguaje |
| FastAPI | 0.143.x | API REST |
| Chainlit | 2.12.x | UI de chat |
| Deep Agents | 0.7.x | Framework agéntico |
| LangChain | 1.4.x | Orquestación de tools |
| SQLAlchemy | 2.1.x | Acceso a SQLite |
| ChromaDB | 1.5.x | Vector store |
| vLLM | 0.31.0 | Servidor de inferencia local |
| Qwen3-4B-AWQ | — | LLM local (4-bit) |
| multilingual-e5-small | — | Embeddings (384 d) |
| pypdf | 6.x | Extracción de texto PDF |
| httpx | 0.28.x | Cliente HTTP async |
| pytest | 9.x | Tests |
| ruff / mypy | — | Lint / tipos |

---

## 8. Archivos Modificados en esta Corrección

| Fichero | Cambio |
|---------|--------|
| `src/agentic_poc/api/schemas.py` | `QueryRequest.doc_ids` y `QueryResponse.doc_ids` |
| `src/agentic_poc/api/app.py` | `lifespan` crea tablas de memoria; pasa `doc_ids` al orquestador y a la respuesta |
| `src/agentic_poc/api/memory.py` | Timestamp del asistente corregido (`now + ".5"`) |
| `src/agentic_poc/api/dependencies.py` | Singleton perezoso para `RagService`, `ConversationStore`, `LocalE5Embeddings`; `clear_caches()` para tests |
| `src/agentic_poc/services/rag.py` | `search`/`search_relevant` aceptan `doc_ids` y filtran con Chroma `filter` |
| `src/agentic_poc/tools/schemas.py` | `SearchDocumentsArgs.doc_ids` |
| `src/agentic_poc/tools/context.py` | `ToolContext.doc_ids` |
| `src/agentic_poc/tools/domain_tools.py` | `search_documents` pasa `doc_ids` al RAG |
| `src/agentic_poc/agents/orchestrator.py` | `answer()` acepta `doc_ids` y los inyecta en `ToolContext` |
| `src/agentic_poc/agents/composer.py` | `compose_response()` acepta `doc_ids` y los incluye en `QueryResponse` |
| `src/agentic_poc/ui/app.py` | Panel de documentos colapsable, acciones de selección, `@cl.on_file_upload`, comando `/documentos`, envío de `doc_ids` en query, paso de "Documentos filtrados" |
| `tests/conftest.py` | `_clear_engine_caches()` llama a `deps.clear_caches()` |
| `tests/test_api.py` | `_StubOrchestrator.answer()` acepta `doc_ids` |
| `docs/FUNCIONAMIENTO_INTERNO.md` | Este documento |

---

## 9. Verificación de Calidad

| Comando | Resultado |
|---------|-----------|
| `uv run pytest -q` | 62 passed, 1 skipped (E2E vLLM requiere servidores locales) |
| `uv run ruff check src/ scripts/ tests/` | All checks passed |
| `uv run ruff format --check src/ scripts/ tests/` | All files formatted |
| `uv run mypy src/` | Success: no issues found |

El único test que no se ejecuta es `test_consulta_documentacion_genera_citas`, que requiere los servidores vLLM locales (`llm` y `embeddings`) levantados y accesibles. Es un test de integración de extremo a extremo, no un problema de código.
