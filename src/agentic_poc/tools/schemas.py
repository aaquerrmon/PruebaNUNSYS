"""Esquemas de argumentos de las herramientas (contrato visible al modelo)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FindCustomersArgs(BaseModel):
    term: str | None = Field(
        default=None,
        description="Texto a buscar en el ID, la razon social o el contacto. Vacio lista todos.",
    )


class CustomerOrdersArgs(BaseModel):
    customer_id: str = Field(description="Identificador del cliente en el ERP, p. ej. 'ALFKI'.")
    only_pending: bool = Field(
        default=True,
        description="Si es true, solo pedidos sin fecha de envio (pendientes).",
    )


class EmptyArgs(BaseModel):
    """Sin argumentos."""


class OrderAmountsArgs(BaseModel):
    order_ids: list[int] = Field(
        min_length=1,
        description="IDs de pedido del ERP cuyos importes se quieren calcular.",
    )


class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=2, description="Pregunta o palabras clave sobre los documentos.")
    k: int = Field(default=5, ge=1, le=10, description="Numero maximo de fragmentos a recuperar.")
