"""Genera PDFs de demostracion en `data/documents` (fpdf2).

Son documentos ficticios de una empresa de electrodomesticos ("Prisma") que
sirven para probar el RAG: plazos de entrega, penalizaciones, motivos de
bloqueo de produccion y politica de calidad.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUTPUT_DIR = Path("data/documents")


class _Doc(FPDF):
    def __init__(self, title: str) -> None:
        super().__init__(format="A4")
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=18)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"PRISMA S.A. - {self.doc_title} - Documento de demostracion", align="R")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")

    def _write(self, height: float, text: str) -> None:
        self.set_x(self.l_margin)
        self.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h1(self, text: str) -> None:
        self.set_font("Helvetica", "B", 15)
        self._write(8, text)
        self.ln(1)

    def h2(self, text: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self._write(6, text)
        self.ln(0.5)

    def paragraph(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self._write(5.4, text)
        self.ln(1.5)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self._write(5.4, f"  - {text}")


def build_contract(path: Path) -> None:
    doc = _Doc("Contrato marco de suministro")
    doc.add_page()
    doc.h1("Contrato marco de suministro y logistica")
    doc.paragraph(
        "Este contrato regula el suministro de componentes y productos terminados entre "
        "Prisma S.A. y sus clientes distribuidos. Las condiciones economicas y los plazos "
        "aqui descritos son de aplicacion general salvo pacto especifico por escrito."
    )
    doc.h2("1. Plazos de entrega")
    doc.paragraph(
        "El plazo de entrega estandar es de 15 dias laborables desde la confirmacion del "
        "pedido en firme. Para pedidos urgentes marcados como prioritarios, el plazo se reduce "
        "a 7 dias laborables, siempre que exista stock de componentes. Si el pedido depende de "
        "fabricacion a medida, el plazo se amplia a 30 dias laborables."
    )
    doc.bullet("Entrega estandar: 15 dias laborables.")
    doc.bullet("Entrega urgente: 7 dias laborables.")
    doc.bullet("Fabricacion a medida: 30 dias laborables.")
    doc.h2("2. Penalizaciones por retraso")
    doc.paragraph(
        "Si el retraso supera los 5 dias laborables respecto a la fecha requerida, se aplicara "
        "una penalizacion del 2 por ciento del importe del pedido por cada semana de retraso, "
        "con un maximo acumulado del 10 por ciento. Los retrasos motivados por causas de fuerza "
        "mayor no generan penalizacion, pero deben comunicarse por escrito en un plazo de 48 horas."
    )
    doc.h2("3. Calidad y devoluciones")
    doc.paragraph(
        "Todo lote rechazado en inspeccion de calidad se considera no conforme y debe ser "
        "reemplazado en un maximo de 10 dias laborables sin coste adicional para el cliente. "
        "El cliente dispone de 30 dias naturales desde la recepcion para notificar defectos "
        "ocultos que no fueran detectables en la inspeccion inicial."
    )
    doc.add_page()
    doc.h2("4. Condiciones economicas")
    doc.paragraph(
        "Los precios unitarios se revisan anualmente segun el indice general de precios. El pago "
        "se realiza a 60 dias fecha factura mediante transferencia bancaria. Los importes no "
        "incluyen impuestos, que se anadiran conforme a la legislacion vigente."
    )
    doc.h2("5. Responsabilidad")
    doc.paragraph(
        "La responsabilidad maxima agregada derivada de este contrato queda limitada al importe "
        "facturado en los doce meses anteriores al evento que genere la reclamacion. Quedan "
        "excluidos los danos indirectos o lucro cesante."
    )
    doc.add_page()
    doc.h2("6. Vigencia y resolucion")
    doc.paragraph(
        "El contrato tiene una vigencia inicial de 24 meses prorrogable de forma automatica por "
        "periodos anuales, salvo denuncia de cualquiera de las partes con 60 dias de antelacion. "
        "El incumplimiento grave de las obligaciones de pago faculta a la resolucion inmediata."
    )
    doc.output(str(path))


def build_production_procedure(path: Path) -> None:
    doc = _Doc("Procedimiento de produccion")
    doc.add_page()
    doc.h1("Procedimiento interno de produccion y escalado")
    doc.paragraph(
        "Este documento describe el flujo de una orden de produccion en Prisma S.A., los motivos "
        "de bloqueo habituales y el procedimiento de escalado ante retrasos."
    )
    doc.h2("1. Estados de una orden de produccion")
    doc.bullet("PENDING: orden planificada, no iniciada.")
    doc.bullet("IN_PROGRESS: orden en fabricacion.")
    doc.bullet(
        "BLOCKED: orden detenida por falta de componentes, averia de maquina o retencion de calidad."
    )
    doc.bullet("DELAYED: orden con riesgo o hecho consumado de retraso respecto al plan.")
    doc.bullet("COMPLETED: orden terminada y entregada a expediciones.")
    doc.h2("2. Motivos de bloqueo")
    doc.paragraph(
        "El bloqueo mas frecuente es la falta de componentes por incumplimiento del proveedor "
        "principal de PCBs. Le siguen las averias de maquina sin plan de contingencia y las "
        "retenciones de calidad cuando una muestra no supera los ensayos de fiabilidad."
    )
    doc.h2("3. Escalado de retrasos")
    doc.paragraph(
        "Todo retraso superior a 3 dias laborables debe escalarse al responsable de produccion, "
        "que evaluara alternativas: reasignar linea, priorizar el pedido o reprogramar la fecha "
        "requerida con el cliente. Las peticiones de cambio de fecha deben registrarse en el ERP."
    )
    doc.output(str(path))


def build_service_levels(path: Path) -> None:
    doc = _Doc("Acuerdo de niveles de servicio")
    doc.add_page()
    doc.h1("Acuerdo de niveles de servicio (SLA)")
    doc.paragraph(
        "El presente acuerdo define los niveles de servicio comprometidos por Prisma S.A. con "
        "sus clientes estrategicos, asi como las metricas de seguimiento trimestral."
    )
    doc.h2("1. Indicadores comprometidos")
    doc.bullet("Cumplimiento de plazo de entrega: minimo 95 por ciento de los pedidos.")
    doc.bullet("Disponibilidad de la API de consulta: minimo 99,5 por ciento mensual.")
    doc.bullet("Tiempo de respuesta de incidencias criticas: menos de 4 horas laborables.")
    doc.h2("2. Retrasos y compensaciones")
    doc.paragraph(
        "Cuando el cumplimiento de plazo baje del 90 por ciento en un trimestre, el cliente "
        "recibira una compensacion equivalente al 3 por ciento de la facturacion del trimestre. "
        "La compensacion por incidencias criticas no resueltas en plazo es de 500 euros por evento."
    )
    doc.h2("3. Revision")
    doc.paragraph(
        "Los niveles de servicio se revisan cada trimestre en comite conjunto. Cualquier "
        "modificacion requiere acuerdo escrito de ambas partes y se aplicara al trimestre siguiente."
    )
    doc.output(str(path))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "contrato_marco_suministro.pdf": build_contract,
        "procedimiento_produccion.pdf": build_production_procedure,
        "acuerdo_niveles_servicio.pdf": build_service_levels,
    }
    for name, builder in builders.items():
        target = OUTPUT_DIR / name
        builder(target)
        print(f"Generado {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
