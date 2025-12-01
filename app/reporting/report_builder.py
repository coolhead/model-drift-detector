from fpdf import FPDF
import tempfile
from pathlib import Path
import pandas as pd

class DriftReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Model Drift Report", ln=True, align="C")
        self.ln(4)

def build_pdf_report(
    drift_df: pd.DataFrame,
    perf_summary: dict,
    out_path: str,
):
    pdf = DriftReport()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "High-level summary:", ln=True)
    for k, v in perf_summary.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)

    pdf.ln(4)
    pdf.cell(0, 8, "Top drifted features (by PSI):", ln=True)

    for _, row in drift_df.sort_values("psi", ascending=False).head(10).iterrows():
        pdf.cell(
            0,
            6,
            f"- {row['feature']}: PSI={row['psi']:.3f}, KS={row['ks']:.3f}, severity={row['severity']}",
            ln=True,
        )

    pdf.output(out_path)


def build_report_tempfile(drift_df: pd.DataFrame, perf_summary: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    build_pdf_report(drift_df, perf_summary, tmp.name)
    return Path(tmp.name)
