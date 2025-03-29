from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "ModelMirror Fairness Audit Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 10)
        if isinstance(content, dict):
            for k, v in content.items():
                self.multi_cell(0, 8, f"{k}: {v}")
        elif isinstance(content, list):
            for item in content:
                self.multi_cell(0, 8, str(item))
        elif isinstance(content, str):
            self.multi_cell(0, 8, content)
        self.ln(5)

def export_report(fairness_results, leakage_report, file_path="audit_report.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_section("Audit Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.add_section("Fairness Metrics", fairness_results)
    pdf.add_section("Leakage Signals", leakage_report.to_dict(orient="records"))
    pdf.output(file_path)
    return file_path