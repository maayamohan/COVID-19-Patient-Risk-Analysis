"""Rebuild verified result summaries, the two-page PDF, and editable PPTX deck."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

import pandas as pd
import numpy as np
import pdfplumber
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
PRESENTATION = ROOT / "presentation"
FIGURES = RESULTS / "figures"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
HISTORICAL_FILES = [
    "raw_baseline_results.csv", "pca_results.csv", "lasso_results.csv",
    "smoteenn_results.csv",
]


def consolidate() -> pd.DataFrame:
    parts = []
    result_files = HISTORICAL_FILES + sorted(p.name for p in RESULTS.glob("*_grouped_results.csv"))
    for name in result_files:
        path = RESULTS / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = name
        frame["split_protocol"] = (
            "stratified_grouped_identical_predictor_rows"
            if "_grouped_results.csv" in name
            else "historical_stratified_random_row_split"
        )
        frame["experiment_id"] = frame.apply(
            lambda r: "|".join(str(r.get(k, "")) for k in
                               ["target", "feature_set", "engineering", "model", "lasso_C", "source_file"]),
            axis=1,
        )
        parts.append(frame)
    results = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    if results.empty:
        raise RuntimeError("No historical result files were found.")
    results.to_csv(RESULTS / "consolidated_results.csv", index=False)
    for key, filename in [
        ("target", "comparison_by_target.csv"),
        ("feature_set", "comparison_by_feature_set.csv"),
        ("model", "comparison_by_model.csv"),
        ("engineering", "comparison_by_engineering.csv"),
    ]:
        columns = [c for c in METRICS if c in results]
        grouped = results.groupby(["split_protocol", key], dropna=False)[columns].agg(["count", "mean", "std"])
        grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
        grouped.reset_index().to_csv(RESULTS / filename, index=False)
    data = pd.read_csv(ROOT / "dataset" / "Covid Data.csv")
    metric_missing = int(results[METRICS].isna().sum().sum())
    metric_nonfinite = int((~np.isfinite(results[METRICS].apply(pd.to_numeric, errors="coerce").to_numpy())).sum())
    audit = {
        "dataset_rows": int(data.shape[0]), "dataset_columns": int(data.shape[1]),
        "exact_duplicate_rows_beyond_first": int(data.duplicated().sum()),
        "consolidated_result_rows": int(len(results)),
        "historical_result_rows": int((results["split_protocol"] == "historical_stratified_random_row_split").sum()),
        "grouped_result_rows": int((results["split_protocol"] == "stratified_grouped_identical_predictor_rows").sum()),
        "historical_duplicate_experiment_ids": int(results["experiment_id"].duplicated().sum()),
        "missing_metric_cells": metric_missing,
        "non_finite_metric_cells": metric_nonfinite,
        "historical_split_caveat": "stratified random row split; duplicate records may cross partitions",
    }
    (RESULTS / "qa_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return results


def build_pdf(results: pd.DataFrame) -> int:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "project_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=3))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#47758A"), spaceAfter=9))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=13, textColor=colors.HexColor("#17324D"), spaceBefore=4, spaceAfter=3))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.1, leading=10.6, spaceAfter=4))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.3, leading=9))
    story = [
        Paragraph("COVID-19 Patient Risk Analysis", styles["Title2"]),
        Paragraph("A tabular adaptation of a CT-based triage research question | UE24CS352A", styles["Sub"]),
        Paragraph("Abstract", styles["H2x"]),
        Paragraph("This project explores whether demographic and recorded clinical variables can rank COVID-19 records by ICU admission, intubation, and recorded death. It adapts the outcomes studied by Zhan and Li using the public Mexican surveillance CSV included in the repository. It does not reproduce their CT/radiomics method. The source has 1,048,575 rows, 21 columns, and 812,049 exact duplicate rows, with no patient identifier. Historical random-row metrics may therefore be optimistic and do not establish patient-level generalization.", styles["Bodyx"]),
        Paragraph("Targets and features", styles["H2x"]),
        Paragraph("ICU and intubation use known codes 1 (positive) and 2 (negative); other codes are unknown and excluded. All known labels for these outcomes occur among hospitalized cases (PATIENT_TYPE=2). Death is 1 for a valid dd/mm/yyyy date and 0 for 9999-99-99; malformed values are excluded. Feature groups are Demo (age, sex), DemoComorb (plus ten comorbidity flags), and Clinical (plus pneumonia). Outcome fields, PATIENT_TYPE, pregnancy, classification, and service/unit variables are excluded.", styles["Bodyx"]),
        Paragraph("Data preparation and models", styles["H2x"]),
        Paragraph("Age sentinels 97/98/99, comorbidity sentinels 98/99, and pneumonia 99 become missing. Training-fitted median/mode imputation, age scaling, and categorical one-hot encoding stay inside the pipeline. New splits group identical predictor rows while stratifying outcomes; this prevents exact feature-vector copies crossing partitions but cannot establish patient identity. Models are Logistic Regression, Random Forest, Linear SVM, MLP, and LightGBM. Independent methods include raw preprocessing, PCA (95% variance), L1 Logistic Regression selection, capped SMOTEENN, mutual-information SelectKBest (a GUS-inspired proxy), and ANOVA F-test SelectFpr. SMOTEENN resamples training data only and caps it at 100,000 rows by default.", styles["Bodyx"]),
        Paragraph("Observed saved results", styles["H2x"]),
    ]
    table_data = [["Outcome", "Highest saved F1 configuration", "Prec.", "Recall", "F1", "ROC-AUC", "AP"]]
    historical = results[results["split_protocol"] == "historical_stratified_random_row_split"]
    best = historical.sort_values("f1", ascending=False).groupby("target", sort=False).head(1)
    target_order = {"ICU": 0, "INTUBED": 1, "DEATH": 2}
    best = best.assign(_o=best.target.map(target_order)).sort_values("_o")
    for _, r in best.iterrows():
        method = str(r.get("engineering", "")).lower()
        model = str(r.get("model", ""))
        cfg = f"{method} / {r.get('feature_set')} / {model}"
        table_data.append([str(r.target), cfg, f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", f"{r.roc_auc:.3f}", f"{r.pr_auc:.3f}"])
    tbl = Table(table_data, colWidths=[0.58*inch, 2.43*inch, 0.55*inch, 0.56*inch, 0.44*inch, 0.62*inch, 0.42*inch], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 7.3),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C6CF")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF4F7")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    cohort_data = [["Target cohort", "Known rows", "Positive rows", "Prevalence"], ["ICU", "192,543", "16,858", "8.75%"], ["Intubation", "192,706", "33,656", "17.46%"], ["Recorded death", "1,048,575", "76,942", "7.34%"]]
    cohort_table = Table(cohort_data, colWidths=[1.55*inch, 1.4*inch, 1.35*inch, 1.0*inch], repeatRows=1)
    cohort_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#47758A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.4),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#B8C6CF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF4F7")]),("ALIGN",(1,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.extend([tbl, Spacer(1, 6), Paragraph("Target cohort counts", styles["H2x"]), cohort_table, Spacer(1, 5), Paragraph("These rows use historical stratified random row splits and default class thresholds. Precision is low for ICU and intubation despite higher recall. Death SMOTEENN used a capped training set and is not directly comparable with uncapped runs.", styles["Smallx"]), PageBreak()])
    story.extend([
        Paragraph("Results, interpretation, and limits", styles["Title2"]),
        Paragraph("Experiment coverage and corrected runs", styles["H2x"]),
    ])
    coverage = [
        ["Saved method", "Rows", "Coverage"],
        ["Raw", "45", "3 targets x 3 feature groups x 5 models"],
        ["PCA", "15", "Clinical group x 3 targets x 5 models"],
        ["LASSO", "15", "Clinical, C=1.0; 5 models for each target"],
        ["SMOTEENN", "42/45", "100,000 training-row cap; 3 class-collapse failures"],
        ["GUS / FPR", "3 each", "New grouped Clinical Logistic Regression runs"],
    ]
    ct = Table(coverage, colWidths=[1.0*inch, 0.65*inch, 4.1*inch], repeatRows=1)
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#47758A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#B8C6CF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF4F7")]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    grouped = results[results["split_protocol"] == "stratified_grouped_identical_predictor_rows"].copy()
    grouped = grouped[(grouped["feature_set"] == "Clinical") & (grouped["model"] == "LogisticRegression")]
    grouped["engineering_norm"] = grouped["engineering"].astype(str).str.lower()
    story.extend([ct, Spacer(1, 8), Paragraph(f"The corrected comparison below contains {len(grouped)} grouped-split runs using Clinical features and Logistic Regression. Each target uses the same predictor-row grouping and seed across methods. SMOTEENN caps training at 100,000 rows; the test rows stay untouched.", styles["Bodyx"]), Paragraph("Grouped-split F1 by feature-engineering method", styles["H2x"])])
    methods = ["raw", "pca", "lasso", "gus", "fpr", "smoteenn"]
    method_labels = ["Raw", "PCA", "LASSO", "GUS", "FPR", "SMOTEENN"]
    f1_table = [["Target", *method_labels]]
    for target in ["ICU", "INTUBED", "DEATH"]:
        row = [target]
        for method in methods:
            sub = grouped[(grouped.target == target) & (grouped.engineering_norm == method)]
            row.append(f"{sub.f1.iloc[0]:.3f}" if len(sub) else "N/R")
        f1_table.append(row)
    ft = Table(f1_table, colWidths=[0.9*inch] + [0.88*inch]*6, repeatRows=1)
    ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17324D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#B8C6CF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF4F7")]),("ALIGN",(1,1),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 38, 20, 380, 92
    chart.data = [[float(grouped[(grouped.target == t) & (grouped.engineering_norm == m)].f1.iloc[0]) if len(grouped[(grouped.target == t) & (grouped.engineering_norm == m)]) else 0 for m in methods] for t in ["ICU", "INTUBED", "DEATH"]]
    chart.categoryAxis.categoryNames = method_labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 0.6, 0.2
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = "%.1f"
    chart.bars[0].fillColor = colors.HexColor("#277DA1")
    chart.bars[1].fillColor = colors.HexColor("#F9C74F")
    chart.bars[2].fillColor = colors.HexColor("#43AA8B")
    drawing = Drawing(500, 132)
    drawing.add(chart)
    legend = Legend()
    legend.x, legend.y = 420, 42
    legend.fontName, legend.fontSize = "Helvetica", 7
    legend.colorNamePairs = [(colors.HexColor("#277DA1"), "ICU"), (colors.HexColor("#F9C74F"), "Intubation"), (colors.HexColor("#43AA8B"), "Death")]
    drawing.add(legend)
    story.extend([ft, Paragraph("All plotted rows use one Logistic Regression model and one feature group. Historical raw-grid mean F1 from Demo to Clinical increased from 0.135 to 0.171 (ICU), 0.255 to 0.274 (intubation), and 0.259 to 0.455 (death); the random split allows duplicate leakage. The historical best rows on page one use that earlier split and are reported separately.", styles["Smallx"]), drawing, Paragraph("Conclusion", styles["H2x"]), Paragraph("Historical runs show stronger ranking for death and weak positive-class precision for ICU and intubation. The data contain 812,049 exact duplicate rows, no patient identifiers, imbalanced outcomes, and predictors whose timing is not always clear. Grouped results compare feature methods for one model and feature group only. A full grouped model grid and external validation are needed before stronger claims. These models must not guide care or resource allocation.", styles["Bodyx"]), Paragraph("References: Zhan, X. and Li, Y., “CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients,” CS229 Final Report (assigned research reference). UE24CS352A, Mini-Project Assignment Guidelines, supplied course handout.", styles["Smallx"])])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C7D4DC"))
        canvas.line(0.65*inch, 0.48*inch, 7.85*inch, 0.48*inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#607985"))
        canvas.drawString(0.68*inch, 0.31*inch, "Course project | exploratory analysis, not clinical validation")
        canvas.drawRightString(7.83*inch, 0.31*inch, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.5*inch, bottomMargin=0.62*inch, title="COVID-19 Patient Risk Analysis")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
    return pages


def pptx_text_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paras: list[str], font_size: int, color: str, bold=False) -> str:
    runs = []
    for line in paras:
        line = escape(line)
        runs.append(f'<a:p><a:pPr algn="l"/><a:r><a:rPr lang="en-US" sz="{font_size}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{line}</a:t></a:r><a:endParaRPr lang="en-US" sz="{font_size}"/></a:p>')
    body = "".join(runs)
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0" tIns="0" bIns="0"><a:spAutoFit/></a:bodyPr><a:lstStyle/>{body}</p:txBody></p:sp>'


def chart_xml(series_values: list[tuple[str, list[float]]]) -> str:
    cats = ["Raw", "PCA", "LASSO", "SMOTEENN"]
    colors_hex = ["277DA1", "F9C74F", "43AA8B"]
    series = []
    for idx, ((name, values), color) in enumerate(zip(series_values, colors_hex)):
        cat_points = "".join(f'<c:pt idx="{i}"><c:v>{c}</c:v></c:pt>' for i, c in enumerate(cats))
        val_points = "".join(f'<c:pt idx="{i}"><c:v>{v:.6f}</c:v></c:pt>' for i, v in enumerate(values))
        series.append(f'<c:ser><c:idx val="{idx}"/><c:order val="{idx}"/><c:tx><c:strRef><c:f>Sheet1!${chr(66 + idx)}$1</c:f><c:strCache><c:ptCount val="1"/><c:pt idx="0"><c:v>{escape(name)}</c:v></c:pt></c:strCache></c:strRef></c:tx><c:spPr><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln></c:spPr><c:cat><c:strLit><c:ptCount val="4"/>{cat_points}</c:strLit></c:cat><c:val><c:numLit><c:formatCode>0.00</c:formatCode><c:ptCount val="4"/>{val_points}</c:numLit></c:val></c:ser>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><c:date1904 val="0"/><c:lang val="en-US"/><c:chart><c:autoTitleDeleted val="0"/><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr/></a:pPr><a:r><a:rPr lang="en-US" sz="1400"/><a:t>Historical best F1 by method</a:t></a:r><a:endParaRPr lang="en-US"/></a:p></c:rich></c:tx><c:overlay val="0"/></c:title><c:plotArea><c:layout/><c:barChart><c:barDir val="col"/><c:grouping val="clustered"/><c:varyColors val="0"/>{''.join(series)}<c:gapWidth val="75"/><c:overlap val="0"/><c:axId val="12345"/><c:axId val="12346"/></c:barChart><c:catAx><c:axId val="12345"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="b"/><c:majorTickMark val="none"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/><c:crossAx val="12346"/><c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/></c:catAx><c:valAx><c:axId val="12346"/><c:scaling><c:orientation val="minMax"/><c:max val="0.6"/></c:scaling><c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/><c:numFmt formatCode="0.0" sourceLinked="0"/><c:majorTickMark val="out"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/><c:crossAx val="12345"/><c:crosses val="autoZero"/><c:crossBetween val="between"/></c:valAx></c:plotArea><c:legend><c:legendPos val="b"/><c:overlay val="0"/></c:legend><c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/><c:showDLblsOverMax val="0"/></c:chart></c:chartSpace>'''


def chart_frame() -> str:
    return '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="20" name="F1 comparison chart"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="4572000" y="1400000"/><a:ext cx="7200000" cy="4700000"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="rId2"/></a:graphicData></a:graphic></p:graphicFrame>'


def slide_xml(title: str, body: list[str], slide_num: int, chart=False) -> str:
    shapes = [pptx_text_shape(2, "Title", 650000, 300000, 11000000, 850000, [title], 2800, "17324D", True)]
    if chart:
        shapes.append(pptx_text_shape(3, "Chart note", 650000, 1300000, 3600000, 4500000, body, 1650, "263746"))
        shapes.append(chart_frame())
    else:
        size = 1900 if len(body) <= 5 else 1650
        shapes.append(pptx_text_shape(3, "Content", 720000, 1450000, 10800000, 5050000, body, size, "263746"))
    shapes.append(pptx_text_shape(30, "Footer", 650000, 6600000, 11000000, 250000, [f"COVID-19 Patient Risk Analysis  |  {slide_num:02d}"], 900, "607985"))
    tree = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>' + "".join(shapes)
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>{tree}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'


def build_pptx(results: pd.DataFrame) -> None:
    PRESENTATION.mkdir(exist_ok=True)
    out = PRESENTATION / "project_presentation.pptx"
    slides = [
        ("COVID-19 Patient Risk Analysis", ["UE24CS352A Machine Learning mini-project", "Tabular adaptation of a CT-based triage research question", "Team details are omitted because they are not recorded in the repository."]),
        ("Prediction question", ["Can demographic and recorded clinical variables rank severe COVID-19 outcomes?", "Outcomes: ICU admission, intubation, and recorded death.", "The work compares ranking quality and default-threshold errors under class imbalance.", "This is a coursework experiment, not a clinical decision system."]),
        ("Research context and dataset adaptation", ["Zhan and Li studied CT radiomics with clinical, laboratory, and radiologist information.", "The paper's patient-level data were unavailable for this project; the approved public surveillance CSV was used.", "Our predictors and design differ from the paper, so this is an adaptation rather than a replication."]),
        ("Dataset and outcome definitions", ["1,048,575 rows and 21 columns in the supplied CSV.", "ICU: 192,543 known labels, 8.75% positive. Intubation: 192,706 known labels, 17.46% positive.", "Death: valid dd/mm/yyyy date means recorded death; 9999-99-99 means no recorded death. Prevalence: 7.34%.", "Known ICU/intubation labels occur only in hospitalized cases (PATIENT_TYPE=2).", "812,049 exact duplicate rows; no patient identifier."]),
        ("Features and leakage controls", ["Demo: age and sex. DemoComorb adds ten comorbidity flags. Clinical adds pneumonia.", "Targets, PATIENT_TYPE, DATE_DIED, pregnancy, classification, and service/unit fields are excluded.", "Age 97/98/99, comorbidity 98/99, pneumonia 99 become missing.", "Imputers, scaling, one-hot encoding, and selectors fit on training data only.", "New split groups identical predictor rows; it cannot verify distinct patients."]),
        ("Model and experiment pipeline", ["Models: Logistic Regression, Random Forest, Linear SVM, MLP, LightGBM.", "Independent methods: raw, PCA, L1 selection, capped SMOTEENN, mutual-information proxy, F-test selection.", "80/20 stratified split, random_state=42. SMOTEENN is training-only with a 100,000-row cap.", "Metrics: precision, recall, F1, ROC-AUC, average precision, accuracy, confusion counts."]),
        ("Historical F1 comparison", ["Highest saved F1 per target and method. Historical split allowed duplicates across partitions.", "SMOTEENN used capped training data and is a different training regime.", "Saved coverage: raw 45; PCA 15 Clinical; LASSO 15 Clinical; SMOTEENN 42/45.", "Three INTUBED/Demo SMOTEENN fits failed after ENN removed one class."], True),
        ("Grouped-split smoke run", ["New raw Clinical Logistic Regression run completed for all three outcomes.", "ICU: precision 0.1221, recall 0.7711, F1 0.2108, ROC-AUC 0.6514, AP 0.1333.", "Intubation: precision 0.2384, recall 0.6904, F1 0.3544, ROC-AUC 0.6382, AP 0.2382.", "Death: precision 0.3337, recall 0.8426, F1 0.4781, ROC-AUC 0.9221, AP 0.4624.", "This is one model and one feature group, not a complete grouped comparison."]),
        ("Conclusion and live demo", ["Historical results show stronger ranking for death, but weak precision for ICU and intubation.", "Repeated rows, no patient ID, outcome imbalance, predictor timing, and dataset substitution limit conclusions.", "Demo: python -m pytest -q", "Then: python src/run_experiments.py --engineering raw --targets ICU --feature-sets Clinical --models LogisticRegression", "Do not use the model for care or resource allocation."]),
    ]
    # Best F1 within each target and historical method, matching the report's grouped comparison.
    x = results[results["split_protocol"] == "historical_stratified_random_row_split"].copy()
    x["eng_norm"] = x.engineering.astype(str).str.lower()
    chart_data = []
    for target in ["ICU", "INTUBED", "DEATH"]:
        vals = []
        for method in ["raw", "pca", "lasso", "smoteenn"]:
            sub = x[(x.target == target) & (x.eng_norm == method)]
            vals.append(float(sub.f1.max()) if len(sub) else 0.0)
        chart_data.append((target, vals))
    grouped_rows = results[results["split_protocol"] == "stratified_grouped_identical_predictor_rows"].copy()
    grouped_rows = grouped_rows[(grouped_rows["feature_set"] == "Clinical") & (grouped_rows["model"] == "LogisticRegression")]
    grouped_rows["eng_norm"] = grouped_rows["engineering"].astype(str).str.lower()
    method_order = ["raw", "pca", "lasso", "gus", "fpr", "smoteenn"]
    grouped_body = ["Same group-aware split, Clinical features, and Logistic Regression. F1 by method (Raw / PCA / LASSO / GUS / FPR / SMOTEENN):"]
    for target in ["ICU", "INTUBED", "DEATH"]:
        values = []
        for method in method_order:
            sub = grouped_rows[(grouped_rows.target == target) & (grouped_rows.eng_norm == method)]
            values.append(f"{sub.f1.iloc[0]:.3f}" if len(sub) else "N/R")
        grouped_body.append(f"{target}: " + " / ".join(values))
    grouped_body.append("SMOTEENN uses a 100,000-row training cap. This 18-run subset covers one of five models and one of three feature groups.")
    grouped_body.append("Historical highest F1: ICU Raw LightGBM 0.220; intubation Raw LightGBM 0.353; death SMOTEENN LightGBM 0.509. Those rows use the duplicate-prone random split.")
    slides[7] = ("Grouped-split comparison and saved best rows", grouped_body)

    content_types = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>','<Default Extension="xml" ContentType="application/xml"/>','<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>','<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>','<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>','<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>','<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>','<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>']
    content_types += [f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides)+1)]
    content_types += ['<Override PartName="/ppt/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>']
    slide_id_xml = "".join('<p:sldId id="%d" r:id="rId%d"/>' % (256 + i, i + 2) for i in range(len(slides)))
    presentation = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{slide_id_xml}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    pres_rels = '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>' + "".join(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(slides)+1))
    master = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    layout = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    theme = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:srgbClr val="17324D"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="263746"/></a:dk2><a:lt2><a:srgbClr val="EEF4F7"/></a:lt2><a:accent1><a:srgbClr val="277DA1"/></a:accent1><a:accent2><a:srgbClr val="F9C74F"/></a:accent2><a:accent3><a:srgbClr val="43AA8B"/></a:accent3><a:accent4><a:srgbClr val="F94144"/></a:accent4><a:accent5><a:srgbClr val="845EC2"/></a:accent5><a:accent6><a:srgbClr val="607985"/></a:accent6><a:hlink><a:srgbClr val="277DA1"/></a:hlink><a:folHlink><a:srgbClr val="845EC2"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>'
    with ZipFile(out, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(content_types) + "</Types>")
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>')
        z.writestr("docProps/core.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>COVID-19 Patient Risk Analysis</dc:title><dc:subject>UE24CS352A machine learning mini-project</dc:subject><dc:creator>Project team</dc:creator></cp:coreProperties>')
        z.writestr("docProps/app.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft PowerPoint</Application><Slides>{len(slides)}</Slides></Properties>')
        z.writestr("ppt/presentation.xml", presentation)
        z.writestr("ppt/_rels/presentation.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{pres_rels}</Relationships>')
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
        z.writestr("ppt/theme/theme1.xml", theme)
        for i, (title, body, *extra) in enumerate(slides, 1):
            has_chart = bool(extra and extra[0])
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(title, body, i, has_chart))
            rels = '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            if has_chart:
                rels += '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart1.xml"/>'
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>')
        z.writestr("ppt/charts/chart1.xml", chart_xml(chart_data))
        # The chart carries literal series values in its chart part.
    # Verify all package XML parts parse before delivering.
    with ZipFile(out) as z:
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(z.read(name))
    # Replace the literal-cache chart with python-pptx's native chart so the
    # deck contains an embedded workbook and the chart data remain editable.
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    deck = Presentation(str(out))
    chart_slide = deck.slides[6]
    for shape in list(chart_slide.shapes):
        if shape.has_chart:
            element = shape._element
            element.getparent().remove(element)
    data = CategoryChartData()
    data.categories = ["Raw", "PCA", "LASSO", "SMOTEENN"]
    for name, values in chart_data:
        data.add_series(name, values)
    chart = chart_slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(5.0), Inches(1.55), Inches(7.8), Inches(5.0), data,
    ).chart
    chart.has_title = True
    chart.chart_title.text_frame.text = "Historical best F1 by method"
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(10)
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 0.6
    chart.value_axis.major_unit = 0.1
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.size = Pt(9)
    chart.category_axis.tick_labels.font.size = Pt(9)
    for series, rgb in zip(chart.series, ["277DA1", "F9C74F", "43AA8B"]):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = RGBColor.from_string(rgb)
        series.format.line.fill.background()
    chart.plots[0].gap_width = 75
    temp_out = out.with_name("project_presentation.native.pptx")
    deck.save(str(temp_out))
    temp_out.replace(out)
    verified = Presentation(str(out))
    charts = [shape.chart for shape in verified.slides[6].shapes if shape.has_chart]
    if len(verified.slides) != 9 or len(charts) != 1 or [s.name for s in charts[0].series] != ["ICU", "INTUBED", "DEATH"]:
        raise RuntimeError("PowerPoint validation failed: expected nine slides and a labelled native chart.")


def main():
    results = consolidate()
    pages = build_pdf(results)
    build_pptx(results)
    print(json.dumps({"consolidated_rows": len(results), "report_pages": pages, "pdf": str(REPORTS / "project_report.pdf"), "pptx": str(PRESENTATION / "project_presentation.pptx")}, indent=2))
    if pages != 2:
        raise RuntimeError(f"Expected a two-page report, generated {pages} pages")


if __name__ == "__main__":
    main()
