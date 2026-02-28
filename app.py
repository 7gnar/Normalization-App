"""
app.py  —  Flask web app for per-group Min-Max score normalization
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
MAX_FILE_MB = 10


# ── normalization helpers ────────────────────────────────────────────────────

def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def normalize_by_group(df, group_col, score_col):
    result = df.copy()
    norm_col = f"Normalized_{score_col}"
    result[norm_col] = (
        result.groupby(group_col)[score_col]
        .transform(minmax_normalize)
        .round(4)
    )
    return result, norm_col


def get_top_n_per_group(df, group_col, norm_col, n=10):
    top_df = (
        df.sort_values([group_col, norm_col], ascending=[True, False])
        .groupby(group_col, sort=False)
        .head(n)
        .copy()
    )
    top_df["Rank"] = (
        top_df.groupby(group_col)[norm_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    cols = [group_col, "Rank"] + [c for c in top_df.columns if c not in (group_col, "Rank")]
    return top_df[cols].sort_values([group_col, "Rank"]).reset_index(drop=True)


def group_stats(df, group_col, score_col, norm_col):
    stats = (
        df.groupby(group_col)[[score_col, norm_col]]
        .agg(["min", "max", "mean"])
        .round(3)
    )
    # Flatten multi-level columns
    stats.columns = [f"{col}_{agg}" for col, agg in stats.columns]
    return stats.reset_index().to_dict(orient="records")


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Receive file, return column metadata so user can pick group/score cols."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file received."}), 400

    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only .csv, .xlsx, or .xls files are supported."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if len(df) == 0:
        return jsonify({"error": "File is empty."}), 400

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols     = df.columns.tolist()

    # Store df in session-like temp storage (simple: re-uploaded each process call)
    # We encode the data as JSON for the next step
    col_info = []
    for col in all_cols:
        col_info.append({
            "name":     col,
            "dtype":    str(df[col].dtype),
            "numeric":  col in numeric_cols,
            "n_unique": int(df[col].nunique()),
            "sample":   [str(v) for v in df[col].dropna().unique()[:4].tolist()]
        })

    return jsonify({
        "rows":        len(df),
        "cols":        len(all_cols),
        "columns":     col_info,
        "numeric_cols": numeric_cols,
        "filename":    file.filename,
    })


@app.route("/process", methods=["POST"])
def process():
    """Re-receive file + user column choices, run normalization, return results."""
    file      = request.files.get("file")
    group_col = request.form.get("group_col")
    score_col = request.form.get("score_col")
    top_n     = int(request.form.get("top_n", 10))

    if not file or not group_col or not score_col:
        return jsonify({"error": "Missing file or column selections."}), 400

    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if group_col not in df.columns or score_col not in df.columns:
        return jsonify({"error": "Selected columns not found in file."}), 400

    result_df, norm_col = normalize_by_group(df, group_col, score_col)
    top_df              = get_top_n_per_group(result_df, group_col, norm_col, n=top_n)
    stats               = group_stats(result_df, group_col, score_col, norm_col)

    # Build per-group table data
    groups = []
    for group_name, grp in top_df.groupby(group_col, sort=False):
        groups.append({
            "name":    str(group_name),
            "columns": grp.columns.tolist(),
            "rows":    grp.fillna("").astype(str).values.tolist(),
        })

    return jsonify({
        "groups":    groups,
        "stats":     stats,
        "norm_col":  norm_col,
        "group_col": group_col,
        "score_col": score_col,
        "top_n":     top_n,
        "total_rows": len(df),
    })


@app.route("/download", methods=["POST"])
def download():
    """Re-run normalization and return an Excel file for download."""
    file      = request.files.get("file")
    group_col = request.form.get("group_col")
    score_col = request.form.get("score_col")
    top_n     = int(request.form.get("top_n", 10))

    filename = file.filename.lower()
    df = pd.read_csv(file) if filename.endswith(".csv") else pd.read_excel(file)

    result_df, norm_col = normalize_by_group(df, group_col, score_col)
    top_df              = get_top_n_per_group(result_df, group_col, norm_col, n=top_n)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="All_Normalized", index=False)
        top_df.to_excel(writer, sheet_name=f"Top{top_n}_Per_Group", index=False)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="normalized_results.xlsx"
    )


if __name__ == "__main__":
    app.run(debug=True)
