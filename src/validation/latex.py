"""LaTeX table generation (booktabs style)."""
from pathlib import Path

import pandas as pd


def to_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_fmt: str = ".3f",
    index: bool = False,
    column_format: "str | None" = None,
) -> str:
    """Render a DataFrame as a booktabs-style LaTeX table string."""
    if column_format is None:
        chars = (["l"] if index else []) + [
            "r" if pd.api.types.is_numeric_dtype(df[c]) else "l"
            for c in df.columns
        ]
        column_format = "".join(chars)

    raw = df.to_latex(
        index=index,
        float_format=f"{{:{float_fmt}}}".format,
        column_format=column_format,
        escape=True,
        na_rep="—",
    )
    # Replace \hline with booktabs rules
    lines = raw.strip().split("\n")
    lines = [l.replace("\\hline", "") for l in lines]

    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{tab:{label}}}\n"
        + "\n".join(lines) + "\n"
        "\\end{table}\n"
    )


def save_table(df: pd.DataFrame, path: "str | Path", caption: str, label: str, **kwargs) -> None:
    """Write a booktabs LaTeX table to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_booktabs(df, caption, label, **kwargs), encoding="utf-8")
    print(f"  LaTeX → {path}")
