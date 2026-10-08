#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    keep_left = left > 0
    keep_right = right > 0
    return float(
        0.5 * np.sum(left[keep_left] * np.log2(left[keep_left] / middle[keep_left]))
        + 0.5 * np.sum(right[keep_right] * np.log2(right[keep_right] / middle[keep_right]))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = str(args.snapshots.resolve() / "snapshot_*.parquet")
    con = duckdb.connect()

    summary = con.execute(
        f"""
        SELECT cast(snapshot_ts AS DATE) snapshot_date,
               count(*) n_snapshots,
               count(DISTINCT user_id) n_users,
               avg((n_labels > 0)::INTEGER) positive_rate,
               avg((n_labels > 1)::INTEGER) multilabel_rate_all,
               sum(n_labels > 1)::DOUBLE / nullif(sum(n_labels > 0), 0) multilabel_rate_positive,
               avg(n_labels) FILTER (WHERE n_labels > 0) mean_labels_positive,
               avg(history_events) mean_history,
               approx_quantile(history_events, 0.5) history_p50,
               approx_quantile(history_events, 0.9) history_p90,
               avg((history_events_before_truncation > history_events)::INTEGER) truncation_rate,
               avg(len(masked_leaf)) mean_masked,
               approx_quantile(len(masked_leaf), 0.9) masked_p90
        FROM read_parquet('{source}')
        GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    target_frequency = con.execute(
        f"""
        SELECT cast(snapshot_ts AS DATE) snapshot_date, target.leaf_id, count(*) n
        FROM read_parquet('{source}') t, unnest(t.label_leaf) AS target(leaf_id)
        GROUP BY 1, 2 ORDER BY 1, n DESC
        """
    ).fetchdf()
    label_counts = con.execute(
        f"""
        SELECT cast(snapshot_ts AS DATE) snapshot_date, n_labels, count(*) n
        FROM read_parquet('{source}') GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchdf()
    action_mix = con.execute(
        f"""
        SELECT cast(snapshot_ts AS DATE) snapshot_date, event.action_id, count(*) n
        FROM read_parquet('{source}') t, unnest(t.action) AS event(action_id)
        WHERE hash(snapshot_id) % 100 = 0
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchdf()
    con.close()

    summary.to_csv(args.output / "snapshot_summary.csv", index=False)
    target_frequency.to_csv(args.output / "target_frequency.csv", index=False)
    label_counts.to_csv(args.output / "label_count_distribution.csv", index=False)
    action_mix.to_csv(args.output / "action_mix_one_percent_sample.csv", index=False)

    dates = sorted(target_frequency["snapshot_date"].unique())
    leaf_ids = sorted(target_frequency["leaf_id"].unique())
    pivot = (
        target_frequency.pivot(index="snapshot_date", columns="leaf_id", values="n")
        .reindex(index=dates, columns=leaf_ids, fill_value=0)
        .fillna(0)
    )
    drift_rows = []
    for first, second in zip(dates[:-1], dates[1:]):
        drift_rows.append(
            {
                "from_snapshot": first,
                "to_snapshot": second,
                "js_divergence": js_divergence(pivot.loc[first].to_numpy(), pivot.loc[second].to_numpy()),
            }
        )
    drift = pd.DataFrame(drift_rows)
    drift.to_csv(args.output / "temporal_target_jsd.csv", index=False)

    concentration_rows = []
    for date, group in target_frequency.groupby("snapshot_date"):
        counts = group["n"].sort_values(ascending=False).to_numpy()
        total = counts.sum()
        concentration_rows.append(
            {
                "snapshot_date": date,
                "active_target_leaf": len(counts),
                "top_10_leaf_share": counts[:10].sum() / total,
                "top_20_percent_leaf_share": counts[: max(1, int(np.ceil(0.2 * len(counts))))].sum() / total,
            }
        )
    concentration = pd.DataFrame(concentration_rows)
    concentration.to_csv(args.output / "target_concentration.csv", index=False)

    x = np.arange(len(summary))
    labels = summary["snapshot_date"].astype(str)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x, summary["n_snapshots"] / 1e6, color="#4C78A8")
    ax1.set_ylabel("Snapshots, млн")
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, summary["positive_rate"], color="#E45756", marker="o")
    ax2.set_ylabel("Доля snapshots с target")
    fig.tight_layout()
    fig.savefig(args.output / "01_snapshots_and_positive_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, summary["history_p50"], marker="o", label="p50")
    ax.plot(x, summary["history_p90"], marker="o", label="p90")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Число событий в истории")
    ax.legend()
    axr = ax.twinx()
    axr.plot(x, summary["truncation_rate"], color="#F58518", marker="s", label="truncation")
    axr.set_ylabel("Доля обрезанных историй")
    fig.tight_layout()
    fig.savefig(args.output / "02_history_length_and_truncation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, summary["multilabel_rate_positive"], marker="o", label="multi-label среди positive")
    ax.plot(x, summary["mean_masked"] / summary["mean_masked"].max(), marker="o", label="mean masked, норм.")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "03_multilabel_and_mask.png", dpi=160)
    plt.close(fig)

    action_names = {1: "view", 2: "favorite", 3: "contact_dab", 4: "contact_dtb"}
    action_plot = action_mix.copy()
    action_plot["snapshot_date"] = action_plot["snapshot_date"].astype(str)
    action_plot["action"] = action_plot["action_id"].map(action_names)
    action_pivot = action_plot.pivot(index="snapshot_date", columns="action", values="n").fillna(0)
    action_share = action_pivot.div(action_pivot.sum(axis=1), axis=0)
    ax = action_share.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20c")
    ax.set_ylabel("Доля событий")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.output / "04_action_mix_sample.png", dpi=160)
    plt.close()

    latest = target_frequency[target_frequency["snapshot_date"] == dates[-1]]["n"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(latest) + 1), latest.to_numpy())
    ax.set_yscale("log")
    ax.set_xlabel("Ранг leaf по частоте target")
    ax.set_ylabel("Число target, log-scale")
    fig.tight_layout()
    fig.savefig(args.output / "05_target_long_tail.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(drift)), drift["js_divergence"], marker="o")
    ax.set_xticks(
        np.arange(len(drift)),
        [f"{a} → {b}" for a, b in zip(drift["from_snapshot"], drift["to_snapshot"])],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Jensen–Shannon divergence")
    fig.tight_layout()
    fig.savefig(args.output / "06_temporal_target_drift.png", dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
