#!/usr/bin/env python3
"""
Compare sigma_star vs vocab gap_mode results from two JSONL files.

Usage:
  # Compare raw inference outputs:
    python compare_gap_modes.py sigma_star.jsonl vocab.jsonl

  # Compare compiled (evaluated) outputs:
    python compare_gap_modes.py sigma_star.compiled.jsonl vocab.compiled.jsonl

  # Save to file:
    python compare_gap_modes.py sigma_star.jsonl vocab.jsonl --output comparison.txt
"""

import json
import sys
import statistics
from collections import defaultdict


def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            iid = data.get("instance_id", "")
            records[iid] = data
    return records


def is_compiled(records):
    sample = next(iter(records.values()))
    return "syntax_ok" in sample


def fmt_pct(n, total):
    if total == 0:
        return "N/A"
    pct = n / total * 100
    return f"{n}/{total} ({pct:.1f}%)"


def compare_raw(rec_a, rec_b, label_a, label_b, out):
    common = sorted(set(rec_a) & set(rec_b))
    only_a = sorted(set(rec_a) - set(rec_b))
    only_b = sorted(set(rec_b) - set(rec_a))

    out.write(f"{'='*70}\n")
    out.write(f"  Raw Inference Output Comparison\n")
    out.write(f"  {label_a} vs {label_b}\n")
    out.write(f"{'='*70}\n\n")

    out.write(f"Total instances:  {label_a}={len(rec_a)}  {label_b}={len(rec_b)}  common={len(common)}\n")
    if only_a:
        out.write(f"Only in {label_a}: {only_a[:10]}{'...' if len(only_a) > 10 else ''}\n")
    if only_b:
        out.write(f"Only in {label_b}: {only_b[:10]}{'...' if len(only_b) > 10 else ''}\n")
    out.write("\n")

    # --- Timed out ---
    to_a = sum(1 for iid in common if rec_a[iid].get("timed_out", False))
    to_b = sum(1 for iid in common if rec_b[iid].get("timed_out", False))
    out.write(f"--- Timed Out (on {len(common)} common instances) ---\n")
    out.write(f"  {label_a}: {fmt_pct(to_a, len(common))}\n")
    out.write(f"  {label_b}: {fmt_pct(to_b, len(common))}\n\n")

    # --- Resamples ---
    def resample_count(v):
        if isinstance(v, list):
            return len(v)
        if isinstance(v, (int, float)):
            return v
        return 0

    res_a = [resample_count(rec_a[iid].get("resamples", 0)) for iid in common]
    res_b = [resample_count(rec_b[iid].get("resamples", 0)) for iid in common]
    if any(r > 0 for r in res_a + res_b):
        out.write(f"--- Resamples ---\n")
        out.write(f"  {label_a}: mean={statistics.mean(res_a):.2f}  median={statistics.median(res_a):.1f}  total={sum(res_a)}\n")
        out.write(f"  {label_b}: mean={statistics.mean(res_b):.2f}  median={statistics.median(res_b):.1f}  total={sum(res_b)}\n")
        diff_resamples = sum(1 for a, b in zip(res_a, res_b) if a != b)
        out.write(f"  Instances with different resample count: {diff_resamples}/{len(common)}\n\n")

    # --- Time taken ---
    times_a = [rec_a[iid].get("time_taken", 0) for iid in common if rec_a[iid].get("time_taken") is not None]
    times_b = [rec_b[iid].get("time_taken", 0) for iid in common if rec_b[iid].get("time_taken") is not None]
    if times_a and times_b:
        out.write(f"--- Time Taken (seconds) ---\n")
        out.write(f"  {label_a}: mean={statistics.mean(times_a):.3f}  median={statistics.median(times_a):.3f}  total={sum(times_a):.1f}\n")
        out.write(f"  {label_b}: mean={statistics.mean(times_b):.3f}  median={statistics.median(times_b):.3f}  total={sum(times_b):.1f}\n")
        diffs = [times_b[i] - times_a[i] for i in range(min(len(times_a), len(times_b)))]
        out.write(f"  Diff ({label_b} - {label_a}): mean={statistics.mean(diffs):+.3f}  median={statistics.median(diffs):+.3f}\n")
        speedup = [times_a[i] / times_b[i] for i in range(min(len(times_a), len(times_b))) if times_b[i] > 0]
        if speedup:
            out.write(f"  Speedup ({label_a}/{label_b}): mean={statistics.mean(speedup):.2f}x  median={statistics.median(speedup):.2f}x\n")
        out.write("\n")

    # --- Per-instance time diff (top 10 biggest differences) ---
    per_inst = []
    for iid in common:
        ta = rec_a[iid].get("time_taken", 0) or 0
        tb = rec_b[iid].get("time_taken", 0) or 0
        per_inst.append((iid, ta, tb, tb - ta))
    per_inst.sort(key=lambda x: abs(x[3]), reverse=True)
    out.write(f"--- Top 10 Biggest Time Differences ---\n")
    out.write(f"  {'instance_id':<40} {label_a:>10} {label_b:>10} {'diff':>10}\n")
    for iid, ta, tb, diff in per_inst[:10]:
        out.write(f"  {iid:<40} {ta:>10.3f} {tb:>10.3f} {diff:>+10.3f}\n")
    out.write("\n")

    # --- Extracted output match ---
    match = 0
    for iid in common:
        ea = rec_a[iid].get("extracted", "")
        eb = rec_b[iid].get("extracted", "")
        if ea == eb:
            match += 1
    out.write(f"--- Extracted Output Match ---\n")
    out.write(f"  Identical outputs: {fmt_pct(match, len(common))}\n\n")


def compare_compiled(rec_a, rec_b, label_a, label_b, out):
    common = sorted(set(rec_a) & set(rec_b))
    only_a = sorted(set(rec_a) - set(rec_b))
    only_b = sorted(set(rec_b) - set(rec_a))

    out.write(f"{'='*70}\n")
    out.write(f"  Compiled (Evaluated) Output Comparison\n")
    out.write(f"  {label_a} vs {label_b}\n")
    out.write(f"{'='*70}\n\n")

    out.write(f"Total instances:  {label_a}={len(rec_a)}  {label_b}={len(rec_b)}  common={len(common)}\n")
    if only_a:
        out.write(f"Only in {label_a}: {only_a[:10]}{'...' if len(only_a) > 10 else ''}\n")
    if only_b:
        out.write(f"Only in {label_b}: {only_b[:10]}{'...' if len(only_b) > 10 else ''}\n")
    out.write("\n")

    # --- Syntax correctness ---
    syn_a = sum(1 for iid in common if rec_a[iid].get("syntax_ok", False))
    syn_b = sum(1 for iid in common if rec_b[iid].get("syntax_ok", False))
    out.write(f"--- Syntax Correctness (on {len(common)} common instances) ---\n")
    out.write(f"  {label_a}: {fmt_pct(syn_a, len(common))}\n")
    out.write(f"  {label_b}: {fmt_pct(syn_b, len(common))}\n\n")

    # --- Functional correctness ---
    test_a = sum(1 for iid in common if rec_a[iid].get("passed_tests", False))
    test_b = sum(1 for iid in common if rec_b[iid].get("passed_tests", False))
    out.write(f"--- Functional Correctness (passed_tests) ---\n")
    out.write(f"  {label_a}: {fmt_pct(test_a, len(common))}\n")
    out.write(f"  {label_b}: {fmt_pct(test_b, len(common))}\n\n")

    # --- Flipped instances ---
    syn_a_only = []
    syn_b_only = []
    test_a_only = []
    test_b_only = []
    for iid in common:
        sa = rec_a[iid].get("syntax_ok", False)
        sb = rec_b[iid].get("syntax_ok", False)
        if sa and not sb:
            syn_a_only.append(iid)
        elif sb and not sa:
            syn_b_only.append(iid)
        ta = rec_a[iid].get("passed_tests", False)
        tb = rec_b[iid].get("passed_tests", False)
        if ta and not tb:
            test_a_only.append(iid)
        elif tb and not ta:
            test_b_only.append(iid)

    out.write(f"--- Syntax Flips ---\n")
    out.write(f"  Only {label_a} correct: {len(syn_a_only)}\n")
    if syn_a_only:
        out.write(f"    {syn_a_only[:15]}{'...' if len(syn_a_only) > 15 else ''}\n")
    out.write(f"  Only {label_b} correct: {len(syn_b_only)}\n")
    if syn_b_only:
        out.write(f"    {syn_b_only[:15]}{'...' if len(syn_b_only) > 15 else ''}\n")
    out.write("\n")

    out.write(f"--- Test Flips ---\n")
    out.write(f"  Only {label_a} passed: {len(test_a_only)}\n")
    if test_a_only:
        out.write(f"    {test_a_only[:15]}{'...' if len(test_a_only) > 15 else ''}\n")
    out.write(f"  Only {label_b} passed: {len(test_b_only)}\n")
    if test_b_only:
        out.write(f"    {test_b_only[:15]}{'...' if len(test_b_only) > 15 else ''}\n")
    out.write("\n")

    # --- Time taken ---
    times_a = [rec_a[iid].get("time_taken", 0) for iid in common if rec_a[iid].get("time_taken") is not None]
    times_b = [rec_b[iid].get("time_taken", 0) for iid in common if rec_b[iid].get("time_taken") is not None]
    if times_a and times_b:
        out.write(f"--- Time Taken (seconds per token) ---\n")
        out.write(f"  {label_a}: mean={statistics.mean(times_a):.4f}  median={statistics.median(times_a):.4f}\n")
        out.write(f"  {label_b}: mean={statistics.mean(times_b):.4f}  median={statistics.median(times_b):.4f}\n\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare sigma_star vs vocab gap_mode JSONL results")
    parser.add_argument("file_a", help="First JSONL file (e.g. sigma_star results)")
    parser.add_argument("file_b", help="Second JSONL file (e.g. vocab results)")
    parser.add_argument("--label-a", default=None, help="Label for file A (auto-detected from gap_mode if omitted)")
    parser.add_argument("--label-b", default=None, help="Label for file B (auto-detected from gap_mode if omitted)")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    rec_a = load_jsonl(args.file_a)
    rec_b = load_jsonl(args.file_b)

    if not rec_a:
        print(f"Error: {args.file_a} is empty or invalid", file=sys.stderr)
        sys.exit(1)
    if not rec_b:
        print(f"Error: {args.file_b} is empty or invalid", file=sys.stderr)
        sys.exit(1)

    # Auto-detect labels from gap_mode field
    sample_a = next(iter(rec_a.values()))
    sample_b = next(iter(rec_b.values()))
    label_a = args.label_a or sample_a.get("gap_mode", args.file_a)
    label_b = args.label_b or sample_b.get("gap_mode", args.file_b)

    out = open(args.output, "w") if args.output else sys.stdout

    compiled_a = is_compiled(rec_a)
    compiled_b = is_compiled(rec_b)

    if compiled_a and compiled_b:
        compare_compiled(rec_a, rec_b, label_a, label_b, out)
    elif not compiled_a and not compiled_b:
        compare_raw(rec_a, rec_b, label_a, label_b, out)
    else:
        print("Error: one file is compiled and the other is raw. Provide two files of the same type.", file=sys.stderr)
        sys.exit(1)

    # --- Summary ---
    out.write(f"{'='*70}\n")
    out.write(f"  Summary\n")
    out.write(f"{'='*70}\n")
    common = set(rec_a) & set(rec_b)
    if compiled_a:
        syn_a = sum(1 for iid in common if rec_a[iid].get("syntax_ok", False))
        syn_b = sum(1 for iid in common if rec_b[iid].get("syntax_ok", False))
        test_a = sum(1 for iid in common if rec_a[iid].get("passed_tests", False))
        test_b = sum(1 for iid in common if rec_b[iid].get("passed_tests", False))
        n = len(common)
        out.write(f"\n  {'Metric':<25} {label_a:>15} {label_b:>15} {'diff':>10}\n")
        out.write(f"  {'-'*65}\n")
        out.write(f"  {'Syntax %':<25} {syn_a/n*100:>14.1f}% {syn_b/n*100:>14.1f}% {(syn_b-syn_a)/n*100:>+9.1f}%\n")
        out.write(f"  {'Tests %':<25} {test_a/n*100:>14.1f}% {test_b/n*100:>14.1f}% {(test_b-test_a)/n*100:>+9.1f}%\n")
    else:
        to_a = sum(1 for iid in common if rec_a[iid].get("timed_out", False))
        to_b = sum(1 for iid in common if rec_b[iid].get("timed_out", False))
        times_a = [rec_a[iid].get("time_taken", 0) or 0 for iid in common]
        times_b = [rec_b[iid].get("time_taken", 0) or 0 for iid in common]
        n = len(common)
        out.write(f"\n  {'Metric':<25} {label_a:>15} {label_b:>15} {'diff':>10}\n")
        out.write(f"  {'-'*65}\n")
        out.write(f"  {'Timed out':<25} {to_a:>15} {to_b:>15} {to_b-to_a:>+10}\n")
        out.write(f"  {'Median time (s)':<25} {statistics.median(times_a):>15.3f} {statistics.median(times_b):>15.3f} {statistics.median(times_b)-statistics.median(times_a):>+10.3f}\n")
        out.write(f"  {'Mean time (s)':<25} {statistics.mean(times_a):>15.3f} {statistics.mean(times_b):>15.3f} {statistics.mean(times_b)-statistics.mean(times_a):>+10.3f}\n")
        out.write(f"  {'Total time (s)':<25} {sum(times_a):>15.1f} {sum(times_b):>15.1f} {sum(times_b)-sum(times_a):>+10.1f}\n")

    out.write("\n")
    if args.output:
        out.close()
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
