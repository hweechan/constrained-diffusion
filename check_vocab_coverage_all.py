#!/usr/bin/env python3
"""
Check vocab terminal coverage for ALL instances across all datasets.
Finds instances where sigma_star and vocab modes are most likely to differ.

Usage:
    python check_vocab_coverage_all.py --model_name Dream-org/Dream-v0-Instruct-7B
    python check_vocab_coverage_all.py --model_name Dream-org/Dream-v0-Instruct-7B --dataset_name smiles
"""

import fire
from collections import defaultdict

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    preprocessed_generate_stuff,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset


def main(model_name="Dream-org/Dream-v0-Instruct-7B", dataset_name=None, device="cpu"):
    from constrained_diffusion.eval.dllm.model import load_model

    eval_model = load_model(model_name)
    tokenizer = eval_model.tokenizer(device)

    datasets_to_check = (
        [dataset_name] if dataset_name else ["jsonschema", "smiles", "THUDM/humaneval-x/cpp"]
    )

    for ds_name in datasets_to_check:
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        try:
            ds = load_dataset(ds_name)
        except Exception as e:
            print(f"  Skipped: {e}")
            continue

        instances = sorted(ds, key=lambda x: x.instance_id())
        gap_stats = []
        prev_lang_id = None

        for i, inst in enumerate(instances):
            try:
                lang, lex_map, subtokens = inst.language_lex_subtokens()
                terminals = set(lang.get_terminals())
                compiled_lex = compile_lex_map(lex_map, subtokens=subtokens)

                lang_id = frozenset(terminals)
                if lang_id == prev_lang_id and not ds.different_grammar_per_instance:
                    gap_stats.append(gap_stats[-1])
                    continue
                prev_lang_id = lang_id

                vocab_lexings, _, _ = preprocessed_generate_stuff(
                    tokenizer, lang, compiled_lex,
                    prelex=inst.prelex(),
                    subtokens=subtokens,
                    strip_chars=inst.strip_chars(),
                    gap_mode="vocab",
                )
                vocab_terms = set()
                for entry in vocab_lexings:
                    for sym in entry[0]:
                        vocab_terms.add(sym)

                uncovered = terminals - vocab_terms
                extra = vocab_terms - terminals
                gap_stats.append({
                    "instance_id": inst.instance_id(),
                    "n_grammar": len(terminals),
                    "n_vocab": len(vocab_terms),
                    "n_uncovered": len(uncovered),
                    "uncovered": sorted(uncovered),
                    "extra": sorted(extra),
                })
            except Exception as e:
                print(f"  Error on {inst.instance_id()}: {e}")
                gap_stats.append({"instance_id": inst.instance_id(), "error": str(e)})

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(instances)}...")

        # Summary
        valid = [s for s in gap_stats if "error" not in s]
        has_gap = [s for s in valid if s["n_uncovered"] > 0]
        no_gap = [s for s in valid if s["n_uncovered"] == 0]

        print(f"\n  Total instances: {len(instances)}")
        print(f"  Successfully checked: {len(valid)}")
        print(f"  Full coverage (no difference expected): {len(no_gap)}")
        print(f"  Partial coverage (difference possible):  {len(has_gap)}")

        if has_gap:
            uncovered_counts = defaultdict(int)
            for s in has_gap:
                for t in s["uncovered"]:
                    uncovered_counts[t] += 1

            print(f"\n  Uncovered terminals (across all instances with gaps):")
            for term, count in sorted(uncovered_counts.items(), key=lambda x: -x[1]):
                print(f"    {term}: {count} instances")

            # Show instances with the most uncovered terminals
            has_gap.sort(key=lambda x: -x["n_uncovered"])
            print(f"\n  Top 10 instances with most uncovered terminals:")
            for s in has_gap[:10]:
                print(f"    {s['instance_id']}: {s['n_uncovered']} uncovered — {s['uncovered']}")

        if not ds.different_grammar_per_instance and valid:
            print(f"\n  (Fixed grammar — all instances share the same terminal set)")


if __name__ == "__main__":
    fire.Fire(main)
