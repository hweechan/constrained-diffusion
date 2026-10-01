#!/usr/bin/env python3
"""
Check whether vocab terminals cover all grammar terminals for a dataset.
If they're the same, sigma_star and vocab gap modes produce identical FSAs.

Usage:
    python check_vocab_coverage.py --dataset jsonschema --model Dream-org/Dream-v0-Instruct-7B
"""

import fire

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    preprocessed_generate_stuff,
)
from constrained_diffusion.eval.dllm.dataset import load_dataset
from constrained_diffusion.eval.dllm.model import load_model


def main(dataset_name="jsonschema", model_name="Dream-org/Dream-v0-Instruct-7B", device="cpu"):
    ds = load_dataset(dataset_name)
    inst = sorted(ds, key=lambda x: x.instance_id())[0]
    lang, lex_map, subtokens = inst.language_lex_subtokens()
    terminals = lang.get_terminals()

    eval_model = load_model(model_name)
    tokenizer = eval_model.tokenizer(device)
    compiled_lex = compile_lex_map(lex_map, subtokens=subtokens)

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

    print(f"Dataset: {dataset_name}")
    print(f"Model:   {model_name}")
    print(f"Grammar terminals ({len(terminals)}): {sorted(terminals)}")
    print(f"Vocab terminals   ({len(vocab_terms)}): {sorted(vocab_terms)}")
    print()

    diff = set(terminals) - vocab_terms
    if diff:
        print(f"Grammar terminals NOT covered by vocab: {sorted(diff)}")
        print("=> sigma_star and vocab modes WILL produce different results")
    else:
        print("All grammar terminals are covered by vocab.")
        print("=> sigma_star and vocab modes produce IDENTICAL FSAs")
        extra = vocab_terms - set(terminals)
        if extra:
            print(f"   (vocab also produces: {sorted(extra)})")


if __name__ == "__main__":
    fire.Fire(main)
