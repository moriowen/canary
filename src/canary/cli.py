"""Entry point: `canary <command>`."""

import argparse
import importlib
import sys

# Imported lazily so `canary peek` doesn't need torch or lightgbm installed.
COMMANDS = {
    "metadata": ("canary.commands.metadata", "pull label columns for all runs and print dataset stats"),
    "sample": ("canary.commands.sample", "build a stratified working sample with transcripts"),
    "peek": ("canary.commands.peek", "print one trajectory as a readable transcript"),
    "build": ("canary.commands.build", "parse a scaffold's runs into turn, feature and split tables"),
    "tabular": ("canary.models.tabular", "train logistic regression and LightGBM on prefix features"),
    "seq": ("canary.models.seq", "train the GRU monitor on per-turn vectors"),
    "text": ("canary.models.text", "fine-tune ModernBERT on transcript prefixes"),
    "taxonomy": ("canary.taxonomy", "LLM-diagnose quiet failures, cluster and name failure modes"),
    "evaluate": ("canary.commands.evaluate", "score predictions and rule baselines, simulate abort policies"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="canary", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    chosen = next((a for a in argv if a in COMMANDS), None)
    for name, (path, help_) in COMMANDS.items():
        p = sub.add_parser(name, help=help_, formatter_class=argparse.RawDescriptionHelpFormatter)
        if name == chosen:
            mod = importlib.import_module(path)
            p.description = mod.__doc__
            mod.add_args(p)
            p.set_defaults(run=mod.run)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
