"""Entry point: `canary <command>`."""

import argparse

from canary.commands import build, evaluate, metadata, peek, sample
from canary.models import seq, tabular, text

COMMANDS = {
    "metadata": (metadata, "pull label columns for all runs and print dataset stats"),
    "sample": (sample, "build a stratified working sample with transcripts"),
    "peek": (peek, "print one trajectory as a readable transcript"),
    "build": (build, "parse a scaffold's runs into turn, feature and split tables"),
    "tabular": (tabular, "train logistic regression and LightGBM on prefix features"),
    "seq": (seq, "train the GRU monitor on per-turn vectors"),
    "text": (text, "fine-tune ModernBERT on transcript prefixes"),
    "evaluate": (evaluate, "score predictions and rule baselines, simulate abort policies"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="canary", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (mod, help_) in COMMANDS.items():
        p = sub.add_parser(name, help=help_, description=mod.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(p)
        p.set_defaults(run=mod.run)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
