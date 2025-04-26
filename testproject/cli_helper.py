# -----------------------------------------------------------------------------
# CLI helper – allow quick actions when run directly
#!/usr/bin/env python3
"""
cli_helper.py
A tiny command-line front-end for common Happyscore tasks.
Run   python cli_helper.py --help   to see the flags.
"""
import argparse, sys

# ── Adjust these imports only if you nested things in a sub-package ──
import data_utils              # provides download_fer2013()
import train                   # provides train_scoring_model()

p = argparse.ArgumentParser(
    description="Convenience wrapper for Happyscore utilities"
)
p.add_argument('--download-data', action='store_true',
               help="Fetch FER2013 with the Kaggle API")
p.add_argument('--train', action='store_true',
               help="Fine-tune the ResNet-based happiness scorer")
args = p.parse_args()

if args.download_data:
    data_utils.download_fer2013()
    sys.exit(0)

if args.train:
    train.train_scoring_model()
    sys.exit(0)

p.print_help()
