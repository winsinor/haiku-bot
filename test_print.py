#!/usr/bin/env python3
"""
test_print.py — sends a short test page with random words to the
Bluetooth receipt printer, to verify connectivity and printing.
"""

import random
import sys

from printer import PRINTER_MAC, ReceiptPrinter

WORDS = [
    "cedar", "ember", "harbor", "lantern", "meadow", "ripple", "summit",
    "willow", "amber", "comet", "drift", "frost", "glow", "hush", "ink",
    "jade", "kite", "lullaby", "mist", "nectar", "orbit", "pebble",
    "quiet", "river", "shadow", "tide", "umbra", "vapor", "whisper",
    "zephyr",
]


def random_words(n=5):
    return " ".join(random.choice(WORDS) for _ in range(n))


def main():
    lines = [
        "=== HAIKU BOT TEST PRINT ===",
        "",
        random_words(),
        random_words(),
        random_words(),
        "",
        "=== END TEST ===",
    ]
    text = "\n".join(lines)
    print(text)

    try:
        print(f"Connecting to printer {PRINTER_MAC}...")
        with ReceiptPrinter() as printer:
            print("Connected. Sending print job...")
            printer.print_text(text)
            printer.feed()
            printer.cut()
            print("Print job sent successfully.")
    except OSError as e:
        print(f"Failed to print: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
