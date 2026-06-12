#!/usr/bin/env python3
"""
printer.py — Bluetooth ESC/POS receipt printer interface.

Targets a PT-210 thermal receipt printer over a Bluetooth RFCOMM
(serial port profile) connection. Requires the printer to be paired
with the host already (e.g. via `bluetoothctl`).
"""

import socket
import textwrap

PRINTER_MAC = "86:67:7A:52:31:A8"
PRINTER_PORT = 1  # standard RFCOMM channel for SPP

ESC = b"\x1b"
GS = b"\x1d"

INIT = ESC + b"@"        # reset printer state
CUT = GS + b"V" + b"\x00"  # partial/full cut, ignored if unsupported

SIZE_NORMAL = GS + b"!" + b"\x00"  # normal width/height

JUSTIFY_LEFT = ESC + b"a" + b"\x00"
JUSTIFY_CENTER = ESC + b"a" + b"\x01"
JUSTIFY_RIGHT = ESC + b"a" + b"\x02"

# Characters per line at normal size on this printer. Tuned from observed
# wrapping behavior rather than the nominal Font A spec.
CHARS_PER_LINE = 16
MAX_SIZE_MULT = 8  # GS ! supports width/height multipliers 1-8


def wrap_text(text, width=CHARS_PER_LINE):
    """Word-wrap text to width, never breaking a word across lines."""
    out = []
    for line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(line, width=width, break_long_words=False,
                                 break_on_hyphens=False) or [""]
        out.extend(wrapped)
    return out


def fit_haiku(lines, chars_per_line=CHARS_PER_LINE, max_mult=MAX_SIZE_MULT):
    """Pick the largest size multiplier that keeps every haiku line on one
    printed line, word-wrapping (without breaking words) if even the
    smallest size can't fit a line."""
    for mult in range(max_mult, 0, -1):
        width = max(1, chars_per_line // mult)
        wrapped = [wrap_text(line, width) for line in lines]
        if all(len(w) == 1 for w in wrapped):
            return mult, [w[0] for w in wrapped]
    width = max(1, chars_per_line)
    flat = [l for line in lines for l in wrap_text(line, width)]
    return 1, flat


class ReceiptPrinter:
    def __init__(self, mac=PRINTER_MAC, port=PRINTER_PORT):
        self.mac = mac
        self.port = port
        self.sock = None

    def connect(self, timeout=10):
        self.sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
        )
        self.sock.settimeout(timeout)
        self.sock.connect((self.mac, self.port))

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        self.sock.sendall(data)

    def init(self):
        self.write(INIT)

    def print_text(self, text):
        self.write(text)
        if not text.endswith("\n"):
            self.write(b"\n")

    def set_size(self, mult=1):
        mult = max(1, min(MAX_SIZE_MULT, mult))
        n = ((mult - 1) << 4) | (mult - 1)  # same width/height multiplier
        self.write(GS + b"!" + bytes([n]))

    def justify(self, mode="left"):
        self.write({"left": JUSTIFY_LEFT, "center": JUSTIFY_CENTER,
                     "right": JUSTIFY_RIGHT}[mode])

    def feed(self, lines=3):
        self.write(b"\n" * lines)

    def cut(self):
        try:
            self.write(CUT)
        except OSError:
            pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()
