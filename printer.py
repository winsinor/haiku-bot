#!/usr/bin/env python3
"""
printer.py — Bluetooth ESC/POS receipt printer interface.

Targets a PT-210 thermal receipt printer over a Bluetooth RFCOMM
(serial port profile) connection. Requires the printer to be paired
with the host already (e.g. via `bluetoothctl`).
"""

import socket

PRINTER_MAC = "86:67:7A:52:31:A8"
PRINTER_PORT = 1  # standard RFCOMM channel for SPP

ESC = b"\x1b"
GS = b"\x1d"

INIT = ESC + b"@"        # reset printer state
CUT = GS + b"V" + b"\x00"  # partial/full cut, ignored if unsupported

SIZE_NORMAL = GS + b"!" + b"\x00"  # normal width/height
SIZE_DOUBLE = GS + b"!" + b"\x11"  # double width + double height


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

    def set_size(self, double=False):
        self.write(SIZE_DOUBLE if double else SIZE_NORMAL)

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
