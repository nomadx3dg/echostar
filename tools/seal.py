"""Seal a trained profile for the hidden opponent.

    python tools/seal.py profiles/fischer.npz data/echo-extra.bin

The code is read from the prompt, so it never lands in your shell history. Anyone with the sealed file and the code can
unseal it; see sealed.py for what that does and does not protect.
"""
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sealed  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    code = getpass.getpass("Code: ")
    if len(sealed.normalize(code)) < 4:
        sys.exit("The code needs at least 4 letters or digits.")
    with open(sys.argv[1], "rb") as fh:
        data = fh.read()
    with open(sys.argv[2], "wb") as fh:
        fh.write(sealed.seal(data, code))
    print(f"Wrote {sys.argv[2]} ({len(data)} bytes sealed).")
