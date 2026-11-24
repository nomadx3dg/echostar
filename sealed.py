"""Seal a file under a short code, using only the standard library.

Used for the hidden extra opponent: the model ships as `data/echo-extra.bin`, which is unreadable noise until the right
code is typed on the page. This is obfuscation for an easter egg, not strong security: the code is short, so anyone
determined can guess it offline. Its job is to keep the model from being found and loaded by browsing the repository.
"""
import hashlib
import hmac
import os

MAGIC = b"ECHOSEAL1"
ITERATIONS = 60_000


def normalize(code):
    """Case-insensitive, letters and digits only: the same rule the page uses when you type the code."""
    return "".join(ch for ch in str(code).lower() if ch.isalnum())


def _key(code, salt):
    return hashlib.pbkdf2_hmac("sha256", normalize(code).encode(), salt, ITERATIONS, dklen=32)


def _xor(data, key, salt):
    stream = hashlib.shake_256(b"stream" + key + salt).digest(len(data))
    return bytes(a ^ b for a, b in zip(data, stream))


def seal(data, code):
    salt = os.urandom(16)
    key = _key(code, salt)
    body = _xor(data, key, salt)
    tag = hmac.new(key, salt + body, hashlib.sha256).digest()
    return MAGIC + salt + tag + body


def unseal(blob, code):
    """The original bytes, or None if `code` is wrong or the blob is damaged."""
    head = len(MAGIC) + 16 + 32
    if len(blob) < head or not blob.startswith(MAGIC):
        return None
    salt = blob[len(MAGIC):len(MAGIC) + 16]
    tag, body = blob[len(MAGIC) + 16:head], blob[head:]
    key = _key(code, salt)
    if not hmac.compare_digest(tag, hmac.new(key, salt + body, hashlib.sha256).digest()):
        return None
    return _xor(body, key, salt)


def unseal_suffixes(blob, typed, lo=4, hi=24):
    """Try each tail of what was typed (the page sends the last keys pressed, which may have noise in front)."""
    text = normalize(typed)[-hi:]
    for n in range(lo, len(text) + 1):
        out = unseal(blob, text[-n:])
        if out is not None:
            return out
    return None
