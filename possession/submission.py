"""Write and check the possession file -- the only thing that is scored."""

import hashlib

from .contract import LETTERS
from .scoring import pm


def write(path, codes):
    """frame,label CSV with '\\n' line endings, byte-identical for identical codes."""
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, LETTERS[code]) for frame, code in enumerate(codes))
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path, n_expected):
    """None if the file passes the official checks, else the scorer's one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_expected)
