"""Text cleaning shared by training (notebooks) and serving (API).

Keep this file tiny and dependency-free: the pickled model refers to
`app.textproc.clean_text` by name, so training and the container must use the
exact same file.
"""
import re
import unicodedata

# Reference codes such as "SIMQ-V9LU2KD" or "simb-gk5c6rl" carry no signal.
_REF = re.compile(r"\b[a-z]{3,5}-[a-z0-9]{6,8}\b", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def clean_text(s) -> str:
    """Normalise one ticket string. Never raises; keeps Sinhala/Tamil/emoji intact."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u200d", "").replace("\u200c", "")
    s = s.lower()
    s = _REF.sub(" refcode ", s)
    s = _DIGITS.sub("0", s)  # keep "a number is here", drop the exact value
    return _SPACES.sub(" ", s).strip()


def build_input(subject, text) -> str:
    """Combine subject (emails only) and body into the single string the model sees."""
    subject = "" if subject is None else str(subject)
    text = "" if text is None else str(text)
    return f"{subject}\n{text}" if subject.strip() else text