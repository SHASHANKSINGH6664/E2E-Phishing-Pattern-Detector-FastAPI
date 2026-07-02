import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlparse

SENSITIVE_KEYWORDS = {
    "secure", "update", "validation", "logon", "login", "appleid", "verify",
    "account", "banking", "confirm", "signin", "webscr", "password",
}
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly", "ow.ly",
    "rebrand.ly", "cutt.ly", "shorte.st",
}

DEFAULT_TLD_LEGIT_PROB = 0.05

# REMOVED: "is_https" and "url_char_prob" to prevent dataset bias/leakage
FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "is_domain_ip",
    "tld_length",
    "subdomain_count",
    "has_obfuscation",
    "obfuscation_ratio",
    "letter_ratio_url",
    "digit_ratio_url",
    "equals_count",
    "qmark_count",
    "ampersand_count",
    "other_special_char_count",
    "special_char_ratio",
    "char_continuation_rate",
    "tld_legitimate_prob",
    "url_entropy",
    "has_sensitive_keyword_or_shortener",
]

_ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")
_HEX_OBFUSCATION_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def calculate_entropy(text: str) -> float:
    if not text:
        return 0.0
    counter = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counter.values())


def _char_continuation_rate(text: str) -> float:
    if len(text) < 2:
        return 1.0

    def char_class(c):
        if c.isalpha():
            return "L"
        if c.isdigit():
            return "D"
        return "S"

    classes = [char_class(c) for c in text]
    same = sum(1 for a, b in zip(classes, classes[1:]) if a == b)
    return same / (len(text) - 1)


def _obfuscation_count(text: str) -> int:
    return len(_HEX_OBFUSCATION_RE.findall(text))


def extract_features(url: str, lookup_tables: dict | None = None) -> dict:
    if not isinstance(url, str):
        url = str(url) if url is not None else ""

    url_lower = url.strip().lower()

    # NORMALIZATION: Strip http://, https://, and www. to prevent structural bias
    norm_url = re.sub(r"^(https?://)?(www\.)?", "", url_lower)

    # Parse the normalized URL by temporarily prepending http:// 
    # to trick urlparse into correctly identifying the hostname and path
    try:
        parsed = urlparse("http://" + norm_url)
        domain = parsed.hostname or ""
    except ValueError:
        domain = ""

    # IP CHECK
    is_ip = False
    if domain:
        try:
            ipaddress.ip_address(domain)
            is_ip = True
        except ValueError:
            pass

    domain_parts = domain.split(".") if domain else []
    tld = domain_parts[-1] if len(domain_parts) > 1 and not is_ip else ""

    # SUBDOMAIN COUNT: Now accurate because 'www.' is gone
    subdomain_count = max(0, len(domain_parts) - 2) if not is_ip else 0

    # CHARACTER COMPOSITION: Analyze the normalized URL to prevent length bias
    n = len(norm_url)
    n_letters = sum(c.isalpha() for c in norm_url)
    n_digits = sum(c.isdigit() for c in norm_url)
    n_equals = norm_url.count("=")
    n_qmark = norm_url.count("?")
    n_amp = norm_url.count("&")
    n_alnum = len(_ALPHANUM_RE.findall(norm_url))
    n_other_special = n - n_alnum

    obf_count = _obfuscation_count(norm_url)

    lookup_tables = lookup_tables or {}
    tld_legit_prob = lookup_tables.get("tld_legit_prob", {}).get(tld, DEFAULT_TLD_LEGIT_PROB)

    has_kw_or_shortener = int(
        any(kw in norm_url for kw in SENSITIVE_KEYWORDS)
        or any(s in domain for s in SHORTENERS)
    )

    return {
        "url_length": n,
        "domain_length": len(domain),
        "is_domain_ip": int(is_ip),
        "tld_length": len(tld),
        "subdomain_count": subdomain_count,
        "has_obfuscation": int(obf_count > 0),
        "obfuscation_ratio": obf_count / n if n else 0.0,
        "letter_ratio_url": n_letters / n if n else 0.0,
        "digit_ratio_url": n_digits / n if n else 0.0,
        "equals_count": n_equals,
        "qmark_count": n_qmark,
        "ampersand_count": n_amp,
        "other_special_char_count": n_other_special,
        "special_char_ratio": n_other_special / n if n else 0.0,
        "char_continuation_rate": _char_continuation_rate(norm_url),
        "tld_legitimate_prob": tld_legit_prob,
        "url_entropy": calculate_entropy(norm_url),
        "has_sensitive_keyword_or_shortener": has_kw_or_shortener,
    }