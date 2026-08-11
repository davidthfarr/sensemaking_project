"""
Off-topic classification rules for the conflict IE pipeline.

Each rule is a dict with:
  case       : case name string ("venezuela" | "iran" | "russia") or "*" for all
  category   : short slug used in offtopic_flagged.csv and STATS
  pattern    : compiled re.Pattern (case-insensitive) matched against cluster theme text
  switchable : if True, inclusion is gated by the INCLUDE_DOMESTIC_POLITICAL flag
               (only used for russia/domestic_political)

Usage
-----
    from analysis.offtopic_rules import classify_cluster

    flag, category = classify_cluster("russia", theme_text,
                                      include_domestic_political=False)
    # flag   : bool  — True means off-topic
    # category : str — slug of matched rule, or "" if on-topic
"""

import re
from typing import Optional

# ── Russia: Tigray / Eritrea / Ethiopia off-topic clusters ────────────────────
_RUSSIA_TIGRAY = re.compile(
    r"""
    tigray | eritrea | ethiopia | addis\s*ababa | abiy\s*ahmed |
    amhara | afar | tplf | oromia | oromo
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Russia: spam / account-recovery clusters ─────────────────────────────────
_RUSSIA_SPAM = re.compile(
    r"""
    @spikeqr | spike\s*qr | account\s*recover | recover\s*account |
    hack(ed|ing)?\s+(account|instagram|facebook|twitter|tiktok) |
    follow\s*back | f4f | l4l |
    dm\s*for\s*promo | onlyfans | cashapp | paypal\s*me
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Russia: domestic US political content (switchable flag) ──────────────────
# Jan 6, Tucker Carlson, Brittney Griner, Ohio train derailment,
# Bigg Boss (Indian reality show), My Chemical Romance (unrelated concerts)
_RUSSIA_DOMESTIC_POLITICAL = re.compile(
    r"""
    jan(uary)?\s*6 | january\s*sixth | capitol\s*(riot|attack|insurrection) |
    tucker\s*carlson | fox\s*news\s*host |
    brittney\s*griner | griner |
    ohio\s*(train)?\s*derailment | east\s*palestine |
    bigg\s*boss |
    my\s*chemical\s*romance\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Venezuela: crypto / financial spam clusters ───────────────────────────────
_VENEZUELA_CRYPTO = re.compile(
    r"""
    \bPURK\b | \bXRP\b | ripple |
    whatsapp\s*(stock|invest|tip|signal) |
    crypto\s*(signal|tip|alert) |
    bitcoin\s*(giveaway|airdrop) |
    forex\s*(signal|tip) |
    pump\s*and\s*dump |
    invest\s+now | guaranteed\s+(profit|return)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Rule table ────────────────────────────────────────────────────────────────
# Order matters: first match wins.

RULES = [
    {
        "case": "russia",
        "category": "tigray_eritrea_ethiopia",
        "pattern": _RUSSIA_TIGRAY,
        "switchable": False,
    },
    {
        "case": "russia",
        "category": "spam_recovery",
        "pattern": _RUSSIA_SPAM,
        "switchable": False,
    },
    {
        "case": "russia",
        "category": "domestic_political",
        "pattern": _RUSSIA_DOMESTIC_POLITICAL,
        "switchable": True,  # gated by include_domestic_political flag
    },
    {
        "case": "venezuela",
        "category": "crypto_spam",
        "pattern": _VENEZUELA_CRYPTO,
        "switchable": False,
    },
]


def classify_cluster(
    case: str,
    theme_text: str,
    include_domestic_political: bool = False,
) -> tuple[bool, str]:
    """
    Returns (is_offtopic, category_slug).

    Parameters
    ----------
    case : str
        Dataset name (lowercase): "venezuela", "iran", "russia".
    theme_text : str
        Cluster theme label to match against.
    include_domestic_political : bool
        When True, domestic_political clusters are treated as ON-topic.
        Default False (exclude them from analysis).

    Returns
    -------
    (True, category_slug)  — cluster is off-topic
    (False, "")            — cluster is on-topic
    """
    for rule in RULES:
        if rule["case"] != "*" and rule["case"] != case:
            continue
        if rule["switchable"] and include_domestic_political:
            # flag says keep domestic political → treat as on-topic
            continue
        if rule["pattern"].search(theme_text or ""):
            return True, rule["category"]
    return False, ""


def apply_to_dataframe(
    df,
    case: str,
    theme_col: str = "theme",
    include_domestic_political: bool = False,
) -> "pd.DataFrame":
    """
    Adds `on_topic` (bool) and `offtopic_category` (str) columns to *df* in-place.
    Returns the modified DataFrame for chaining.
    """
    results = df[theme_col].fillna("").apply(
        lambda t: classify_cluster(case, t, include_domestic_political)
    )
    df["on_topic"] = [not r[0] for r in results]
    df["offtopic_category"] = [r[1] for r in results]
    return df
