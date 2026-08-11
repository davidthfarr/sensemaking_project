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
    # flag     : bool — True means off-topic
    # category : str  — slug of matched rule, or "" if on-topic
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# RUSSIA rules
# ─────────────────────────────────────────────────────────────────────────────

# Tigray / Eritrea / Ethiopia — Horn of Africa conflict entirely unrelated to
# the Russia-Ukraine war.  ~41 expected clusters.
_RUSSIA_TIGRAY = re.compile(
    r"""
    tigray | eritrea | ethiopia | addis\s*ababa | abiy\s*ahmed |
    amhara | afar | tplf | oromia | oromo
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Spam / account-recovery — bot-like clusters, unrelated to any conflict topic.
# ~29 expected clusters.
_RUSSIA_SPAM = re.compile(
    r"""
    @spikeqr | spike\s*qr |
    account\s*recov | recov\s*account |
    hack(ed|ing)?\s+(account|instagram|facebook|twitter|tiktok|telegram) |
    follow\s*back | f4f | l4l |
    dm\s*for\s*promo | onlyfans | cashapp | paypal\s*me |
    get\s+followers | buy\s+followers | gain\s+followers
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Domestic US political content — unrelated to Russia-Ukraine war.
# Switchable: set include_domestic_political=True to keep these for sensitivity analysis.
# ~20 expected clusters (previously matched ~0 due to insufficient patterns).
# Covers: Jan 6, Tucker Carlson, Griner, Ohio train derailment, US border/immigration
# debate (framed domestically), US gas prices, Bigg Boss, My Chemical Romance.
_RUSSIA_DOMESTIC_POLITICAL = re.compile(
    r"""
    # January 6 / Capitol attack
    jan(uary)?\s*6\b | january\s*sixth | capitol\s*(riot|attack|insurrection|breach) |
    stop\s*the\s*steal |

    # Tucker Carlson / Fox News hosts (not mentioning Ukraine/Russia)
    tucker\s*carlson | ingraham\s*angle | jesse\s*watters | sean\s*hannity\b |

    # Brittney Griner
    brittney\s*griner | \bgriner\b |

    # Ohio train derailment / East Palestine
    ohio\s*(train)?\s*derailment | east\s*palestine | norfolk\s*southern\s*(derail|spill) |

    # US domestic border / immigration framing (not refugee / Ukrainian context)
    \bus\s+border\s+(crisis|policy|security|wall)\b |
    migrant\s*(caravan|invasion|influx) |
    title\s*42\b |

    # US domestic energy / gas prices (not sanctions framing)
    gas\s*prices?\s*(at|in|under|rise|surging|spike) |
    price\s*at\s+the\s+pump |
    fuel\s*prices?\s+(in\s+the\s+us|domestic|american) |

    # US Supreme Court / Dobbs / abortion rights
    roe\s*v\.?\s*wade | dobbs\b | abortion\s*(ban|rights|ruling|overturn) |
    supreme\s+court\s+(overturns?|strikes?|rules?|decision) |

    # Ron DeSantis / Florida political storylines (US-domestic)
    \bdesantis\b | florida\s*governor\s*ron |

    # Gun control / school shootings
    uvalde | school\s*shooting | gun\s*control\s*(debate|legislation|bill) |
    second\s*amendment\s*(rights|debate) |

    # Student loan forgiveness
    student\s*loan\s*(forgiveness|cancel|debt\s*relief) |

    # Bigg Boss (Indian reality TV)
    bigg\s*boss |

    # My Chemical Romance (band, concert announcements)
    my\s*chemical\s*romance\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ─────────────────────────────────────────────────────────────────────────────
# VENEZUELA rules
# ─────────────────────────────────────────────────────────────────────────────

# Crypto / financial promotion and spam — not related to Venezuela political
# conflict.  ~14 expected clusters covering PURK ecosystem, XRP airdrop,
# Tether/USDT, WhatsApp/Telegram stock tips, AI model promotion, generic
# crypto-trading-advice content, and stock-market spam.
_VENEZUELA_CRYPTO_PROMO = re.compile(
    r"""
    # Named tokens / ecosystems
    \bPURK\b | \bXRP\b | ripple | \bUSDT\b | tether\s+(\w+\s+)?stable |

    # Airdrop / giveaway / free crypto
    (crypto|bitcoin|token)\s*(airdrop|giveaway|faucet) |
    free\s+crypto | earn\s+free\s+(crypto|bitcoin|token) |

    # WhatsApp / Telegram promotion channels
    whatsapp\s*(stock|invest|tip|signal|group|channel) |
    telegram\s*(stock|invest|signal|channel|group) |

    # Crypto trading advice / signals
    crypto\s*(trading\s+advice|trading\s+signal|market\s+insight|guidance\s+for\s+beginner) |
    (trading|invest)\s*(advice|signal|tip|community\s+support) |

    # AI model creation / promotion services (clearly off-topic)
    ai\s+model\s+(creation|management|services?) |
    promotion\s+of\s+ai\s+model |

    # Stock market / investment content linked to Venezuela (spam)
    stock\s+market\s+insight | market\s+observation.*?(stock|crypto) |

    # Generic investment spam
    guaranteed\s+(profit|return|gain) |
    invest\s+now\b | pump\s+and\s+dump |
    forex\s*(signal|tip|robot|ea\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ─────────────────────────────────────────────────────────────────────────────
# IRAN rules
# ─────────────────────────────────────────────────────────────────────────────

# Entertainment / sports content unrelated to the Iran conflict (Mahsa Amini
# protests and related government response).  Iran competed in FIFA 2022 and
# there were notable protest incidents at the tournament, so bare "FIFA" or
# "World Cup" is NOT flagged here — only unambiguously off-topic sports content.
# Low expected count; Iran corpus appears genuinely cleaner than Russia.
_IRAN_ENTERTAINMENT = re.compile(
    r"""
    # Indian cricket — unambiguously off-topic for Iran conflict
    \bIPL\b | indian\s+premier\s+league | \bBCCI\b |
    virat\s+kohli | rohit\s+sharma | ms\s+dhoni | sachin\s+tendulkar |

    # Bollywood / Indian cinema
    bollywood | \bSRK\b | shah\s+rukh\s+khan | salman\s+khan | deepika\s+padukone |
    karan\s+johar | dharma\s+production |

    # K-pop / Korean entertainment
    \bBTS\b | blackpink | exo\b | k[-\s]*pop | kpop |
    korean\s+(drama|series|entertainment|idol) |

    # Pakistani entertainment / drama (when not conflict-related)
    pakistani\s+(drama|celebrity|actor|showbiz)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Crypto / investment spam for Iran — flag only obvious spam signals, NOT
# content about Iran crypto sanctions evasion (which IS on-topic).
_IRAN_MARKETS_CRYPTO = re.compile(
    r"""
    # Trading signal / tip channels
    crypto\s+(signal|tip|alert|call)\b |
    (forex|fx)\s+(signal|tip|robot|ea\b) |

    # Airdrop / giveaway spam
    (crypto|token|nft)\s*(airdrop|giveaway|faucet) |
    free\s+crypto | earn\s+free\s+(crypto|token) |

    # Generic investment spam
    guaranteed\s+(profit|return|gain) |
    invest\s+now\b | pump\s+and\s+dump |

    # Specific platforms used as spam vectors
    \bBinance\s+(referral|promo|bonus)\b |
    copy\s+trading\s+(service|platform|bot)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rule table — order matters: first match wins.
# Expected counts are advisory; the notebook cell checks actuals against them.
# ─────────────────────────────────────────────────────────────────────────────

RULES = [
    # ── Russia ──────────────────────────────────────────────────────────────
    {
        "case": "russia",
        "category": "tigray_eritrea_ethiopia",
        "pattern": _RUSSIA_TIGRAY,
        "switchable": False,
        "expected_count": 41,
    },
    {
        "case": "russia",
        "category": "spam_recovery",
        "pattern": _RUSSIA_SPAM,
        "switchable": False,
        "expected_count": 29,
    },
    {
        "case": "russia",
        "category": "domestic_political",
        "pattern": _RUSSIA_DOMESTIC_POLITICAL,
        "switchable": True,   # gated by include_domestic_political flag
        "expected_count": 20,
    },
    # ── Venezuela ────────────────────────────────────────────────────────────
    {
        "case": "venezuela",
        "category": "crypto_promo",
        "pattern": _VENEZUELA_CRYPTO_PROMO,
        "switchable": False,
        "expected_count": 14,
    },
    # ── Iran ─────────────────────────────────────────────────────────────────
    {
        "case": "iran",
        "category": "entertainment_sports",
        "pattern": _IRAN_ENTERTAINMENT,
        "switchable": False,
        "expected_count": 3,   # low expected — Iran corpus appears cleaner
    },
    {
        "case": "iran",
        "category": "markets_crypto",
        "pattern": _IRAN_MARKETS_CRYPTO,
        "switchable": False,
        "expected_count": 5,   # low expected
    },
]

# ─────────────────────────────────────────────────────────────────────────────

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
        When True, domestic_political clusters are treated as ON-topic (kept in
        analysis).  Default False (exclude them).

    Returns
    -------
    (True, category_slug)  — cluster is off-topic
    (False, "")            — cluster is on-topic
    """
    for rule in RULES:
        if rule["case"] != "*" and rule["case"] != case:
            continue
        if rule["switchable"] and include_domestic_political:
            continue
        if rule["pattern"].search(theme_text or ""):
            return True, rule["category"]
    return False, ""


def apply_to_dataframe(
    df,
    case: str,
    theme_col: str = "theme",
    include_domestic_political: bool = False,
):
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
