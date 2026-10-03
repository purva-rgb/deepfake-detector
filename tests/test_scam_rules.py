import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scam import load_rules, scan, score

RULES = load_rules()


def cats(text):
    return {h.category for h in scan(text, RULES)}


def test_rules_file_shape():
    by_cat = {}
    for r in RULES.rules:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + len(r["patterns"])
    assert set(by_cat) == set(RULES.categories) and len(by_cat) == 6
    assert all(n >= 6 for n in by_cat.values()), by_cat


def test_never_share_otp_does_not_match():
    assert "payment_credentials" not in cats("Never share your OTP with anyone")
    hits = scan("Never share your OTP with anyone", RULES, include_suppressed=True)
    assert hits and all(h.suppressed for h in hits)  # it WAS seen, then suppressed by the guard


@pytest.mark.parametrize("t", [
    "Do not share your OTP with anyone.",
    "Beware of callers who ask you to send your OTP.",
    "Please avoid anyone who says send your card number.",
    "We don't ask you to share your password.",
])
def test_warnings_are_suppressed(t):
    assert "payment_credentials" not in cats(t)


def test_send_otp_now_matches():
    assert "payment_credentials" in cats("send your OTP now")
    assert "payment_credentials" in cats("Beware! Send your OTP now to claim the prize.")  # warning in a previous sentence only


def test_whatsapp_support_not_redirect():
    assert "off_platform_redirect" not in cats("WhatsApp us for support")
    assert "off_platform_redirect" not in cats("Message us on WhatsApp for customer care.")


def test_real_redirect_matches():
    assert "off_platform_redirect" in cats("WhatsApp me to claim your reward")
    assert "off_platform_redirect" in cats("Join our private Telegram group")


def test_all_six_categories_can_fire():
    samples = {
        "financial_promise": "Guaranteed returns on every deposit",
        "urgency": "Act now before it is too late",
        "authority_impersonation": "This is an officer from the income tax department, a notice has been issued",
        "payment_credentials": "Complete your KYC today",
        "off_platform_redirect": "Join our Telegram group",
        "fake_branding": "Amazon lucky draw winners announced",
    }
    for c, t in samples.items():
        assert c in cats(t), (c, t)


def test_benign_text_has_no_hits():
    for t in ["Save twenty percent on running shoes this weekend.",
              "Thanks for watching, please subscribe to our channel.",
              "Call our customer care for any support."]:
        assert scan(t, RULES) == [], t
        assert score(scan(t, RULES))["S"] == 0.0


def test_scoring_formula_hand_computed():
    # guaranteed returns = 0.85 (financial_promise), act now = 0.55 (urgency)
    # S_max = 0.85, N_categories = 2 -> 0.85 + 0.10 * (2 - 1) = 0.95
    s = score(scan("Guaranteed returns, act now", RULES))
    assert s["n_categories"] == 2
    assert s["S_max"] == pytest.approx(0.85)
    assert s["S"] == pytest.approx(0.95)
    # single category: no bonus
    assert score(scan("Guaranteed returns", RULES))["S"] == pytest.approx(0.85)
    # three categories, capped at 1.0: 0.85 + 0.2 = 1.05 -> 1.0
    s3 = score(scan("Guaranteed returns! Act now and send your OTP.", RULES))
    assert s3["n_categories"] == 3 and s3["S"] == pytest.approx(1.0)
