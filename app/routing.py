"""Fixed category-to-team table and the consistency rules the harness checks."""

from __future__ import annotations

from app.schemas import Category, Team

CATEGORY_TO_TEAM: dict[str, str] = {
    category.value: team.value
    for category, team in (
        (Category.payment_refund, Team.Payments___Refunds),
        (Category.ride_trip_issue, Team.Ride_Operations),
        (Category.lost_item, Team.Lost___Found),
        (Category.order_missing_wrong, Team.Food_Operations),
        (Category.delivery_delay, Team.Delivery_Operations),
        (Category.food_quality, Team.Restaurant_Quality),
        (Category.account_promo, Team.Account_Services),
        (Category.safety_conduct, Team.Trust___Safety),
        (Category.app_technical, Team.Tech_Support),
        (Category.general_inquiry, Team.Front_line_Support),
        (Category.spam_irrelevant, Team.Auto_close___Spam_Filter),
    )
}
assert set(CATEGORY_TO_TEAM) == {category.value for category in Category}

# Tickets below this confidence get needs_human_review=true (optional bonus field).
REVIEW_THRESHOLD = 0.45


def finalize_prediction(ticket: dict, raw: dict, model_version: str) -> dict:
    """Turn a model guess into a response the schema accepts.

    Team is looked up, never predicted. Spam is never urgent and has no second
    category. A second category equal to the first is dropped. Anything the
    model returns outside the enum falls back to general_inquiry.
    """
    category = raw.get("category")
    if category not in CATEGORY_TO_TEAM:
        category = "general_inquiry"

    secondary = raw.get("secondary_category")
    if secondary not in CATEGORY_TO_TEAM or secondary == category:
        secondary = None

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence != confidence:  # NaN
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    is_urgent = bool(raw.get("is_urgent", False))
    if category == "spam_irrelevant":
        secondary = None
        is_urgent = False

    body = {
        "category": category,
        "secondary_category": secondary,
        "team": CATEGORY_TO_TEAM[category],
        "is_urgent": is_urgent,
        "confidence": round(confidence, 4),
        "model_version": model_version,
        "needs_human_review": confidence < REVIEW_THRESHOLD,
    }
    ticket_id = ticket.get("ticket_id")
    if isinstance(ticket_id, str):
        body = {"ticket_id": ticket_id, **body}
    return body
