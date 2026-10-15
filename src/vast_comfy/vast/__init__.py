"""Vast.ai REST API adapters (offers, instances, SSH keys, logs)."""

from .client import VastClient
from .offers import build_search_query, evaluate_offer, filter_offers, quote_from_offer, select_best
from .rental import build_create_body, create_instance, wait_for_running

__all__ = [
    "VastClient",
    "build_search_query",
    "evaluate_offer",
    "filter_offers",
    "quote_from_offer",
    "select_best",
    "build_create_body",
    "create_instance",
    "wait_for_running",
]
