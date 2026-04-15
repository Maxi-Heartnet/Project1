"""tucasard.com site parser.

Thin wrapper around scraper.sites._domiclick — all parsing logic lives there.
Structurally identical to apartamentosrd.com.do (same Domiclick platform).
Selectors spiked on 2026-04-10.
"""
from scraper.sites._domiclick import scrape as _domiclick_scrape

TUCASARD_BASE = 'https://www.tucasard.com'


def scrape(max_pages: int = 150, browser=None) -> list:
    """Scrape tucasard.com and return a list of listing dicts."""
    return _domiclick_scrape(TUCASARD_BASE, max_pages=max_pages, browser=browser)
