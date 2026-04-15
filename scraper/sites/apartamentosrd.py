"""apartamentosrd.com.do site parser.

Thin wrapper around scraper.sites._domiclick — all parsing logic lives there.
Selectors spiked on 2026-04-10.
"""
from scraper.sites._domiclick import scrape as _domiclick_scrape

APARTAMENTOSRD_BASE = 'https://www.apartamentosrd.com.do'


def scrape(max_pages: int = 150, browser=None) -> list:
    """Scrape apartamentosrd.com.do and return a list of listing dicts."""
    return _domiclick_scrape(APARTAMENTOSRD_BASE, max_pages=max_pages, browser=browser)
