# Scrapability Spike Results

**Date:** 2026-04-08  
**Tested:** 9 sites (corotos.com.do confirmed Phase 1 from prior implementation)  
**Result:** 6 Phase 1, 3 Phase 2, 0 Blocked

---

## Phase 1 Sites (requests + BeautifulSoup sufficient)

### supercasas.com
- **Listing URL:** `https://www.supercasas.com/buscar/?Tipo=2&PagingPageSkip=0`
- **Pagination:** `&PagingPageSkip=N` (N=0,1,2…)
- **Cards/page:** 18
- **Card selector:** `li.normal`
- **Price selector:** `.title3` — e.g. `US$ 140,000` or `RD$ 9,500,000`
- **Location selector:** `.title2` — sector name, e.g. `Piantini`
- **Anchor href:** `li.normal > a[href]` — relative path, e.g. `/apartamentos-venta-piantini/1385955/`
- **Base URL for canonical:** `https://www.supercasas.com`

### miscasasrd.com
- **Listing URL:** `https://www.miscasasrd.com/properties`
- **Pagination:** `/properties?page=N&web_page=properties` (~449 pages)
- **Cards/page:** 18
- **Card selector:** `li.property-listing.clearfix`
- **Price selector:** `span.listing-type-price` — e.g. `$420,000 USD` or `RD$ 5,900,000 DOP`
- **Location selector:** `data-popover-data` JSON attribute → `"location"` key; fallback: `.description`
- **Anchor href:** `a.related-property[href]` — relative path
- **Base URL for canonical:** `https://www.miscasasrd.com`
- **Note:** Runs on EasyBroker platform — same HTML as casaspb.com

### indominicana.com
- **Listing URL:** `https://indominicana.com/propiedades/venta/apartamentos`
- **Pagination:** `/propiedades.php?status=sale&type[0]=apartamentos&page_no=N` (~216 pages)
- **Cards/page:** 12
- **Card selector:** `div.property-container`
- **Price selector:** text within `div.property-container` matching `US$` or `RD$` pattern
- **Location selector:** `a` tags inside `.property-container` containing location text
- **Anchor href:** `div.property-container > a[href*="/propiedades/"]`
- **Base URL for canonical:** `https://indominicana.com`

### inmuebles.mercadolibre.com.do
- **Listing URL:** `https://inmuebles.mercadolibre.com.do/`
- **Pagination:** `https://inmuebles.mercadolibre.com.do/_Desde_N_NoIndex_True` (N=49,97,145…)
- **Cards/page:** 48 (highest volume)
- **Card selector:** `div.ui-search-result__wrapper`
- **Price selector (currency):** `span.andes-money-amount__currency-symbol`
- **Price selector (amount):** `span.andes-money-amount__fraction`
- **Location selector:** `span.poly-component__location`
- **Anchor href:** `a.poly-component__title[href]` — absolute URL
- **Note:** Largest inventory. Currency symbol separate from amount — parse together for full price.

### plusval.com.do
- **Listing URL:** `https://plusval.com.do/propiedades/venta`
- **Pagination:** `/propiedades/venta?page=N`
- **Cards/page:** 28
- **Card selector:** `li.featured-property`
- **Price selector:** `p.lead.font-pulpBold.text-primary-100` — e.g. `US$140,000`
- **Location selector:** `span.label.medium-label` (short name) or `h4.lead.font-pulpRegular` (full title)
- **Anchor href:** `a.property[href*="/propiedad/"]` — relative path
- **Base URL for canonical:** `https://plusval.com.do`
- **Note:** All USD pricing — no DOP listings observed.

### casaspb.com
- **Listing URL:** `https://www.casaspb.com/properties`
- **Pagination:** `/properties?page=N&web_page=properties` (~149 pages)
- **Cards/page:** 18
- **Card selector:** `div.thumbnail`
- **Price selector:** `span.listing-type-price` — e.g. `$625,000 USD` or `RD$ 56,480,000 DOP`
- **Location selector:** `div.caption > span` — e.g. `Apartamento en Playa Cosón, Las Terrenas`
- **Anchor href:** `a.related-property[href*="/property/"]` — relative path
- **Base URL for canonical:** `https://www.casaspb.com`
- **Note:** Runs on EasyBroker platform — same HTML structure as miscasasrd.com

---

## Phase 2 Sites (JS rendering required — deferred)

### remaxrd.com
- **Platform:** Next.js + React (Material UI)
- **Static body:** ~879 chars — loading placeholder only
- **Blocker:** Listings fetched fully client-side; no accessible REST API found
- **Recommendation:** Playwright required; target `/propiedades` after JS renders

### apartamentosrd.com.do
- **Platform:** Next.js (Domiclick platform, shared with tucasard.com)
- **Static body:** ~1,147 chars — search filter shell only
- **Blocker:** `__NEXT_DATA__` contains metadata only, no listings
- **Recommendation:** Playwright required; same platform as tucasard

### tucasard.com
- **Platform:** Next.js (Domiclick platform, shared with apartamentosrd.com.do)
- **Static body:** ~796 chars
- **Blocker:** Identical to apartamentosrd — same Domiclick backend
- **Recommendation:** One Playwright implementation would cover both sites

---

## Summary

| Site | Class | Cards/pg | Notes |
|---|---|---|---|
| corotos.com.do | Phase 1 | 30 | Existing implementation |
| supercasas.com | Phase 1 | 18 | |
| miscasasrd.com | Phase 1 | 18 | EasyBroker (same as casaspb) |
| indominicana.com | Phase 1 | 12 | 216 pages of apartments |
| inmuebles.mercadolibre.com.do | Phase 1 | 48 | Largest inventory |
| plusval.com.do | Phase 1 | 28 | USD only |
| casaspb.com | Phase 1 | 18 | EasyBroker (same as miscasasrd) |
| remaxrd.com | Phase 2 | — | Next.js, JS-rendered |
| apartamentosrd.com.do | Phase 2 | — | Domiclick platform |
| tucasard.com | Phase 2 | — | Domiclick platform (same as above) |

**Implementation order for Unit 5 (cap at 5):**
1. supercasas.com — clean selectors, straightforward
2. miscasasrd.com — EasyBroker (covers casaspb pattern for free later)
3. inmuebles.mercadolibre.com.do — largest inventory, highest value
4. plusval.com.do — all USD, simpler price parsing
5. indominicana.com — largest catalog depth
6. casaspb.com — deferred (EasyBroker clone of miscasasrd; trivial once miscasasrd done)

---

## Phase 2 — Playwright Spike (2026-04-10)

All three Phase 2 sites confirmed renderable with `wait_until='networkidle'` (resolved within 30 s in testing). Playwright `sync_api` with `page.content()` → BeautifulSoup is the confirmed pattern.

### remaxrd.com

- **Listing URL:** `https://www.remaxrd.com/propiedades`
- **Pagination:** None — single page only. URL params `?page=2` return the same 16 cards; stop after page 1.
- **Cards/page:** 16
- **Card selector:** `a[target="_blank"][href*="/propiedad/"]`
- **Source URL:** relative `/propiedad/...` → prepend `https://www.remaxrd.com`
- **Property type:** `h3` text within card
- **Sector:** `span` sibling immediately after `h3` — e.g. `"ENSANCHE NACO, SANTO DOMINGO DE GUZMÁN"` → first comma-segment (lowercased to title case)
- **Price:** first `<span>` in the price container containing `"US$"` — format `"US$565,000"` (parse with `parse_price_usd`)
- **Bedrooms:** `<li>` containing `img[src*="icon_bed_remaxrd.svg"]` — text is the bare integer
- **Area:** `<li>` containing `img[src*="icon_rule_remaxrd.svg"]` — inner `<span>` text e.g. `"233.60 M2"` (use `get_text(strip=True)`)
- **networkidle:** Confirmed reliable within 30 s
- **Note:** CSS-in-JS class names (`sc-*`, `css-*`) are unstable across deploys. All selectors above use stable semantic attributes only.

### apartamentosrd.com.do

- **Base URL:** `https://www.apartamentosrd.com.do`
- **Listing URL:** `https://www.apartamentosrd.com.do/propiedades?listing_type=1&page={page}`
- **Pagination:** `?listing_type=1&page=N`; stop when page yields no `div.card.h-100` cards
- **Cards/page:** 30
- **Card selector:** `div.card.h-100`
- **Anchor:** `a[href*="/propiedad/"]` inside card — relative href, prepend base URL
- **Property type:** `<h5>` inside `div.property-content`
- **Sector:** `<span>` containing `<i class="fas fa-map-marker-alt">` — e.g. `"Serrallés, Santo Domingo D.N."` → first comma-segment
- **Price:** `<span>` in the price `<ul>` — `"US$ 165,000"` or `"RD$ X,XXX,XXX"` (skip DOP via `parse_price_usd`)
- **Bedrooms:** `<li>` with `<i class="fas fa-bed">` — e.g. `"1 Hab."` or `"Desde 1 hasta 3 Hab."` → extract first integer
- **Area:** `<li>` with `<i class="fas fa-arrows-alt">` — e.g. `"58 Mt2"` (use `_parse_area` with `mt2` pattern)
- **networkidle:** Confirmed reliable within 30 s

### tucasard.com

- **Base URL:** `https://www.tucasard.com`
- **Structural identity:** 100% identical to apartamentosrd.com.do (same Domiclick platform)
- **All selectors, pagination, and field patterns:** identical to apartamentosrd above
- **networkidle:** Confirmed reliable within 30 s
- **Implementation:** Single `_domiclick.py` helper parameterised by `base_url` covers both sites
