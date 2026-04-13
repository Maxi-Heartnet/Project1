/* map.js — bidirectional Google Maps ↔ sector form integration.
 *
 * Globals expected (injected by server into page HTML):
 *   window.SECTOR_COORDS  — { "Piantini": { lat, lng }, ... }
 *   window.MAPS_MAP_ID    — Google Cloud Map ID string
 *
 * Globals exposed (called by main.js fillRandom / clearForm):
 *   window.updateMapForSector(name)
 *   window.resetMap()
 *
 * window.initMap  — Maps SDK async defer callback. Must be a global BEFORE the
 * SDK script executes (this file is loaded before the SDK script tag).
 *
 * window.gm_authFailure — assigned at TOP LEVEL (outside initMap) so it is
 * registered before the SDK initialises and can catch auth failures during
 * SDK load itself.
 */

(function () {
  'use strict';

  var map = null;
  var markerMap = {};         // { sectorName: AdvancedMarkerElement }
  var selectedSector = null;  // currently highlighted sector name

  var DEFAULT_CENTER = { lat: 18.486, lng: -69.931 };
  var DEFAULT_ZOOM   = 11;
  var ZOOM_SELECTED  = 14;

  // ---- DOM helpers --------------------------------------------------------

  function getMapContainer() {
    return document.getElementById('map-container');
  }

  function getSectorInput() {
    return document.getElementById('sector');
  }

  function showMapError() {
    var container = getMapContainer();
    var fallback  = document.getElementById('map-fallback');
    if (container) container.style.display = 'none';
    if (fallback)  fallback.hidden = false;
  }

  // ---- Pin content factories ----------------------------------------------

  function makePinContent(selected) {
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'width:44px;height:44px;display:flex;align-items:center;justify-content:center;cursor:pointer;';

    var circle = document.createElement('div');
    if (selected) {
      circle.style.cssText = 'width:18px;height:18px;border-radius:50%;background:#c0392b;box-shadow:0 1px 3px rgba(0,0,0,.4);';
    } else {
      circle.style.cssText = 'width:12px;height:12px;border-radius:50%;background:#1a6fbd;box-shadow:0 1px 3px rgba(0,0,0,.3);';
    }
    wrapper.appendChild(circle);
    return wrapper;
  }

  // ---- Map state ----------------------------------------------------------

  function unhighlightAll() {
    Object.keys(markerMap).forEach(function (name) {
      markerMap[name].content = makePinContent(false);
    });
    selectedSector = null;
  }

  function highlightMarker(name) {
    if (!markerMap[name]) return;
    if (selectedSector && selectedSector !== name && markerMap[selectedSector]) {
      markerMap[selectedSector].content = makePinContent(false);
    }
    markerMap[name].content = makePinContent(true);
    selectedSector = name;
  }

  // ---- Public API (called by main.js) ------------------------------------

  window.updateMapForSector = function (name) {
    if (!map) return;
    // No-op if the same sector is already selected (R8)
    if (name && name === selectedSector) return;
    var coords = (window.SECTOR_COORDS || {})[name];
    if (coords) {
      map.panTo(coords);
      map.setZoom(ZOOM_SELECTED);
      highlightMarker(name);
    } else {
      window.resetMap();
    }
  };

  window.resetMap = function () {
    if (!map) return;
    unhighlightAll();
    map.panTo(DEFAULT_CENTER);
    map.setZoom(DEFAULT_ZOOM);
  };

  // ---- Auth failure (must be top-level, before SDK init) -----------------

  window.gm_authFailure = showMapError;

  // ---- SDK callback -------------------------------------------------------

  window.initMap = function () {
    try {
      var container = getMapContainer();
      if (!container) return;

      map = new google.maps.Map(container, {
        center:          DEFAULT_CENTER,
        zoom:            DEFAULT_ZOOM,
        mapId:           window.MAPS_MAP_ID || 'DEMO_MAP_ID',
        gestureHandling: 'cooperative',
      });

      var coords = window.SECTOR_COORDS || {};
      Object.keys(coords).forEach(function (sector) {
        var pos = coords[sector];
        var marker = new google.maps.marker.AdvancedMarkerElement({
          map:      map,
          position: pos,
          content:  makePinContent(false),
          title:    sector,
        });

        marker.addListener('click', function () {
          var input = getSectorInput();
          if (!input) return;
          // No-op if clicking the already-selected pin (R8)
          if (sector === selectedSector) return;
          input.value = sector;
          window.updateMapForSector(sector);
          input.focus();
        });

        markerMap[sector] = marker;
      });

      // Wire sector input change → map (R5, R6)
      var sectorInput = getSectorInput();
      if (sectorInput) {
        sectorInput.addEventListener('change', function () {
          window.updateMapForSector(sectorInput.value);
        });
      }

      // R2: if sector input is pre-filled on load (e.g. browser autofill),
      // initialize map centered on that sector
      if (sectorInput && sectorInput.value) {
        window.updateMapForSector(sectorInput.value);
      }

    } catch (err) {
      console.error('Map initialization error:', err);
      showMapError();
    }
  };

}());
