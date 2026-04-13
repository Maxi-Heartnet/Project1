(function () {
  'use strict';
  // SECTORS is injected as a global by the inline script in the page HTML

  // Populate sector datalist
  var dl = document.getElementById('sector-list');
  SECTORS.forEach(function (s) {
    var opt = document.createElement('option');
    opt.value = s;
    dl.appendChild(opt);
  });

  // Validation rules
  function validateSector(v) {
    return v.trim() ? null : 'Please enter a sector or neighborhood.';
  }
  function validatePropertyType(v) {
    return v ? null : 'Please select a property type.';
  }
  function validateBedrooms(v) {
    var n = parseInt(v, 10);
    return (v !== '' && !isNaN(n) && n >= 1) ? null
      : 'Enter a whole number of bedrooms (1 or more).';
  }
  function validateArea(v) {
    var n = parseFloat(v);
    return (v !== '' && !isNaN(n) && n > 0) ? null
      : 'Enter a positive area in m\u00b2.';
  }

  var fields = [
    { id: 'sector',        validate: validateSector },
    { id: 'property_type', validate: validatePropertyType },
    { id: 'bedrooms',      validate: validateBedrooms },
    { id: 'area_m2',       validate: validateArea },
  ];

  function setError(fieldId, msg) {
    var el = document.getElementById(fieldId + '-error');
    if (el) el.textContent = msg || '';
  }

  // Blur validation
  fields.forEach(function (f) {
    var el = document.getElementById(f.id);
    if (!el) return;
    el.addEventListener('blur', function () {
      setError(f.id, f.validate(el.value) || '');
    });
  });

  // Formatting helpers
  function fmtFull(n) {
    return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
  }
  function fmtK(n) {
    return '$' + Math.round(n / 1000) + 'k';
  }

  // Whitelist for tier badge CSS classes (prevents raw API value in className)
  var TIER_CLASS = {
    'Budget':    'badge-Budget',
    'Mid-Range': 'badge-MidRange',
    'Luxury':    'badge-Luxury',
  };

  function clearResults() {
    var r = document.getElementById('results');
    while (r.firstChild) r.removeChild(r.firstChild);
    return r;
  }

  function renderResults(data) {
    var r = clearResults();

    var headline = document.createElement('div');
    headline.className = 'price-headline';
    headline.textContent = '$' + fmtFull(data.price_low) + ' \u2013 $' + fmtFull(data.price_high) + ' USD';
    r.appendChild(headline);

    var badge = document.createElement('span');
    badge.className = 'badge ' + (TIER_CLASS[data.market_tier] || 'badge-MidRange');
    badge.textContent = data.market_tier;
    r.appendChild(badge);

    var s = data.tier_stats;
    var statsEl = document.createElement('div');
    statsEl.className = 'tier-stats';

    var rows = [
      ['Typical price range', fmtK(s.price_p10) + ' \u2013 ' + fmtK(s.price_p90) + ' USD'],
      ['Typical area',        Math.round(s.area_p10) + '\u2013' + Math.round(s.area_p90) + ' m\u00b2'],
      ['Typical bedrooms',    Math.round(s.beds_p10) + '\u2013' + Math.round(s.beds_p90) + ' beds'],
    ];
    rows.forEach(function (row) {
      var lbl = document.createElement('span');
      lbl.className = 'stat-label';
      lbl.textContent = row[0];
      var val = document.createElement('span');
      val.className = 'stat-value';
      val.textContent = row[1];
      statsEl.appendChild(lbl);
      statsEl.appendChild(val);
    });
    r.appendChild(statsEl);
  }

  function renderError(msg) {
    var r = clearResults();
    var p = document.createElement('p');
    p.className = 'result-error';
    p.textContent = msg;
    r.appendChild(p);
  }

  var form     = document.getElementById('predict-form');
  var btn      = document.getElementById('submit-btn');
  var clearBtn = document.getElementById('clear-btn');
  var fillBtn  = document.getElementById('fill-btn');

  function clearForm() {
    fields.forEach(function (f) {
      var el = document.getElementById(f.id);
      if (el) { el.value = ''; }
      setError(f.id, '');
    });
    var r = document.getElementById('results');
    while (r.firstChild) { r.removeChild(r.firstChild); }
    r.textContent = 'Fill in the form above to see a price estimate.';
    btn.disabled = false;
    btn.textContent = 'Estimate Price';
    if (window.resetMap) window.resetMap();
  }

  function fillRandom() {
    if (!SECTORS.length) { return; }
    var randomSector = SECTORS[Math.floor(Math.random() * SECTORS.length)];
    document.getElementById('sector').value = randomSector;
    document.getElementById('property_type').value =
      ['apartment', 'house'][Math.floor(Math.random() * 2)];
    document.getElementById('bedrooms').value =
      Math.floor(Math.random() * 5) + 1;
    document.getElementById('area_m2').value =
      Math.round((Math.random() * 350 + 50) * 10) / 10;
    fields.forEach(function (f) { setError(f.id, ''); });
    var r = document.getElementById('results');
    while (r.firstChild) { r.removeChild(r.firstChild); }
    r.textContent = 'Fill in the form above to see a price estimate.';
    btn.disabled = false;
    btn.textContent = 'Estimate Price';
    if (window.updateMapForSector) window.updateMapForSector(randomSector);
  }

  clearBtn.addEventListener('click', clearForm);
  fillBtn.addEventListener('click', fillRandom);

  form.addEventListener('submit', function (e) {
    e.preventDefault();

    // Validate all
    var valid = true;
    fields.forEach(function (f) {
      var el  = document.getElementById(f.id);
      var err = f.validate(el.value);
      setError(f.id, err || '');
      if (err) { valid = false; }
    });
    if (!valid) { return; }

    // Loading state — disable all action buttons while fetch is in-flight
    btn.disabled = true;
    btn.textContent = 'Estimating...';
    clearBtn.disabled = true;
    fillBtn.disabled = true;
    var r = clearResults();
    var loadingMsg = document.createElement('p');
    loadingMsg.textContent = 'Getting your estimate\u2026';
    r.appendChild(loadingMsg);

    var payload = {
      sector:        document.getElementById('sector').value.trim(),
      property_type: document.getElementById('property_type').value,
      bedrooms:      parseInt(document.getElementById('bedrooms').value, 10),
      area_m2:       parseFloat(document.getElementById('area_m2').value),
    };

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    .then(function (response) {
      if (!response.ok) { throw { isServer: true }; }
      return response.json();
    })
    .then(function (data) {
      renderResults(data);
      btn.disabled = false;
      btn.textContent = 'Estimate Price';
      clearBtn.disabled = false;
      fillBtn.disabled = false;
    })
    .catch(function (err) {
      if (err && err.isServer) {
        renderError('Something went wrong. Please try again.');
      } else {
        renderError('Could not reach the server. Check your connection and try again.');
      }
      btn.disabled = false;
      btn.textContent = 'Estimate Price';
      clearBtn.disabled = false;
      fillBtn.disabled = false;
    });
  });
}());
