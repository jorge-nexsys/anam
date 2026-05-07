// ═══════════════════════════════════════════════════════════════════════
// AnamDB Playground — Frontend Logic & WebSocket Bridge
// ═══════════════════════════════════════════════════════════════════════

// ── State ────────────────────────────────────────────────────────────
const state = {
  connected: false,
  socket: null,
  wsUrl: 'ws://localhost:8080/ws', // The WebSocket bridge URL
  queryCount: 0,
  latencyMs: 0,
  currentResult: null,
  paretoData: []
};

// ── DOM Elements ─────────────────────────────────────────────────────
const els = {
  status: document.getElementById('connection-status'),
  statusText: document.querySelector('#connection-status .status-text'),
  editor: document.getElementById('sql-editor'),
  btnExecute: document.getElementById('btn-execute'),
  btnExplain: document.getElementById('btn-explain'),
  queryTime: document.getElementById('query-time'),
  resultsGrid: document.getElementById('results-grid'),
  traceContent: document.getElementById('trace-content'),
  resultCount: document.getElementById('result-count'),
  provenanceViewer: document.getElementById('provenance-viewer'),
  triagePanel: document.getElementById('triage-panel'),
  statusRows: document.getElementById('status-rows'),
  statusLatency: document.getElementById('status-latency'),
  tabs: document.querySelectorAll('.tab'),
  tabContents: document.querySelectorAll('.tab-content'),
  presets: document.getElementById('query-presets')
};

// ── WebSocket Bridge ─────────────────────────────────────────────────
function connect() {
  updateStatus(false, 'Connecting...');
  
  // Create WebSocket connection
  state.socket = new WebSocket(state.wsUrl);

  state.socket.onopen = () => {
    updateStatus(true, 'Connected');
    fetchHealth();
  };

  state.socket.onmessage = (event) => {
    try {
      const resp = JSON.parse(event.data);
      handleResponse(resp);
    } catch (e) {
      console.error('Failed to parse response:', e);
    }
  };

  state.socket.onclose = () => {
    updateStatus(false, 'Disconnected');
    // Try to reconnect after 3 seconds
    setTimeout(connect, 3000);
  };

  state.socket.onerror = (err) => {
    console.error('WebSocket error:', err);
    state.socket.close();
  };
}

function updateStatus(connected, text) {
  state.connected = connected;
  els.statusText.textContent = text;
  if (connected) {
    els.status.classList.add('connected');
  } else {
    els.status.classList.remove('connected');
  }
}

function sendCommand(cmd) {
  if (!state.connected || !state.socket) {
    console.warn('Cannot send command: not connected');
    return;
  }
  state.socket.send(JSON.stringify(cmd));
}

function handleResponse(resp) {
  // Handle responses from the server.
  // The server sends back JSON with "ok", "error", or specific fields depending on the method.
  
  if (resp.status && resp.version) {
    // Health response
    console.log('Server health:', resp);
    return;
  }
  
  if (resp.error) {
    showError(resp.error);
    return;
  }
  
  // Query response
  if (resp.ok && resp.reasoning_tree !== undefined) {
    const end = performance.now();
    state.latencyMs = Math.round(end - state.queryStartTime);
    
    // In a real app, we'd decode the Arrow IPC bytes here using apache-arrow JS.
    // For the demo playground, we'll simulate the decoded result since we can't easily
    // load the WASM Arrow decoder via CDN in a standalone HTML file without a bundler.
    simulateDecodeIpc(resp);
  }
}

// ── Query Execution ──────────────────────────────────────────────────
function executeQuery() {
  const sql = els.editor.value.trim();
  if (!sql) return;

  if (!state.connected) {
    // For demo purposes if server isn't running, show simulated results
    console.log('Not connected, simulating query execution');
    simulateQuery(sql);
    return;
  }

  els.btnExecute.innerHTML = '<span class="btn-icon-inline">⏳</span> Running...';
  els.btnExecute.disabled = true;
  
  state.queryStartTime = performance.now();
  
  sendCommand({
    method: 'query',
    sql: sql
  });
}

function fetchHealth() {
  sendCommand({ method: 'health' });
}

// ── UI Updates ───────────────────────────────────────────────────────
function renderResults(data) {
  if (!data || !data.columns || data.rows.length === 0) {
    els.resultsGrid.innerHTML = `
      <div class="empty-state animate-in">
        <div class="empty-icon">✓</div>
        <p>Query executed successfully (0 rows)</p>
      </div>`;
    updateStatusBar(0);
    return;
  }

  let tableHtml = '<table class="results-table animate-in"><thead><tr>';
  
  // Headers
  data.columns.forEach(col => {
    tableHtml += `<th>${escapeHtml(col)}</th>`;
  });
  tableHtml += '</tr></thead><tbody>';

  // Rows
  data.rows.forEach((row, rowIndex) => {
    tableHtml += `<tr onclick="showProvenance(${rowIndex})">`;
    row.forEach((val, colIndex) => {
      const isNum = typeof val === 'number';
      const isAnomaly = data.columns[colIndex] === 'fraud_prob' && val > 0.8;
      const cls = `${isNum ? 'num' : ''} ${isAnomaly ? 'high-risk' : ''}`;
      
      const displayVal = isNum && !Number.isInteger(val) ? val.toFixed(4) : escapeHtml(String(val));
      tableHtml += `<td class="${cls}">${displayVal}</td>`;
    });
    tableHtml += '</tr>';
  });
  
  tableHtml += '</tbody></table>';
  
  els.resultsGrid.innerHTML = tableHtml;
  els.resultCount.textContent = `${data.rows.length} rows`;
  updateStatusBar(data.rows.length);
  
  // Also build reasoning trace
  if (data.trace) {
    els.traceContent.innerHTML = `<div class="trace-block animate-in">${escapeHtml(data.trace)}</div>`;
  } else {
    els.traceContent.innerHTML = `<div class="empty-state"><p>No reasoning trace available</p></div>`;
  }
  
  // Update Triage Panel
  updateTriagePanel(data);
}

function showError(msg) {
  els.resultsGrid.innerHTML = `
    <div class="empty-state animate-in" style="color: var(--danger)">
      <div class="empty-icon">⚠</div>
      <p>Error executing query</p>
      <span class="empty-hint" style="color: var(--danger)">${escapeHtml(msg)}</span>
    </div>`;
  els.btnExecute.innerHTML = '<span class="btn-icon-inline">▶</span> Execute';
  els.btnExecute.disabled = false;
  els.queryTime.textContent = '';
  els.resultCount.textContent = 'Error';
  updateStatusBar(0);
}

function updateStatusBar(rows) {
  els.statusRows.textContent = `${rows} rows`;
  els.statusLatency.textContent = `${state.latencyMs}ms`;
  els.queryTime.textContent = `${state.latencyMs}ms`;
  els.btnExecute.innerHTML = '<span class="btn-icon-inline">▶</span> Execute';
  els.btnExecute.disabled = false;
}

function showProvenance(rowIndex) {
  // Select row in grid
  document.querySelectorAll('.results-table tbody tr').forEach((tr, i) => {
    if (i === rowIndex) tr.classList.add('selected');
    else tr.classList.remove('selected');
  });

  // Switch to provenance tab
  document.querySelector('[data-tab="provenance"]').click();
  
  const row = state.currentResult.rows[rowIndex];
  
  // Render a mock tree based on the row data
  els.provenanceViewer.innerHTML = `
    <div class="animate-in">
      <h3 style="margin-bottom: 12px; font-size: 14px;">Derivation Tree (Row ${rowIndex})</h3>
      <ul class="prov-tree">
        <li class="prov-node">
          <div class="prov-node-label">Session::sql()</div>
          <div class="prov-node-detail">polynomial_semiring token generated</div>
          <ul class="prov-tree prov-children">
            <li class="prov-node" style="border-left-color: var(--warning)">
              <div class="prov-node-label">LogicEngine::filter()</div>
              <div class="prov-node-detail">Passed constraint: fraud_prob > 0.8</div>
            </li>
            <li class="prov-node" style="border-left-color: var(--info)">
              <div class="prov-node-label">FaoScalarUdf::evaluate()</div>
              <div class="prov-node-detail">Model: fraud_detector v1.0.0 (ONNX)</div>
              <div class="prov-node-detail">Output score: ${row[1] || '0.95'}</div>
            </li>
            <li class="prov-node" style="border-left-color: var(--success)">
              <div class="prov-node-label">LanceStreamingProvider::scan()</div>
              <div class="prov-node-detail">Source dataset: txns.lance</div>
              <div class="prov-node-detail">Record ID: txn_${rowIndex}_v2</div>
            </li>
          </ul>
        </li>
      </ul>
    </div>
  `;
}

function updateTriagePanel(data) {
  // Find rows with high fraud prob
  const fraudIdx = data.columns.indexOf('fraud_prob');
  if (fraudIdx === -1) {
    els.triagePanel.innerHTML = '<div class="empty-state"><p>No anomalies detected</p></div>';
    return;
  }
  
  let anomalies = [];
  data.rows.forEach((row, i) => {
    if (row[fraudIdx] > 0.9) {
      anomalies.push({ rowIndex: i, score: row[fraudIdx], amount: row[0] });
    }
  });
  
  if (anomalies.length === 0) {
    els.triagePanel.innerHTML = '<div class="empty-state"><div class="empty-icon">✓</div><p>No anomalies detected</p></div>';
    return;
  }
  
  let html = '<div class="animate-in"><h3 style="margin-bottom: 12px; font-size: 13px; color: var(--warning);">HITL Triage Queue</h3>';
  
  anomalies.forEach((a, i) => {
    html += `
      <div class="triage-card" id="triage-card-${i}">
        <div class="triage-card-title">High Confidence Anomaly Flagged</div>
        <div class="triage-card-detail">Model returned score ${a.score.toFixed(4)} for transaction amount $${a.amount}. Semantic monitor flagged this as requiring human review.</div>
        <div class="triage-card-meta">Row ${a.rowIndex} • Model: fraud_detector v1.0.0</div>
        <div class="triage-actions">
          <button class="btn btn-sm btn-accept" onclick="resolveTriage(${i}, 'Accept')">Accept</button>
          <button class="btn btn-sm btn-correct" onclick="resolveTriage(${i}, 'Correct')">Correct</button>
          <button class="btn btn-sm btn-retry" onclick="resolveTriage(${i}, 'Retry')">Retry (Diff Model)</button>
        </div>
      </div>
    `;
  });
  
  html += '</div>';
  els.triagePanel.innerHTML = html;
}

window.resolveTriage = function(id, action) {
  const card = document.getElementById(`triage-card-${id}`);
  if (card) {
    card.innerHTML = `<div class="triage-card-title" style="color: var(--success);">✓ Resolved: ${action}</div>`;
    setTimeout(() => {
      card.style.display = 'none';
    }, 1500);
  }
};

function renderParetoChart() {
  const ctx = document.getElementById('pareto-chart');
  if (!ctx) return;
  
  // Simulated pareto data for models
  const data = {
    datasets: [{
      label: 'Available Models',
      data: [
        { x: 2.1, y: 0.85, label: 'fraud_fast (v1)' },
        { x: 5.4, y: 0.92, label: 'fraud_detector (v2)' },
        { x: 12.0, y: 0.96, label: 'fraud_deep (v3)' },
        { x: 28.5, y: 0.98, label: 'fraud_ensemble (v4)' }
      ],
      backgroundColor: 'rgba(99, 102, 241, 0.6)',
      borderColor: '#6366f1',
      pointRadius: 8,
      pointHoverRadius: 10
    }]
  };
  
  new Chart(ctx, {
    type: 'scatter',
    data: data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.label}: ${ctx.raw.x}ms, ${(ctx.raw.y * 100).toFixed(1)}% Acc`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Latency (ms) ↓', color: '#8b97a8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          title: { display: true, text: 'Accuracy ↑', color: '#8b97a8' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      }
    }
  });
}

// ── Event Listeners ──────────────────────────────────────────────────

els.btnExecute.addEventListener('click', executeQuery);

els.editor.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    executeQuery();
  }
});

els.tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    // Remove active from all
    els.tabs.forEach(t => t.classList.remove('active'));
    els.tabContents.forEach(c => c.classList.remove('active'));
    
    // Add active to clicked
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
    
    // Lazy render chart
    if (tab.dataset.tab === 'pareto' && !window.paretoRendered) {
      renderParetoChart();
      window.paretoRendered = true;
    }
  });
});

els.presets.addEventListener('change', (e) => {
  const val = e.target.value;
  if (!val) return;
  
  const presets = {
    'basic': 'SELECT * FROM txns LIMIT 10;',
    'filter': 'SELECT amount, region, merchant_type FROM txns WHERE amount > 5000 ORDER BY amount DESC;',
    'udf': 'SELECT \n  transaction_id,\n  amount, \n  fraud_detector(amount, fraud_prob, hour_of_day) AS score \nFROM txns \nWHERE score > 0.8;',
    'aggregate': 'SELECT region, COUNT(*) as cnt, AVG(amount) as avg_amt \nFROM txns \nGROUP BY region \nORDER BY cnt DESC;',
    'provenance': 'SELECT amount, fraud_prob, provenance \nFROM txns \nWHERE fraud_prob > 0.9;'
  };
  
  if (presets[val]) {
    els.editor.value = presets[val];
  }
  
  e.target.value = '';
});

// ── Utilities & Simulation ───────────────────────────────────────────

function escapeHtml(unsafe) {
  return String(unsafe)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Since we can't easily parse Arrow IPC bytes in the browser without a bundler to include 
// apache-arrow, we simulate the decoded output for the playground demo when running locally.
function simulateQuery(sql) {
  els.btnExecute.innerHTML = '<span class="btn-icon-inline">⏳</span> Running...';
  els.btnExecute.disabled = true;
  
  const start = performance.now();
  
  setTimeout(() => {
    state.latencyMs = Math.round(performance.now() - start);
    
    let mockData = {
      columns: ['amount', 'fraud_prob', 'region'],
      rows: [
        [42000.50, 0.95, 'EU'],
        [38500.00, 0.92, 'APAC'],
        [27000.00, 0.88, 'US'],
        [15000.25, 0.82, 'US'],
        [12500.00, 0.81, 'EU']
      ],
      trace: "── Batch 0 (5 rows) ──\n  row 0: PolynomialSemiring(model_ver_id='query_pipeline', func_id='sql', source_record_ids=['row_0'])\n  row 1: PolynomialSemiring(model_ver_id='query_pipeline', func_id='sql', source_record_ids=['row_1'])\n  row 2: PolynomialSemiring(model_ver_id='query_pipeline', func_id='sql', source_record_ids=['row_2'])"
    };
    
    if (sql.toLowerCase().includes('count')) {
      mockData = {
        columns: ['cnt', 'avg_amt'],
        rows: [[100000, 254.50]],
        trace: "── Batch 0 (1 rows) ──\n  row 0: PolynomialSemiring(Aggregate[COUNT,AVG])"
      };
    }
    
    state.currentResult = mockData;
    renderResults(mockData);
  }, 120); // Simulate network + execution latency
}

function simulateDecodeIpc(resp) {
  // In a real app: const table = Table.from(resp.arrow_ipc_batch);
  // We just pass through to the simulator for the demo UI
  simulateQuery(els.editor.value);
}

// ── Init ─────────────────────────────────────────────────────────────
console.log('AnamDB Playground UI initializing...');
// Attempt connection (will gracefully fail to simulated mode if server isn't running)
connect();
