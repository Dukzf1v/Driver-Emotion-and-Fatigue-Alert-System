let vehicles = {}; // vehicleId -> vehicleData
let selectedVehicleId = null;
let chartInstance = null;
let currentTab = 'tabLive';

// History
const MAX_CHART_POINTS = 60;
let vehicleHistories = {}; // vehicleId -> Array of { time: string, f_score: number, anger_level: number, fear_level: number }

// Trend
const HISTORY_LEN = 10;
let vehicleTrendHistories = {}; // vehicleId -> { f_score: [], anger: [], fear: [] }

// DOM
const vehicleList = document.getElementById('vehicleList');
const noVehiclesPlaceholder = document.getElementById('noVehiclesPlaceholder');
const vehicleDetails = document.getElementById('vehicleDetails');
const connBadge = document.getElementById('connBadge');

// Update
const detailVehicleId = document.getElementById('detailVehicleId');
const detailDriverName = document.getElementById('detailDriverName');
const detailConnStatus = document.getElementById('detailConnStatus');
const stateBanner = document.getElementById('stateBanner');
const stateIcon = document.getElementById('stateIcon');
const stateText = document.getElementById('stateText');
const stateAction = document.getElementById('stateAction');
const lastUpdate = document.getElementById('lastUpdate');

const fatigueBar = document.getElementById('fatigueBar');
const fatigueVal = document.getElementById('fatigueVal');
const fatigueTrend = document.getElementById('fatigueTrend');

const angryBar = document.getElementById('angryBar');
const angryVal = document.getElementById('angryVal');
const angryTrend = document.getElementById('angryTrend');

const fearBar = document.getElementById('fearBar');
const fearVal = document.getElementById('fearVal');
const fearTrend = document.getElementById('fearTrend');

// Init
function initChart() {
    const ctx = document.getElementById('liveChart').getContext('2d');

    Chart.defaults.color = '#8b949e';
    Chart.defaults.borderColor = 'rgba(139, 148, 158, 0.15)';

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Fatigue (F-Score)',
                    data: [],
                    borderColor: '#F0A030',
                    backgroundColor: 'rgba(240, 160, 48, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Anger',
                    data: [],
                    borderColor: '#E85A5A',
                    backgroundColor: 'rgba(232, 90, 90, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Fear',
                    data: [],
                    borderColor: '#8888F8',
                    backgroundColor: 'rgba(136, 136, 248, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        callback: function (value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            family: "'Outfit', sans-serif",
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

// Select
function selectVehicle(vehicleId) {
    selectedVehicleId = vehicleId;

    // Update
    document.querySelectorAll('.vehicle-item').forEach(el => {
        if (el.dataset.id === vehicleId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });

    // Show details, hide placeholder
    noVehiclesPlaceholder.style.display = 'none';
    vehicleDetails.style.display = 'block';

    // Update
    const history = vehicleHistories[vehicleId] || [];
    chartInstance.data.labels = history.map(h => h.time);
    chartInstance.data.datasets[0].data = history.map(h => h.f_score * 100);
    chartInstance.data.datasets[1].data = history.map(h => h.anger_level * 100);
    chartInstance.data.datasets[2].data = history.map(h => h.fear_level * 100);
    chartInstance.update('none');

    // Update
    const vdata = vehicles[vehicleId];
    if (vdata) {
        updateVehicleDetailUI(vdata);
    }
}

// Render Sidebar
function renderSidebar() {
    const scrollTop = vehicleList.scrollTop;
    vehicleList.innerHTML = '';

    const activeVehicles = Object.values(vehicles);

    if (activeVehicles.length === 0) {
        vehicleList.innerHTML = '<div style="color: var(--text3); font-size:12px; text-align:center; padding: 20px 0;">No vehicle connected</div>';
        return;
    }

    // Online first
    activeVehicles.sort((a, b) => {
        if (a.status === 'online' && b.status !== 'online') return -1;
        if (a.status !== 'online' && b.status === 'online') return 1;
        return a.vehicle_id.localeCompare(b.vehicle_id);
    });

    activeVehicles.forEach(v => {
        const item = document.createElement('div');
        item.className = `vehicle-item ${v.vehicle_id === selectedVehicleId ? 'active' : ''}`;
        item.dataset.id = v.vehicle_id;

        // Badge
        const badgeClass = `badge-${v.state}`;
        const badgeText = v.status === 'online' ? (v.state === 'NORMAL' ? 'NORMAL' : v.state) : 'OFFLINE';

        item.innerHTML = `
            <div class="vehicle-info">
                <span class="vehicle-id-lbl"><i class="fa-solid fa-truck-moving me-1"></i> ${v.vehicle_id}</span>
                <span class="vehicle-driver-lbl">${v.driver_name}</span>
            </div>
            <div class="vehicle-status">
                <span class="vehicle-state-badge ${v.status === 'online' ? badgeClass : 'badge-UNKNOWN'}">${badgeText}</span>
                <span class="vehicle-status-dot ${v.status === 'online' ? 'status-online' : 'status-offline'}"></span>
            </div>
        `;

        item.addEventListener('click', () => selectVehicle(v.vehicle_id));
        vehicleList.appendChild(item);
    });

    vehicleList.scrollTop = scrollTop;
}

// Trend
function updateTrendUI(element, current, historyArr) {
    if (historyArr.length < HISTORY_LEN) {
        historyArr.push(current);
        element.className = 'trend-indicator trend-stable';
        element.innerText = '--';
        return;
    }
    const oldVal = historyArr.shift();
    historyArr.push(current);
    const diff = current - oldVal;

    if (Math.abs(diff) < 0.5) {
        element.className = 'trend-indicator trend-stable';
        element.innerText = 'Stable';
    } else if (diff > 0) {
        element.className = 'trend-indicator trend-up';
        element.innerText = `+${diff.toFixed(1)}% ↗`;
    } else {
        element.className = 'trend-indicator trend-down';
        element.innerText = `${diff.toFixed(1)}% ↘`;
    }
}

// Update Detail Vehicle
function updateVehicleDetailUI(vdata) {
    detailVehicleId.innerText = vdata.vehicle_id;
    detailDriverName.innerText = vdata.driver_name;

    if (vdata.status === 'online') {
        detailConnStatus.innerHTML = '<span style="color:#3DCC8E; font-weight:600;"><i class="fa-solid fa-circle-dot me-1"></i> Active</span>';
    } else {
        detailConnStatus.innerHTML = '<span style="color:#8b949e; font-weight:600;"><i class="fa-solid fa-circle-dot me-1"></i> Offline</span>';
        stateBanner.className = 'state-UNKNOWN';
        stateIcon.className = 'fa-solid fa-circle-question me-2';
        stateText.innerText = 'OFFLINE';
        stateAction.innerText = 'No data received from vehicle.';
        return;
    }

    // Update Status Banner
    stateBanner.className = 'state-' + vdata.state;
    if (vdata.state === 'NORMAL') {
        stateIcon.className = 'fa-solid fa-check-circle me-2';
        stateText.innerText = 'NORMAL (OK)';
        stateAction.innerText = 'Driver is focused, no abnormal signs detected.';
    } else if (vdata.state === 'FATIGUE') {
        stateIcon.className = 'fa-solid fa-triangle-exclamation me-2';
        stateText.innerText = 'FATIGUE - WARNING!';
        stateAction.innerText = 'RECOMMENDATION: Request driver to pull over and rest for 15 minutes.';
    } else if (vdata.state === 'ANGRY') {
        stateIcon.className = 'fa-solid fa-face-angry me-2';
        stateText.innerText = 'ANGER - WARNING!';
        stateAction.innerText = 'RECOMMENDATION: Contact driver via intercom to de-escalate situation.';
    } else if (vdata.state === 'FEAR') {
        stateIcon.className = 'fa-solid fa-face-surprise me-2';
        stateText.innerText = 'FEAR - WARNING!';
        stateAction.innerText = 'RECOMMENDATION: Check dashcam or radio/intercom immediately.';
    } else if (vdata.state === 'DISTRACTED') {
        stateIcon.className = 'fa-solid fa-eye-slash me-2';
        stateText.innerText = 'DISTRACTED - WARNING!';
        stateAction.innerText = 'RECOMMENDATION: Advise driver to focus on the road ahead.';
    }

    // Progress
    const f_pct = Math.min(vdata.f_score * 100, 100);
    const a_pct = Math.min(vdata.anger_level * 100, 100);
    const fear_pct = Math.min(vdata.fear_level * 100, 100);

    fatigueBar.style.width = f_pct + '%';
    fatigueVal.innerText = f_pct.toFixed(1) + '%';

    angryBar.style.width = a_pct + '%';
    angryVal.innerText = a_pct.toFixed(1) + '%';

    fearBar.style.width = fear_pct + '%';
    fearVal.innerText = fear_pct.toFixed(1) + '%';

    // Trend
    if (!vehicleTrendHistories[vdata.vehicle_id]) {
        vehicleTrendHistories[vdata.vehicle_id] = { f_score: [], anger: [], fear: [] };
    }
    const trends = vehicleTrendHistories[vdata.vehicle_id];
    updateTrendUI(fatigueTrend, f_pct, trends.f_score);
    updateTrendUI(angryTrend, a_pct, trends.anger);
    updateTrendUI(fearTrend, fear_pct, trends.fear);

    // Time
    const updateTime = new Date(vdata.timestamp * 1000);
    lastUpdate.innerText = "Updated at: " + updateTime.toLocaleTimeString();
}

// Switch Tab
function switchTab(tabId) {
    currentTab = tabId;

    // Active
    document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');

    // Elements
    const sidebar = document.getElementById('sidebarSection');
    const liveView = document.getElementById('liveView');
    const historyView = document.getElementById('historyView');
    const driversView = document.getElementById('driversView');

    if (tabId === 'tabLive') {
        sidebar.style.display = 'flex';
        liveView.style.display = 'block';
        historyView.style.display = 'none';
        driversView.style.display = 'none';
    } else if (tabId === 'tabHistory') {
        sidebar.style.display = 'none';
        liveView.style.display = 'none';
        historyView.style.display = 'block';
        driversView.style.display = 'none';
        fetchHistory();
    } else if (tabId === 'tabDrivers') {
        sidebar.style.display = 'none';
        liveView.style.display = 'none';
        historyView.style.display = 'none';
        driversView.style.display = 'block';
        fetchDrivers();
    }
}

// Fetch History
function fetchHistory() {
    const searchVal = document.getElementById('historySearch').value;
    const filterVal = document.getElementById('historyFilter').value;

    fetch(`/api/history?vehicle_id=${encodeURIComponent(searchVal)}&state=${encodeURIComponent(filterVal)}`)
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('historyTableBody');
            tableBody.innerHTML = '';

            if (data.length === 0) {
                tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text3); padding: 30px;">No warning history found.</td></tr>`;
                return;
            }

            data.forEach(r => {
                const dateStr = new Date(r.timestamp * 1000).toLocaleString();
                let stateText = r.state;
                let badgeClass = `badge-${r.state}`;

                if (r.state === 'FATIGUE') stateText = 'Fatigue';
                else if (r.state === 'ANGRY') stateText = 'Anger';
                else if (r.state === 'FEAR') stateText = 'Fear';
                else if (r.state === 'DISTRACTED') stateText = 'Distracted';

                tableBody.innerHTML += `
                    <tr>
                        <td style="font-weight: 500;">${dateStr}</td>
                        <td style="font-family: 'Outfit', sans-serif; font-weight: 700; color: var(--sky);">${r.vehicle_id}</td>
                        <td>${r.driver_name}</td>
                        <td><span class="vehicle-state-badge ${badgeClass}">${stateText}</span></td>
                        <td>${(r.f_score * 100).toFixed(1)}%</td>
                        <td>${(r.anger_level * 100).toFixed(1)}%</td>
                        <td>${(r.fear_level * 100).toFixed(1)}%</td>
                    </tr>
                `;
            });
        })
        .catch(err => console.error("Error loading history:", err));
}

// Fetch Driver
function fetchDrivers() {
    fetch('/api/drivers')
        .then(response => response.json())
        .then(data => {
            const grid = document.getElementById('driverGrid');
            grid.innerHTML = '';

            if (data.length === 0) {
                grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text3); padding: 40px;">No driver profile found.</div>`;
                return;
            }

            data.forEach(d => {
                // Score class
                let scoreClass = 'score-good';
                let statusClass = 'status-good';
                if (d.safety_score < 50) {
                    scoreClass = 'score-crit';
                    statusClass = 'status-danger';
                } else if (d.safety_score < 80) {
                    scoreClass = 'score-warn';
                    statusClass = 'status-warning';
                }

                grid.innerHTML += `
                    <div class="driver-card">
                        <div class="driver-status-badge ${statusClass}">${d.status}</div>
                        <div class="driver-card-header">
                            <div class="driver-avatar"><i class="fa-solid fa-user-tie"></i></div>
                            <div class="driver-card-title">
                                <span class="driver-name-text">${d.driver_name}</span>
                                <span class="driver-vehicle-text"><i class="fa-solid fa-truck-moving me-1"></i> Plate: ${d.vehicle_id}</span>
                            </div>
                        </div>
                        <div class="driver-score-section">
                            <span class="driver-score-lbl">Safety Score</span>
                            <span class="driver-score-val ${scoreClass}">${d.safety_score.toFixed(0)}</span>
                        </div>
                        <div class="driver-stats-section">
                            <div class="driver-stat-row">
                                <span>Total warnings:</span>
                                <span class="driver-stat-count" style="color: var(--sky);">${d.total_warnings}</span>
                            </div>
                            <div class="driver-stat-row">
                                <span>Fatigue:</span>
                                <span class="driver-stat-count">${d.fatigue_count}</span>
                            </div>
                            <div class="driver-stat-row">
                                <span>Distracted:</span>
                                <span class="driver-stat-count">${d.distracted_count}</span>
                            </div>
                            <div class="driver-stat-row">
                                <span>Anger:</span>
                                <span class="driver-stat-count">${d.angry_count}</span>
                            </div>
                            <div class="driver-stat-row">
                                <span>Fear:</span>
                                <span class="driver-stat-count">${d.fear_count}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
        })
        .catch(err => console.error("Error loading driver info:", err));
}

// SSE
function initSSE() {
    const source = new EventSource('/api/stream');

    source.onopen = function () {
        connBadge.className = '';
        connBadge.style.borderColor = '#3DCC8E';
        connBadge.style.color = '#3DCC8E';
        connBadge.style.background = 'rgba(61,204,142,.1)';
        connBadge.innerHTML = '<i class="fa-solid fa-wifi me-1"></i> Central Monitor';
    };

    source.onerror = function () {
        connBadge.className = '';
        connBadge.style.borderColor = '#E85A5A';
        connBadge.style.color = '#E85A5A';
        connBadge.style.background = 'rgba(232,90,90,.1)';
        connBadge.innerHTML = '<i class="fa-solid fa-link-slash me-1"></i> Connection lost';
    };

    source.onmessage = function (event) {
        const msg = JSON.parse(event.data);

        // Init
        if (msg.type === 'init') {
            vehicles = msg.vehicles;

            // History
            Object.keys(vehicles).forEach(vid => {
                if (!vehicleHistories[vid]) {
                    vehicleHistories[vid] = [];
                }
            });

            renderSidebar();

            // Auto select first vehicle
            const vids = Object.keys(vehicles);
            if (vids.length > 0 && !selectedVehicleId) {
                selectVehicle(vids[0]);
            }

            // Sync sub-tab
            if (currentTab === 'tabHistory') fetchHistory();
            if (currentTab === 'tabDrivers') fetchDrivers();
        }

        // Update
        else if (msg.type === 'update') {
            const vid = msg.vehicle_id;
            const isFirstTime = !vehicles[vid];

            vehicles[vid] = msg.data;

            // Chart history
            if (!vehicleHistories[vid]) {
                vehicleHistories[vid] = [];
            }

            const now = new Date(msg.data.timestamp * 1000);
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            vehicleHistories[vid].push({
                time: timeStr,
                f_score: msg.data.f_score,
                anger_level: msg.data.anger_level,
                fear_level: msg.data.fear_level
            });

            if (vehicleHistories[vid].length > MAX_CHART_POINTS) {
                vehicleHistories[vid].shift();
            }

            renderSidebar();

            // If no vehicle selected or this is the only new vehicle -> auto select
            if (!selectedVehicleId || isFirstTime) {
                selectVehicle(vid);
            }

            // If this vehicle is selected -> update detail + push new score to chart
            if (selectedVehicleId === vid) {
                updateVehicleDetailUI(msg.data);

                // Update Chart.js
                chartInstance.data.labels.push(timeStr);
                chartInstance.data.datasets[0].data.push(msg.data.f_score * 100);
                chartInstance.data.datasets[1].data.push(msg.data.anger_level * 100);
                chartInstance.data.datasets[2].data.push(msg.data.fear_level * 100);

                if (chartInstance.data.labels.length > MAX_CHART_POINTS) {
                    chartInstance.data.labels.shift();
                    chartInstance.data.datasets[0].data.shift();
                    chartInstance.data.datasets[1].data.shift();
                    chartInstance.data.datasets[2].data.shift();
                }
                chartInstance.update('none');
            }

            // If warning and in sub-tab -> auto update without manual reload
            if (msg.data.state !== 'NORMAL') {
                if (currentTab === 'tabHistory') fetchHistory();
                if (currentTab === 'tabDrivers') fetchDrivers();
            }
        }

        // Status Change
        else if (msg.type === 'status_change') {
            Object.keys(msg.updates).forEach(vid => {
                if (vehicles[vid]) {
                    vehicles[vid].status = msg.updates[vid];

                    // If offline -> clear old data, keep basic structure
                    if (msg.updates[vid] === 'offline') {
                        vehicles[vid].state = 'UNKNOWN';
                        vehicles[vid].f_score = 0;
                        vehicles[vid].anger_level = 0;
                        vehicles[vid].fear_level = 0;

                        // Reset trend
                        if (vehicleTrendHistories[vid]) {
                            vehicleTrendHistories[vid] = { f_score: [], anger: [], fear: [] };
                        }
                    }
                }
            });

            renderSidebar();

            // If the selected vehicle's status has changed -> update detail
            if (selectedVehicleId && msg.updates[selectedVehicleId]) {
                updateVehicleDetailUI(vehicles[selectedVehicleId]);
            }

            // Sync sub-tab when status changes
            if (currentTab === 'tabHistory') fetchHistory();
            if (currentTab === 'tabDrivers') fetchDrivers();
        }
    };
}

// Init when document loaded
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initSSE();

    // Tabs event
    document.getElementById('tabLive').addEventListener('click', (e) => {
        e.preventDefault();
        switchTab('tabLive');
    });
    document.getElementById('tabHistory').addEventListener('click', (e) => {
        e.preventDefault();
        switchTab('tabHistory');
    });
    document.getElementById('tabDrivers').addEventListener('click', (e) => {
        e.preventDefault();
        switchTab('tabDrivers');
    });

    // History search & filter
    document.getElementById('historySearch').addEventListener('input', fetchHistory);
    document.getElementById('historyFilter').addEventListener('change', fetchHistory);

    // Manual reload
    document.getElementById('btnReloadHistory').addEventListener('click', fetchHistory);
    document.getElementById('btnReloadDrivers').addEventListener('click', fetchDrivers);
});
