function esc(v) {
    return String(v)
        .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function fmt(v, d = 3) {
    const n = Number(v);
    return typeof v !== "number" && Number.isNaN(n) ? "—" : Number.isNaN(n) ? "—" : n.toFixed(d);
}

function fmtPt(pt) {
    if (!Array.isArray(pt) || pt.length !== 2) return "—";
    return pt.map(v => fmt(Number(v), 1)).join(", ");
}

function fmtPoly(poly) {
    if (!Array.isArray(poly) || poly.length === 0) return "—";
    const full = poly.map(p => `(${fmtPt(p)})`).join(" ");
    return `<span title="${esc(full)}">${poly.length} вершин</span>`;
}

function getLabel(d) { return d?.label ?? d?.class_name ?? "—"; }
function getScore(d) { const n = Number(d?.score ?? d?.confidence); return Number.isFinite(n) ? n : null; }
function getAngle(d) { const n = Number(d?.angle_deg ?? d?.angle); return Number.isFinite(n) ? n : null; }
function getTrack(d) { return d?.track_id != null ? String(d.track_id) : "—"; }

export function setStatusChip(el, tone, text) {
    el.className = `status-chip status-chip--${tone}`;
    el.textContent = text;
}

export function setStateBox(el, tone, text) {
    el.className = `state-box state-box--${tone}`;
    el.textContent = text;
}

export function renderSummary(el, rows) {
    el.innerHTML = rows.map(({ label, value }) =>
        `<div class="summary-row"><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`
    ).join("");
}

export function renderJson(el, data) {
    el.textContent = JSON.stringify(data, null, 2);
}

export function clearTable(tbody, colspan = 7, msg = "Данные отсутствуют") {
    tbody.innerHTML = `<tr><td class="table-empty" colspan="${colspan}">${esc(msg)}</td></tr>`;
}

export function renderImageDetectionsTable(tbody, detections) {
    if (!Array.isArray(detections) || detections.length === 0) {
        clearTable(tbody, 7, "Объекты не обнаружены");
        return;
    }
    tbody.innerHTML = detections.map((d, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${esc(getLabel(d))}</td>
            <td>${fmt(getScore(d), 3)}</td>
            <td>${esc(fmtPt(d?.center))}</td>
            <td>${esc(fmtPt(d?.size))}</td>
            <td>${fmt(getAngle(d), 1)}</td>
            <td>${fmtPoly(d?.polygon)}</td>
        </tr>`).join("");
}

export function renderVideoDetectionsTable(tbody, frames) {
    if (!Array.isArray(frames) || frames.length === 0) {
        clearTable(tbody, 7, "Обработанные кадры отсутствуют");
        return;
    }
    const rows = [];
    for (const f of frames) {
        const dets = Array.isArray(f?.detections) ? f.detections : [];
        if (dets.length === 0) {
            rows.push(`<tr>
                <td>${f?.frame_index ?? "—"}</td>
                <td>${fmt(Number(f?.timestamp_sec), 2)}</td>
                <td colspan="5" class="table-empty">Детекции не обнаружены</td>
            </tr>`);
            continue;
        }
        for (const d of dets) {
            rows.push(`<tr>
                <td>${f?.frame_index ?? "—"}</td>
                <td>${fmt(Number(f?.timestamp_sec), 2)}</td>
                <td>${esc(getLabel(d))}</td>
                <td>${fmt(getScore(d), 3)}</td>
                <td>${getTrack(d)}</td>
                <td>${fmt(getAngle(d), 1)}</td>
                <td>${fmtPoly(d?.polygon)}</td>
            </tr>`);
        }
    }
    tbody.innerHTML = rows.join("");
}

export function renderPreview(container, type, url, placeholder) {
    container.innerHTML = "";
    if (!url) {
        const p = document.createElement("p");
        p.className = "placeholder-text";
        p.textContent = placeholder;
        container.appendChild(p);
        return;
    }
    if (type === "image") {
        const img = document.createElement("img");
        img.src = url;
        img.alt = "Предварительный просмотр";
        container.appendChild(img);
        return;
    }
    const vid = document.createElement("video");
    vid.src = url;
    vid.controls = true;
    vid.playsInline = true;
    container.appendChild(vid);
}

export function pushLog(container, message) {
    const item = document.createElement("div");
    item.className = "log-entry";
    item.textContent = `[${new Date().toLocaleTimeString("ru-RU")}] ${message}`;
    container.prepend(item);
}
