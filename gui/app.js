import {
    checkHealth, inferImage, analyzeVideo,
    fetchAnnotatedImageUrl, fetchAnnotatedVideoUrl,
} from "/frontend/api.js";
import {
    setStatusChip, setStateBox, renderSummary, renderJson,
    renderImageDetectionsTable, renderVideoDetectionsTable,
    renderPreview, pushLog, clearTable,
} from "/frontend/renderers.js";

const COLS = 7;
const MAX_IMG = 10 * 1024 * 1024;
const MAX_VID = 200 * 1024 * 1024;

const THEAD = {
    image: `<tr><th>№</th><th>Класс</th><th>Уверенность</th><th>Центр</th><th>Размер</th><th>Угол, °</th><th>Полигон</th></tr>`,
    video: `<tr><th>Кадр</th><th>Время, с</th><th>Класс</th><th>Уверенность</th><th>Идентификатор трека</th><th>Угол, °</th><th>Полигон</th></tr>`,
};

const el = {
    modeSelect:       document.getElementById("modeSelect"),
    fileInput:        document.getElementById("fileInput"),
    sampleEveryInput: document.getElementById("sampleEveryInput"),
    maxFramesInput:   document.getElementById("maxFramesInput"),
    useTrackingInput: document.getElementById("useTrackingInput"),
    analyzeButton:    document.getElementById("analyzeButton"),
    annotateButton:   document.getElementById("annotateButton"),
    cancelButton:     document.getElementById("cancelButton"),
    healthButton:     document.getElementById("healthButton"),
    themeButton:      document.getElementById("themeButton"),
    resetButton:      document.getElementById("resetButton"),
    healthChip:       document.getElementById("healthChip"),
    appVersion:       document.getElementById("appVersion"),
    modelPath:        document.getElementById("modelPath"),
    stateBox:         document.getElementById("stateBox"),
    summaryList:      document.getElementById("summaryList"),
    jsonBlock:        document.getElementById("jsonBlock"),
    resultTableHead:  document.getElementById("resultTableHead"),
    resultTableBody:  document.getElementById("resultTableBody"),
    inputPreview:     document.getElementById("inputPreview"),
    outputPreview:    document.getElementById("outputPreview"),
    logBox:           document.getElementById("logBox"),
};

const state = { inputUrl: null, outputUrl: null, ctrl: null };

const mode = () => el.modeSelect.value;
const file = () => el.fileInput.files?.[0] || null;

function revoke(url) { if (url) URL.revokeObjectURL(url); }

function cleanUrls() {
    revoke(state.inputUrl);
    revoke(state.outputUrl);
    state.inputUrl = state.outputUrl = null;
}

function trimPos(v) { return typeof v === "string" ? v.trim() : ""; }
function posInt(v, fb = null) { const n = Number(v); return Number.isFinite(n) && n >= 1 ? Math.floor(n) : fb; }
function fmtNum(v, d = null, sfx = "") {
    const n = Number(v);
    return Number.isFinite(n) ? (d === null ? `${n}${sfx}` : `${n.toFixed(d)}${sfx}`) : "—";
}

function previewType(f) {
    if (!f?.type) return null;
    return f.type.startsWith("image/") ? "image" : f.type.startsWith("video/") ? "video" : null;
}

function videoOpts() {
    return {
        sampleEvery: posInt(trimPos(el.sampleEveryInput.value), 1),
        maxFrames: trimPos(el.maxFramesInput.value) ? posInt(trimPos(el.maxFramesInput.value), null) : null,
        useTracking: el.useTrackingInput.checked,
    };
}

function setLoading(on) {
    [el.modeSelect, el.fileInput, el.analyzeButton, el.annotateButton, el.healthButton, el.resetButton]
        .forEach(e => e.disabled = on);
    el.sampleEveryInput.disabled = on || mode() !== "video";
    el.maxFramesInput.disabled   = on || mode() !== "video";
    el.useTrackingInput.disabled = on || mode() !== "video";
    el.cancelButton.classList.toggle("is-hidden", !on);
}

function thead(m) { el.resultTableHead.innerHTML = THEAD[m] || THEAD.image; }

function syncMode() {
    const v = mode() === "video";
    el.sampleEveryInput.disabled = !v;
    el.maxFramesInput.disabled   = !v;
    el.useTrackingInput.disabled = !v;
    el.fileInput.accept = v ? "video/*" : "image/*";
    thead(v ? "video" : "image");
}

function resetResults() {
    revoke(state.outputUrl);
    state.outputUrl = null;
    renderSummary(el.summaryList, []);
    renderJson(el.jsonBlock, {});
    thead(mode());
    clearTable(el.resultTableBody, COLS, "Данные отсутствуют");
    renderPreview(el.outputPreview, null, null, "После аннотирования здесь отобразится результат.");
}

function resetAll() {
    abort();
    cleanUrls();
    el.fileInput.value = "";
    renderPreview(el.inputPreview,  null, null, "Предварительный просмотр исходного файла.");
    renderPreview(el.outputPreview, null, null, "После аннотирования здесь отобразится результат.");
    renderSummary(el.summaryList, []);
    renderJson(el.jsonBlock, {});
    thead(mode());
    clearTable(el.resultTableBody, COLS, "Данные отсутствуют");
    setStateBox(el.stateBox, "neutral", "Состояние сброшено. Выберите файл и режим обработки.");
    pushLog(el.logBox, "Состояние интерфейса сброшено");
}

function abort() {
    if (state.ctrl) { state.ctrl.abort(); state.ctrl = null; }
}

function begin() {
    abort();
    state.ctrl = new AbortController();
    return state.ctrl.signal;
}

function end() { state.ctrl = null; setLoading(false); }

function validate(f, m) {
    if (!f) throw new Error("Файл не выбран.");
    const mime = f.type || "";
    if (m === "image") {
        if (!mime.startsWith("image/")) throw new Error("Для режима «Изображение» нужен файл image/*.");
        if (f.size > MAX_IMG) throw new Error(`Изображение слишком большое: ${(f.size/1024/1024).toFixed(1)} МБ (макс. ${MAX_IMG/1024/1024} МБ).`);
    }
    if (m === "video") {
        if (!mime.startsWith("video/")) throw new Error("Для режима «Видеозапись» нужен файл video/*.");
        if (f.size > MAX_VID) throw new Error(`Видео слишком большое: ${(f.size/1024/1024).toFixed(1)} МБ (макс. ${MAX_VID/1024/1024} МБ).`);
    }
}

function previewInput() {
    const f = file();
    revoke(state.inputUrl);
    state.inputUrl = null;
    if (!f) { renderPreview(el.inputPreview, null, null, "Предварительный просмотр исходного файла."); return; }
    state.inputUrl = URL.createObjectURL(f);
    renderPreview(el.inputPreview, previewType(f), state.inputUrl, "Предварительный просмотр недоступен.");
}

function imgSummary(data, f) {
    return [
        { label: "Тип данных",         value: "Изображение" },
        { label: "Имя файла",           value: f?.name || "—" },
        { label: "Ширина, пкс",         value: fmtNum(data?.width) },
        { label: "Высота, пкс",         value: fmtNum(data?.height) },
        { label: "Количество детекций", value: fmtNum(data?.count) },
    ];
}

function vidSummary(data, f, opts) {
    const m = data?.metadata ?? {};
    return [
        { label: "Тип данных",              value: "Видеозапись" },
        { label: "Имя файла",               value: f?.name || "—" },
        { label: "Частота кадров",          value: fmtNum(m.fps, 2) },
        { label: "Разрешение",              value: Number.isFinite(Number(m.width)) && Number.isFinite(Number(m.height)) ? `${m.width} × ${m.height} пкс` : "—" },
        { label: "Кадров в записи",         value: fmtNum(m.total_frames) },
        { label: "Обработано кадров",       value: fmtNum(m.processed_frames) },
        { label: "Шаг дискретизации",       value: String(opts.sampleEvery) },
        { label: "Предел кадров",           value: opts.maxFrames == null ? "Не задан" : String(opts.maxFrames) },
        { label: "Трекинг объектов",        value: opts.useTracking ? "Включён" : "Отключён" },
        { label: "Длительность, с",         value: fmtNum(m.duration_sec, 2) },
        { label: "Всего детекций",          value: fmtNum(data?.total_detections) },
        { label: "Уникальных треков",       value: fmtNum(data?.unique_track_ids) },
    ];
}

async function refreshHealth() {
    setStatusChip(el.healthChip, "neutral", "Проверка…");
    try {
        const data = await checkHealth();
        const s = data?.status || "unknown";
        const labels = { ok: "В норме", degraded: "Деградация", not_initialized: "Не инициализирован" };
        const tones  = { ok: "success", degraded: "warning" };
        setStatusChip(el.healthChip, tones[s] || "danger", labels[s] ?? s);
        el.appVersion.textContent = data?.version    || "—";
        el.modelPath.textContent  = data?.model_path || "—";
        pushLog(el.logBox, `Состояние сервера: ${labels[s] ?? s}`);
    } catch (e) {
        setStatusChip(el.healthChip, "danger", "Недоступен");
        el.appVersion.textContent = "—";
        el.modelPath.textContent  = "—";
        pushLog(el.logBox, `Ошибка проверки: ${e instanceof Error ? e.message : String(e)}`);
    }
}

async function handleAnalyze() {
    const f = file(), m = mode();
    try {
        validate(f, m);
        resetResults();
        const signal = begin();
        setLoading(true);
        setStateBox(el.stateBox, "warning", "Выполняется обработка запроса…");
        pushLog(el.logBox, `Анализ: ${m === "image" ? "изображение" : "видеозапись"} — ${f.name}`);

        if (m === "image") {
            const data = await inferImage(f, signal);
            thead("image");
            renderSummary(el.summaryList, imgSummary(data, f));
            renderImageDetectionsTable(el.resultTableBody, data?.detections ?? []);
            renderJson(el.jsonBlock, data ?? {});
            setStateBox(el.stateBox, "success", `Обработано. Детекций: ${fmtNum(data?.count)}.`);
            pushLog(el.logBox, `Анализ завершён. Детекций: ${data?.count ?? 0}`);
            return;
        }

        const opts = videoOpts();
        const data = await analyzeVideo({ file: f, ...opts, signal });
        thead("video");
        renderSummary(el.summaryList, vidSummary(data, f, opts));
        renderVideoDetectionsTable(el.resultTableBody, data?.frames ?? []);
        renderJson(el.jsonBlock, data ?? {});
        setStateBox(el.stateBox, "success",
            `Обработано. Кадров: ${fmtNum(data?.metadata?.processed_frames)}, детекций: ${fmtNum(data?.total_detections)}.`);
        pushLog(el.logBox, `Анализ завершён. Кадров: ${data?.metadata?.processed_frames ?? 0}, детекций: ${data?.total_detections ?? 0}`);
    } catch (e) {
        if (e?.name === "AbortError") { setStateBox(el.stateBox, "neutral", "Запрос отменён."); pushLog(el.logBox, "Отменено"); return; }
        const msg = e instanceof Error ? e.message : String(e);
        setStateBox(el.stateBox, "danger", msg);
        pushLog(el.logBox, `Ошибка: ${msg}`);
    } finally { end(); }
}

async function handleAnnotate() {
    const f = file(), m = mode();
    try {
        validate(f, m);
        const signal = begin();
        setLoading(true);
        setStateBox(el.stateBox, "warning", "Генерируется аннотированный результат…");
        pushLog(el.logBox, `Аннотирование: ${f.name}`);
        revoke(state.outputUrl);
        state.outputUrl = null;
        renderPreview(el.outputPreview, null, null, "Генерируется…");

        if (m === "image") {
            state.outputUrl = await fetchAnnotatedImageUrl(f, signal);
            renderPreview(el.outputPreview, "image", state.outputUrl, "Изображение отсутствует.");
            setStateBox(el.stateBox, "success", "Аннотированное изображение готово.");
            pushLog(el.logBox, "Аннотированное изображение получено");
            return;
        }

        const opts = videoOpts();
        state.outputUrl = await fetchAnnotatedVideoUrl({ file: f, ...opts, signal });
        renderPreview(el.outputPreview, "video", state.outputUrl, "Видеозапись отсутствует.");
        setStateBox(el.stateBox, "success", "Аннотированная видеозапись готова.");
        pushLog(el.logBox, "Аннотированная видеозапись получена");
    } catch (e) {
        if (e?.name === "AbortError") { setStateBox(el.stateBox, "neutral", "Запрос отменён."); pushLog(el.logBox, "Отменено"); return; }
        const msg = e instanceof Error ? e.message : String(e);
        setStateBox(el.stateBox, "danger", msg);
        pushLog(el.logBox, `Ошибка аннотирования: ${msg}`);
    } finally { end(); }
}

el.modeSelect.addEventListener("change", () => {
    syncMode(); previewInput(); resetResults();
    pushLog(el.logBox, `Режим: ${mode() === "image" ? "изображение" : "видеозапись"}`);
});

el.fileInput.addEventListener("change", () => {
    previewInput(); resetResults();
    const f = file();
    if (f) pushLog(el.logBox, `Файл: ${f.name} (${(f.size/1024).toFixed(0)} КБ)`);
});

el.analyzeButton.addEventListener("click",  handleAnalyze);
el.annotateButton.addEventListener("click", handleAnnotate);
el.healthButton.addEventListener("click",   refreshHealth);
el.resetButton.addEventListener("click",    resetAll);

el.cancelButton.addEventListener("click", () => {
    abort();
    setStateBox(el.stateBox, "neutral", "Запрос отменён.");
    pushLog(el.logBox, "Отменено");
    setLoading(false);
});

el.themeButton.addEventListener("click", () => {
    document.body.classList.toggle("theme-dark");
    pushLog(el.logBox, `Тема: ${document.body.classList.contains("theme-dark") ? "тёмная" : "светлая"}`);
});

window.addEventListener("beforeunload", cleanUrls);
window.addEventListener("pagehide", cleanUrls);

function init() {
    syncMode();
    renderPreview(el.inputPreview, null, null, "Предварительный просмотр исходного файла.");
    renderPreview(el.outputPreview, null, null, "После аннотирования здесь отобразится результат.");
    renderSummary(el.summaryList, []);
    renderJson(el.jsonBlock, {});
    clearTable(el.resultTableBody, COLS, "Данные отсутствуют");
    setStateBox(el.stateBox, "neutral", "Выберите файл и режим обработки.");
    el.cancelButton.classList.add("is-hidden");
    refreshHealth();
}

init();
