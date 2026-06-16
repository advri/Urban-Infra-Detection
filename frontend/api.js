const API_BASE = String(window.__APP_API_PREFIX__ ?? "/api").replace(/\/+$/, "");

const TIMEOUTS = {
    default:       30_000,
    video_analyze: 600_000,
    video_annotate: 600_000,
};

function normalizePath(path) {
    if (!path) return "/";
    return path.startsWith("/") ? path : `/${path}`;
}

function buildCandidateUrls(path) {
    const normalized = normalizePath(path);
    const candidates = [];
    if (API_BASE) candidates.push(`${API_BASE}${normalized}`);
    candidates.push(normalized);
    return [...new Set(candidates)];
}

function collectValidationMessages(payload) {
    const detail = payload?.detail;

    if (Array.isArray(detail)) {
        return detail
            .map((item) => {
                if (typeof item === "string") return item;
                if (item?.msg) {
                    const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
                    return loc ? `${loc}: ${item.msg}` : item.msg;
                }
                return JSON.stringify(item);
            })
            .join("; ");
    }

    if (Array.isArray(payload?.errors)) {
        return payload.errors
            .map((item) => item.msg || item.message || JSON.stringify(item))
            .join("; ");
    }

    return "";
}

async function extractErrorMessage(response) {
    let payload = null;
    try {
        payload = await response.clone().json();
    } catch (_) {
        payload = null;
    }

    const validationMessage = collectValidationMessages(payload);
    if (validationMessage) return validationMessage;

    if (typeof payload?.detail === "string" && payload.detail.trim()) {
        return payload.detail;
    }

    const statusMessages = {
        400: "Некорректный запрос (400)",
        413: "Файл превышает допустимый размер (413)",
        415: "Неподдерживаемый тип файла (415)",
        500: "Внутренняя ошибка сервера (500)",
        503: "Сервис временно недоступен (503)",
        524: "Соединение прервано туннелем (524) — возможно, истёк таймаут ngrok",
    };
    return statusMessages[response.status]
        || response.statusText
        || `Ошибка HTTP ${response.status}`;
}

async function fetchWithFallback(path, options = {}, timeoutMs = TIMEOUTS.default) {
    const urls = buildCandidateUrls(path);

    const timeoutController = new AbortController();
    const timeoutId = setTimeout(
        () => timeoutController.abort(
            new DOMException(`Превышено время ожидания (${Math.round(timeoutMs / 1000)} с)`, "TimeoutError")
        ),
        timeoutMs,
    );

    const signals = [timeoutController.signal];
    if (options.signal) signals.push(options.signal);

    const combinedSignal = typeof AbortSignal.any === "function"
        ? AbortSignal.any(signals)
        : timeoutController.signal;

    const fetchOptions = { ...options, signal: combinedSignal };
    let lastResponse = null;

    try {
        for (const url of urls) {
            const response = await fetch(url, fetchOptions);
            if (response.ok) return response;
            lastResponse = response;
            if (response.status !== 404) {
                throw new Error(await extractErrorMessage(response));
            }
        }
    } catch (err) {
        if (err.name === "AbortError" || err.name === "TimeoutError") throw err;
        if (err.name === "TypeError" && err.message.includes("fetch")) {
            throw new Error("Не удалось подключиться к серверу. Проверьте, что сервер запущен и туннель активен.");
        }
        throw err;
    } finally {
        clearTimeout(timeoutId);
    }

    if (lastResponse) throw new Error(await extractErrorMessage(lastResponse));
    throw new Error("Не удалось получить ответ от сервера.");
}

async function fetchJson(path, options = {}, timeoutMs = TIMEOUTS.default) {
    const response = await fetchWithFallback(path, options, timeoutMs);
    return response.json();
}

async function fetchBlob(path, options = {}, timeoutMs = TIMEOUTS.default) {
    const response = await fetchWithFallback(path, options, timeoutMs);
    return response.blob();
}

function buildVideoQuery({ sampleEvery = 1, maxFrames = null, useTracking = true }) {
    const params = new URLSearchParams();
    params.set("sample_every", String(sampleEvery));
    params.set("use_tracking", String(Boolean(useTracking)));
    if (maxFrames !== null && maxFrames !== undefined && maxFrames !== "") {
        params.set("max_frames", String(maxFrames));
    }
    return params.toString();
}

export async function checkHealth(signal) {
    return fetchJson("/health", { signal });
}

export async function inferImage(file, signal) {
    const formData = new FormData();
    formData.append("file", file);
    return fetchJson("/inference/image", { method: "POST", body: formData, signal });
}

export async function fetchAnnotatedImageUrl(file, signal) {
    const formData = new FormData();
    formData.append("file", file);
    const blob = await fetchBlob(
        "/inference/annotated",
        { method: "POST", body: formData, signal },
    );
    return URL.createObjectURL(blob);
}

export async function analyzeVideo({ file, sampleEvery = 1, maxFrames = null, useTracking = true, signal }) {
    const formData = new FormData();
    formData.append("file", file);
    return fetchJson(
        `/video/analyze?${buildVideoQuery({ sampleEvery, maxFrames, useTracking })}`,
        { method: "POST", body: formData, signal },
        TIMEOUTS.video_analyze,
    );
}

export async function fetchAnnotatedVideoUrl({ file, sampleEvery = 1, maxFrames = null, useTracking = true, signal }) {
    const formData = new FormData();
    formData.append("file", file);
    const blob = await fetchBlob(
        `/video/annotated?${buildVideoQuery({ sampleEvery, maxFrames, useTracking })}`,
        { method: "POST", body: formData, signal },
        TIMEOUTS.video_annotate,
    );
    return URL.createObjectURL(blob);
}
