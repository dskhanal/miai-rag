// ============================================================================
// MiAi AUDIO MANAGER
// ============================================================================
// Main-thread audio controller kept outside index.html.
//
// Responsibilities:
//   - sentence chunking
//   - Audio Worker lifecycle / messaging
//   - Kitten audio playback
//   - browser speechSynthesis fallback
//   - stop/cancel state
//   - playback callbacks for UI highlighting / microphone coordination
//
// The actual Kitten-TTS synthesis runs in audio-worker.js.
// ============================================================================

export class AudioManager {
    constructor({ workerUrl, onStateChange = null, onLog = null, onError = null } = {}) {
        if (!workerUrl) throw new Error("AudioManager requires workerUrl.");

        this.workerUrl = workerUrl;
        this.onStateChange = typeof onStateChange === "function" ? onStateChange : null;
        this.onLog = typeof onLog === "function" ? onLog : null;
        this.onError = typeof onError === "function" ? onError : null;

        this.worker = null;
        this.audio = new Audio();
        this.audio.preload = "auto";

        this.enabled = true;
        this.voice = "Bella";
        this.model = "nano";
        this.speed = 1;

        this.playing = false;
        this.runId = 0;
        this.requestId = 0;
        this.pending = new Map();
        this.activeObjectUrl = null;
        this.requestTimeoutMs = 120000;
    }

    setEnabled(enabled) {
        this.enabled = Boolean(enabled);
    }

    setVoice(voice) {
        this.voice = String(voice || "Bella");
    }

    setModel(model) {
        const value = String(model || "").trim();
        if (value) this.model = value;
    }

    setSpeed(speed) {
        const value = Number(speed);
        if (Number.isFinite(value) && value > 0) this.speed = value;
    }

    isPlaying() {
        return this.playing;
    }

    getSentenceChunks(text) {
        const source = String(text || "");
        const matches = source.match(/[^.!?\n]+[.!?\n]+|[^.!?\n]+$/g) || [source];
        return matches.map(s => s.trim()).filter(Boolean);
    }

    ensureWorker() {
        if (this.worker) return this.worker;

        this.worker = new Worker(this.workerUrl, { type: "module" });

        this.worker.addEventListener("message", (event) => {
            const message = event.data || {};

            if (message.type === "AUDIO_LOG") {
                this.onLog?.(`[Audio Worker] ${message.message}`);
                return;
            }

            const pending = this.pending.get(message.requestId);
            if (!pending) return;

            if (pending.timeoutId) {
                clearTimeout(pending.timeoutId);
                pending.timeoutId = null;
            }

            if (message.type === "SYNTHESIS_RESULT") {
                this.pending.delete(message.requestId);
                pending.resolve({
                    ok: true,
                    index: message.index,
                    buffer: message.buffer,
                    mimeType: message.mimeType || "audio/wav"
                });
                return;
            }

            if (message.type === "SYNTHESIS_CANCELLED") {
                this.pending.delete(message.requestId);
                pending.resolve({ ok: false, cancelled: true, index: message.index });
                return;
            }

            if (message.type === "SYNTHESIS_ERROR") {
                this.pending.delete(message.requestId);
                pending.resolve({
                    ok: false,
                    cancelled: false,
                    index: message.index,
                    error: message.error || "Audio synthesis failed"
                });
            }
        });

        this.worker.addEventListener("error", (event) => {
            const error = new Error(event.message || "Audio Worker failed");
            this.onError?.(error);

            for (const pending of this.pending.values()) {
                if (pending.timeoutId) clearTimeout(pending.timeoutId);
                pending.resolve({ ok: false, cancelled: false, error: error.message });
            }
            this.pending.clear();

            try { this.worker?.terminate(); } catch (_) {}
            this.worker = null;
        });

        this.onLog?.("Audio Worker created.");
        return this.worker;
    }

    requestSynthesis(text, index, runId) {
        const worker = this.ensureWorker();
        const requestId = ++this.requestId;

        return new Promise((resolve) => {
            const timeoutId = setTimeout(() => {
                const pending = this.pending.get(requestId);
                if (!pending) return;

                this.pending.delete(requestId);

                resolve({
                    ok: false,
                    cancelled: runId !== this.runId,
                    index,
                    error: "Audio synthesis timed out."
                });

                this.onLog?.(
                    `[Audio Worker] Synthesis timeout for chunk ${index + 1}; using browser voice fallback.`
                );
            }, this.requestTimeoutMs);

            this.pending.set(requestId, {
                resolve,
                runId,
                index,
                timeoutId
            });

            worker.postMessage({
                type: "SYNTHESIZE_CHUNK",
                requestId,
                runId,
                index,
                text,
                voice: this.voice,
                model: this.model,
                speed: this.speed
            });
        });
    }

    async playBlob(buffer, mimeType, runId) {
        if (runId !== this.runId) return;

        this.revokeObjectUrl();
        const blob = new Blob([buffer], { type: mimeType || "audio/wav" });
        const url = URL.createObjectURL(blob);
        this.activeObjectUrl = url;
        this.audio.src = url;

        await new Promise((resolve) => {
            const finish = () => {
                this.audio.onended = null;
                this.audio.onerror = null;
                resolve();
            };

            this.audio.onended = finish;
            this.audio.onerror = finish;

            const playPromise = this.audio.play();
            if (playPromise?.catch) playPromise.catch(finish);
        });

        this.revokeObjectUrl();
    }

    async playBrowserFallback(text, runId) {
        if (runId !== this.runId) return;
        if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") return;

        await new Promise((resolve) => {
            const utterance = new SpeechSynthesisUtterance(String(text || ""));
            utterance.rate = Math.max(0.5, Math.min(2, Number(this.speed) || 1));
            utterance.onend = resolve;
            utterance.onerror = resolve;
            window.speechSynthesis.speak(utterance);
        });
    }

    async speak(text, {
        force = false,
        onPrepareChunks = null,
        onChunkChange = null
    } = {}) {
        if (!this.enabled && !force) return false;

        this.stop();

        const clean = String(text || "").replace(/[*#_`]/g, "").trim();
        if (!clean) return false;

        const sentences = this.getSentenceChunks(clean);
        if (!sentences.length) return false;

        const runId = ++this.runId;
        this.worker?.postMessage({ type: "RESET_CANCELLED", runId });
        this.setPlaying(true);
        onPrepareChunks?.(sentences);

        try {
            let nextSynthesis = this.requestSynthesis(sentences[0], 0, runId);

            for (let i = 0; i < sentences.length; i++) {
                if (runId !== this.runId) break;

                const result = await nextSynthesis;
                if (runId !== this.runId || result?.cancelled) break;

                nextSynthesis = i + 1 < sentences.length
                    ? this.requestSynthesis(sentences[i + 1], i + 1, runId)
                    : Promise.resolve(null);

                onChunkChange?.(i, true, sentences[i]);

                if (result?.ok && result.buffer) {
                    await this.playBlob(result.buffer, result.mimeType, runId);
                } else {
                    if (result?.error) {
                        this.onLog?.(`[Audio Worker] Kitten fallback for chunk ${i + 1}: ${result.error}`);
                    }
                    await this.playBrowserFallback(sentences[i], runId);
                }

                onChunkChange?.(i, false, sentences[i]);
            }
        } catch (error) {
            this.onError?.(error);
        } finally {
            if (runId === this.runId) {
                onChunkChange?.(-1, false, "");
                this.setPlaying(false);
            }
        }

        return true;
    }

    stop() {
        const cancelledRun = this.runId;
        this.runId++;

        if (this.worker && cancelledRun > 0) {
            try {
                this.worker.postMessage({
                    type: "CANCEL_RUN",
                    runId: cancelledRun
                });
            } catch (_) {}
        }

        // Release main-thread callers immediately. The synthesis worker may
        // still finish an already-running WebGPU operation, but any late
        // result is ignored because its request is no longer pending.
        for (const [requestId, pending] of this.pending.entries()) {
            if (pending.runId !== cancelledRun) continue;
            if (pending.timeoutId) clearTimeout(pending.timeoutId);
            pending.resolve({
                ok: false,
                cancelled: true,
                index: pending.index
            });
            this.pending.delete(requestId);
        }

        try {
            this.audio.pause();
            this.audio.currentTime = 0;
        } catch (_) {}

        this.revokeObjectUrl();

        if ("speechSynthesis" in window) {
            try { window.speechSynthesis.cancel(); } catch (_) {}
        }

        this.setPlaying(false);
    }

    setPlaying(playing) {
        const next = Boolean(playing);
        if (this.playing === next) return;
        this.playing = next;
        this.onStateChange?.({ playing: next });
    }

    revokeObjectUrl() {
        if (!this.activeObjectUrl) return;
        try { URL.revokeObjectURL(this.activeObjectUrl); } catch (_) {}
        this.activeObjectUrl = null;
    }

    destroy() {
        this.stop();
        for (const pending of this.pending.values()) {
            if (pending.timeoutId) clearTimeout(pending.timeoutId);
            pending.resolve({ ok: false, cancelled: true });
        }
        this.pending.clear();
        try { this.worker?.terminate(); } catch (_) {}
        this.worker = null;
        this.audio.removeAttribute("src");
    }
}
