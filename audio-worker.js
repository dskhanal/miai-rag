// ============================================================================
// MiAi AUDIO SYNTHESIS WORKER
// ============================================================================
// Purpose:
//   - Keep Kitten-TTS model loading and synthesis out of the main HTML file.
//   - Return encoded audio bytes to audio-manager.js.
//   - Never touches DOM, <audio>, microphone, or speechSynthesis APIs.
//
// Browser fallback speech synthesis is intentionally handled by audio-manager.js
// because speechSynthesis / HTMLAudioElement are main-thread browser APIs.
// ============================================================================

let kittenModulePromise = null;
const cancelledRuns = new Set();

async function getKittenTTS() {
    if (!kittenModulePromise) {
        kittenModulePromise = import(
            "https://cdn.jsdelivr.net/npm/kitten-tts-webgpu@0.1.1/+esm"
        ).catch(error => {
            // Allow a later request to retry if the CDN/module load failed
            // transiently rather than keeping a permanently rejected promise.
            kittenModulePromise = null;
            throw error;
        });
    }

    return kittenModulePromise;
}

function postWorkerLog(message, runId = null) {
    self.postMessage({
        type: "AUDIO_LOG",
        runId,
        message: String(message || "")
    });
}

function pruneCancelledRuns(currentRunId = null) {
    // runId is monotonic in AudioManager. Keep only a small recent window so
    // a long session cannot grow this Set indefinitely.
    const current = Number(currentRunId);

    if (Number.isFinite(current)) {
        for (const value of cancelledRuns) {
            const numeric = Number(value);
            if (Number.isFinite(numeric) && numeric < current - 20) {
                cancelledRuns.delete(value);
            }
        }
    }

    if (cancelledRuns.size > 64) {
        const keep = [...cancelledRuns].slice(-32);
        cancelledRuns.clear();
        keep.forEach(value => cancelledRuns.add(value));
    }
}

self.addEventListener("message", async (event) => {
    const data = event.data || {};

    if (data.type === "CANCEL_RUN") {
        cancelledRuns.add(data.runId);
        pruneCancelledRuns(data.runId);
        return;
    }

    if (data.type === "RESET_CANCELLED") {
        cancelledRuns.delete(data.runId);
        pruneCancelledRuns(data.runId);
        return;
    }

    if (data.type !== "SYNTHESIZE_CHUNK") return;

    const {
        requestId,
        runId,
        index = 0,
        text = "",
        voice = "Bella",
        model = "nano",
        speed = 1
    } = data;

    if (!String(text).trim() || cancelledRuns.has(runId)) {
        self.postMessage({
            type: "SYNTHESIS_CANCELLED",
            requestId,
            runId,
            index
        });
        return;
    }

    try {
        const { textToSpeech } = await getKittenTTS();

        if (cancelledRuns.has(runId)) {
            self.postMessage({ type: "SYNTHESIS_CANCELLED", requestId, runId, index });
            return;
        }

        const blob = await textToSpeech(String(text), {
            model,
            voice,
            speed,
            onProgress: (message) => postWorkerLog(message, runId)
        });

        if (cancelledRuns.has(runId)) {
            self.postMessage({ type: "SYNTHESIS_CANCELLED", requestId, runId, index });
            return;
        }

        if (!(blob instanceof Blob)) {
            throw new Error("Kitten-TTS did not return an audio Blob.");
        }

        const buffer = await blob.arrayBuffer();

        self.postMessage({
            type: "SYNTHESIS_RESULT",
            requestId,
            runId,
            index,
            mimeType: blob.type || "audio/wav",
            buffer
        }, [buffer]);
    } catch (error) {
        self.postMessage({
            type: "SYNTHESIS_ERROR",
            requestId,
            runId,
            index,
            error: error?.message || String(error)
        });
    }
});
