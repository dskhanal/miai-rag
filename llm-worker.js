import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm";

// ============================================================================
// MiAi WebLLM Execution Worker
// ============================================================================
//
// Keep this worker deliberately small.
//
// Responsibilities:
// - Receive WebLLM's worker protocol from CreateWebWorkerMLCEngine().
// - Execute model loading / inference work away from the main UI thread.
//
// Not responsibilities:
// - Model catalogue or cache-status UI.
// - RAG retrieval/ranking.
// - Prompt construction.
// - Audio.
// - Arena storage.
//
// Those responsibilities remain in their existing MiAi components so the
// WebWorkerMLCEngineHandler protocol is not mixed with custom application
// messages.
// ============================================================================

const handler = new WebWorkerMLCEngineHandler();

self.onmessage = (event) => {
    handler.onmessage(event);
};
