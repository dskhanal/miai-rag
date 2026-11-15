// ============================================================================
// MiAi REASONING / BRANCH WORKER
// Deterministic branch-state engine for source-centred reasoning.
//
// Responsibilities:
// - Maintain reasoning sessions and parent/child branch relationships.
// - Persist branch nodes in IndexedDB.
// - Keep source evidence separate from MiAi-derived reasoning.
// - Build compact branch context for deep ANALYSE / EVALUATE / APPLY / COMPARE.
// - Return inherited evidence references for the next retrieval/generation step.
//
// Non-responsibilities:
// - Does NOT run WebLLM.
// - Does NOT retrieve source passages.
// - Does NOT build model prompts.
// - Does NOT decide whether a source claim is true.
// ============================================================================

const REASONING_VERSION = "miai-reasoning-1.1.0";
const DB_NAME = "miai-reasoning-db";
const DB_VERSION = 1;
const SESSION_STORE = "sessions";
const NODE_STORE = "nodes";

const OPERATIONS = Object.freeze([
    "source_answer",
    "super_inquiry",
    "analyse",
    "evaluate",
    "apply",
    "compare"
]);


const RETRIEVAL_STOP_WORDS = new Set([
    "the","and","for","that","this","with","from","into","than","then","are","was","were",
    "been","being","have","has","had","will","would","could","should","may","might","can",
    "their","there","they","them","our","your","you","about","also","such","these","those",
    "using","used","use","reading","source","branch","analyse","analyze","evaluate","apply","compare",
    "further","focus","evidence","response","result","context","derived","miai"
]);

const LIMITS = Object.freeze({
    MAX_RESPONSE_CHARS: 14000,
    MAX_PROMPT_CHARS: 5000,
    MAX_CONTEXT_CHARS: 9000,
    MAX_EVIDENCE_ITEMS: 16,
    MAX_EVIDENCE_TEXT_CHARS: 2600,
    MAX_LINEAGE_NODES: 6,
    MAX_NODE_EXCERPT_CHARS: 1800,
    MAX_SESSIONS_RETURNED: 20
});

let dbPromise = null;

function clean(value) {
    return String(value ?? "").trim();
}

function nowISO() {
    return new Date().toISOString();
}

function uid(prefix = "id") {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
        return `${prefix}_${crypto.randomUUID()}`;
    }
    return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function clampText(value, maxChars) {
    const text = clean(value);
    if (text.length <= maxChars) return text;
    return `${text.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}

function normalizeOperation(value) {
    const operation = clean(value).toLowerCase();
    if (!OPERATIONS.includes(operation)) {
        throw new Error(`Unsupported reasoning operation "${operation}".`);
    }
    return operation;
}

function normalizeEvidenceItem(item = {}) {
    const text = clampText(item.text, LIMITS.MAX_EVIDENCE_TEXT_CHARS);
    if (!text) return null;

    const sentenceNumber = Number(item.sentenceNumber);
    const wordPos = Number(item.wordPos);
    const pageNumber = item.pageNumber === null || item.pageNumber === undefined
        ? null
        : item.pageNumber;

    const fallbackId = [
        clean(item.sourceFingerprint) || "source",
        clean(item.sectionId) || "section",
        Number.isFinite(sentenceNumber) ? `sentence_${sentenceNumber}` : "sentence",
        Number.isFinite(wordPos) ? `word_${wordPos}` : "word"
    ].join(":");

    return {
        evidenceId: clean(item.evidenceId) || clean(item.id) || fallbackId,
        sourceFingerprint: clean(item.sourceFingerprint) || null,
        sourceTitle: clean(item.sourceTitle) || null,
        sectionId: clean(item.sectionId) || null,
        sectionHeading: clean(item.sectionHeading) || null,
        pageNumber,
        sentenceNumber: Number.isFinite(sentenceNumber) ? sentenceNumber : null,
        wordPos: Number.isFinite(wordPos) ? wordPos : null,
        contextualNeighbour: Boolean(item.contextualNeighbour),
        text
    };
}

function normalizeEvidence(value) {
    if (!Array.isArray(value)) return [];

    const seen = new Set();
    const out = [];

    for (const raw of value) {
        if (out.length >= LIMITS.MAX_EVIDENCE_ITEMS) break;
        const item = normalizeEvidenceItem(raw);
        if (!item || seen.has(item.evidenceId)) continue;
        seen.add(item.evidenceId);
        out.push(item);
    }

    return out;
}

function normalizeRetrievalMeta(value = {}) {
    const confidenceValue = Number(value.confidence);
    return {
        confidence: Number.isFinite(confidenceValue)
            ? Math.max(0, Math.min(1, confidenceValue))
            : null,
        coverage: clean(value.coverage) || null,
        quality: clean(value.quality) || null,
        resultCount: Number.isFinite(Number(value.resultCount))
            ? Math.max(0, Math.floor(Number(value.resultCount)))
            : null
    };
}

function openDB() {
    if (dbPromise) return dbPromise;

    dbPromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onupgradeneeded = () => {
            const db = request.result;

            if (!db.objectStoreNames.contains(SESSION_STORE)) {
                const sessions = db.createObjectStore(SESSION_STORE, { keyPath: "id" });
                sessions.createIndex("sourceFingerprint", "sourceFingerprint", { unique: false });
                sessions.createIndex("updatedAt", "updatedAt", { unique: false });
            }

            if (!db.objectStoreNames.contains(NODE_STORE)) {
                const nodes = db.createObjectStore(NODE_STORE, { keyPath: "id" });
                nodes.createIndex("sessionId", "sessionId", { unique: false });
                nodes.createIndex("parentId", "parentId", { unique: false });
                nodes.createIndex("createdAt", "createdAt", { unique: false });
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("Could not open reasoning database."));
    });

    return dbPromise;
}

function requestAsPromise(request) {
    return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("IndexedDB request failed."));
    });
}

async function getRecord(storeName, key) {
    const db = await openDB();
    const tx = db.transaction(storeName, "readonly");
    return requestAsPromise(tx.objectStore(storeName).get(key));
}

async function putRecord(storeName, value) {
    const db = await openDB();
    const tx = db.transaction(storeName, "readwrite");
    await requestAsPromise(tx.objectStore(storeName).put(value));
    return value;
}

async function deleteRecord(storeName, key) {
    const db = await openDB();
    const tx = db.transaction(storeName, "readwrite");
    await requestAsPromise(tx.objectStore(storeName).delete(key));
}

async function getNodesForSession(sessionId) {
    const db = await openDB();
    const tx = db.transaction(NODE_STORE, "readonly");
    const index = tx.objectStore(NODE_STORE).index("sessionId");
    const rows = await requestAsPromise(index.getAll(sessionId));
    return rows.sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
}

async function getChildren(parentId) {
    const db = await openDB();
    const tx = db.transaction(NODE_STORE, "readonly");
    const index = tx.objectStore(NODE_STORE).index("parentId");
    const rows = await requestAsPromise(index.getAll(parentId));
    return rows.sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
}

async function startSession(payload = {}) {
    const sourceFingerprint = clean(payload.sourceFingerprint) || "unindexed-source";
    const sourceTitle = clean(payload.sourceTitle) || "Current Reading";

    const session = {
        id: uid("session"),
        sourceFingerprint,
        sourceTitle,
        createdAt: nowISO(),
        updatedAt: nowISO(),
        rootNodeId: null,
        activeNodeId: null,
        nodeCount: 0
    };

    await putRecord(SESSION_STORE, session);
    return session;
}

async function touchSession(sessionId, nodeId = null, rootNodeId = null) {
    const session = await getRecord(SESSION_STORE, sessionId);
    if (!session) throw new Error("Reasoning session not found.");

    session.updatedAt = nowISO();
    if (nodeId) session.activeNodeId = nodeId;
    if (rootNodeId && !session.rootNodeId) session.rootNodeId = rootNodeId;
    session.nodeCount = (await getNodesForSession(sessionId)).length;
    await putRecord(SESSION_STORE, session);
    return session;
}

async function saveNode(payload = {}) {
    const sessionId = clean(payload.sessionId);
    if (!sessionId) throw new Error("SAVE_NODE requires sessionId.");

    const session = await getRecord(SESSION_STORE, sessionId);
    if (!session) throw new Error("Reasoning session not found.");

    const operation = normalizeOperation(payload.operation || "source_answer");
    const parentId = clean(payload.parentId) || null;
    const parent = parentId ? await getRecord(NODE_STORE, parentId) : null;

    if (parentId && (!parent || parent.sessionId !== sessionId)) {
        throw new Error("Parent reasoning node was not found in this session.");
    }

    const userPrompt = clampText(payload.userPrompt, LIMITS.MAX_PROMPT_CHARS);
    const response = clampText(payload.response, LIMITS.MAX_RESPONSE_CHARS);
    if (!response) throw new Error("SAVE_NODE requires a response.");

    const evidence = normalizeEvidence(payload.evidence);
    const createdAt = nowISO();

    const node = {
        id: uid("node"),
        sessionId,
        parentId,
        rootId: parent ? (parent.rootId || parent.id) : null,
        depth: parent ? (Number(parent.depth) || 0) + 1 : 0,
        operation,
        userPrompt,
        applicationContext: clampText(payload.applicationContext, LIMITS.MAX_PROMPT_CHARS) || null,
        response,
        evidence,
        evidenceIds: evidence.map(item => item.evidenceId),
        retrievalMeta: normalizeRetrievalMeta(payload.retrievalMeta),
        mode: clean(payload.mode) || "study",
        sourceFingerprint: session.sourceFingerprint,
        sourceTitle: session.sourceTitle,
        createdAt,
        updatedAt: createdAt
    };

    if (!node.rootId) node.rootId = node.id;

    await putRecord(NODE_STORE, node);
    const updatedSession = await touchSession(sessionId, node.id, node.rootId);

    return {
        node,
        session: updatedSession
    };
}

async function getLineage(nodeId) {
    const lineage = [];
    const visited = new Set();
    let currentId = clean(nodeId);

    while (currentId && lineage.length < 40) {
        if (visited.has(currentId)) break;
        visited.add(currentId);

        const node = await getRecord(NODE_STORE, currentId);
        if (!node) break;

        lineage.push(node);
        currentId = node.parentId;
    }

    return lineage.reverse();
}

function mergeInheritedEvidence(lineage) {
    const seen = new Set();
    const out = [];

    // Newer evidence has priority because it is usually more specific to the
    // active branch. Traverse backwards, then restore a readable order.
    for (let i = lineage.length - 1; i >= 0; i--) {
        const evidence = normalizeEvidence(lineage[i].evidence);
        for (const item of evidence) {
            if (out.length >= LIMITS.MAX_EVIDENCE_ITEMS) break;
            if (seen.has(item.evidenceId)) continue;
            seen.add(item.evidenceId);
            out.push(item);
        }
    }

    return out.reverse();
}

function compactLineageContext(lineage) {
    const recent = lineage.slice(-LIMITS.MAX_LINEAGE_NODES);
    const blocks = [];

    for (const node of recent) {
        const label = node.operation === "source_answer"
            ? "SOURCE ANSWER"
            : String(node.operation || "reasoning").toUpperCase();

        const prompt = clampText(node.userPrompt, 700);
        const response = clampText(node.response, LIMITS.MAX_NODE_EXCERPT_CHARS);

        const lines = [
            `[${label} | depth ${node.depth || 0}]`,
            prompt ? `User focus: ${prompt}` : null,
            `MiAi-derived result (NOT source evidence): ${response}`
        ].filter(Boolean);

        blocks.push(lines.join("\n"));
    }

    const text = blocks.join("\n\n");
    return clampText(text, LIMITS.MAX_CONTEXT_CHARS);
}

function extractRetrievalKeywords(text, maxTerms = 18) {
    const tokens = clean(text).toLowerCase().match(/[a-z][a-z0-9'-]{2,}/g) || [];
    const counts = new Map();

    for (const token of tokens) {
        const term = token.replace(/^'+|'+$/g, "");
        if (!term || term.length < 4 || RETRIEVAL_STOP_WORDS.has(term)) continue;
        counts.set(term, (counts.get(term) || 0) + 1);
    }

    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
        .slice(0, maxTerms)
        .map(([term]) => term);
}

function extractRetrievalSeed(lineage, operation, userPrompt) {
    const parent = lineage[lineage.length - 1];
    const grandparent = lineage[lineage.length - 2];

    const inheritedText = [
        parent ? parent.userPrompt : "",
        parent ? parent.response : "",
        grandparent ? grandparent.userPrompt : ""
    ].filter(Boolean).join(" ");

    const inheritedTerms = extractRetrievalKeywords(inheritedText, 18);

    const operationCue = {
        analyse: "relationship implication mechanism assumption",
        evaluate: "strength limitation adequacy uncertainty support",
        apply: "principle requirement action condition",
        compare: "similarity difference principle condition"
    }[operation] || "";

    return clampText(
        [clean(userPrompt), operationCue, inheritedTerms.join(" ")]
            .filter(Boolean)
            .join(" "),
        1800
    );
}

async function prepareBranch(payload = {}) {
    const sessionId = clean(payload.sessionId);
    const parentNodeId = clean(payload.parentNodeId);
    const operation = normalizeOperation(payload.operation);
    const userPrompt = clampText(payload.userPrompt, LIMITS.MAX_PROMPT_CHARS);

    if (!sessionId) throw new Error("PREPARE_BRANCH requires sessionId.");
    if (!parentNodeId) throw new Error("PREPARE_BRANCH requires parentNodeId.");
    if (!["analyse", "evaluate", "apply", "compare"].includes(operation)) {
        throw new Error("PREPARE_BRANCH requires analyse, evaluate, apply, or compare.");
    }

    const parent = await getRecord(NODE_STORE, parentNodeId);
    if (!parent || parent.sessionId !== sessionId) {
        throw new Error("Parent reasoning node was not found in this session.");
    }

    const lineage = await getLineage(parentNodeId);
    const inheritedEvidence = mergeInheritedEvidence(lineage);
    const branchContext = compactLineageContext(lineage);
    const retrievalQuery = extractRetrievalSeed(lineage, operation, userPrompt);

    return {
        reasoningVersion: REASONING_VERSION,
        sessionId,
        parentNodeId,
        rootNodeId: parent.rootId || parent.id,
        nextDepth: (Number(parent.depth) || 0) + 1,
        operation,
        branchContext,
        inheritedEvidence,
        inheritedEvidenceIds: inheritedEvidence.map(item => item.evidenceId),
        retrievalQuery,
        applicationContext: operation === "apply" || operation === "compare"
            ? userPrompt
            : ""
    };
}

async function getTree(payload = {}) {
    const sessionId = clean(payload.sessionId);
    if (!sessionId) throw new Error("GET_TREE requires sessionId.");

    const session = await getRecord(SESSION_STORE, sessionId);
    if (!session) throw new Error("Reasoning session not found.");

    const nodes = await getNodesForSession(sessionId);
    return {
        reasoningVersion: REASONING_VERSION,
        session,
        nodes
    };
}

async function getNode(payload = {}) {
    const nodeId = clean(payload.nodeId);
    if (!nodeId) throw new Error("GET_NODE requires nodeId.");
    const node = await getRecord(NODE_STORE, nodeId);
    if (!node) throw new Error("Reasoning node not found.");
    return node;
}

async function listChildren(payload = {}) {
    const parentNodeId = clean(payload.parentNodeId);
    if (!parentNodeId) throw new Error("LIST_CHILDREN requires parentNodeId.");
    return getChildren(parentNodeId);
}

async function clearSession(payload = {}) {
    const sessionId = clean(payload.sessionId);
    if (!sessionId) throw new Error("CLEAR_SESSION requires sessionId.");

    const nodes = await getNodesForSession(sessionId);
    for (const node of nodes) {
        await deleteRecord(NODE_STORE, node.id);
    }
    await deleteRecord(SESSION_STORE, sessionId);

    return { cleared: true, sessionId, deletedNodes: nodes.length };
}

function getInfo() {
    return {
        reasoningVersion: REASONING_VERSION,
        operations: [...OPERATIONS],
        limits: { ...LIMITS },
        persistence: "IndexedDB"
    };
}

self.addEventListener("message", async (event) => {
    const message = event.data || {};
    const { id, type, payload } = message;

    try {
        if (id === undefined || id === null) {
            throw new Error("Reasoning Worker request requires an id.");
        }

        let result;

        switch (type) {
            case "START_SESSION":
                result = await startSession(payload);
                break;
            case "SAVE_NODE":
                result = await saveNode(payload);
                break;
            case "PREPARE_BRANCH":
                result = await prepareBranch(payload);
                break;
            case "GET_TREE":
                result = await getTree(payload);
                break;
            case "GET_NODE":
                result = await getNode(payload);
                break;
            case "LIST_CHILDREN":
                result = await listChildren(payload);
                break;
            case "CLEAR_SESSION":
                result = await clearSession(payload);
                break;
            case "GET_REASONING_INFO":
                result = getInfo();
                break;
            default:
                throw new Error(`Unknown Reasoning Worker request: ${String(type)}`);
        }

        self.postMessage({ id, ok: true, result });
    } catch (error) {
        self.postMessage({
            id,
            ok: false,
            error: error?.message || String(error)
        });
    }
});
