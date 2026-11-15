// ============================================================================
// MiAi RETRIEVAL WORKER — Retrieval V2
// ============================================================================
// Lightweight, browser-local retrieval planning and chunk ranking.
//
// Responsibilities:
// - Rule-based query intent detection and lightweight decomposition.
// - Document-aware query expansion using vocabulary already in the source.
// - BM25 lexical ranking over ingestion-worker chunks.
// - Optional MiniLM semantic embeddings via Transformers.js (user opt-in).
// - Reciprocal Rank Fusion (RRF) across query variants and retrieval lanes.
// - A second algorithmic retrieval round for Super Inquiry / deep mode.
// - Section-diverse candidate selection.
//
// Non-responsibilities:
// - Does NOT run the answer LLM.
// - Does NOT decide final sentence citations.
// - Does NOT generate source facts or synonyms from outside the document.
// - Semantic retrieval is optional; lexical/planner mode works by itself.
// ============================================================================

const RETRIEVAL_VERSION = "miai-retrieval-2.3.0";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const SEMANTIC_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const RRF_K = 60;

const MESH = Object.freeze({
    NORMAL_DEPTH: 1,
    DEEP_DEPTH: 2,
    MAX_SEEDS_NORMAL: 5,
    MAX_SEEDS_DEEP: 8,
    MAX_NEIGHBOURS_PER_NODE: 14,
    DEPTH_DECAY: 0.58,
    MIN_PROPAGATED_SCORE: 0.0008,
    RELATION_WEIGHTS: Object.freeze({
        same_heading: 0.96,
        same_section: 0.82,
        shared_phrase: 0.78,
        adjacent: 0.70,
        rare_terms: 0.58,
        human_relation: 0.98
    })
});

const STOP_WORDS = new Set([
    "the","and","for","that","this","with","from","what","when","where","which","who","whom",
    "whose","why","how","does","did","are","was","were","will","would","could","should","can",
    "may","might","have","has","had","into","about","your","their","there","then","than","they",
    "them","his","her","its","our","you","use","using","used","explain","describe","tell","give",
    "show","reading","document","source","please","also","more","most","some","any","all","other",
    "further","current","response","answer"
]);

const QUERY_FILLER = new Set([
    "explain","describe","tell","show","give","discuss","outline","identify","summarise","summarize",
    "analyze","analyse","evaluate","assess","compare","apply"
]);

const state = {
    source: null,
    semantic: {
        enabled: false,
        allowDownload: false,
        status: "off",
        message: "Semantic lane off",
        extractor: null,
        embeddings: null,
        loadingPromise: null,
        indexedFingerprint: null,
        failed: false
    }
};

function clean(value) {
    return String(value ?? "").replace(/\s+/g, " ").trim();
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function normalizeToken(token) {
    let t = String(token || "").toLowerCase().replace(/[^a-z0-9']/g, "");
    if (t.length > 6 && t.endsWith("ingly")) t = t.slice(0, -5);
    else if (t.length > 5 && t.endsWith("ation")) t = t.slice(0, -5);
    else if (t.length > 5 && t.endsWith("ing")) t = t.slice(0, -3);
    else if (t.length > 4 && t.endsWith("ied")) t = `${t.slice(0, -3)}y`;
    else if (t.length > 4 && t.endsWith("ed")) t = t.slice(0, -2);
    else if (t.length > 4 && t.endsWith("es")) t = t.slice(0, -2);
    else if (t.length > 3 && t.endsWith("s")) t = t.slice(0, -1);
    return t;
}

function tokens(text, { unique = false, keepFiller = false } = {}) {
    const raw = String(text || "").toLowerCase().match(/[a-z0-9']+/g) || [];
    const out = raw
        .map(normalizeToken)
        .filter(token => token.length > 2 && !STOP_WORDS.has(token) && (keepFiller || !QUERY_FILLER.has(token)));
    return unique ? [...new Set(out)] : out;
}

function tokenFrequency(items) {
    const map = new Map();
    for (const item of items) map.set(item, (map.get(item) || 0) + 1);
    return map;
}

function detectIntent(query) {
    const q = clean(query).toLowerCase();
    const meaningful = tokens(q, { unique: true });

    const compare = /\b(compare|comparison|versus|vs\.?|difference|differences|similarit|contrast)\b/.test(q);
    const evaluate = /\b(evaluate|evaluation|assess|assessment|adequate|effective|effectiveness|strength|weakness|limitation|limitations|critique|quality)\b/.test(q);
    const apply = /\b(apply|application|scenario|context|practice|transfer|use this|in a new)\b/.test(q);
    const procedural = /\b(how (?:do|does|should|can|to)|steps?|process|procedure|method)\b/.test(q);
    const causal = /\b(why|cause|causes|causal|effect|effects|impact|influence|lead to|result in)\b/.test(q);
    const evidence = /\b(evidence|support|finding|findings|data|result|results|showed|demonstrated)\b/.test(q);
    const limitation = /\b(limitation|limitations|caution|uncertain|uncertainty|missing|gap|gaps)\b/.test(q);
    const explicitMulti = /\b(?:benefits?|advantages?|strengths?)\b.*\b(?:limitations?|disadvantages?|weaknesses?)\b/.test(q)
        || /\b(?:and how|and why|and what|as well as|but also)\b/.test(q);
    const multiPart = countQuestionParts(q) > 1 || explicitMulti || (/\band\b/.test(q) && meaningful.length >= 7);

    let id = "factual";
    if (compare) id = "compare";
    else if (evaluate) id = "evaluate";
    else if (apply) id = "apply";
    else if (procedural) id = "procedural";
    else if (causal) id = "causal";
    else if (evidence) id = "evidence";
    else if (limitation) id = "limitation";
    else if (multiPart) id = "multi-part";

    return {
        id,
        compare,
        evaluate,
        apply,
        procedural,
        causal,
        evidence,
        limitation,
        multiPart
    };
}

function countQuestionParts(query) {
    const punctuationParts = String(query || "").split(/[?;]+/).map(clean).filter(Boolean);
    if (punctuationParts.length > 1) return punctuationParts.length;

    const clauses = String(query || "")
        .split(/\b(?:and also|as well as|while also|but also)\b/i)
        .map(clean)
        .filter(part => tokens(part).length >= 2);
    return clauses.length;
}

function splitSubqueries(query, intent, maxParts = 4) {
    const q = clean(query);
    if (!q) return [];

    const candidates = [];
    const push = value => {
        const v = clean(value).replace(/^[,;:\-]+|[,;:\-]+$/g, "").trim();
        if (tokens(v).length < 2) return;
        if (!candidates.some(existing => existing.toLowerCase() === v.toLowerCase())) candidates.push(v);
    };

    if (intent.compare) {
        const vsParts = q.split(/\b(?:versus|vs\.?)\b/i).map(clean).filter(Boolean);
        if (vsParts.length === 2) {
            push(vsParts[0]);
            push(vsParts[1]);
        }

        const compareMatch = q.match(/\bcompare\s+(.+?)\s+(?:and|with)\s+(.+?)(?:[?;]|$)/i)
            || q.match(/\bdifferences?\s+between\s+(.+?)\s+and\s+(.+?)(?:[?;]|$)/i);
        if (compareMatch) {
            push(compareMatch[1]);
            push(compareMatch[2]);
        }
    }

    const paired = q.match(/\b(benefits?|advantages?|strengths?)\s+and\s+(limitations?|disadvantages?|weaknesses?)\s+(?:of|for)\s+(.+?)(?=\s+and\s+(?:how|why|what)\b|[?;]|$)/i);
    if (paired) {
        const topic = clean(paired[3]);
        push(`${paired[1]} of ${topic}`);
        push(`${paired[2]} of ${topic}`);

        const tailMatch = q.match(/\band\s+(how|why|what)\s+(.+?)(?:[?;]|$)/i);
        if (tailMatch) push(`${topic} ${tailMatch[1]} ${tailMatch[2]}`);
    }

    q.split(/\b(?:and how|and why|and what)\b/i).forEach(push);
    q.split(/[?;]+/).forEach(push);

    if (intent.multiPart || intent.evaluate || intent.apply || intent.compare) {
        q.split(/\b(?:and also|as well as|while also|but also)\b/i).forEach(push);
    }

    // Conservative split for long "X and Y" questions. Avoid splitting short
    // noun phrases such as "health and wellbeing".
    if (candidates.length < 2 && tokens(q).length >= 9) {
        const andParts = q.split(/\band\b/i).map(clean).filter(Boolean);
        if (andParts.length === 2 && andParts.every(part => tokens(part).length >= 3)) {
            andParts.forEach(push);
        }
    }

    return candidates.slice(0, maxParts);
}

function looksLikeConversationalFollowup(query) {
    const q = clean(query).toLowerCase();
    const t = tokens(q, { unique: true });
    return /^(what|how) about\b|^(and|also|then)\b|\b(it|this|that|these|those|they|them)\b/.test(q) || t.length <= 3;
}

function sourceExpansionTerms(queryTokens, limit = 8) {
    const src = state.source;
    if (!src || !queryTokens.length) return [];

    const scores = new Map();
    const querySet = new Set(queryTokens);

    const hitCounts = src.chunkTokenSets.map(chunkTokenSet => {
        let hits = 0;
        for (const queryToken of queryTokens) if (chunkTokenSet.has(queryToken)) hits++;
        return hits;
    });
    const maxHits = hitCounts.length ? Math.max(...hitCounts) : 0;
    const minHits = queryTokens.length >= 4 && maxHits >= 2 ? 2 : 1;

    for (let i = 0; i < src.chunks.length; i++) {
        const queryHits = hitCounts[i] || 0;
        if (queryHits < minHits) continue;

        const tf = src.chunkTermFrequencies[i];
        for (const [term, count] of tf) {
            if (querySet.has(term) || STOP_WORDS.has(term) || term.length < 4) continue;
            const df = src.documentFrequency.get(term) || 1;
            const idf = Math.log(1 + (src.chunks.length + 1) / (df + 1));
            const score = queryHits * idf * (1 + Math.log1p(count));
            scores.set(term, (scores.get(term) || 0) + score);
        }

        const headingTerms = tokens(src.chunks[i].sectionHeading || "", { unique: true });
        for (const term of headingTerms) {
            if (querySet.has(term) || STOP_WORDS.has(term)) continue;
            scores.set(term, (scores.get(term) || 0) + 2.5 * queryHits);
        }
    }

    return [...scores.entries()]
        .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
        .slice(0, limit)
        .map(([term]) => term);
}

function planQuery(query, { deep = false, conversationHint = "" } = {}) {
    const original = clean(query);
    const intent = detectIntent(original);
    const originalTokens = tokens(original, { unique: true });
    const followup = Boolean(conversationHint && looksLikeConversationalFollowup(original));

    const hintTokens = followup ? tokens(conversationHint, { unique: true }).slice(0, 10) : [];
    const planningTokens = [...new Set([...originalTokens, ...hintTokens])];
    const expansionTerms = sourceExpansionTerms(planningTokens, deep ? 10 : 6);
    const subqueries = splitSubqueries(original, intent, deep ? 4 : 2);

    const variants = [];
    const pushVariant = (text, kind, weight = 1) => {
        const value = clean(text);
        if (!value || tokens(value).length < 1) return;
        if (variants.some(item => item.text.toLowerCase() === value.toLowerCase())) return;
        variants.push({ text: value, kind, weight });
    };

    pushVariant(original, "original", 1.15);

    if (followup) {
        pushVariant(`${original} ${hintTokens.join(" ")}`, "conversation-aware", 1.0);
    }

    if (originalTokens.length) {
        pushVariant(originalTokens.join(" "), "normalized", 0.95);
    }

    if (expansionTerms.length) {
        pushVariant(`${originalTokens.join(" ")} ${expansionTerms.slice(0, deep ? 8 : 5).join(" ")}`, "document-expanded", 1.0);
    }

    for (const subquery of subqueries) pushVariant(subquery, "subquery", deep ? 1.05 : 0.9);

    const maxVariants = deep ? 7 : 4;

    return {
        original,
        intent,
        followup,
        conversationHint: followup ? clean(conversationHint) : "",
        originalTokens,
        expansionTerms,
        subqueries,
        variants: variants.slice(0, maxVariants),
        deep
    };
}

function bm25ScoreForTerms(queryTerms, chunkIndex) {
    const src = state.source;
    if (!src || !queryTerms.length) return 0;

    const tf = src.chunkTermFrequencies[chunkIndex];
    const docLength = src.chunkTokenLengths[chunkIndex];
    const k1 = 1.2;
    const b = 0.75;
    let score = 0;

    for (const term of queryTerms) {
        const frequency = tf.get(term) || 0;
        if (!frequency) continue;
        const df = src.documentFrequency.get(term) || 0;
        const idf = Math.log(1 + ((src.chunks.length - df + 0.5) / (df + 0.5)));
        const denominator = frequency + k1 * (1 - b + b * (docLength / Math.max(1, src.avgChunkLength)));
        score += idf * ((frequency * (k1 + 1)) / Math.max(0.0001, denominator));
    }

    return score;
}

function lexicalRank(queryText, limit = 20) {
    const queryTerms = tokens(queryText, { unique: true });
    if (!state.source || !queryTerms.length) return [];

    const scored = state.source.chunks
        .map((chunk, index) => {
            const tf = state.source.chunkTermFrequencies[index];
            const matchedTerms = queryTerms.reduce((count, term) => count + (tf.has(term) ? 1 : 0), 0);
            const coverage = matchedTerms / Math.max(1, queryTerms.length);
            let score = bm25ScoreForTerms(queryTerms, index) * (0.35 + 0.9 * coverage);
            if (queryTerms.length >= 4 && matchedTerms < 2) score = 0;
            return {
                chunkId: chunk.id,
                index,
                score,
                matchedTerms,
                coverage
            };
        })
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score || a.index - b.index)
        .slice(0, limit);

    return scored;
}

function addRankListToRRF(scoreMap, rankList, { lane, variant, weight = 1 } = {}) {
    rankList.forEach((item, i) => {
        const rank = i + 1;
        const rrf = weight / (RRF_K + rank);
        let entry = scoreMap.get(item.chunkId);
        if (!entry) {
            entry = {
                chunkId: item.chunkId,
                rrfScore: 0,
                lexicalHits: [],
                semanticHits: []
            };
            scoreMap.set(item.chunkId, entry);
        }
        entry.rrfScore += rrf;
        const hit = { rank, rawScore: item.score, variant: variant || "query" };
        if (lane === "semantic") entry.semanticHits.push(hit);
        else entry.lexicalHits.push(hit);
    });
}

function topBridgeTerms(candidateIds, queryTokens, limit = 8) {
    const src = state.source;
    if (!src) return [];
    const querySet = new Set(queryTokens);
    const scores = new Map();

    for (const chunkId of candidateIds.slice(0, 2)) {
        const index = src.chunkIndexById.get(chunkId);
        if (index === undefined) continue;
        for (const [term, count] of src.chunkTermFrequencies[index]) {
            if (querySet.has(term) || STOP_WORDS.has(term) || term.length < 4) continue;
            const df = src.documentFrequency.get(term) || 1;
            const idf = Math.log(1 + (src.chunks.length + 1) / (df + 1));
            scores.set(term, (scores.get(term) || 0) + idf * (1 + Math.log1p(count)));
        }
    }

    return [...scores.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([term]) => term);
}

async function importTransformers() {
    return import(TRANSFORMERS_URL);
}

function postStatus(status, message, extra = {}) {
    state.semantic.status = status;
    state.semantic.message = message;
    self.postMessage({
        type: "RETRIEVAL_STATUS",
        status,
        message,
        semanticEnabled: state.semantic.enabled,
        modelId: SEMANTIC_MODEL_ID,
        ...extra
    });
}

async function ensureSemanticExtractor() {
    if (!state.semantic.enabled) return null;
    if (state.semantic.failed) return null;
    if (state.semantic.extractor) return state.semantic.extractor;
    if (state.semantic.loadingPromise) return state.semantic.loadingPromise;

    state.semantic.loadingPromise = (async () => {
        try {
            postStatus("loading", state.semantic.allowDownload
                ? "Loading optional MiniLM semantic retrieval model…"
                : "Checking cached MiniLM semantic retrieval model…");

            const { pipeline, env } = await importTransformers();
            env.useBrowserCache = true;
            env.allowRemoteModels = Boolean(state.semantic.allowDownload);

            const extractor = await pipeline(
                "feature-extraction",
                SEMANTIC_MODEL_ID,
                {
                    progress_callback: progress => {
                        if (!progress) return;
                        const pct = Number(progress.progress);
                        if (Number.isFinite(pct)) {
                            postStatus("loading", `MiniLM semantic model ${Math.round(pct)}%`, { progress: pct });
                        }
                    }
                }
            );

            state.semantic.extractor = extractor;
            postStatus("model-ready", "MiniLM semantic model ready.");
            return extractor;
        } catch (error) {
            state.semantic.extractor = null;
            state.semantic.embeddings = null;
            state.semantic.indexedFingerprint = null;
            state.semantic.failed = true;
            postStatus("unavailable", `Semantic retrieval unavailable: ${error?.message || error}`);
            return null;
        } finally {
            state.semantic.loadingPromise = null;
        }
    })();

    return state.semantic.loadingPromise;
}

function tensorRows(tensor) {
    const dims = tensor?.dims || [];
    const data = tensor?.data;
    if (!data || !dims.length) return [];

    const rows = dims.length === 1 ? 1 : dims[0];
    const width = dims.length === 1 ? dims[0] : dims[dims.length - 1];
    const output = [];
    for (let r = 0; r < rows; r++) {
        output.push(Float32Array.from(data.slice(r * width, (r + 1) * width)));
    }
    return output;
}

async function ensureSemanticIndex() {
    if (!state.semantic.enabled || !state.source) return false;
    if (
        state.semantic.embeddings &&
        state.semantic.indexedFingerprint === state.source.fingerprint
    ) return true;

    const extractor = await ensureSemanticExtractor();
    if (!extractor) return false;

    try {
        postStatus("indexing", `Indexing ${state.source.chunks.length} source chunks semantically…`);
        const texts = state.source.chunks.map(chunk =>
            clean(`${chunk.sectionHeading || ""} ${chunk.text}`).slice(0, 2200)
        );
        const batchSize = 8;
        const allRows = [];

        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            const tensor = await extractor(batch, { pooling: "mean", normalize: true });
            allRows.push(...tensorRows(tensor));
            postStatus("indexing", `Semantic index ${Math.min(texts.length, i + batch.length)}/${texts.length}`, {
                progress: ((i + batch.length) / Math.max(1, texts.length)) * 100
            });
        }

        if (allRows.length !== state.source.chunks.length) {
            throw new Error(`Expected ${state.source.chunks.length} embeddings but received ${allRows.length}.`);
        }

        state.semantic.embeddings = allRows;
        state.semantic.indexedFingerprint = state.source.fingerprint;
        postStatus("ready", `Hybrid retrieval ready · ${allRows.length} semantic chunks indexed.`);
        return true;
    } catch (error) {
        state.semantic.embeddings = null;
        state.semantic.indexedFingerprint = null;
        postStatus("unavailable", `Semantic indexing failed: ${error?.message || error}`);
        return false;
    }
}

function dotProduct(a, b) {
    const n = Math.min(a?.length || 0, b?.length || 0);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

async function semanticRank(queryText, limit = 20) {
    if (!state.semantic.enabled || !state.source) return [];
    const ready = await ensureSemanticIndex();
    if (!ready) return [];

    const extractor = state.semantic.extractor;
    const tensor = await extractor(clean(queryText).slice(0, 1800), { pooling: "mean", normalize: true });
    const queryVector = tensorRows(tensor)[0];
    if (!queryVector) return [];

    return state.semantic.embeddings
        .map((embedding, index) => ({
            chunkId: state.source.chunks[index].id,
            index,
            score: dotProduct(queryVector, embedding)
        }))
        .filter(item => Number.isFinite(item.score))
        .sort((a, b) => b.score - a.score || a.index - b.index)
        .slice(0, limit);
}

function normalizeMeshPhrase(value) {
    return clean(value)
        .toLowerCase()
        .replace(/[^a-z0-9&' -]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function extractNamedPhrases(text, max = 14) {
    const source = String(text || "").replace(/\s+/g, " ").trim();
    if (!source) return [];

    const out = [];
    const seen = new Set();
    const push = value => {
        const phrase = clean(value).replace(/^[\s,;:()\[\]-]+|[\s,;:()\[\]-]+$/g, "");
        if (phrase.length < 4 || phrase.length > 90) return;
        const key = normalizeMeshPhrase(phrase);
        if (!key || seen.has(key)) return;
        const words = key.split(/\s+/).filter(Boolean);
        if (!words.length || words.length > 9) return;
        seen.add(key);
        out.push({ text: phrase, key });
    };

    // Headings/titles frequently survive ingestion in title case or capitals.
    const titlePattern = /\b(?:[A-Z][A-Za-z0-9'&/-]{2,})(?:\s+(?:of|the|and|for|in|on|to|with|[A-Z][A-Za-z0-9'&/-]{2,})){0,7}\b/g;
    for (const match of source.matchAll(titlePattern)) push(match[0]);

    // Acronyms and formal abbreviated names can be strong document anchors.
    const acronymPattern = /\b[A-Z]{2,8}(?:\s+[A-Z]{2,8}){0,3}\b/g;
    for (const match of source.matchAll(acronymPattern)) push(match[0]);

    return out.slice(0, max);
}

function addMeshEdge(mesh, fromId, toId, type, weight, detail = null) {
    if (!fromId || !toId || fromId === toId) return;
    if (!mesh.has(fromId)) mesh.set(fromId, []);
    const list = mesh.get(fromId);
    const existing = list.find(item => item.targetId === toId);
    if (existing) {
        if (!Array.isArray(existing.relatedTypes)) existing.relatedTypes = [existing.type];
        if (!existing.relatedTypes.includes(type)) existing.relatedTypes.push(type);
        if (weight > existing.weight) {
            existing.weight = weight;
            existing.type = type;
            if (detail) existing.detail = detail;
        } else if (detail && !existing.detail) {
            existing.detail = detail;
        }
        return;
    }
    list.push({ targetId: toId, type, weight, detail, relatedTypes: [type] });
}

function connectMeshPair(mesh, aId, bId, type, weight, detail = null) {
    addMeshEdge(mesh, aId, bId, type, weight, detail);
    addMeshEdge(mesh, bId, aId, type, weight, detail);
}

function buildContextMesh(chunks, chunkTokenSets, documentFrequency) {
    const mesh = new Map(chunks.map(chunk => [chunk.id, []]));
    const phraseIndex = new Map();
    const headingGroups = new Map();
    const sectionGroups = new Map();
    const recordIdIndex = new Map();

    const groupPush = (map, key, id) => {
        if (!key) return;
        if (!map.has(key)) map.set(key, []);
        map.get(key).push(id);
    };

    // 1) Structural groups and named/title phrase index.
    chunks.forEach((chunk, index) => {
        const headingKey = normalizeMeshPhrase(
            [chunk.chapter, chunk.heading, chunk.subheading, chunk.sectionHeading]
                .filter(Boolean)
                .join(" ")
        );
        const sectionKey = clean(chunk.sectionId) || headingKey;
        groupPush(headingGroups, headingKey, chunk.id);
        groupPush(sectionGroups, sectionKey, chunk.id);
        for (const recordId of chunk.recordIds || []) groupPush(recordIdIndex, clean(recordId), chunk.id);

        const phraseSource = `${chunk.chapter || ""}. ${chunk.heading || ""}. ${chunk.subheading || ""}. ${chunk.sectionHeading || ""}. ${chunk.aliases.join(" ")}. ${chunk.keywords.join(" ")}. ${chunk.citations.join(" ")}. ${chunk.referenceMaterials.join(" ")}. ${chunk.domain || ""}. ${chunk.domainOther || ""}. ${chunk.coreConcept || ""}. ${chunk.documentType || ""}. ${chunk.chunkType || ""}. ${chunk.text || ""}`;
        for (const phrase of extractNamedPhrases(phraseSource)) {
            if (!phraseIndex.has(phrase.key)) phraseIndex.set(phrase.key, []);
            const ids = phraseIndex.get(phrase.key);
            if (!ids.includes(chunk.id)) ids.push(chunk.id);
        }

        // Sequential chunks remain useful even when section detection is weak.
        if (index > 0) {
            connectMeshPair(
                mesh,
                chunks[index - 1].id,
                chunk.id,
                "adjacent",
                MESH.RELATION_WEIGHTS.adjacent,
                "document-neighbour"
            );
        }
    });

    // 2) Same-heading links. Prefer nearby chunks within the heading family.
    for (const [heading, ids] of headingGroups) {
        if (!heading || ids.length < 2) continue;
        for (let i = 0; i < ids.length; i++) {
            for (let j = Math.max(0, i - 3); j <= Math.min(ids.length - 1, i + 3); j++) {
                if (i === j) continue;
                connectMeshPair(
                    mesh,
                    ids[i],
                    ids[j],
                    "same_heading",
                    MESH.RELATION_WEIGHTS.same_heading,
                    heading
                );
            }
        }
    }

    // 3) Same-section links, slightly weaker than exact same-heading links.
    for (const [section, ids] of sectionGroups) {
        if (!section || ids.length < 2) continue;
        for (let i = 0; i < ids.length; i++) {
            for (let j = Math.max(0, i - 2); j <= Math.min(ids.length - 1, i + 2); j++) {
                if (i === j) continue;
                connectMeshPair(
                    mesh,
                    ids[i],
                    ids[j],
                    "same_section",
                    MESH.RELATION_WEIGHTS.same_section,
                    section
                );
            }
        }
    }

    // 4) Human-curated CSV relations. These are explicit HITL links and are
    // stronger than inferred rare-term/semantic relationships.
    for (const chunk of chunks) {
        for (const relatedRecordId of chunk.relatedRecordIds || []) {
            const targets = recordIdIndex.get(clean(relatedRecordId)) || [];
            for (const targetId of targets) {
                if (targetId === chunk.id) continue;
                connectMeshPair(
                    mesh,
                    chunk.id,
                    targetId,
                    "human_relation",
                    0.98,
                    relatedRecordId
                );
            }
        }
    }

    const chunkIdIndex = new Map(chunks.map(chunk => [chunk.id, chunk.id]));
    for (const chunk of chunks) {
        for (const relatedChunkId of chunk.relatedChunkIds || []) {
            const targetId = chunkIdIndex.get(clean(relatedChunkId));
            if (!targetId || targetId === chunk.id) continue;
            connectMeshPair(
                mesh,
                chunk.id,
                targetId,
                "human_chunk_relation",
                1.0,
                relatedChunkId
            );
        }
    }

    // 5) Shared names/titles/phrases. Ignore phrases occurring almost everywhere.
    for (const [phrase, ids] of phraseIndex) {
        if (ids.length < 2 || ids.length > Math.max(8, Math.ceil(chunks.length * 0.45))) continue;
        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < Math.min(ids.length, i + 5); j++) {
                connectMeshPair(
                    mesh,
                    ids[i],
                    ids[j],
                    "shared_phrase",
                    MESH.RELATION_WEIGHTS.shared_phrase,
                    phrase
                );
            }
        }
    }

    // 5) Rare-term similarity. This creates weak cross-section links without an ML model.
    const rareTerms = chunkTokenSets.map(tokenSet => {
        const scored = [...tokenSet]
            .filter(term => term.length >= 5 && !STOP_WORDS.has(term))
            .map(term => ({ term, df: documentFrequency.get(term) || 1 }))
            .filter(item => item.df <= Math.max(4, Math.ceil(chunks.length * 0.28)))
            .sort((a, b) => a.df - b.df || b.term.length - a.term.length)
            .slice(0, 18)
            .map(item => item.term);
        return new Set(scored);
    });

    for (let i = 0; i < chunks.length; i++) {
        const candidates = [];
        for (let j = i + 1; j < chunks.length; j++) {
            let overlap = 0;
            for (const term of rareTerms[i]) if (rareTerms[j].has(term)) overlap++;
            if (overlap < 2) continue;
            const denominator = Math.max(1, Math.min(rareTerms[i].size, rareTerms[j].size));
            const ratio = overlap / denominator;
            const weight = Math.min(0.72, MESH.RELATION_WEIGHTS.rare_terms + Math.min(0.14, ratio * 0.22));
            candidates.push({ j, overlap, ratio, weight });
        }
        candidates
            .sort((a, b) => b.overlap - a.overlap || b.ratio - a.ratio)
            .slice(0, 3)
            .forEach(item => {
                connectMeshPair(
                    mesh,
                    chunks[i].id,
                    chunks[item.j].id,
                    "rare_terms",
                    item.weight,
                    `${item.overlap} shared rare terms`
                );
            });
    }

    // Keep the strongest/most useful relationships per node.
    for (const [chunkId, links] of mesh) {
        links.sort((a, b) => b.weight - a.weight || a.targetId.localeCompare(b.targetId));
        mesh.set(chunkId, links.slice(0, MESH.MAX_NEIGHBOURS_PER_NODE));
    }

    const counts = { adjacent: 0, same_heading: 0, same_section: 0, shared_phrase: 0, rare_terms: 0, human_relation: 0, human_chunk_relation: 0 };
    let directedEdges = 0;
    for (const links of mesh.values()) {
        directedEdges += links.length;
        for (const link of links) if (counts[link.type] !== undefined) counts[link.type]++;
    }

    return {
        mesh,
        phraseIndex,
        stats: {
            nodes: chunks.length,
            directedEdges,
            undirectedApprox: Math.round(directedEdges / 2),
            phraseAnchors: phraseIndex.size,
            relationCounts: counts
        }
    };
}

function relationPriorityForIntent(type, intentId) {
    let factor = 1;
    if (intentId === "factual") {
        if (type === "adjacent") factor = 0.92;
        if (type === "same_section") factor = 0.86;
        if (type === "rare_terms") factor = 0.72;
    }
    if (type === "human_relation") factor = 1.15;
    if (["analyse", "evaluate", "compare", "apply", "multi-part"].includes(intentId)) {
        if (type === "same_heading") factor = 1.06;
        if (type === "same_section") factor = 1.03;
        if (type === "shared_phrase") factor = 1.04;
        if (type === "rare_terms") factor = 0.95;
    }
    return factor;
}

function expandScoreMapThroughMesh(scoreMap, plan, deep) {
    const src = state.source;
    if (!src?.contextMesh?.size || !scoreMap.size) {
        return { expanded: 0, maxDepth: 0, relationCounts: {} };
    }

    const maxDepth = deep ? MESH.DEEP_DEPTH : MESH.NORMAL_DEPTH;
    const seedLimit = deep ? MESH.MAX_SEEDS_DEEP : MESH.MAX_SEEDS_NORMAL;
    const seeds = [...scoreMap.values()]
        .sort((a, b) => b.rrfScore - a.rrfScore)
        .slice(0, seedLimit)
        .map(item => ({ chunkId: item.chunkId, score: item.rrfScore }));

    const bestPropagation = new Map();
    const relationCounts = {};
    let expanded = 0;

    for (const seed of seeds) {
        const queue = [{ chunkId: seed.chunkId, score: seed.score, depth: 0, path: [] }];
        const seenDepth = new Map([[seed.chunkId, 0]]);

        while (queue.length) {
            const current = queue.shift();
            if (current.depth >= maxDepth) continue;
            const links = src.contextMesh.get(current.chunkId) || [];

            for (const link of links) {
                const nextDepth = current.depth + 1;
                const intentFactor = relationPriorityForIntent(link.type, plan?.intent?.id || "factual");
                const propagated = current.score * link.weight * intentFactor * Math.pow(MESH.DEPTH_DECAY, nextDepth - 1);
                if (propagated < MESH.MIN_PROPAGATED_SCORE) continue;

                const propagationKey = `${seed.chunkId}->${link.targetId}`;
                if ((bestPropagation.get(propagationKey) || 0) >= propagated) continue;
                bestPropagation.set(propagationKey, propagated);

                let target = scoreMap.get(link.targetId);
                if (!target) {
                    target = {
                        chunkId: link.targetId,
                        rrfScore: 0,
                        lexicalHits: [],
                        semanticHits: [],
                        meshHits: []
                    };
                    scoreMap.set(link.targetId, target);
                    expanded++;
                }
                if (!Array.isArray(target.meshHits)) target.meshHits = [];

                // Mesh support is deliberately additive but capped per path so
                // related context cannot overpower a strong direct match.
                const meshContribution = propagated * 0.64;
                target.rrfScore += meshContribution;
                target.meshHits.push({
                    seedChunkId: seed.chunkId,
                    fromChunkId: current.chunkId,
                    type: link.type,
                    weight: link.weight,
                    depth: nextDepth,
                    contribution: meshContribution,
                    detail: link.detail || null,
                    relatedTypes: Array.isArray(link.relatedTypes) ? [...link.relatedTypes] : [link.type]
                });
                relationCounts[link.type] = (relationCounts[link.type] || 0) + 1;

                const previousDepth = seenDepth.get(link.targetId);
                if (nextDepth < maxDepth && (previousDepth === undefined || nextDepth < previousDepth)) {
                    seenDepth.set(link.targetId, nextDepth);
                    queue.push({
                        chunkId: link.targetId,
                        score: propagated,
                        depth: nextDepth,
                        path: [...current.path, link.type]
                    });
                }
            }
        }
    }

    return { expanded, maxDepth, relationCounts };
}

function selectDiverseCandidates(entries, maxCandidates, deep) {
    if (!state.source) return [];

    const sorted = entries
        .map(entry => {
            const index = state.source.chunkIndexById.get(entry.chunkId);
            const chunk = state.source.chunks[index];
            return {
                ...entry,
                index,
                chunk,
                direct: Boolean(entry.lexicalHits?.length || entry.semanticHits?.length)
            };
        })
        .filter(item => item.chunk)
        .sort((a, b) => b.rrfScore - a.rrfScore || a.index - b.index);

    if (!sorted.length) return [];

    const selected = [];
    const selectedIds = new Set();
    const sectionCounts = new Map();
    const maxPerSection = deep ? 4 : 3;
    const topScore = Math.max(0.000001, sorted[0].rrfScore || 0.000001);

    const add = item => {
        if (!item || selectedIds.has(item.chunkId) || selected.length >= maxCandidates) return false;
        const section = clean(item.chunk.sectionHeading) || `section:${item.chunk.sectionId || item.index}`;
        const count = sectionCounts.get(section) || 0;
        if (count >= maxPerSection) return false;
        selected.push(item);
        selectedIds.add(item.chunkId);
        sectionCounts.set(section, count + 1);
        return true;
    };

    const direct = sorted.filter(item => item.direct);

    // First preserve the strongest direct evidence across sections.
    const directSections = new Set();
    for (const item of direct) {
        if (selected.length >= maxCandidates) break;
        const section = clean(item.chunk.sectionHeading) || `section:${item.chunk.sectionId || item.index}`;
        if (directSections.has(section)) continue;
        if (add(item)) directSections.add(section);
    }

    // Then fill with the remaining strongest direct results.
    for (const item of direct) {
        if (selected.length >= maxCandidates) break;
        add(item);
    }

    // Mesh-only chunks supplement direct retrieval. Relation-specific minimum
    // ratios stop a weak adjacent chunk from being selected merely because it
    // lives in a different section.
    const meshOnly = sorted.filter(item => !item.direct && item.meshHits?.length);
    const meshBudget = deep ? Math.min(7, maxCandidates) : Math.min(4, maxCandidates);
    let meshAdded = 0;

    for (const item of meshOnly) {
        if (selected.length >= maxCandidates || meshAdded >= meshBudget) break;
        const ratio = item.rrfScore / topScore;
        const strongest = [...item.meshHits]
            .sort((a, b) => (b.contribution || 0) - (a.contribution || 0))[0];
        const type = strongest?.type || "mesh";
        const minRatio = {
            same_heading: 0.12,
            shared_phrase: 0.16,
            same_section: 0.18,
            rare_terms: 0.25,
            adjacent: 0.34,
            human_relation: 0.08
        }[type] ?? 0.28;

        if (ratio < minRatio) continue;
        if (add(item)) meshAdded++;
    }

    return selected
        .sort((a, b) => b.rrfScore - a.rrfScore)
        .map((item, rankIndex) => ({
            chunkId: item.chunkId,
            rank: rankIndex + 1,
            rrfScore: item.rrfScore,
            startSentence: item.chunk.startSentence,
            endSentence: item.chunk.endSentence,
            startWord: item.chunk.startWord,
            endWord: item.chunk.endWord,
            sectionId: item.chunk.sectionId || null,
            sectionHeading: item.chunk.sectionHeading || null,
            chapter: item.chunk.chapter || null,
            heading: item.chunk.heading || null,
            subheading: item.chunk.subheading || null,
            chunkType: item.chunk.chunkType || "content",
            pageNumber: item.chunk.pageNumber || null,
            lexicalHits: item.lexicalHits,
            semanticHits: item.semanticHits,
            meshHits: Array.isArray(item.meshHits) ? item.meshHits : [],
            retrievalReason: item.direct
                ? "direct"
                : (item.meshHits?.[0]?.type || "mesh")
        }));
}

async function retrieve(payload = {}) {
    if (!state.source) throw new Error("Retrieval source has not been indexed.");

    const query = clean(payload.query);
    if (!query) throw new Error("RETRIEVE requires query.");

    const deep = Boolean(payload.deep);
    const maxCandidates = clamp(Number(payload.maxCandidates) || (deep ? 14 : 10), 4, 24);
    const conversationHint = clean(payload.conversationHint);
    const useSemantic = state.semantic.enabled && payload.semantic !== false;
    const plan = planQuery(query, { deep, conversationHint });
    const scoreMap = new Map();
    const laneLimit = Math.max(12, maxCandidates * 2);

    // Round 1: original + normalized + document-aware expansions/subqueries.
    for (const variant of plan.variants) {
        const lexical = lexicalRank(variant.text, laneLimit);
        addRankListToRRF(scoreMap, lexical, {
            lane: "lexical",
            variant: variant.kind,
            weight: variant.weight
        });
    }

    let semanticUsed = false;
    if (useSemantic) {
        const semanticQueries = plan.variants
            .filter(item => ["original", "conversation-aware", "subquery", "document-expanded"].includes(item.kind))
            .slice(0, deep ? 4 : 2);

        for (const variant of semanticQueries) {
            const semantic = await semanticRank(variant.text, laneLimit);
            if (semantic.length) semanticUsed = true;
            addRankListToRRF(scoreMap, semantic, {
                lane: "semantic",
                variant: variant.kind,
                weight: variant.kind === "original" ? 1.1 : 0.9
            });
        }
    }

    let rounds = 1;
    const secondPassVariants = [];

    // Round 2 for Super/deep mode: use high-IDF vocabulary from the strongest
    // first-round source regions as bridge terms. This is deterministic and
    // never introduces outside vocabulary.
    if (deep && scoreMap.size) {
        const firstRoundIds = [...scoreMap.values()]
            .sort((a, b) => b.rrfScore - a.rrfScore)
            .slice(0, 5)
            .map(item => item.chunkId);

        const bridgeTerms = topBridgeTerms(
            firstRoundIds,
            [...new Set([...plan.originalTokens, ...plan.expansionTerms])],
            8
        );

        if (bridgeTerms.length) {
            const bridgeQuery = `${plan.originalTokens.join(" ")} ${bridgeTerms.slice(0, 6).join(" ")}`.trim();
            if (bridgeQuery) secondPassVariants.push({ text: bridgeQuery, kind: "bridge", weight: 0.85 });

            for (const subquery of plan.subqueries.slice(0, 3)) {
                const subTerms = tokens(subquery, { unique: true });
                const subExpansion = sourceExpansionTerms(subTerms, 4);
                if (subExpansion.length) {
                    secondPassVariants.push({
                        text: `${subquery} ${subExpansion.join(" ")}`,
                        kind: "subquery-bridge",
                        weight: 0.85
                    });
                }
            }
        }

        if (secondPassVariants.length) {
            rounds = 2;
            for (const variant of secondPassVariants.slice(0, 4)) {
                const lexical = lexicalRank(variant.text, laneLimit);
                addRankListToRRF(scoreMap, lexical, {
                    lane: "lexical",
                    variant: variant.kind,
                    weight: variant.weight
                });
            }

            if (useSemantic) {
                for (const variant of secondPassVariants.slice(0, 2)) {
                    const semantic = await semanticRank(variant.text, laneLimit);
                    if (semantic.length) semanticUsed = true;
                    addRankListToRRF(scoreMap, semantic, {
                        lane: "semantic",
                        variant: variant.kind,
                        weight: 0.75
                    });
                }
            }
        }
    }

    const mesh = expandScoreMapThroughMesh(scoreMap, plan, deep);
    const candidates = selectDiverseCandidates([...scoreMap.values()], maxCandidates, deep);

    return {
        retrievalVersion: RETRIEVAL_VERSION,
        sourceFingerprint: state.source.fingerprint,
        query,
        plan: {
            intent: plan.intent.id,
            multiPart: plan.intent.multiPart,
            followup: plan.followup,
            expansionTerms: plan.expansionTerms,
            subqueries: plan.subqueries,
            variants: plan.variants.map(item => ({ kind: item.kind, text: item.text })),
            secondPassVariants: secondPassVariants.map(item => ({ kind: item.kind, text: item.text })),
            rounds
        },
        semantic: {
            enabled: state.semantic.enabled,
            used: semanticUsed,
            status: state.semantic.status,
            modelId: SEMANTIC_MODEL_ID
        },
        mesh: {
            enabled: Boolean(state.source?.contextMesh?.size),
            maxDepth: mesh.maxDepth,
            expandedCandidates: mesh.expanded,
            relationCounts: mesh.relationCounts,
            stats: state.source?.meshStats || null
        },
        candidates
    };
}

function indexSource(payload = {}) {
    const fingerprint = clean(payload.fingerprint);
    const chunks = Array.isArray(payload.chunks) ? payload.chunks : [];
    if (!fingerprint || !chunks.length) {
        throw new Error("INDEX_SOURCE requires fingerprint and chunks.");
    }

    const safeChunks = chunks
        .map((chunk, index) => ({
            id: clean(chunk.id) || `chunk_${index + 1}`,
            text: clean(chunk.text),
            wordCount: Number(chunk.wordCount) || tokens(chunk.text).length,
            startSentence: Number(chunk.startSentence) || 1,
            endSentence: Number(chunk.endSentence) || Number(chunk.startSentence) || 1,
            startWord: Number(chunk.startWord) || 1,
            endWord: Number(chunk.endWord) || Number(chunk.startWord) || 1,
            sectionHeading: clean(chunk.sectionHeading) || "Document",
            sectionId: clean(chunk.sectionId) || null,
            chunkId: clean(chunk.chunkId || chunk.id) || `chunk_${index + 1}`,
            chunkOrder: Number(chunk.chunkOrder) || null,
            chapter: clean(chunk.chapter) || "",
            heading: clean(chunk.heading) || "",
            subheading: clean(chunk.subheading) || "",
            pageNumber: chunk.pageNumber ?? null,
            recordIds: Array.isArray(chunk.recordIds) ? chunk.recordIds.map(clean).filter(Boolean) : [],
            recordType: clean(chunk.recordType) || "section",
            chunkType: clean(chunk.chunkType) || "content",
            keywords: Array.isArray(chunk.keywords) ? chunk.keywords.map(clean).filter(Boolean) : [],
            aliases: Array.isArray(chunk.aliases) ? chunk.aliases.map(clean).filter(Boolean) : [],
            relatedRecordIds: Array.isArray(chunk.relatedRecordIds) ? chunk.relatedRecordIds.map(clean).filter(Boolean) : [],
            relatedChunkIds: Array.isArray(chunk.relatedChunkIds) ? chunk.relatedChunkIds.map(clean).filter(Boolean) : [],
            citations: Array.isArray(chunk.citations) ? chunk.citations.map(clean).filter(Boolean) : [],
            referenceMaterials: Array.isArray(chunk.referenceMaterials) ? chunk.referenceMaterials.map(clean).filter(Boolean) : [],
            links: Array.isArray(chunk.links) ? chunk.links.map(clean).filter(Boolean) : [],
            images: Array.isArray(chunk.images) ? chunk.images.map(clean).filter(Boolean) : [],
            domain: clean(chunk.domain) || "",
            domainOther: clean(chunk.domainOther) || "",
            coreConcept: clean(chunk.coreConcept) || "",
            documentType: clean(chunk.documentType) || "",
            humanCorrected: Boolean(chunk.humanCorrected)
        }))
        .filter(chunk => chunk.text);

    if (!safeChunks.length) throw new Error("INDEX_SOURCE contains no usable chunks.");

    const sourceTitle = clean(payload.title) || "Current Reading";
    const chunkTokens = safeChunks.map(chunk =>
        tokens(`${sourceTitle} ${chunk.chapter || ""} ${chunk.heading || ""} ${chunk.subheading || ""} ${chunk.sectionHeading || ""} ${chunk.keywords.join(" ")} ${chunk.aliases.join(" ")} ${chunk.citations.join(" ")} ${chunk.referenceMaterials.join(" ")} ${chunk.recordType || ""} ${chunk.chunkType || ""} ${chunk.domain || ""} ${chunk.domainOther || ""} ${chunk.coreConcept || ""} ${chunk.documentType || ""} ${chunk.text}`)
    );
    const chunkTokenSets = chunkTokens.map(items => new Set(items));
    const chunkTermFrequencies = chunkTokens.map(tokenFrequency);
    const chunkTokenLengths = chunkTokens.map(items => items.length);
    const avgChunkLength = chunkTokenLengths.reduce((a, b) => a + b, 0) / Math.max(1, chunkTokenLengths.length);
    const documentFrequency = new Map();

    for (const tokenSet of chunkTokenSets) {
        for (const term of tokenSet) documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
    }

    const meshBuild = buildContextMesh(safeChunks, chunkTokenSets, documentFrequency);

    state.source = {
        fingerprint,
        title: sourceTitle,
        chunks: safeChunks,
        chunkTokens,
        chunkTokenSets,
        chunkTermFrequencies,
        chunkTokenLengths,
        avgChunkLength,
        documentFrequency,
        chunkIndexById: new Map(safeChunks.map((chunk, index) => [chunk.id, index])),
        contextMesh: meshBuild.mesh,
        phraseIndex: meshBuild.phraseIndex,
        meshStats: meshBuild.stats
    };

    if (state.semantic.indexedFingerprint !== fingerprint) {
        state.semantic.embeddings = null;
        state.semantic.indexedFingerprint = null;
    }

    return {
        retrievalVersion: RETRIEVAL_VERSION,
        fingerprint,
        chunkCount: safeChunks.length,
        vocabularySize: documentFrequency.size,
        semanticEnabled: state.semantic.enabled,
        mesh: meshBuild.stats
    };
}

async function setSemantic(payload = {}) {
    state.semantic.enabled = Boolean(payload.enabled);
    state.semantic.allowDownload = Boolean(payload.allowDownload);
    if (state.semantic.enabled) state.semantic.failed = false;

    if (!state.semantic.enabled) {
        state.semantic.status = "off";
        state.semantic.message = "Semantic lane off";
        state.semantic.embeddings = null;
        state.semantic.indexedFingerprint = null;
        if (state.semantic.extractor?.dispose) {
            try { await state.semantic.extractor.dispose(); } catch (_) {}
        }
        state.semantic.extractor = null;
        state.semantic.failed = false;
        postStatus("off", "Semantic lane off · planner + BM25 + RRF remain active.");
        return getInfo();
    }

    const ready = await ensureSemanticIndex();
    if (!ready && !state.semantic.allowDownload) {
        postStatus("unavailable", "No cached semantic model was available. Enable download to add MiniLM.");
    }

    return getInfo();
}

function clearSource() {
    state.source = null;
    state.semantic.embeddings = null;
    state.semantic.indexedFingerprint = null;
    return { cleared: true };
}

function getInfo() {
    return {
        retrievalVersion: RETRIEVAL_VERSION,
        indexed: Boolean(state.source),
        sourceFingerprint: state.source?.fingerprint || null,
        chunkCount: state.source?.chunks?.length || 0,
        mesh: state.source?.meshStats || null,
        semantic: {
            enabled: state.semantic.enabled,
            status: state.semantic.status,
            message: state.semantic.message,
            modelId: SEMANTIC_MODEL_ID,
            cachedIndexReady: Boolean(
                state.semantic.embeddings &&
                state.semantic.indexedFingerprint === state.source?.fingerprint
            )
        }
    };
}

self.addEventListener("message", async event => {
    const message = event.data || {};
    const { id, type, payload } = message;

    try {
        if (id === undefined || id === null) throw new Error("Retrieval Worker request requires an id.");

        let result;
        switch (type) {
            case "INDEX_SOURCE":
                result = indexSource(payload);
                break;
            case "CLEAR_SOURCE":
                result = clearSource();
                break;
            case "SET_SEMANTIC":
                result = await setSemantic(payload);
                break;
            case "RETRIEVE":
                result = await retrieve(payload);
                break;
            case "GET_RETRIEVAL_INFO":
                result = getInfo();
                break;
            default:
                throw new Error(`Unknown Retrieval Worker request: ${String(type)}`);
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
