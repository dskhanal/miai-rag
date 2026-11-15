// ============================================================================
// MiAi INGESTION WORKER
// Browser-local document normalisation and retrieval preparation.
//
// Inspired by successful ingestion ideas used by systems such as Crawl4AI:
// - keep raw, cleaned and retrieval-ready representations conceptually separate
// - remove obvious boilerplate/noise conservatively
// - preserve document structure and provenance
// - de-duplicate repeated content
// - prepare overlapping chunks and sentence records before retrieval
// - avoid using an LLM for ingestion
//
// This worker does NOT parse PDF/DOCX binary files. index.html extracts those
// with the browser libraries it already uses, then sends structured text here.
// ============================================================================

const INGESTION_VERSION = "miai-ingestion-2.0.0";

const DEFAULTS = Object.freeze({
    maxInputChars: 2_500_000,
    wordLimit: 3000,
    chunkTargetWords: 60,
    chunkMinWords: 42,
    chunkMaxWords: 78,
    chunkOverlapPercent: 0.15,
    chunkOverlapMinPercent: 0.10,
    chunkOverlapMaxPercent: 0.20,
    minParagraphWordsForDedupe: 8,
    keywordMin: 5,
    keywordMax: 10
});

const STOP_WORDS = new Set([
    "the","and","is","are","of","for","with","a","an","to","in","on","at","by","or","as",
    "that","this","from","into","onto","than","then","was","were","been","being","have","has","had",
    "will","would","could","should","may","might","can","their","there","they","them","our","your","you",
    "its","his","her","who","what","when","where","which","why","how","about","also","such","these",
    "those","using","used","use","between","within","without","through","during","over","under","after",
    "before","each","other","more","most","some","any","all","both","not","no","if","but","so","do",
    "does","did","be","it","we","i","he","she","them","because","while","whereas"
]);

function cleanString(value) {
    return String(value ?? "");
}

function wordCount(text) {
    return cleanString(text).trim().split(/\s+/).filter(Boolean).length;
}

function normalizeUnicode(text) {
    return cleanString(text)
        .normalize("NFKC")
        .replace(/[\u00A0\u2007\u202F]/g, " ")
        .replace(/[\u200B-\u200D\uFEFF]/g, "")
        .replace(/[\u2018\u2019]/g, "'")
        .replace(/[\u201C\u201D]/g, '"')
        .replace(/[\u2013\u2014]/g, "-")
        .replace(/\r\n?/g, "\n");
}

function normalizeLine(line) {
    return normalizeUnicode(line)
        .replace(/[\t ]+/g, " ")
        .replace(/\s+([,.;:!?])/g, "$1")
        .trim();
}

function canonicalLine(line) {
    return normalizeLine(line)
        .toLowerCase()
        .replace(/\bpage\s+\d+(?:\s+of\s+\d+)?\b/g, "page #")
        .replace(/\b\d+\b/g, "#")
        .replace(/[^a-z#]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function canonicalParagraph(text) {
    return normalizeLine(text)
        .toLowerCase()
        .replace(/\d+/g, "#")
        .replace(/[^a-z0-9#]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function isPageNumberNoise(line) {
    const value = normalizeLine(line);
    if (!value) return false;
    return /^\d{1,4}$/.test(value) ||
        /^page\s+\d+(?:\s+(?:of|\/)\s*\d+)?$/i.test(value) ||
        /^\d+\s*\/\s*\d+$/.test(value);
}

function isLikelyHeading(line) {
    const value = normalizeLine(line);
    if (!value || looksLikeReferenceEntry(value)) return false;

    const words = value.split(/\s+/).filter(Boolean);
    if (words.length > 14 || value.length > 140) return false;
    if (/^[•\-*]\s+/.test(value)) return false;
    if (/^[A-Za-z]\)$/.test(value)) return false;

    if (/^(?:\d+(?:\.\d+){0,4}|[A-Z])\s*[.)-]?\s+\S+/.test(value)) return true;
    if (/^[A-Z][A-Z0-9 &/(),:'-]{3,}$/.test(value) && /[A-Z]/.test(value)) return true;

    // Common document headings are sometimes a single title-cased word
    // (e.g. Introduction, Assessment, Conclusion). Treat these as headings
    // only when they occur as their own extracted line.
    if (words.length === 1 && /^[A-Z][A-Za-z-]{3,}$/.test(value)) return true;

    const terminalSentencePunctuation = /[.!?]$/.test(value);
    if (terminalSentencePunctuation) return false;

    let titleLike = 0;
    let alphaWords = 0;
    for (const word of words) {
        const alpha = word.replace(/[^A-Za-z]/g, "");
        if (!alpha) continue;
        alphaWords++;
        if (/^[A-Z][a-z]/.test(alpha) || /^[A-Z]{2,}$/.test(alpha)) titleLike++;
    }

    return alphaWords >= 2 && titleLike / alphaWords >= 0.7;
}

function looksLikeReferenceEntry(line) {
    const value = normalizeLine(line);
    return /^\s*(?:\[?\d+\]?\s*[.)-]?\s+).*(?:\b(?:19|20)\d{2}\b|https?:\/\/|doi\b)/i.test(value);
}

function detectHeadingLevel(line) {
    const value = normalizeLine(line);
    if (!value || looksLikeReferenceEntry(value)) return 0;

    const chapterMatch = value.match(/^chapter\s+\d+\b/i);
    if (chapterMatch) return 1;

    const numbered = value.match(/^(\d+(?:\.\d+){0,4})\s*[.)-]?\s+\S+/);
    if (numbered) {
        const depth = numbered[1].split(".").length;
        return Math.min(3, depth);
    }

    if (/^[A-Z][A-Z0-9 &/(),:'-]{3,}$/.test(value) && wordCount(value) <= 10) return 1;
    if (isLikelyHeading(value)) return 2;
    return 0;
}

function parseStructuredMarker(rawLine) {
    const raw = cleanString(rawLine).trim();
    if (!raw.startsWith("[[MIAI:")) return null;

    const heading = raw.match(/^\[\[MIAI:H([1-6])\]\]\s*(.*)$/i);
    if (heading) {
        return {
            type: "heading",
            level: Math.max(1, Math.min(6, Number(heading[1]) || 2)),
            text: normalizeLine(heading[2])
        };
    }

    const quote = raw.match(/^\[\[MIAI:QUOTE\]\]\s*(.*)$/i);
    if (quote) return { type: "quote", text: normalizeLine(quote[1]) };

    const caption = raw.match(/^\[\[MIAI:CAPTION\]\]\s*(.*)$/i);
    if (caption) return { type: "image_caption", text: normalizeLine(caption[1]) };

    const link = raw.match(/^\[\[MIAI:LINK\]\]\s*([^\t|]+)(?:[\t|]+(.*))?$/i);
    if (link) {
        return {
            type: "link",
            url: cleanString(link[1]).trim(),
            label: normalizeLine(link[2] || "")
        };
    }

    return null;
}

function isReferenceHeading(value) {
    const text = normalizeLine(value).toLowerCase();
    return /^(references|bibliography|reference materials|further reading|works cited|sources)$/.test(text);
}

function looksLikeImageCaption(value) {
    const text = normalizeLine(value);
    return /^(?:figure|fig\.?|image|illustration|plate)\s*\d+[a-z]?\s*[:.\-]?\s*\S+/i.test(text);
}

function extractInlineMetadata(value) {
    let text = normalizeUnicode(value || "").trim();
    const links = [];
    const citations = [];

    text = text.replace(/\bhttps?:\/\/[^\s<>()\[\]{}]+|\bwww\.[^\s<>()\[\]{}]+/gi, match => {
        links.push(match.replace(/[.,;:!?]+$/, ""));
        return " ";
    });

    text = text.replace(/\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]/g, match => {
        citations.push(match);
        return " ";
    });

    text = text.replace(/\(([^()]{0,100}\b(?:19|20)\d{2}[a-z]?[^()]*)\)/g, (match, inner) => {
        if (/[A-Za-z]/.test(inner)) {
            citations.push(match);
            return " ";
        }
        return match;
    });

    return {
        cleanText: text.replace(/\s+/g, " ").replace(/\s+([,.;:!?])/g, "$1").trim(),
        links: [...new Set(links)],
        citations: [...new Set(citations)]
    };
}

function citationLookupKeys(citation) {
    const value = normalizeLine(citation);
    const keys = new Set();

    const numeric = value.match(/\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]/);
    if (numeric) {
        const numbers = numeric[1].match(/\d+/g) || [];
        numbers.forEach(number => keys.add(`n:${number}`));
    }

    const year = value.match(/\b((?:19|20)\d{2}[a-z]?)\b/i)?.[1]?.toLowerCase();
    const surname = value.match(/\(?\s*([A-Z][A-Za-z'’-]{2,})/)?.[1]?.toLowerCase();
    if (year && surname) keys.add(`ay:${surname}:${year}`);

    return [...keys];
}

function referenceLookupKeys(citationText) {
    const value = normalizeLine(citationText);
    const keys = new Set();

    const numeric = value.match(/^\s*\[?(\d+)\]?\s*[.)\-:]?/);
    if (numeric) keys.add(`n:${numeric[1]}`);

    const year = value.match(/\b((?:19|20)\d{2}[a-z]?)\b/i)?.[1]?.toLowerCase();
    const surname = value.match(/^\s*(?:\[?\d+\]?\s*[.)\-:]?\s*)?([A-Z][A-Za-z'’-]{2,})/)?.[1]?.toLowerCase();
    if (year && surname) keys.add(`ay:${surname}:${year}`);

    return [...keys];
}

function sourceMetadata(payload = {}) {
    const meta = payload.sourceMetadata || payload.metadata || {};
    return {
        domain: normalizeLine(meta.domain || payload.domain || ""),
        domainOther: normalizeLine(meta.domainOther || meta.domain_other || payload.domainOther || ""),
        coreConcept: normalizeLine(meta.coreConcept || meta.core_concept || payload.coreConcept || ""),
        documentType: normalizeLine(meta.documentType || meta.document_type || payload.documentType || "")
    };
}

function contentTokens(text) {
    return (normalizeUnicode(text).toLowerCase().match(/[a-z][a-z0-9'-]{2,}/g) || [])
        .map(token => token.replace(/^'+|'+$/g, ""))
        .filter(token => token && !STOP_WORDS.has(token));
}

function tokenSimilarity(a, b) {
    const A = new Set(contentTokens(a));
    const B = new Set(contentTokens(b));
    if (!A.size || !B.size) return 0;
    let shared = 0;
    for (const term of A) if (B.has(term)) shared++;
    return shared / Math.max(1, Math.min(A.size, B.size));
}

function cleanKeywordPhrase(value) {
    return normalizeLine(value)
        .toLowerCase()
        .split(/\s+/)
        .map(token => token.replace(/^[^a-z0-9]+|[^a-z0-9'-]+$/g, ""))
        .filter(token => token && !STOP_WORDS.has(token) && token.length > 1)
        .join(" ")
        .trim();
}

function nounishScore(token) {
    const t = String(token || "").toLowerCase();
    let score = 0;
    if (/(?:tion|sion|ment|ness|ity|ance|ence|ism|ist|ship|hood|ology|graphy|therapy|care|model|theory|framework|system|process|method|practice|policy|standard|assessment|management|research|education|learning|health|nursing)$/.test(t)) score += 1.5;
    if (t.length >= 7) score += 0.35;
    return score;
}

function extractChunkKeywords(text, metadata = {}, hierarchy = {}, minKeywords = DEFAULTS.keywordMin, maxKeywords = DEFAULTS.keywordMax) {
    const source = normalizeUnicode(text || "");
    const words = source.match(/[A-Za-z][A-Za-z0-9'’-]*/g) || [];
    const candidates = new Map();

    const add = (phrase, score) => {
        const cleaned = cleanKeywordPhrase(phrase);
        if (!cleaned || cleaned.split(/\s+/).length > 4) return;
        candidates.set(cleaned, Math.max(candidates.get(cleaned) || 0, score));
    };

    // Capitalised multi-word phrases are strong noun-phrase candidates.
    const capitalised = source.match(/\b(?:[A-Z][A-Za-z0-9'’-]{2,}(?:\s+|$)){2,4}/g) || [];
    capitalised.forEach((phrase, index) => add(phrase, 6 - index * 0.02));

    const lowered = words.map(word => word.toLowerCase());
    for (let n = 3; n >= 2; n--) {
        for (let i = 0; i <= lowered.length - n; i++) {
            const slice = lowered.slice(i, i + n);
            if (slice.some(token => STOP_WORDS.has(token))) continue;
            const phrase = slice.join(" ");
            let score = n * 1.2;
            score += slice.reduce((sum, token) => sum + nounishScore(token), 0);
            add(phrase, score);
        }
    }

    const frequencies = new Map();
    for (const token of lowered) {
        const clean = token.replace(/^'+|'+$/g, "");
        if (!clean || STOP_WORDS.has(clean) || clean.length < 3) continue;
        frequencies.set(clean, (frequencies.get(clean) || 0) + 1);
    }
    for (const [term, count] of frequencies) {
        add(term, Math.log1p(count) * 1.7 + nounishScore(term));
    }

    const highValueMetadata = [
        hierarchy.subheading,
        hierarchy.heading,
        hierarchy.chapter,
        metadata.coreConcept,
        metadata.domainOther,
        metadata.domain
    ].filter(Boolean);

    for (const value of highValueMetadata) {
        const cleaned = cleanKeywordPhrase(value);
        if (!cleaned) continue;
        const terms = cleaned.split(/\s+/);
        const appears = terms.some(term => source.toLowerCase().includes(term));
        if (appears || value === hierarchy.subheading || value === hierarchy.heading) add(cleaned, 7.5);
    }

    const sorted = [...candidates.entries()]
        .sort((a, b) => b[1] - a[1] || b[0].split(" ").length - a[0].split(" ").length || b[0].length - a[0].length)
        .map(([term]) => term);

    const selected = [];
    for (const candidate of sorted) {
        if (selected.length >= maxKeywords) break;
        if (selected.some(existing => existing.includes(candidate) || candidate.includes(existing))) {
            if (candidate.split(" ").length === 1) continue;
        }
        selected.push(candidate);
    }

    return selected.slice(0, Math.max(minKeywords, Math.min(maxKeywords, selected.length)));
}

function joinDehyphenatedLines(lines) {
    const out = [];
    for (let i = 0; i < lines.length; i++) {
        const current = lines[i];
        const next = lines[i + 1];

        if (
            current &&
            next &&
            /[A-Za-z]-$/.test(current.text) &&
            /^[a-z]/.test(next.text) &&
            current.pageNumber === next.pageNumber
        ) {
            out.push({
                ...current,
                text: current.text.slice(0, -1) + next.text,
                joinedHyphenation: true
            });
            i++;
            continue;
        }
        out.push(current);
    }
    return out;
}

function makePageLines(payload) {
    const pages = Array.isArray(payload.pages) && payload.pages.length
        ? payload.pages
        : [{ pageNumber: null, text: payload.rawText || "" }];

    const result = [];

    for (const page of pages) {
        const pageNumber = Number.isFinite(Number(page.pageNumber))
            ? Number(page.pageNumber)
            : null;

        const raw = normalizeUnicode(page.text || "");
        const split = raw.split("\n");

        for (let i = 0; i < split.length; i++) {
            const marker = parseStructuredMarker(split[i]);
            result.push({
                pageNumber,
                sourceLine: i + 1,
                text: marker ? normalizeLine(marker.text || marker.label || marker.url || "") : normalizeLine(split[i]),
                markerType: marker?.type || null,
                markerLevel: marker?.level || null,
                markerUrl: marker?.url || null,
                markerLabel: marker?.label || null
            });
        }
    }

    return result;
}

function findRepeatedPageLines(lines) {
    const pages = new Map();

    for (const line of lines) {
        if (!line.pageNumber || !line.text) continue;
        if (!pages.has(line.pageNumber)) pages.set(line.pageNumber, []);
        pages.get(line.pageNumber).push(line);
    }

    if (pages.size < 3) return new Set();

    const occurrences = new Map();
    for (const [pageNumber, pageLines] of pages) {
        const candidateLines = [
            ...pageLines.slice(0, 4),
            ...pageLines.slice(Math.max(0, pageLines.length - 4))
        ];
        const seenOnPage = new Set();

        for (const item of candidateLines) {
            const canonical = canonicalLine(item.text);
            if (!canonical || canonical.length < 3 || canonical.length > 120) continue;
            if (wordCount(item.text) > 18) continue;
            if (seenOnPage.has(canonical)) continue;
            seenOnPage.add(canonical);

            if (!occurrences.has(canonical)) occurrences.set(canonical, new Set());
            occurrences.get(canonical).add(pageNumber);
        }
    }

    const threshold = Math.max(3, Math.ceil(pages.size * 0.45));
    const repeated = new Set();

    for (const [canonical, pageSet] of occurrences) {
        if (pageSet.size >= threshold) repeated.add(canonical);
    }

    return repeated;
}

function prepareCleanLines(payload) {
    const sourceLines = joinDehyphenatedLines(makePageLines(payload));
    const repeatedPageLines = findRepeatedPageLines(sourceLines);
    const cleanLines = [];
    const removed = {
        pageNumbers: 0,
        repeatedHeadersFooters: 0,
        duplicateBlankLines: 0,
        emptyLines: 0
    };

    let previousBlank = false;

    for (const item of sourceLines) {
        const text = normalizeLine(item.text);

        if (!text) {
            if (previousBlank) {
                removed.duplicateBlankLines++;
                continue;
            }
            cleanLines.push({ ...item, text: "", blank: true });
            previousBlank = true;
            removed.emptyLines++;
            continue;
        }

        previousBlank = false;

        if (isPageNumberNoise(text)) {
            removed.pageNumbers++;
            continue;
        }

        const canonical = canonicalLine(text);
        if (repeatedPageLines.has(canonical)) {
            removed.repeatedHeadersFooters++;
            continue;
        }

        const explicitHeading = item.markerType === "heading";
        const headingLevel = explicitHeading
            ? Math.max(1, Math.min(6, Number(item.markerLevel) || 2))
            : detectHeadingLevel(text);

        cleanLines.push({
            ...item,
            text,
            blank: false,
            heading: explicitHeading || headingLevel > 0,
            headingLevel,
            specialType: item.markerType && item.markerType !== "heading" ? item.markerType : null
        });
    }

    return { cleanLines, removed };
}

function parseBool(value, defaultValue = true) {
    if (value === undefined || value === null || value === "") return defaultValue;
    const v = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "y", "include", "included"].includes(v)) return true;
    if (["0", "false", "no", "n", "exclude", "excluded"].includes(v)) return false;
    return defaultValue;
}

function splitList(value) {
    if (Array.isArray(value)) return value.map(cleanString).map(v => v.trim()).filter(Boolean);
    return cleanString(value).split(/[|;]+/).map(v => v.trim()).filter(Boolean);
}

function normalizeStructuredRecords(records = [], payload = {}) {
    const meta = sourceMetadata(payload);

    return records.map((raw, index) => {
        if (!parseBool(raw.include, true)) return null;

        const textChunk = normalizeUnicode(raw.text_chunk || raw.chunk_text || raw.text || raw.content || "").trim();
        if (!textChunk) return null;

        const orderValue = Number(raw.order);
        const sectionOrderValue = Number(raw.section_order ?? raw.sectionOrder);
        const chunkOrderValue = Number(raw.chunk_order ?? raw.chunkOrder);
        const pageValue = Number(raw.page_number ?? raw.pageNumber);
        const sectionId = normalizeLine(raw.section_id || raw.sectionId || raw.record_id || raw.recordId || `section_${index + 1}`) || `section_${index + 1}`;
        const recordId = normalizeLine(raw.record_id || raw.recordId || sectionId) || sectionId;
        const chunkId = normalizeLine(raw.chunk_id || raw.chunkId || "") || null;

        return {
            id: sectionId,
            sectionId,
            recordId,
            chunkId,
            chunkOrder: Number.isFinite(chunkOrderValue) ? chunkOrderValue : null,
            sectionOrder: Number.isFinite(sectionOrderValue) ? sectionOrderValue : (Number.isFinite(orderValue) ? orderValue : index + 1),
            parentSectionId: normalizeLine(raw.parent_section_id || raw.parentSectionId || "") || null,
            chapter: normalizeLine(raw.chapter || ""),
            heading: normalizeLine(raw.heading || raw.title || "Document") || "Document",
            subheading: normalizeLine(raw.subheading || ""),
            pageNumber: Number.isFinite(pageValue) && pageValue > 0 ? pageValue : null,
            text: textChunk,
            textChunk,
            order: Number.isFinite(orderValue) ? orderValue : index + 1,
            recordType: normalizeLine(raw.record_type || raw.recordType || "section").toLowerCase() || "section",
            chunkType: normalizeLine(raw.chunk_type || raw.chunkType || raw.record_type || raw.recordType || "content").toLowerCase() || "content",
            keywords: splitList(raw.keywords),
            aliases: splitList(raw.aliases),
            relatedRecordIds: splitList(raw.related_record_ids || raw.relatedRecordIds),
            relatedChunkIds: splitList(raw.related_chunk_ids || raw.relatedChunkIds),
            citations: splitList(raw.citations),
            referenceMaterials: splitList(raw.reference_materials || raw.referenceMaterials),
            links: splitList(raw.links),
            images: splitList(raw.images || raw.image_captions || raw.imageCaptions),
            reviewStatus: normalizeLine(raw.review_status || raw.reviewStatus || "unreviewed").toLowerCase() || "unreviewed",
            reviewerNotes: cleanString(raw.reviewer_notes || raw.reviewerNotes || "").trim(),
            humanCorrected: parseBool(raw.human_corrected ?? raw.humanCorrected, false),
            domain: normalizeLine(raw.domain || meta.domain || ""),
            domainOther: normalizeLine(raw.domain_other || raw.domainOther || meta.domainOther || ""),
            coreConcept: normalizeLine(raw.core_concept || raw.coreConcept || meta.coreConcept || ""),
            documentType: normalizeLine(raw.document_type || raw.documentType || meta.documentType || ""),
            sourceRow: index + 2
        };
    }).filter(Boolean).sort((a, b) =>
        a.sectionOrder - b.sectionOrder ||
        (a.chunkOrder ?? Number.MAX_SAFE_INTEGER) - (b.chunkOrder ?? Number.MAX_SAFE_INTEGER) ||
        a.order - b.order ||
        a.sourceRow - b.sourceRow
    );
}

function canonicalSentence(text) {
    return normalizeLine(text)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function mergeOverlappingChunkTexts(rows) {
    const merged = [];

    for (const row of rows) {
        const incoming = sentenceSplit(row.textChunk || row.text);
        if (!incoming.length) continue;

        let overlapCount = 0;
        const maxCheck = Math.min(4, merged.length, incoming.length);
        for (let size = maxCheck; size >= 1; size--) {
            const left = merged.slice(-size).map(canonicalSentence);
            const right = incoming.slice(0, size).map(canonicalSentence);
            if (left.every((value, index) => value && value === right[index])) {
                overlapCount = size;
                break;
            }
        }

        merged.push(...incoming.slice(overlapCount));
    }

    return merged.join(" ").trim();
}

function structuredRowsToSections(rows) {
    const groups = new Map();

    for (const row of rows) {
        const key = row.sectionId || row.recordId || `section_${groups.size + 1}`;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(row);
    }

    return [...groups.entries()].map(([sectionId, sectionRows], index) => {
        sectionRows.sort((a, b) => (a.chunkOrder ?? a.order) - (b.chunkOrder ?? b.order));
        const first = sectionRows[0];
        const text = sectionRows.some(row => row.chunkId)
            ? mergeOverlappingChunkTexts(sectionRows)
            : sectionRows.map(row => row.text).join("\n\n").trim();

        const blocks = [];
        if (text) {
            blocks.push({
                type: "content",
                text,
                paragraphIndex: 1,
                citations: [...new Set(sectionRows.flatMap(row => row.citations || []))],
                links: [...new Set(sectionRows.flatMap(row => row.links || []))],
                images: [...new Set(sectionRows.flatMap(row => row.images || []))],
                referenceMaterials: [...new Set(sectionRows.flatMap(row => row.referenceMaterials || []))]
            });
        }

        return {
            id: sectionId,
            sectionId,
            recordId: first.recordId || sectionId,
            parentSectionId: first.parentSectionId || null,
            chapter: first.chapter || "",
            heading: first.heading || first.subheading || first.chapter || "Document",
            subheading: first.subheading || "",
            pageNumber: first.pageNumber || null,
            text,
            blocks,
            order: first.sectionOrder || first.order || index + 1,
            recordType: first.recordType || "section",
            keywords: [...new Set(sectionRows.flatMap(row => row.keywords || []))],
            aliases: [...new Set(sectionRows.flatMap(row => row.aliases || []))],
            relatedRecordIds: [...new Set(sectionRows.flatMap(row => row.relatedRecordIds || []))],
            relatedChunkIds: [...new Set(sectionRows.flatMap(row => row.relatedChunkIds || []))],
            reviewStatus: first.reviewStatus || "reviewed",
            reviewerNotes: sectionRows.map(row => row.reviewerNotes).filter(Boolean).join(" | "),
            humanCorrected: sectionRows.some(row => row.humanCorrected),
            domain: first.domain || "",
            domainOther: first.domainOther || "",
            coreConcept: first.coreConcept || "",
            documentType: first.documentType || "",
            sourceRows: sectionRows
        };
    }).filter(section => section.text);
}

function structuredRowsToChunks(rows, sectionSentences) {
    const chunkRows = rows.filter(row => row.chunkId);
    if (!chunkRows.length) return [];

    const sentenceBySection = new Map();
    for (const sentence of sectionSentences) {
        if (!sentenceBySection.has(sentence.sectionId)) sentenceBySection.set(sentence.sectionId, []);
        sentenceBySection.get(sentence.sectionId).push(sentence);
    }

    return chunkRows.map((row, index) => {
        const sectionItems = sentenceBySection.get(row.sectionId) || [];
        const rowSentences = sentenceSplit(row.textChunk);
        const startMatch = rowSentences.length
            ? sectionItems.find(item => canonicalSentence(item.text) === canonicalSentence(rowSentences[0]))
            : null;
        const endMatch = rowSentences.length
            ? [...sectionItems].reverse().find(item => canonicalSentence(item.text) === canonicalSentence(rowSentences[rowSentences.length - 1]))
            : null;
        const first = startMatch || sectionItems[0] || null;
        const last = endMatch || sectionItems[sectionItems.length - 1] || first;

        return {
            id: row.chunkId,
            chunkId: row.chunkId,
            chunkOrder: row.chunkOrder || index + 1,
            text: row.textChunk,
            textChunk: row.textChunk,
            wordCount: wordCount(row.textChunk),
            startSentence: first?.sentenceNumber || 1,
            endSentence: last?.sentenceNumber || first?.sentenceNumber || 1,
            startWord: first?.wordPos || 1,
            endWord: last ? last.wordPos + last.wordCount - 1 : wordCount(row.textChunk),
            sectionId: row.sectionId,
            sectionHeading: row.subheading || row.heading || row.chapter || "Document",
            chapter: row.chapter || "",
            heading: row.heading || "",
            subheading: row.subheading || "",
            pageNumber: row.pageNumber || null,
            recordIds: [row.recordId || row.sectionId].filter(Boolean),
            recordType: row.recordType || "section",
            chunkType: row.chunkType || "content",
            keywords: row.keywords || [],
            aliases: row.aliases || [],
            relatedRecordIds: row.relatedRecordIds || [],
            relatedChunkIds: row.relatedChunkIds || [],
            citations: row.citations || [],
            referenceMaterials: row.referenceMaterials || [],
            links: row.links || [],
            images: row.images || [],
            humanCorrected: Boolean(row.humanCorrected),
            domain: row.domain || "",
            domainOther: row.domainOther || "",
            coreConcept: row.coreConcept || "",
            documentType: row.documentType || "",
            overlapWords: 0,
            overlapPercent: 0
        };
    });
}

function buildBlocksFromLines(lines, { referenceMode = false } = {}) {
    const blocks = [];
    const sectionLinks = [];
    const images = [];
    let paragraph = [];
    let paragraphIndex = 0;

    const flushParagraph = () => {
        if (!paragraph.length) return;
        paragraphIndex++;
        const raw = paragraph.join(" ").replace(/\s+/g, " ").trim();
        paragraph = [];
        if (!raw) return;

        const inline = extractInlineMetadata(raw);
        if (!inline.cleanText) {
            sectionLinks.push(...inline.links);
            return;
        }

        blocks.push({
            type: "content",
            text: inline.cleanText,
            paragraphIndex,
            citations: inline.citations,
            links: inline.links,
            images: [],
            referenceMaterials: []
        });
        sectionLinks.push(...inline.links);
    };

    for (const line of lines) {
        if (line.blank) {
            flushParagraph();
            continue;
        }

        if (line.specialType === "link") {
            flushParagraph();
            const url = cleanString(line.markerUrl || line.text).trim();
            if (url) sectionLinks.push(url);
            continue;
        }

        if (line.specialType === "quote" || /^>\s+/.test(line.text) || (/^[“"]/.test(line.text) && /[”"]$/.test(line.text))) {
            flushParagraph();
            paragraphIndex++;
            const quoteText = line.text.replace(/^>\s+/, "");
            const inline = extractInlineMetadata(quoteText);
            if (inline.cleanText) {
                blocks.push({
                    type: "quote",
                    text: inline.cleanText,
                    paragraphIndex,
                    citations: inline.citations,
                    links: inline.links,
                    images: [],
                    referenceMaterials: []
                });
            }
            sectionLinks.push(...inline.links);
            continue;
        }

        if (line.specialType === "image_caption" || looksLikeImageCaption(line.text)) {
            flushParagraph();
            paragraphIndex++;
            const inline = extractInlineMetadata(line.text);
            if (inline.cleanText) {
                images.push(inline.cleanText);
                blocks.push({
                    type: "image_caption",
                    text: inline.cleanText,
                    paragraphIndex,
                    citations: inline.citations,
                    links: inline.links,
                    images: [inline.cleanText],
                    referenceMaterials: []
                });
            }
            sectionLinks.push(...inline.links);
            continue;
        }

        if (referenceMode) {
            flushParagraph();
            paragraphIndex++;
            const inline = extractInlineMetadata(line.text);
            if (inline.cleanText || inline.links.length) {
                blocks.push({
                    type: "reference",
                    text: inline.cleanText || line.text,
                    paragraphIndex,
                    citations: [],
                    links: inline.links,
                    images: [],
                    referenceMaterials: []
                });
            }
            sectionLinks.push(...inline.links);
            continue;
        }

        paragraph.push(line.text);
    }

    flushParagraph();

    return {
        blocks,
        links: [...new Set(sectionLinks)],
        images: [...new Set(images)]
    };
}

function linesToSections(cleanLines, metadata = {}) {
    const sections = [];
    const hierarchy = { chapter: "", heading: "", subheading: "" };
    let current = {
        pageNumber: cleanLines.find(x => x.pageNumber)?.pageNumber || null,
        lines: [],
        chapter: "",
        heading: "Document",
        subheading: "",
        headingLevel: 0,
        sectionTitle: "Document"
    };

    const pushSection = () => {
        if (!current.lines.some(line => line.text || line.specialType)) return;
        const title = current.subheading || current.heading || current.chapter || current.sectionTitle || "Document";
        const referenceMode = isReferenceHeading(title);
        const built = buildBlocksFromLines(current.lines, { referenceMode });
        const text = built.blocks.map(block => block.text).filter(Boolean).join("\n\n").trim();
        if (!text && !built.links.length && !built.images.length) return;

        const id = `section_${String(sections.length + 1).padStart(3, "0")}`;
        sections.push({
            id,
            sectionId: id,
            recordId: id,
            parentSectionId: null,
            chapter: current.chapter || "",
            heading: current.heading || current.chapter || "Document",
            subheading: current.subheading || "",
            sectionTitle: title,
            headingLevel: current.headingLevel || 0,
            pageNumber: current.pageNumber || null,
            text,
            blocks: built.blocks,
            links: built.links,
            images: built.images,
            isReferenceSection: referenceMode,
            order: sections.length + 1,
            recordType: referenceMode ? "reference" : "section",
            keywords: [],
            aliases: [],
            relatedRecordIds: [],
            relatedChunkIds: [],
            reviewStatus: "machine_processed",
            reviewerNotes: "",
            humanCorrected: false,
            domain: metadata.domain || "",
            domainOther: metadata.domainOther || "",
            coreConcept: metadata.coreConcept || "",
            documentType: metadata.documentType || ""
        });
    };

    const startSection = ({ pageNumber = null, level = 0, title = "Document" } = {}) => {
        current = {
            pageNumber: pageNumber || current.pageNumber || null,
            lines: [],
            chapter: hierarchy.chapter,
            heading: hierarchy.heading || hierarchy.chapter || "Document",
            subheading: hierarchy.subheading,
            headingLevel: level,
            sectionTitle: title
        };
    };

    for (const line of cleanLines) {
        if (
            line.pageNumber &&
            current.pageNumber &&
            line.pageNumber !== current.pageNumber &&
            current.lines.some(item => item.text || item.specialType)
        ) {
            pushSection();
            startSection({ pageNumber: line.pageNumber, level: current.headingLevel, title: current.sectionTitle });
        }

        if (line.heading && line.text) {
            pushSection();
            const level = Math.max(1, Math.min(6, Number(line.headingLevel) || 2));

            if (level === 1) {
                hierarchy.chapter = line.text;
                hierarchy.heading = "";
                hierarchy.subheading = "";
            } else if (level === 2) {
                hierarchy.heading = line.text;
                hierarchy.subheading = "";
            } else {
                hierarchy.subheading = line.text;
            }

            startSection({ pageNumber: line.pageNumber || current.pageNumber, level, title: line.text });
            continue;
        }

        if (!current.pageNumber && line.pageNumber) current.pageNumber = line.pageNumber;
        current.lines.push(line);
    }

    pushSection();

    if (!sections.length) {
        const built = buildBlocksFromLines(cleanLines, { referenceMode: false });
        const text = built.blocks.map(block => block.text).filter(Boolean).join("\n\n").trim();
        if (text) {
            sections.push({
                id: "section_001",
                sectionId: "section_001",
                recordId: "section_001",
                parentSectionId: null,
                chapter: "",
                heading: "Document",
                subheading: "",
                sectionTitle: "Document",
                headingLevel: 0,
                pageNumber: cleanLines.find(x => x.pageNumber)?.pageNumber || null,
                text,
                blocks: built.blocks,
                links: built.links,
                images: built.images,
                isReferenceSection: false,
                order: 1,
                recordType: "section",
                keywords: [],
                aliases: [],
                relatedRecordIds: [],
                relatedChunkIds: [],
                reviewStatus: "machine_processed",
                reviewerNotes: "",
                humanCorrected: false,
                domain: metadata.domain || "",
                domainOther: metadata.domainOther || "",
                coreConcept: metadata.coreConcept || "",
                documentType: metadata.documentType || ""
            });
        }
    }

    return sections;
}

function extractReferenceTable(sections) {
    const references = [];
    const contentSections = [];

    for (const section of sections) {
        if (!section.isReferenceSection) {
            contentSections.push(section);
            continue;
        }

        for (const block of section.blocks || []) {
            const citationText = normalizeLine(block.text);
            if (!citationText) continue;
            const idMatch = citationText.match(/^\s*\[?(\d+)\]?\s*[.)\-:]?/);
            const referenceId = idMatch
                ? `ref_${String(idMatch[1]).padStart(3, "0")}`
                : `ref_${String(references.length + 1).padStart(3, "0")}`;
            references.push({
                referenceId,
                citationText,
                links: [...new Set(block.links || [])],
                keys: referenceLookupKeys(citationText),
                pageNumber: section.pageNumber || null
            });
        }
    }

    return { contentSections, references };
}

function buildReferenceIndex(references) {
    const index = new Map();
    for (const reference of references || []) {
        for (const key of reference.keys || []) {
            if (!index.has(key)) index.set(key, reference.referenceId);
        }
    }
    return index;
}

function attachReferenceLinks(sections, references) {
    const index = buildReferenceIndex(references);
    return sections.map(section => ({
        ...section,
        blocks: (section.blocks || []).map(block => {
            const ids = new Set(block.referenceMaterials || []);
            for (const citation of block.citations || []) {
                for (const key of citationLookupKeys(citation)) {
                    const id = index.get(key);
                    if (id) ids.add(id);
                }
            }
            return { ...block, referenceMaterials: [...ids] };
        })
    }));
}

function splitParagraphs(text) {
    return normalizeUnicode(text)
        .split(/\n\s*\n+/)
        .flatMap(block => {
            const lines = block.split("\n").map(normalizeLine).filter(Boolean);
            if (!lines.length) return [];

            const paragraphs = [];
            let buffer = [];

            const flush = () => {
                const joined = buffer.join(" ").replace(/\s+/g, " ").trim();
                if (joined) paragraphs.push(joined);
                buffer = [];
            };

            for (const line of lines) {
                if (isLikelyHeading(line) && buffer.length) {
                    flush();
                    paragraphs.push(line);
                    continue;
                }
                buffer.push(line);
            }
            flush();
            return paragraphs;
        });
}

function dedupeSections(sections, minWords = DEFAULTS.minParagraphWordsForDedupe) {
    const seenParagraphs = new Set();
    let duplicateParagraphs = 0;

    const result = sections.map(section => {
        if (!Array.isArray(section.blocks) || !section.blocks.length) return section;
        const keptBlocks = [];

        for (const block of section.blocks) {
            const canonical = canonicalParagraph(block.text || "");
            const words = wordCount(block.text || "");
            const dedupeEligible = block.type === "content" && words >= minWords;

            if (dedupeEligible && canonical && seenParagraphs.has(canonical)) {
                duplicateParagraphs++;
                continue;
            }

            if (dedupeEligible && canonical) seenParagraphs.add(canonical);
            keptBlocks.push(block);
        }

        const text = keptBlocks.map(block => block.text).filter(Boolean).join("\n\n").trim();
        return { ...section, blocks: keptBlocks, text };
    }).filter(section => section.text || section.links?.length || section.images?.length);

    return { sections: result, duplicateParagraphs };
}

function truncateBlockAtSentenceBoundary(block, maxWords) {
    if (maxWords <= 0) return null;
    const sentences = sentenceSplit(block.text || "");
    const kept = [];
    let words = 0;

    for (const sentence of sentences) {
        const count = wordCount(sentence);
        if (kept.length && words + count > maxWords) break;
        if (!kept.length && count > maxWords) {
            // Preserve the sentence boundary even when the final sentence is
            // longer than the remaining nominal word budget.
            kept.push(sentence);
            words += count;
            break;
        }
        kept.push(sentence);
        words += count;
    }

    if (!kept.length) return null;
    return { ...block, text: kept.join(" "), _limitedWords: words };
}

function limitSectionsByWords(sections, limit) {
    const safeLimit = Number.isFinite(Number(limit)) && Number(limit) > 0
        ? Math.floor(Number(limit))
        : DEFAULTS.wordLimit;

    let remaining = safeLimit;
    const limited = [];
    let truncated = false;

    for (const section of sections) {
        if (remaining <= 0) {
            truncated = true;
            break;
        }

        const sectionWords = wordCount(section.text || "");
        if (sectionWords <= remaining) {
            limited.push(section);
            remaining -= sectionWords;
            continue;
        }

        const keptBlocks = [];
        let used = 0;
        for (const block of section.blocks || []) {
            if (used >= remaining) break;
            const count = wordCount(block.text || "");
            if (used + count <= remaining) {
                keptBlocks.push(block);
                used += count;
                continue;
            }

            const partial = truncateBlockAtSentenceBoundary(block, remaining - used);
            if (partial) {
                used += partial._limitedWords || wordCount(partial.text);
                delete partial._limitedWords;
                keptBlocks.push(partial);
            }
            break;
        }

        if (keptBlocks.length) {
            limited.push({
                ...section,
                blocks: keptBlocks,
                text: keptBlocks.map(block => block.text).filter(Boolean).join("\n\n").trim()
            });
        }

        remaining -= used;
        truncated = true;
        break;
    }

    return { sections: limited, truncated, usedWords: safeLimit - Math.max(0, remaining) };
}

function sectionsToCleanText(sections) {
    return sections.map(section => {
        const hierarchy = [section.chapter, section.heading, section.subheading]
            .map(normalizeLine)
            .filter(Boolean)
            .filter((value, index, arr) => arr.indexOf(value) === index && value !== "Document");
        const title = hierarchy.length ? `${hierarchy.join(" > ")}\n` : "";
        return `${title}${section.text || ""}`.trim();
    }).filter(Boolean).join("\n\n").trim();
}

function sentenceSplit(text) {
    const source = cleanString(text).trim();
    if (!source) return [];

    try {
        if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
            const segmenter = new Intl.Segmenter("en", { granularity: "sentence" });
            return [...segmenter.segment(source)]
                .map(part => part.segment.trim())
                .filter(Boolean);
        }
    } catch (_) {}

    return (source.match(/[^.!?\n]+[.!?]+|[^.!?\n]+$/g) || [source])
        .map(s => s.trim())
        .filter(Boolean);
}

function buildSentenceRecords(sections) {
    const records = [];
    let wordPos = 1;

    for (const section of sections) {
        const blocks = Array.isArray(section.blocks) && section.blocks.length
            ? section.blocks
            : [{ type: "content", text: section.text, paragraphIndex: 1, citations: [], links: [], images: [], referenceMaterials: [] }];

        for (const block of blocks) {
            if (!block?.text || block.type === "reference" || block.type === "link") continue;
            const sentences = block.type === "image_caption"
                ? [normalizeLine(block.text)].filter(Boolean)
                : sentenceSplit(block.text);

            for (let localIndex = 0; localIndex < sentences.length; localIndex++) {
                const sentence = sentences[localIndex];
                const count = wordCount(sentence);
                if (!count) continue;

                records.push({
                    id: `sentence_${records.length + 1}`,
                    text: sentence,
                    sentenceNumber: records.length + 1,
                    sentenceIndexInSection: localIndex + 1,
                    wordPos,
                    wordCount: count,
                    paragraphIndex: Number(block.paragraphIndex) || 1,
                    chunkType: block.type || "content",
                    sectionId: section.id,
                    sectionHeading: section.subheading || section.heading || section.chapter || "Document",
                    chapter: section.chapter || "",
                    heading: section.heading || "",
                    subheading: section.subheading || "",
                    pageNumber: section.pageNumber || null,
                    recordId: section.recordId || section.id || null,
                    parentSectionId: section.parentSectionId || null,
                    recordType: section.recordType || "section",
                    keywords: Array.isArray(section.keywords) ? [...section.keywords] : [],
                    aliases: Array.isArray(section.aliases) ? [...section.aliases] : [],
                    relatedRecordIds: Array.isArray(section.relatedRecordIds) ? [...section.relatedRecordIds] : [],
                    relatedChunkIds: Array.isArray(section.relatedChunkIds) ? [...section.relatedChunkIds] : [],
                    citations: [...new Set(block.citations || [])],
                    referenceMaterials: [...new Set(block.referenceMaterials || [])],
                    links: [...new Set(block.links || [])],
                    images: [...new Set(block.images || [])],
                    reviewStatus: section.reviewStatus || null,
                    humanCorrected: Boolean(section.humanCorrected),
                    domain: section.domain || "",
                    domainOther: section.domainOther || "",
                    coreConcept: section.coreConcept || "",
                    documentType: section.documentType || ""
                });
                wordPos += count;
            }
        }
    }

    return records;
}

function chooseOverlapSentences(previousSentences, targetWords = DEFAULTS.chunkTargetWords) {
    const minWords = Math.max(1, Math.ceil(targetWords * DEFAULTS.chunkOverlapMinPercent));
    const maxWords = Math.max(minWords, Math.floor(targetWords * DEFAULTS.chunkOverlapMaxPercent));
    const ideal = Math.round(targetWords * DEFAULTS.chunkOverlapPercent);
    let best = [];
    let bestDistance = Infinity;

    for (let start = previousSentences.length - 1; start >= 0; start--) {
        const candidate = previousSentences.slice(start);
        const words = candidate.reduce((sum, sentence) => sum + sentence.wordCount, 0);
        if (words > maxWords) continue;
        if (words < minWords) continue;
        const distance = Math.abs(words - ideal);
        if (distance < bestDistance) {
            best = candidate;
            bestDistance = distance;
        }
    }

    return best;
}

function buildSemanticSectionChunks(sectionSentences, targetWords = DEFAULTS.chunkTargetWords) {
    if (!sectionSentences.length) return [];

    const target = Math.max(40, Number(targetWords) || DEFAULTS.chunkTargetWords);
    const minWords = Math.max(25, Math.round(target * 0.60));
    const maxWords = Math.max(target + 10, Math.round(target * 1.30));
    const bases = [];
    let current = [];
    let currentWords = 0;

    const flush = () => {
        if (!current.length) return;
        bases.push({ sentences: current, wordCount: currentWords });
        current = [];
        currentWords = 0;
    };

    for (let i = 0; i < sectionSentences.length; i++) {
        const sentence = sectionSentences[i];
        const previous = current[current.length - 1] || null;
        const next = sectionSentences[i + 1] || null;

        const typeChanged = previous && sentence.chunkType !== previous.chunkType;
        const newParagraph = previous && sentence.paragraphIndex !== previous.paragraphIndex;
        const wouldExceed = current.length && currentWords + sentence.wordCount > maxWords;

        if (typeChanged || (wouldExceed && currentWords >= minWords) || (newParagraph && currentWords >= minWords)) {
            flush();
        }

        current.push(sentence);
        currentWords += sentence.wordCount;

        const nextTypeChanged = next && next.chunkType !== sentence.chunkType;
        const nextParagraph = next && next.paragraphIndex !== sentence.paragraphIndex;
        const coherentWithNext = next ? tokenSimilarity(current.map(item => item.text).join(" "), next.text) : 0;

        if (
            nextTypeChanged ||
            currentWords >= maxWords ||
            (currentWords >= target && (nextParagraph || !next || coherentWithNext < 0.08))
        ) {
            flush();
        }
    }

    flush();

    // Avoid orphan tail chunks where a short paragraph follows a coherent block.
    if (bases.length >= 2) {
        const last = bases[bases.length - 1];
        const previous = bases[bases.length - 2];
        const sameType = last.sentences[0]?.chunkType === previous.sentences[0]?.chunkType;
        if (sameType && last.wordCount < minWords && previous.wordCount + last.wordCount <= Math.round(maxWords * 1.25)) {
            previous.sentences.push(...last.sentences);
            previous.wordCount += last.wordCount;
            bases.pop();
        }
    }

    return bases;
}

function buildChunks(sentences, targetWords = DEFAULTS.chunkTargetWords) {
    const grouped = new Map();
    for (const sentence of sentences) {
        const key = sentence.sectionId || sentence.recordId || "document";
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(sentence);
    }

    const chunks = [];

    for (const [sectionId, sectionSentences] of grouped) {
        const bases = buildSemanticSectionChunks(sectionSentences, targetWords);
        let sectionChunkOrder = 0;

        for (let i = 0; i < bases.length; i++) {
            const base = bases[i];
            const firstBaseSentence = base.sentences[0];
            const overlap = i > 0 && bases[i - 1].sentences[0]?.chunkType === firstBaseSentence?.chunkType
                ? chooseOverlapSentences(bases[i - 1].sentences, Number(targetWords) || DEFAULTS.chunkTargetWords)
                : [];
            const collected = [...overlap, ...base.sentences];
            if (!collected.length) continue;

            sectionChunkOrder++;
            const first = collected[0];
            const last = collected[collected.length - 1];
            const overlapWords = overlap.reduce((sum, sentence) => sum + sentence.wordCount, 0);
            const baseWords = Math.max(1, base.wordCount);
            const text = collected.map(sentence => sentence.text).join(" ").trim();
            const hierarchy = {
                chapter: first.chapter || "",
                heading: first.heading || "",
                subheading: first.subheading || ""
            };
            const metadata = {
                domain: first.domain || "",
                domainOther: first.domainOther || "",
                coreConcept: first.coreConcept || "",
                documentType: first.documentType || ""
            };
            const inheritedKeywords = [...new Set(collected.flatMap(sentence => sentence.keywords || []))];
            const generatedKeywords = extractChunkKeywords(text, metadata, hierarchy, DEFAULTS.keywordMin, DEFAULTS.keywordMax);
            const keywords = [...new Set([...inheritedKeywords, ...generatedKeywords])].slice(0, DEFAULTS.keywordMax);
            const chunkId = `${sectionId}_chunk_${String(sectionChunkOrder).padStart(3, "0")}`;

            chunks.push({
                id: chunkId,
                chunkId,
                chunkOrder: sectionChunkOrder,
                text,
                textChunk: text,
                wordCount: wordCount(text),
                baseWordCount: baseWords,
                overlapWords,
                overlapPercent: Math.round((overlapWords / baseWords) * 1000) / 10,
                startSentence: first.sentenceNumber,
                endSentence: last.sentenceNumber,
                startWord: first.wordPos,
                endWord: last.wordPos + last.wordCount - 1,
                sectionId,
                sectionHeading: first.subheading || first.heading || first.chapter || "Document",
                chapter: first.chapter || "",
                heading: first.heading || "",
                subheading: first.subheading || "",
                pageNumber: first.pageNumber || null,
                recordIds: [...new Set(collected.map(sentence => sentence.recordId).filter(Boolean))],
                recordType: first.recordType || "section",
                chunkType: first.chunkType || "content",
                keywords,
                aliases: [...new Set(collected.flatMap(sentence => sentence.aliases || []))],
                relatedRecordIds: [...new Set(collected.flatMap(sentence => sentence.relatedRecordIds || []))],
                relatedChunkIds: [...new Set(collected.flatMap(sentence => sentence.relatedChunkIds || []))],
                citations: [...new Set(collected.flatMap(sentence => sentence.citations || []))],
                referenceMaterials: [...new Set(collected.flatMap(sentence => sentence.referenceMaterials || []))],
                links: [...new Set(collected.flatMap(sentence => sentence.links || []))],
                images: [...new Set(collected.flatMap(sentence => sentence.images || []))],
                humanCorrected: collected.some(sentence => sentence.humanCorrected),
                domain: first.domain || "",
                domainOther: first.domainOther || "",
                coreConcept: first.coreConcept || "",
                documentType: first.documentType || ""
            });
        }
    }

    return chunks;
}

function buildStructuredChunks(sentences, targetWords) {
    return buildChunks(sentences, targetWords);
}

function extractKeywords(text, max = 20) {
    const terms = extractChunkKeywords(text, {}, {}, Math.min(5, max), Math.min(max, 20));
    return terms.map(term => ({ term, count: 1 }));
}

function fnv1a(text) {
    let hash = 0x811c9dc5;
    const input = cleanString(text);
    for (let i = 0; i < input.length; i++) {
        hash ^= input.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, "0");
}

function processSource(payload = {}) {
    const sourceType = normalizeLine(payload.sourceType || "text").toLowerCase() || "text";
    const title = normalizeLine(payload.title || "Current Reading") || "Current Reading";
    const metadata = sourceMetadata(payload);

    const pages = Array.isArray(payload.pages) ? payload.pages : null;
    const assembledRaw = pages?.length
        ? pages.map(p => cleanString(p.text)).join("\n\n")
        : cleanString(payload.rawText);

    if (!assembledRaw.trim()) {
        throw new Error("PROCESS_SOURCE requires source text.");
    }

    if (assembledRaw.length > DEFAULTS.maxInputChars) {
        throw new Error(
            `Source is too large for browser ingestion (${assembledRaw.length.toLocaleString()} characters). ` +
            `Maximum is ${DEFAULTS.maxInputChars.toLocaleString()}.`
        );
    }

    const rawWords = wordCount(assembledRaw);
    const { cleanLines, removed } = prepareCleanLines({ rawText: assembledRaw, pages });
    const hierarchySections = linesToSections(cleanLines, metadata);
    const extracted = extractReferenceTable(hierarchySections);
    const linkedSections = attachReferenceLinks(extracted.contentSections, extracted.references);
    const deduped = dedupeSections(linkedSections);
    const limited = limitSectionsByWords(deduped.sections, payload.wordLimit || DEFAULTS.wordLimit);

    const cleanText = sectionsToCleanText(limited.sections);
    const sentences = buildSentenceRecords(limited.sections);
    const chunks = buildChunks(
        sentences,
        payload.chunkTargetWords || DEFAULTS.chunkTargetWords
    );

    const cleanWords = wordCount(cleanText);
    const reductionWords = Math.max(0, rawWords - cleanWords);

    return {
        ingestionVersion: INGESTION_VERSION,
        structuredSchemaVersion: "miai-source-csv-2.0",
        sourceType,
        title,
        sourceMetadata: metadata,
        fingerprint: fnv1a(`${JSON.stringify(metadata)}\n${cleanText}`),
        cleanText,
        fitText: cleanText,
        references: extracted.references.map(reference => ({
            referenceId: reference.referenceId,
            citationText: reference.citationText,
            links: reference.links || [],
            pageNumber: reference.pageNumber || null
        })),
        sections: limited.sections.map(section => ({
            id: section.id,
            chapter: section.chapter || "",
            heading: section.heading || "Document",
            subheading: section.subheading || "",
            pageNumber: section.pageNumber,
            recordType: section.recordType || "section",
            wordCount: wordCount(section.text)
        })),
        sectionRecords: limited.sections.map((section, index) => ({
            recordId: section.recordId || section.id || `record_${index + 1}`,
            sectionId: section.id || `section_${index + 1}`,
            parentSectionId: section.parentSectionId || null,
            chapter: section.chapter || "",
            heading: section.heading || "Document",
            subheading: section.subheading || "",
            pageNumber: section.pageNumber || null,
            recordType: section.recordType || "section",
            text: section.text,
            keywords: Array.isArray(section.keywords) ? [...section.keywords] : [],
            aliases: Array.isArray(section.aliases) ? [...section.aliases] : [],
            relatedRecordIds: Array.isArray(section.relatedRecordIds) ? [...section.relatedRecordIds] : [],
            relatedChunkIds: Array.isArray(section.relatedChunkIds) ? [...section.relatedChunkIds] : [],
            links: [...new Set(section.links || [])],
            images: [...new Set(section.images || [])],
            reviewStatus: section.reviewStatus || "machine_processed",
            reviewerNotes: section.reviewerNotes || "",
            humanCorrected: Boolean(section.humanCorrected),
            include: true,
            order: Number(section.order) || index + 1,
            domain: metadata.domain,
            domainOther: metadata.domainOther,
            coreConcept: metadata.coreConcept,
            documentType: metadata.documentType,
            wordCount: wordCount(section.text)
        })),
        sentences,
        chunks,
        keywords: extractKeywords(cleanText),
        stats: {
            rawChars: assembledRaw.length,
            rawWords,
            cleanWords,
            reductionWords,
            reductionPercent: rawWords ? Math.round((reductionWords / rawWords) * 1000) / 10 : 0,
            sections: limited.sections.length,
            sentences: sentences.length,
            chunks: chunks.length,
            references: extracted.references.length,
            chunkTargetWords: payload.chunkTargetWords || DEFAULTS.chunkTargetWords,
            chunkOverlapPercent: DEFAULTS.chunkOverlapPercent,
            truncated: limited.truncated,
            removedPageNumbers: removed.pageNumbers,
            removedRepeatedHeadersFooters: removed.repeatedHeadersFooters,
            removedDuplicateParagraphs: deduped.duplicateParagraphs,
            removedDuplicateBlankLines: removed.duplicateBlankLines
        }
    };
}

function processStructuredSource(payload = {}) {
    const title = normalizeLine(payload.title || "Structured CSV Source") || "Structured CSV Source";
    const sourceType = normalizeLine(payload.sourceType || "csv").toLowerCase() || "csv";
    const rows = normalizeStructuredRecords(payload.records, payload);

    if (!rows.length) {
        throw new Error("PROCESS_STRUCTURED_SOURCE requires at least one included row with text_chunk/text.");
    }

    const metadata = {
        domain: rows.find(row => row.domain)?.domain || sourceMetadata(payload).domain,
        domainOther: rows.find(row => row.domainOther)?.domainOther || sourceMetadata(payload).domainOther,
        coreConcept: rows.find(row => row.coreConcept)?.coreConcept || sourceMetadata(payload).coreConcept,
        documentType: rows.find(row => row.documentType)?.documentType || sourceMetadata(payload).documentType
    };

    const referenceRows = rows.filter(row => row.chunkType === "reference" || row.recordType === "reference");
    const contentRows = rows.filter(row => row.chunkType !== "reference" && row.recordType !== "reference");
    if (!contentRows.length) throw new Error("Structured CSV contains no included content chunks.");

    const references = [
        ...referenceRows.map((row, index) => ({
            referenceId: row.referenceMaterials?.[0] || row.recordId || `ref_${String(index + 1).padStart(3, "0")}`,
            citationText: row.textChunk,
            links: row.links || [],
            pageNumber: row.pageNumber || null
        })),
        ...(Array.isArray(payload.references) ? payload.references.map((reference, index) => ({
            referenceId: normalizeLine(reference.referenceId || reference.reference_id || `ref_${String(index + 1).padStart(3, "0")}`),
            citationText: normalizeLine(reference.citationText || reference.citation_text || ""),
            links: splitList(reference.links),
            pageNumber: Number(reference.pageNumber || reference.page_number) || null,
            reviewStatus: normalizeLine(reference.reviewStatus || reference.review_status || ""),
            reviewerNotes: cleanString(reference.reviewerNotes || reference.reviewer_notes || "").trim()
        })).filter(reference => reference.referenceId && reference.citationText) : [])
    ];

    const sections = structuredRowsToSections(contentRows).map(section => ({ ...section, ...metadata }));
    const assembledRaw = sections.map(section => section.text).join("\n\n");
    if (assembledRaw.length > DEFAULTS.maxInputChars) {
        throw new Error(
            `Structured source is too large for browser ingestion (${assembledRaw.length.toLocaleString()} characters). Maximum is ${DEFAULTS.maxInputChars.toLocaleString()}.`
        );
    }

    const limited = limitSectionsByWords(sections, payload.wordLimit || DEFAULTS.wordLimit);
    const cleanText = sectionsToCleanText(limited.sections);
    const sentences = buildSentenceRecords(limited.sections);
    const totalWords = wordCount(assembledRaw);
    const withinLimit = totalWords <= Number(payload.wordLimit || DEFAULTS.wordLimit);
    let chunks = withinLimit ? structuredRowsToChunks(contentRows, sentences) : [];

    if (!chunks.length) {
        chunks = buildStructuredChunks(
            sentences,
            payload.chunkTargetWords || DEFAULTS.chunkTargetWords
        );
    } else {
        chunks = chunks.map(chunk => {
            const generated = extractChunkKeywords(
                chunk.text,
                metadata,
                { chapter: chunk.chapter, heading: chunk.heading, subheading: chunk.subheading },
                DEFAULTS.keywordMin,
                DEFAULTS.keywordMax
            );
            return {
                ...chunk,
                keywords: [...new Set([...(chunk.keywords || []), ...generated])].slice(0, DEFAULTS.keywordMax)
            };
        });
    }

    const cleanWords = wordCount(cleanText);

    return {
        ingestionVersion: INGESTION_VERSION,
        structuredSchemaVersion: cleanString(payload.schemaVersion || "miai-source-csv-2.0"),
        sourceType,
        title,
        sourceMetadata: metadata,
        fingerprint: fnv1a(`${JSON.stringify(metadata)}\n${cleanText}`),
        cleanText,
        fitText: cleanText,
        references,
        sections: limited.sections.map(section => ({
            id: section.id,
            chapter: section.chapter || "",
            heading: section.heading || "Document",
            subheading: section.subheading || "",
            pageNumber: section.pageNumber,
            recordType: section.recordType || "section",
            wordCount: wordCount(section.text)
        })),
        sectionRecords: limited.sections.map((section, index) => ({
            recordId: section.recordId || section.id || `record_${index + 1}`,
            sectionId: section.id || `section_${index + 1}`,
            parentSectionId: section.parentSectionId || null,
            chapter: section.chapter || "",
            heading: section.heading || "Document",
            subheading: section.subheading || "",
            pageNumber: section.pageNumber || null,
            recordType: section.recordType || "section",
            text: section.text,
            keywords: Array.isArray(section.keywords) ? [...section.keywords] : [],
            aliases: Array.isArray(section.aliases) ? [...section.aliases] : [],
            relatedRecordIds: Array.isArray(section.relatedRecordIds) ? [...section.relatedRecordIds] : [],
            relatedChunkIds: Array.isArray(section.relatedChunkIds) ? [...section.relatedChunkIds] : [],
            reviewStatus: section.reviewStatus || "reviewed",
            reviewerNotes: section.reviewerNotes || "",
            humanCorrected: Boolean(section.humanCorrected),
            include: true,
            order: Number(section.order) || index + 1,
            domain: metadata.domain,
            domainOther: metadata.domainOther,
            coreConcept: metadata.coreConcept,
            documentType: metadata.documentType,
            wordCount: wordCount(section.text)
        })),
        sentences,
        chunks,
        keywords: extractKeywords(cleanText),
        stats: {
            rawChars: assembledRaw.length,
            rawWords: totalWords,
            cleanWords,
            reductionWords: 0,
            reductionPercent: 0,
            sections: limited.sections.length,
            sentences: sentences.length,
            chunks: chunks.length,
            references: references.length,
            truncated: limited.truncated,
            structuredRecords: contentRows.length,
            humanCorrectedRecords: contentRows.filter(row => row.humanCorrected).length,
            chunkTargetWords: payload.chunkTargetWords || DEFAULTS.chunkTargetWords,
            chunkOverlapPercent: DEFAULTS.chunkOverlapPercent,
            removedPageNumbers: 0,
            removedRepeatedHeadersFooters: 0,
            removedDuplicateParagraphs: 0,
            removedDuplicateBlankLines: 0
        }
    };
}

self.addEventListener("message", (event) => {
    const message = event.data || {};
    const { id, type, payload } = message;

    try {
        if (id === undefined || id === null) {
            throw new Error("Ingestion Worker request requires an id.");
        }

        let result;

        switch (type) {
            case "PROCESS_SOURCE":
                result = processSource(payload);
                break;

            case "PROCESS_STRUCTURED_SOURCE":
                result = processStructuredSource(payload);
                break;

            case "GET_INGESTION_INFO":
                result = {
                    ingestionVersion: INGESTION_VERSION,
                    defaults: { ...DEFAULTS }
                };
                break;

            default:
                throw new Error(`Unknown Ingestion Worker request: ${String(type)}`);
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
