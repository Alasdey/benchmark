EVENTSTORYLINE — DATASET NOTES
================================
Generated: 2026-05-24
Source: cltl/EventStoryLine (GitHub)
Format used: v0.9 XML, annotated_data/v0.9/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WHAT IT IS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EventStoryLine (ESL) is a corpus of news articles grouped into
"topics" (event clusters), annotated for event mentions and
inter-event causal and temporal relations.

The corpus exists in two versions:
  v0.9  — the version benchmarked by ALL published ECI papers
          (258 documents, 22 topic folders)
  v1.5  — extended annotation (ECB+), different schema,
          not used by ECI literature

Each document is an XML file with three annotation layers:

  Markables   event mentions of type ACTION_* (triggers) plus
              entity mentions (LOC_*, TIME_*, HUMAN_*, NON_HUMAN_*)
              — only ACTION_* are used for ECI

  Relations   PLOT_LINK   causal relations between mentions
                          relType ∈ {PRECONDITION, FALLING_ACTION}
              TLINK       temporal relations (not used for ECI)
              COREF       coreference chains within a topic

  Tokens      flat list of <token t_id sentence="N"> elements;
              the `sentence` attribute gives the sentence index


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CORPUS STATISTICS (v0.9, measured from files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total documents:              258
  Topic folders:                22  (1,3,4,5,7,8,12,13,14,16,
                                     18,19,20,22,23,24,30,32,33,35,37,41)
  Train documents:              233  (topics ≠ {37, 41})
  Dev documents:                 25  (topics 37 and 41)

  ACTION_* mentions (event triggers):  ~4,316
  PLOT_LINK annotations total:         ~5,519
    of which intra-sentence:           ~1,770  (~32%)
    of which inter-sentence:           ~3,749  (~68%)

  CLINK : mention ratio:         ~1.3 per mention  (much denser than CTB)

  Topic 37 (dev):  13 documents
  Topic 41 (dev):  12 documents


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. ANNOTATION SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Unlike Causal-TimeBank, ESL annotation is NOT restricted to
intra-sentence pairs. Caselli & Vossen (2017) annotated causal
relations across sentence boundaries wherever the relation was
clear and direct.

Key structural properties:

  Cross-sentence majority: ~68% of PLOT_LINKs span multiple sentences.
    This is the dominant case, unlike Causal-TimeBank where 90% are
    intra-sentence.

  Both relTypes map to the same causal direction:
    PRECONDITION   A is a precondition for B  →  A causes B
    FALLING_ACTION A leads to B as consequence →  A causes B
    Both are stored as causes=[src, tgt] / caused_by=[tgt, src].

  Coreference cycles: v0.9 contains directed COREF chains that
    form cycles (A→B→C→A). Propagating relations through coreference
    is therefore unsafe without cycle detection. No published paper
    does coreference propagation on ESL v0.9.

  No closed-world assumption: unannotated pairs are not guaranteed
    non-causal. They were simply not selected for annotation.

  Annotation is topic-local: relations connect mentions within the
    same topic (document cluster), not across topics.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. DATAPREP SCRIPT — EventStoryLine_dataprep_master.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF repo default:   Nofing/EventStoryLine-master
Data path default: data/EventStoryLine/annotated_data/v0.9

── Output schema (one row per document) ──

  id          str          doc_name attribute or filename stem
  doc_idx     int          0-based index, sorted by (topic_id, filename)
  topic_id    int          numeric topic folder (1, 3, … 41)
  split       str          "train" | "dev"
  tokens      list[str]    flat token list (doc-level)
  mentions    list[str]    m_id strings, sorted by first token position
  spans       list[list[int]]  token indices per mention (0-based, doc-level)
  relations   dict         {"causes":    [[src_idx, tgt_idx], …],
                             "caused_by": [[tgt_idx, src_idx], …]}
  sentences   list[list[int]]  [[start, end), …] token boundary slices

── Key decisions ──

  Version            v0.9 (the only version benchmarked by ECI papers)
  Mention types      ACTION_* only (drops LOC/TIME/HUMAN/NON_HUMAN)
  Relations kept     PLOT_LINK where relType ∈ {PRECONDITION, FALLING_ACTION}
  Sentence filter    None — all annotated pairs regardless of distance
  Coref propagation  No — v0.9 has directed coreference cycles
  NoRel stored       No — only positive annotations stored
  pair_list stored   No — consumers enumerate pairs at training time
  Dev split          topics {37, 41} → split="dev"
                     Source: Caselli & Vossen (2017); confirmed in HOTECI
  Fold assignment    doc_idx % 5  (train docs only)
  doc_idx ordering   sorted by (topic_id, filename) for stability

── Fold usage ──

  train = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 != k)
  test  = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 == k)
  dev   = ds.filter(lambda x: x["split"] == "dev")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. DIFFERENCES WITH SOTA EVALUATION SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset version          v0.9 ✓  (same as all published papers)
  Dev topics {37, 41}      ✓  (same as Caselli 2017, HOTECI, DiffusECI)
  5-fold CV                doc_idx % 5 ✓

  Relation scope:          This script keeps ALL annotated pairs (intra +
                           inter-sentence). Most SOTA papers also evaluate
                           all pairs. A minority restricts to intra-sentence
                           (e.g. some CATENA variants) — results not
                           directly comparable if filtered.

  Negative pairs:          NOT stored. SOTA trains classifiers over all
                           enumerated event pairs, including negatives
                           labeled "None" / "NoRel". Consumers of this
                           dataset must enumerate pairs from sentences +
                           spans and generate negatives at training time.

  pair_list:               NOT stored. SOTA evaluation iterates over
                           all ordered event pairs in a document or a
                           sentence window as classification candidates.

  Context window:          NOT built. SOTA models (DiffusECI, CLECI, iLIF,
                           HOTECI) construct a 5-sentence context window
                           per event pair at inference time. This must be
                           built by the consumer from the sentences field.

  Label schema:            This script stores causes / caused_by (positives
                           only). SOTA models output a 3-way label:
                           cause / effect / None. The None class must be
                           derived at training time from unannotated pairs.

  Coreference propagation: Not applied here (cycles in v0.9).
                           Some older papers (pre-2022) propagate through
                           coref; most recent work does not.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. HOW SOTA LITERATURE USES THIS CORPUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Standard classification setup ──

  All papers treat ECI as binary or ternary pair classification:
    Input:  (event_a, event_b, context)
    Output: cause / caused_by / None

  Candidates: all ordered pairs (i, j) of ACTION mentions in a document
    (or within a sentence window). Positive rate ~5–10% depending on
    whether all pairs or only intra-sentence pairs are evaluated.

── Context encoding ──

  Typical input: the sentence(s) containing each event, plus surrounding
    context (usually a ±2 sentence window). Events are marked with special
    tokens (e.g., <e1>trigger</e1>).

── Evaluation ──

  Metric: F1 on the positive class (cause/caused_by) only.
    NoRel pairs are excluded from the F1 numerator.
    Binary F1: treat cause + caused_by as a single positive class.
    Macro F1 over cause + caused_by: also common in recent papers.

  5-fold CV on train topics; report mean ± std.
  Dev topics {37, 41} used as held-out validation during training.

── Key papers ──

  Caselli & Vossen (ESC, ACL 2017)       https://aclanthology.org/W17-2711
    Defines the standard benchmark split (dev topics 37, 41).

  Man et al. (DiffusECI, AAAI 2024)
    Diffusion-based event representation; SOTA at publication.
    Uses full ESL v0.9 with 5-fold CV and dev topics 37/41.

  Liao et al. (CLECI, Information 2026)
    Contrastive learning for ECI; reports on ESL + CTB + MAVEN-ERE.

  HOTECI (unreferenced preprint, confirmed impl)
    Source of confirmed dev-topic logic:
      if '37/' in doc_id or '41/' in doc_id → dev


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. PRACTICAL IMPLICATIONS FOR THIS PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • The stored dataset is a positive-only annotation list. At training
    time, enumerate all ordered mention pairs per document, look up
    which are in causes / caused_by, and label the rest as NoRel.

  • Use sentences + spans to build context windows for model input.
    Standard: take the sentence containing each event and ±2 neighbours.

  • Do NOT filter to intra-sentence pairs unless replicating a specific
    paper that does so — most SOTA results include cross-sentence pairs.

  • Do NOT propagate coreference — v0.9 has cycles.

  • Report F1 on the causal class (positive F1), as all papers do,
    to stay comparable. Binary F1 (causes + caused_by as one class)
    is the most common metric in recent work.

  • The 5-fold train/test split uses doc_idx % 5. Keep topics 37 and 41
    as a fixed dev set across all folds.
