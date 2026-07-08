EVENTSTORYLINE — DATASET NOTES
================================
Generated: 2026-05-25
Source: cltl/EventStoryLine (GitHub)
Format used: v0.9 XML + pre-computed evaluation_format/


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

The repository also ships a pre-computed evaluation format
(evaluation_format/full_corpus/v0.9/) produced by create_gold_document.py,
which merges ECB+ coreference chains with ESC PLOT_LINKs to give the
ground-truth pair set used by all published ECI papers.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CORPUS STATISTICS (v0.9, measured from files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Total documents:              258
  Topic folders:                22  (1,3,4,5,7,8,12,13,14,16,
                                     18,19,20,22,23,24,30,32,33,35,37,41)
  Train documents:              233  (topics ≠ {37, 41})
  Dev documents:                 25  (topics 37 and 41)

  ACTION_* mentions (event triggers):  ~5,334
  Sentences:                           ~4,316

  Raw XML PLOT_LINKs (before coref propagation):   2,266
    of which intra-sentence:                        1,770  (~78%)
    of which inter-sentence:                          496  (~22%)

  Evaluation format pairs (after coref propagation): 5,625
    of which intra-sentence:                          1,770  (~31%)
    of which inter-sentence:                          3,855  (~69%)
    — 5 documents have no tab file; those use raw XML fallback

  The coreference expansion (2,266 → 5,625) adds ~3,359 cross-sentence
  pairs by taking the Cartesian product of coreferent mention clusters
  for each PLOT_LINK.

  Topic 37 (dev):  13 documents
  Topic 41 (dev):  12 documents


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. ANNOTATION SCOPE AND COREFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Unlike Causal-TimeBank, ESL annotation is NOT restricted to
intra-sentence pairs. Caselli & Vossen (2017) annotated causal
relations across sentence boundaries wherever the relation was
clear and direct.

Key structural properties:

  Two-stage gold standard:
    Stage 1 — raw PLOT_LINKs in the XML (2,266 pairs). These are the
              direct human annotations.
    Stage 2 — evaluation format tab files (5,625 pairs). Produced by
              create_gold_document.py: for each PLOT_LINK, the script
              looks up each endpoint in the ECB+ coreference dictionary
              and emits the Cartesian product of coreferential mentions.
              This is the ground truth used by all published ECI papers.

  Both relTypes map to the same causal direction:
    PRECONDITION   A is a precondition for B  →  A causes B
    FALLING_ACTION A leads to B as consequence →  A causes B
    In the evaluation format, pairs are reordered so src token < tgt
    token and relType is swapped accordingly (both still mean causes).

  No closed-world assumption: unannotated pairs are not guaranteed
    non-causal. They were simply not selected for annotation.

  Conflict between coreference and causality (Luo et al. ACL 2024):
    "the expansion process introduces false causal relations … some
    co-referenced event pairs are incorrectly labeled as causality."
    After removing conflicts + DAG enforcement: 5,478 clean pairs.
    This cleaning step is NOT applied in our dataprep (we use the
    raw evaluation format at 5,625 pairs, matching DiffusECI/CLECI).

  Annotation is topic-local: relations connect mentions within the
    same topic (document cluster), not across topics.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. DATAPREP SCRIPT — EventStoryLine_dataprep_master.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HF repo default:   Nofing/EventStoryLine-master
Data path default: data/EventStoryLine/annotated_data/v0.9
Eval format:       data/EventStoryLine/evaluation_format/full_corpus/
                   v0.9/event_mentions_extended/  (auto-derived)

── Output schema (one row per document) ──

  id          str          doc_name attribute or filename stem
  doc_idx     int          contiguous 0-based index (no gaps from skipped
                           files), sorted by (topic_id, filename)
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
  Relations source   evaluation_format tab files (coreference-propagated,
                     5,602 causes in output — ~23 fewer than 5,625 tab
                     lines due to mentions filtered by ACTION_TAGS)
                     Fallback to raw XML PLOT_LINKs for 5 docs with no tab
  Sentence filter    None — all pairs regardless of distance
  Coref propagation  Yes — via pre-computed evaluation format
  NoRel stored       No — only positive annotations stored
  pair_list stored   No — consumers enumerate pairs at training time
  Dev split          topics {37, 41} → split="dev"
                     Source: Caselli & Vossen (2017); confirmed in HOTECI
  Fold assignment    doc_idx % 5  (train docs only)
  doc_idx            contiguous — incremented only for successfully parsed docs

── Split logic ──

  Train vs dev: topic_id ∈ {37, 41} → "dev", else → "train"
  5-fold CV within train (consumer-side, not stored):
    train = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 != k)
    test  = ds.filter(lambda x: x["split"] == "train" and x["doc_idx"] % 5 == k)
    dev   = ds.filter(lambda x: x["split"] == "dev")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. ALIGNMENT WITH SOTA EVALUATION SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset version          v0.9 ✓
  Dev topics {37, 41}      ✓  (Caselli 2017, HOTECI, DiffusECI, CLECI)
  5-fold CV                doc_idx % 5 ✓
  Coreference propagation  ✓  evaluation format used (same as all SOTA)
  Pair counts              ~5,602 causes ✓  (DiffusECI: 5,625; Luo: 5,478)
  Relation scope           intra + inter-sentence ✓

  Remaining gaps vs SOTA:

  pair_list not stored     SOTA trains over all enumerated ordered pairs
                           (i, j) labeled cause / caused_by / None.
                           Consumers must enumerate pairs from sentences +
                           spans and generate NoRel at training time.

  Context window           NOT built in dataset. DiffusECI / CLECI build
                           a ±2-sentence window per event pair at training
                           time from the raw token/sentence fields.

  Label schema             causes / caused_by (positives only) stored.
                           SOTA outputs a 3-way label: cause / effect / None.
                           None class derived at training time.

  Conflict cleaning        Luo et al. (ACL 2024) remove 147 conflicting pairs
                           (coref vs causality) and enforce DAG structure
                           (5,625 → 5,478). Not applied here.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. HOW SOTA LITERATURE USES THIS CORPUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Standard classification setup ──

  All papers treat ECI as binary or ternary pair classification:
    Input:  (event_a, event_b, context)
    Output: cause / caused_by / None

  Candidates: all ordered pairs (i, j) of ACTION mentions in a document.
    Positive rate ~5–10% depending on intra vs full-document scope.

── Context encoding (DiffusECI standard) ──

  For each event mention eᵢ: take a window of 5 sentences — the hosting
  sentence plus ±2 neighbours. Concatenate the two windows (C₁, C₂) into
  a single context C, preserving document order, without repeating
  sentences that appear in both windows.

── Evaluation ──

  Metric: F1 on the positive class (cause/caused_by) only.
    NoRel pairs are excluded from the F1 numerator.
    Binary F1: treat cause + caused_by as one positive class (most common).
    Directional macro F1 over cause + caused_by: used in Luo et al. 2024.

  5-fold CV on train topics; report mean ± std.
  Dev topics {37, 41} used as held-out validation during training.

── Confirmed paper setups ──

  Caselli & Vossen 2017   Created evaluation format (5,625 propagated pairs)
  DiffusECI AAAI 2024     5,625 pairs (1,770 intra + 3,855 inter); ±2 sent
                          window; 5-fold; binary F1
  CLECI EMNLP 2024        5,625 pairs; 5-fold; binary F1
  Luo et al. ACL 2024     5,478 pairs (conflict-cleaned); 5-fold;
                          directional 3-way F1
  HOTECI AAAI 2023        1,770 pairs (intra only); 5-fold; binary F1
  SemDI EMNLP 2024        1,770 pairs (intra only); drops aspectual/
                          causative/perception/reporting mentions; 5-fold


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. PRACTICAL IMPLICATIONS FOR THIS PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • The stored dataset has ~5,602 positive pairs (causes + caused_by).
    At training time, enumerate all ordered mention pairs per document,
    look up which are in causes / caused_by, and label the rest as NoRel.

  • Use sentences + spans to build context windows for model input.
    Standard (DiffusECI): hosting sentence of each event ± 2 neighbours,
    concatenated, no sentence repeated.

  • Do NOT filter to intra-sentence pairs unless replicating HOTECI or
    SemDI — most SOTA results include cross-sentence pairs.

  • Report F1 on the causal class (positive F1), as all papers do,
    to stay comparable. Binary F1 (causes + caused_by as one class)
    is the most common metric in recent work.

  • The 5-fold train/test split uses doc_idx % 5 on train documents only.
    Topics 37 and 41 are always dev, never in any fold.
