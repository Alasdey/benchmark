
# SERE: Data Preparation Report

## Paper Under Analysis

**SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification**
Hao et al., 2026 (arXiv:2605.03701v1)

---

## Overview

SERE conducts experiments on three Event Causality Identification (ECI) datasets:

1. **EventStoryLine v0.9 (ESC)** (Caselli & Vossen, 2017)
2. **Causal-TimeBank (CTB)** (Mirza et al., 2014)
3. **MAVEN-ERE** (Wang et al., 2022)

The SERE paper states:

> *"Data preprocessing follows Gao et al. (2023)."* — Section 4.2

This report traces the full reference chain to document exactly what preprocessing is applied to each dataset.

---

## Reference Chain

```
SERE (Hao et al., 2026)
  └─ "follows Gao et al. (2023)"
       └─ Gao et al. (2023) - "Is ChatGPT a Good Causal Reasoner?"
            └─ "Following previous works (Gao et al., 2019; Choubey & Huang, 2017),
                only the top 20 topics of ESC are used for evaluation."
                 ├─ Gao et al. (2019) - "Modeling Document-level Causal Structures..." (NAACL)
                 └─ Choubey & Huang (2017) - "A Sequential Model..." (EMNLP)
                      [Note: This paper is about temporal relations on TimeBank,
                       not ESC. Cited for the convention of topic-level filtering.]
```

---

## Dataset 1: EventStoryLine v0.9 (ESC)

### 1.1 Original Dataset Description

| Property | Value |
|---|---|
| **Reference** | Caselli & Vossen (2017), *Events and Stories in the News Workshop* |
| **Source corpus** | ECB+ corpus (news articles about calamity events) |
| **Topics** | 22 topics (natural disasters, shootings, killings, accidents, trials) |
| **Documents** | 258 |
| **Event mentions** | 7,275 total (inherited from ECB+); 191 negated mentions |
| **Temporal expressions** | 1,297 |
| **Temporal links (TLINKs)** | 6,904 (event-to-DCT and event-to-TIMEX3 anchoring relations) |
| **Causal relations (PLOT_LINKs)** | 2,265 manually annotated; extended to **5,519** via within-document event coreference chains |
| **Explicit causal relations** | Only 117 out of the total PLOT_LINKs |
| **Relation types** | `rising_action` (1,147 manual / 2,653 extended) and `falling_action` (1,118 manual / 2,844 extended) |

### 1.2 What PLOT_LINKs Are

PLOT_LINKs are **not standard causal links**. They are broader "explanatory relations" defined as:

> *"A loose causal and temporal relation between a pair of event mentions, where one event mention explains/justifies the occurrence of the other event mention."* — Caselli & Vossen (2017)

They encompass:
- Standard causal relations (cause, enablement, prevention)
- Contingency, sub-event, entailment, and co-participation relations
- Both **explicit** (with linguistic markers) and **implicit** (no overt causal signal) relations
- Both **intra-sentence** and **cross-sentence** relations

PLOT_LINKs are **asymmetrical** and **non-transitive**.

### 1.3 Annotation Scope

- Events eligible for PLOT_LINKs are restricted to three classes:
  - `ACTION_OCCURRENCE`
  - `ACTION_PERCEPTION`
  - `ACTION_STATE`
- The following event types are **excluded** from PLOT_LINK annotation:
  - `ACTION_ASPECTUAL`
  - `ACTION_CAUSATIVE`
  - `ACTION_REPORTING`
  - `ACTION_GENERIC`
- Annotators freely selected event pairs (no predefined dense pairing scheme).
- Annotation was conducted by 2 experts using the CAT tool.

### 1.4 Original Train/Test Split

The ESC paper defines:
- **Development set:** 6 topics (T5, T7, T8, T32, T33, T35)
- **Test set:** 16 topics (T1, T3, T4, T12, T13, T14, T16, T18, T19, T20, T22, T23, T24, T30, T37, T41)

### 1.5 Data Preparation for ECI (from Gao et al., 2019)

Gao et al. (2019) established the standard preprocessing for ESC in the ECI task:

#### Topic Selection & Split

- 22 topics ordered by topic ID.
- **Last 2 topics** → reserved as **development set** (for parameter tuning).
- **Remaining 20 topics** → used for **5-fold cross-validation** (training and evaluation).
- Gao et al. (2023) confirms: *"only the top 20 topics of ESC are used for evaluation."*

#### Event Filtering

- **Gold event mentions** from ESC annotations are used (no automatic event detection).
- **Aspectual, causative, perception, and reporting event mentions are excluded** — 639 event mentions removed.
- Justification: *"most of which were not annotated with any causal relation according to Caselli and Vossen (2017)."*

#### Positive & Negative Pair Construction

**Intra-sentence pairs:**
- All event mention pairs within the same sentence are considered.
- Total: ~7,805 intra-sentence event mention pairs.
- Of these, **1,770 are annotated as causal** (PLOT_LINKs).
- POS:NEG ratio ≈ **1:3**.

**Cross-sentence pairs:**
- Event mentions from different sentences are paired (one mention from each sentence).
- Total: ~46,521 cross-sentence event mention pairs.
- Of these, **3,855 are annotated as causal**.
- POS:NEG ratio ≈ **1:10**.

**Task formulation:**
- Binary classification: causal vs. non-causal.
- **Direction is not distinguished** — if events A and B have any causal link (rising_action or falling_action), the pair is positive regardless of direction.

#### Class Imbalance Handling (in Gao et al., 2019)

- Gao et al. (2019) applied the `"balanced"` class weight option in logistic regression classifiers to handle class imbalance.
- Separate classifiers were trained for intra-sentence and cross-sentence cases.

### 1.6 What SERE Uses

**Main experiments (Table 2):** SERE reports a single ESC result (not split by intra/inter), following the Gao et al. (2023) preprocessing. This likely includes both intra- and cross-sentence pairs from the top 20 topics.

**CPATT settings (Table 3, Section 5.1):** SERE additionally reports ESC-intra and ESC-inter separately, using Zhang et al. (2023)'s alternative preprocessing:
- For long contexts, only sentences containing the target events are retained.
- Negative samples are constructed by randomly pairing non-causal events with their corresponding sentences.

---

## Dataset 2: Causal-TimeBank (CTB)

### 2.1 Original Dataset Description

| Property | Value |
|---|---|
| **Reference** | Mirza et al. (2014), *EACL 2014 Workshop on Computational Approaches to Causality in Language* |
| **Source corpus** | TempEval-3 TBAQ-cleaned corpus (TimeBank + AQUAINT) |
| **Total text** | ~100K words |
| **Documents** | 183 |
| **Event mentions** | 6,811 (gold TimeML annotations) |
| **Temporal links (TLINKs)** | ~5,118 between event pairs |
| **Causal links (CLINKs)** | 318 total; **296 intra-sentence**, ~18-22 cross-sentence |
| **Average CLINKs per document** | ~1.4 |

### 2.2 Annotation Scheme

CTB annotates causality using the **CLINK** tag, inspired by TimeML:

- **CLINK** is a directional one-to-one relation: the **causing event** is the source (S) and the **caused event** is the target (T).
- **C-SIGNAL** tags mark textual indicators of causal relations (e.g., *because of*, *due to*, *as a result of*).

#### Restriction to Explicit Causality

> *"As causal relations are often not overtly expressed in text, we restrict the annotation of CLINKs to the presence of an **explicit causal construction linking two events in the same sentence**."* — Mirza et al. (2014)

Annotated constructions include:
- **Basic constructions** with CAUSE/ENABLE/PREVENT verbs (e.g., *caused*, *enabled*, *prevented*)
- **Affect verbs** (e.g., *affect*, *influence*, *determine*, *change*)
- **Link verbs** (e.g., *link*, *lead*, *depend on*)
- **Periphrastic causatives** (e.g., *caused the boat to heel*)
- **Causal signals** — conjunctions (*because*, *since*, *so that*), prepositions (*because of*, *due to*, *by*, *from*), adverbial connectors (*as a result*, *therefore*, *thus*)

#### What Is NOT Annotated

- **Implicit causality** is excluded (e.g., lexical causatives like *kill* = cause to die)
- **Cross-sentence causal relations** are not systematically annotated (only ~18-22 exist incidentally)
- The conjunction *and* implying causation is not annotated due to ambiguity
- Temporal conjunctions *after* and *when* are not treated as causal signals

### 2.3 Data Preparation for ECI

**Standard practice** (confirmed by the MAVEN-ERE paper):

> *"Due to the small data scale of Causal-TB and EventStoryLine, previous works (Gao et al., 2019; Cao et al., 2021) typically adopt **5-fold cross-validation** on them and only do **causality identification**, which ignores the directions of causal relations."* — Wang et al. (2022)

#### Evaluation Scope: Intra-Sentence Only

- The original annotation restricts CLINKs to explicit constructions within the same sentence.
- Only ~18-22 cross-sentence CLINKs exist — negligibly few.
- Therefore, CTB evaluation is **effectively intra-sentence only**.

#### Pair Construction

- **Positive pairs:** Event pairs with annotated CLINKs (direction ignored for binary ECI).
- **Negative pairs:** All non-annotated intra-sentence event pairs.
- **Class imbalance:** With 318 causal pairs out of thousands of total intra-sentence event pairs, the dataset is **highly imbalanced** (approximately 1:23 POS:NEG ratio).

#### Split

- **5-fold cross-validation** (standard); some works use 10-fold.
- No separate held-out test set.

### 2.4 What Gao et al. (2023) Says

Gao et al. (2023) provides no specific preprocessing description for CTB beyond using it as-is and noting:
- 184 documents, 6,813 events, 318 causal event pairs
- Evaluation under zero-shot settings with ChatGPT

### 2.5 What SERE Uses

SERE follows Gao et al. (2023), which follows standard practice: 5-fold CV on intra-sentence event pairs, binary causality identification (direction-agnostic).

---

## Dataset 3: MAVEN-ERE

### 3.1 Original Dataset Description

| Property | Value |
|---|---|
| **Reference** | Wang et al. (2022), *EMNLP 2022* |
| **Source corpus** | MAVEN event detection dataset → English Wikipedia articles |
| **Topics** | 90 topics |
| **Documents** | 4,480 |
| **Event mentions** | 112,276 |
| **Event coreference chains** | 103,193 |
| **Temporal relations** | 1,216,217 |
| **Causal relations** | 57,992 (10,617 CAUSE + 47,375 PRECONDITION) |
| **Subevent relations** | 15,841 |
| **Event type coverage** | 168 fine-grained types |

### 3.2 Causal Relation Annotation Details

**Relation types:**
- **CAUSE:** *"the tail event is inevitable given the head event"*
- **PRECONDITION:** *"the tail event would not have happened if the head event had not happened"*
- Negative events (events that did not actually happen) are also covered.

**Annotation scope:**
- Causal relations are annotated **only for event pairs with BEFORE and OVERLAP temporal relations** (leveraging the temporal annotation done in a prior stage).
- Annotators consider **transitivity** to make minimal annotations; discarded relations are automatically completed afterward.
- Both intra- and cross-sentence relations are annotated.

**Distance distribution of causal relations:**

| Distance | MAVEN-ERE | CTB | ESC |
|---|---|---|---|
| < 50 words | 48.6% | 100% | 59.3% |
| 50–200 words | 37.6% | 0% | 32.2% |
| > 200 words | 13.7% | 0% | 8.4% |
| Average distance | 92 words | 11 words | 76 words |

**Annotation quality:**
- 58 trained and qualified annotators.
- Each document annotated by 3 independent annotators.
- Final results obtained via majority voting.
- Inter-annotator agreement: 69.5% (Cohen's kappa) for causal relations.

### 3.3 Data Split

From MAVEN-ERE Appendix D (Table 15):

| Split | #Documents | #Event Mentions | #Causal Links |
|---|---|---|---|
| **Train** | 2,913 | 73,939 | 36,316 |
| **Dev** | 710 | 17,780 | 9,698 |
| **Test** | 857 | 20,557 | 11,978 |

Splits follow the original MAVEN dataset split (Wang et al., 2020b).

### 3.4 Test Set Availability

> *"Since MAVEN-ERE did not release the test set, we evaluate ChatGPT on its development set."* — Gao et al. (2023)

### 3.5 What SERE Uses

SERE follows Gao et al. (2023), which uses the **development set** for evaluation since the test set is not publicly released. No additional preprocessing is described for MAVEN-ERE in any paper in the chain.

---

## Additional Experimental Settings in SERE

### CPATT Settings (Section 5.1)

For the fine-tuning comparison experiments, SERE uses **different preprocessing** from the main experiments, following Zhang et al. (2023):

- For long contexts, **only sentences containing the target events are retained**.
- **Negative samples** are constructed by **randomly pairing non-causal events** with their corresponding sentences.
- ESC is split into **ESC-intra** (both events in the same sentence) and **ESC-inter** (events in different sentences).
- CTB does not distinguish between intra and inter cases in this setting.
- SERE directly performs **inference on the test split** (no fine-tuning for their method).

The authors note:

> *"CPATT's preprocessing is primarily designed for training, where random negative sampling helps reduce causal hallucination; therefore, these results are not included in the main experiments."*

### Fine-Tuning Experiments (Appendix B)

- Qwen2.5-3B-Inst is fine-tuned using supervised instruction-based LoRA via the LlamaFactory library.
- Training instances use the **same input format as the API-based setup**.
- The underlying data splits follow the CPATT setting.

---

## Summary Comparison Table

| Aspect | ESC | CTB | MAVEN-ERE |
|---|---|---|---|
| **Source genre** | News (ECB+ calamity events) | News (TimeBank + AQUAINT) | Wikipedia (general domain, 90 topics) |
| **Causal relation type** | PLOT_LINKs (broad explanatory, implicit + explicit) | CLINKs (explicit causal constructions only) | CAUSE + PRECONDITION (over temporal backbone) |
| **Annotation scope** | Freely selected event pairs, intra + cross-sentence | **Same-sentence only**, explicit constructions only | Event pairs with BEFORE/OVERLAP temporal relations, intra + cross-sentence |
| **Total causal links** | 5,519 (extended via coref); 1,770 intra-sentence | 318 (296 intra-sentence) | 57,992 |
| **Explicit vs implicit** | Both (only 117/5,519 explicit) | **Explicit only** | Both |
| **Train/eval split** | Top 20 topics; 5-fold CV | 5-fold CV | MAVEN split; dev set used (test unreleased) |
| **Event filtering** | Gold mentions; exclude aspectual/causative/perception/reporting (639 removed) | Gold TimeML events as-is | Gold MAVEN triggers |
| **Negative pairs** | All non-causal event pairs in scope | All non-causal intra-sentence event pairs | Event pairs with temporal relations but no causal annotation |
| **POS:NEG ratio** | ~1:3 (intra), ~1:10 (cross) | ~1:23 | Not explicitly stated |
| **Task formulation** | Binary (direction-agnostic) | Binary (direction-agnostic) | Binary (direction-agnostic per Gao et al. 2023 convention) |
| **SERE preprocessing source** | Gao et al. (2023) → Gao et al. (2019) | Gao et al. (2023) → standard practice | Gao et al. (2023) → dev set |

---

## External Resources Used in SERE Pipeline

These are not datasets for evaluation but are used as part of the SERE retrieval framework:

| Resource | Role in SERE |
|---|---|
| **ConceptNet** (Speer et al., 2018) | Localized on Neo4j; nodes encoded with Contriever-msmarco; shortest paths queried via Cypher for the Conceptual Path Metric |
| **spaCy** (Honnibal & Montani, 2017) | Dependency syntax tree construction for the Syntactic Metric |
| **Contriever-msmarco** (Izacard et al., 2021) | Encoder for matching event mentions to ConceptNet nodes |

---

## Key Observations

1. **No paper in the chain fully specifies all preprocessing details in one place.** The full picture requires tracing through 4+ papers.

2. **ESC's "causal relations" are actually PLOT_LINKs**, which are broader than standard causality — they include contingency, sub-event, entailment, and co-participation relations. Most are implicit (only 117/5,519 are explicit).

3. **CTB is restricted to explicit, intra-sentence causality only**, making it fundamentally different from ESC and MAVEN-ERE in scope and difficulty. Its high class imbalance (~1:23) partly explains why all methods achieve very low F1 on this dataset.

4. **MAVEN-ERE's test set is not publicly available**, so evaluation uses the development set. This is an important caveat when comparing results across papers.

5. **The "Gao et al. (2023) preprocessing" is essentially a pass-through** — it delegates to Gao et al. (2019) for ESC and uses standard conventions for CTB and MAVEN-ERE, without introducing novel preprocessing steps of its own.
