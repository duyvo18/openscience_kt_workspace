# Consolidated Peer Review & Revision Guide
## *Distributional Pedagogical Alignment: An Evidence-Centered Framework for Knowledge Tracing*

> **Venue:** ACOMPA 2026 — International Conference on Advanced Computing and Analytics
>
> **Paper type:** Short paper — Position / Vision (no experiments; equations and results deferred to a future implementation paper)
>
> **Document status:** This is a *consolidated* review synthesizing two independent reviews (an ACOMPA reviewer report and a CMT review-form report). It is written to be self-contained and to drive the revision directly. Every consolidated concern is cross-referenced to its source(s) and paired with a concrete revision action.

---

## 0. How to use this document

- **Section 3** is the *verdict* (both reviews' scores reconciled).
- **Section 5** is the **required-revision list**, ordered by priority. Each item has: the concern, why it matters, and a concrete *Action* the team can execute. Treat the six MAJOR items as blocking for camera-ready.
- **Section 6** is a **checklist** you can copy into an issue tracker.
- **Section 7** collects the authors' rebuttal questions in one place.
- Where the two reviews used different labels for the same issue, we merged them; the source tags `[R1]` (ACOMPA report) and `[R2]` (CMT form) show provenance.

---

## 1. Summary of the paper (agreed by both reviews)

The paper proposes a **conceptual framework** for Knowledge Tracing (KT) built on an *evidence-centered* view: a learner interaction does not directly update the knowledge state; it contributes to an update only after being **aggregated and interpreted through a pedagogical lens**. This interpretation is elevated to an **explicit intermediate stage** rather than being entangled in the mastery update.

The framework is organized as four representation spaces — **Interaction → Distribution → Knowledge → Prediction** — realized by four sequential modules and mapped onto the **student / evidence / task** models of Evidence-Centered Design (ECD, Mislevy):

- **Module 1 — Interaction Representation Learning:** two parallel branches (interaction context: question semantics, response, difficulty; knowledge context: localized mastery read from memory `M_{t-1}` via the knowledge graph + concept difficulty), fused into one interaction representation.
- **Module 2 — Distributional Pedagogical Alignment (DPA)** *(the named core contribution)*: (i) **distribution projection** via a statistical-descriptor decoder (mean/variance); (ii) **pattern construction** with fixed operators — temporal pooling, same-KC pooling, G-aware pooling; (iii) **pedagogical alignment** — soft, differentiable "nudges" toward principles (guessing/slipping; monotonicity); (iv) **pattern readout** to a structured object `z′` preserving the statistical descriptor.
- **Module 3 — Mastery State Tracking:** uncertainty-weighted, residual update of an explicit per-KC memory `M ∈ R^{C×d}`, plus a learned pattern-to-KC gating.
- **Module 4 — Prediction Aggregation:** next-step probability from the updated state and the target question, plus a KC-to-prediction contribution.

**Planned evaluation:** mathematics-domain datasets partitioned by native graph availability (graph-native: Junyi, XES3G5M, Eedi/NIPS34; flat-tag: ASSISTments, Algebra/Bridge-to-Algebra with a CMDKT-constructed graph); a baseline set spanning BKT and all three themes (DKT, DKVMN, AKT, GKT, CMDKT, UKT, KeenKT, S2KT, PSI-KT, NSKT, PLKT, LPKT, MemoryKT); metrics AUC/ACC/RMSE plus ECE and NLL; three ablations (point-vector vs distribution; remove alignment; enable operators one at a time); four hypotheses H1–H4.

---

## 2. Points of agreement between the two reviews

Both reviewers **converge** on the following, which raises confidence that they are real:

- **Accept as a position paper, conditional on revision.** Both land at *Weak Accept* (borderline), above threshold for a vision track, blocked mainly by under-specification of the core mechanism.
- **The named core contribution (DPA / pedagogical alignment) is the least-specified part of the paper.** This is the single most important issue in both reports.
- **The novelty delta versus existing memory / attention / distributional / gate-based KT is not yet sharply drawn.**
- **The monotonicity principle is a strong, contestable claim** that collides with the forgetting-curve literature and must be reframed as a testable hypothesis.
- **"Structural consistency across learners" is asserted, not established.**
- **The theory grounding (ECD), the honest positioning, the interpretability caution, and the evaluation plan are genuine strengths.**

No substantive **disagreement** exists between the two reviews. They differ only in emphasis: R1 stresses the evaluation-plan rigor (graph confound, statistical testing, reproduction burden) and scalability; R2 stresses the formal novelty delta versus specific architectures (DKVMN/AKT/LPKT) and the classical guess/slip formalism (BKT/IRT). Both sets are incorporated below.

---

## 3. Reconciled recommendation

| | R1 (ACOMPA report) | R2 (CMT form) | Consolidated |
|---|---|---|---|
| **Recommendation** | Weak Accept (accept w/ mandatory revisions) | 4/6 — Weak Accept (borderline) | **Weak Accept — conditional on the MAJOR revisions in §5** |
| **Reviewer confidence** | High | 4/5 | High |

**Sub-scores (from R2's rubric, endorsed by R1):**

| Criterion | Score (1–5) | Consolidated comment |
|---|---|---|
| Originality / Novelty | **3** | Elegant ECD framing; core-mechanism novelty under-differentiated from prior work. |
| Technical Soundness | **3** | No formalization yet; several load-bearing claims asserted rather than argued. |
| Significance / Relevance | **4** | Clear relevance to KT and to the conference's analytics/ML scope. |
| Clarity / Presentation | **5** | Very well written and organized; figure and structure aid comprehension. |
| Related Work / Positioning | **4** | Careful three-theme positioning with direct contrast to closest prior work. |
| Evaluation Plan (vision paper) | **4** | Thorough, falsifiable, tied to the framework; some hypotheses under-argued; graph confound unaddressed. |

**Bottom line for the team:** The perspective is worth publishing, but the paper currently *names* a contribution it does not yet *specify*. The revision must make the alignment mechanism concrete enough to assess in principle, and must sharpen why the framework differs from — and improves on — the models it is built from. If the venue turns out to expect at least preliminary results, add a minimal proof-of-concept on one graph-native dataset (this also directly answers the novelty and mechanism concerns).

---

## 4. Strengths (preserve these in revision — do not dilute)

1. **Theory grounding.** Mapping ECD's student/evidence/task models onto the Knowledge/Distribution/Interaction spaces grounds an architecture in educational-measurement theory — uncommon and valuable in an engineering-driven KT literature. `[R1, R2]`
2. **Honest, integrative positioning.** The three-theme taxonomy is used to show how DPA *integrates* prior work, with direct contrast to the closest prior work (PLKT) and to LPKT, and an explicit statement of what each ingredient borrows. `[R1, R2]`
3. **Mature interpretability stance.** Citing both *Attention is not Explanation* and *Attention is not not Explanation* and downgrading attribution to *behavioral consistency* rather than a claim about internals is appropriately cautious. `[R1, R2]`
4. **Credible evaluation plan.** Datasets partitioned by native graph availability; baselines across all three themes plus BKT; calibration-aware metrics (ECE, NLL); ablations mapped one-to-one onto the precondition chain; four falsifiable hypotheses. `[R1, R2]`
5. **Clarity and structure.** The four-module/four-space organization and the precondition-chain spine (distribution → shared coordinate → alignment → per-KC attribution) are easy to follow; Figure 1 supports the exposition. `[R1, R2]`

---

## 5. Required and suggested revisions

### MAJOR — treat as blocking

**J1. Specify the alignment mechanism concretely (at least schematically).** `[R1-M1, R2-M1/Q1]`
*Concern.* "Nudges the patterns toward those principles as a soft, differentiable adjustment" never says what the object is — a regularization loss, a projection onto a constraint set, an auxiliary objective, or a constraint layer. Because alignment **is** the named contribution, this is the paper's central gap.
*Why it matters.* Without it, a reader cannot tell whether DPA is a substantive mechanism or a re-labeling of an existing encoder→pool→memory→readout pipeline; the entire contribution rests here.
*Action.*
- Give one concrete instantiation (equation or precise schematic) for **at least one** principle: what quantity is computed, where it enters the graph, and through what it is differentiable.
- State how alignment is **prevented from conflicting with the prediction objective** (e.g., weighting, stop-gradient, curriculum, or a stated trade-off).
- For **guess/slip**: define differentiably how an "outlier response" is detected and how the distribution's expectation/variance is adjusted to "absorb" it. Then relate it to classical formalisms — **what does the distributional treatment add over BKT's explicit guess/slip probabilities or IRT 3PL/4PL asymptotes?** `[R1-M1, R2-M4/Q4]`
- For **monotonicity**: show how the two hand-authored rules become soft, differentiable constraints without collapsing into brittle thresholds/hyperparameters.

**J2. Sharpen the novelty of the "explicit evidence-interpretation stage."** `[R1-M4, R2-M2]`
*Concern.* Memory- (DKVMN), attention- (AKT), and gate-based (LPKT) models already carry intermediate representations readable as evidence interpretation; distributional models (UKT/KeenKT/S2KT) already do uncertainty-weighted updates and dispersion preservation. Novelty is "their interpretation as an explicit alignment stage," which by itself is thin.
*Action.*
- Provide a **side-by-side** showing where a representative existing model *entangles* evidence interpretation with the mastery update, versus where DPA *separates* it — and, crucially, **what that separation changes in behavior or prediction**.
- Sharpen the **PLKT** distinction specifically: PLKT's temporal sliding windows are themselves a pooling operator, so "we use pooling operators" is not the differentiator. State the difference (organizing by pedagogical operators + aligning against principles) as a case where DPA and PLKT would make **different predictions**. `[R1-M4]`
- Clarify **what placing the distribution *before* pattern pooling and alignment** buys over UKT/KeenKT/S2KT, which already assume dispersion/uncertainty. `[R2-M6]`

**J3. Reframe the monotonicity principle as a testable hypothesis, and reconcile with forgetting.** `[R1-M2, R2-M3/Q3]`
*Concern.* Calling time-gap-as-forgetting a "dataset-specific bias" runs against a substantial forgetting-curve / spacing-effect literature and against LPKT's explicit forgetting gate (a listed baseline). A blanket monotonicity prior risks **suppressing genuine forgetting signal** — trading one bias for another.
*Action.*
- Present monotonicity as a **hypothesis to be validated**, not a design axiom.
- Add an **ablation that separates "debiasing" from "signal suppression"** (e.g., compare with/without the monotonicity prior on datasets with strong vs weak spacing structure).
- Explicitly **reconcile with LPKT's forgetting mechanism** in the text rather than only listing LPKT as a baseline.

**J4. Establish (or downgrade) "structural consistency across learners."** `[R1-M3, R2-M5/Q2]`
*Concern.* A fixed operator applied to heterogeneous histories yields *different* distributions; a shared operator does not guarantee *semantically comparable* patterns across learners — which is exactly what makes "one principle for every history" well-posed.
*Action.*
- Either argue this theoretically, or **explicitly downgrade it to a working assumption/hypothesis to be tested.**
- Propose a concrete **consistency metric** (e.g., alignment of pattern-space geometry across learner cohorts) that the implementation paper can report.

**J5. Close the evaluation-plan gaps.** `[R1-M6, R2-M3/m4]`
*Actions (each is independently checkable):*
1. **Graph confound.** For flat-tag datasets the graph `G` is *constructed* (CMDKT), so gains could stem from the injected graph, not from DPA. Add an ablation that **holds the graph fixed while removing alignment (and vice versa)** so the two contributions are isolated. `[R1-M6a, R2-M3]`
2. **Hypothesis falsifiability.** H1 ("competitive … improves most where reliable structure exists") is nearly unfalsifiable as worded — define success thresholds. For **H2** (data-scarce gains), give an argument beyond "priors act as regularizers," and state the protocol for constructing scarce regimes (learner subsampling? cold-start?). For **H4**, specify the graph-noise model used to test graceful degradation. `[R1-M6b, R2-m4]`
3. **Statistical rigor.** Commit to **multiple seeds, variance reporting, and significance testing** across the large baseline set — expected in the pyKT community to support "competitive" claims. `[R1-M6c]`
4. **Reproduction burden.** Several 2024–2026 baselines (UKT, KeenKT, S2KT, NSKT, PLKT, MemoryKT) are likely **not in pyKT**; acknowledge the reproduction effort and state how identical splits/preprocessing will be guaranteed. `[R1-M6d, R2-m5]`

**J6. Add scalability/complexity and cold-start discussion.** `[R1-M7, R2-m3/Q5]`
*Concern.* Multiple pooling operators + distributional projection + a per-KC recurrent memory `M ∈ R^{C×d}` may be costly on large-KC datasets (e.g., XES3G5M); pattern construction also depends on history length.
*Action.* Add a paragraph on **expected complexity vs. AKT/DKVMN**, the cost of per-step pooling over full history at ASSISTments/Algebra scale, and **short-history / cold-start behavior** on which pattern construction depends.

> **Optional but strongly recommended:** a **minimal proof-of-concept** experiment on one graph-native dataset (e.g., XES3G5M or Junyi) implementing just the alignment layer. Even a single result would simultaneously address J1, J2, and J3, and would move the paper from "borderline" toward "clear accept." If the venue expects preliminary results, this is effectively required.

---

### MINOR — polish for camera-ready

- **N1 — Notation.** Unify `r_i ∈ {0,1}` vs `r_t` and `q_i` vs `q_t`; and disambiguate **"evidence"**, which is used both colloquially (introduction) and in a special *post-alignment* sense (Sec. III). Define terms early and once. `[R1-minor, R2-m1]`
- **N2 — Interpretability outputs overlap.** Distinguish the **pattern-to-KC gating** (Module 3) from the **KC-to-prediction contribution** (Module 4); and note that the Section V caveat partially undercuts their earlier presentation as architectural features — align the framing. `[R2-m2]`
- **N3 — Contribution phrasing.** "Perspective" and "architecture" are largely ECD-applied-to-KT, and the "core mechanism" is not yet specified — **soften the abstract/introduction to match the current level of specification.** `[R1-M5, R2-m7]`
- **N4 — ECD as a binding constraint.** Strengthen the ECD framing by naming **at least one concrete design decision that follows *necessarily* from ECD** and would be "wrong" in a non-ECD model; otherwise the mapping reads as an attractive relabeling. `[R1-M5]`
- **N5 — "Pedagogical principles" scope.** Only two principles are instantiated (guess/slip, monotonicity). Either enumerate the intended principle space or scope the claim to "a small, extensible set." `[R1-minor]`
- **N6 — Preprint-based positioning & anonymity.** Several key references, including the **closest prior work (PLKT [11])**, are 2026 preprints not yet community-vetted; state this so readers can calibrate, and verify the positioning survives if a preprint changes. Also confirm the arXiv preprints cited do not **de-anonymize** the authors under the venue's blind-review policy (placeholders "Author One / University Name" suggest double-blind). `[R1-minor, R2-m5]`
- **N7 — Figure 1.** The 1→2→3→4 U-shaped layout is mildly counterintuitive — add an explicit ordering cue. Ensure legibility in **grayscale** and that the four-space color-coding is defined in the caption. `[R1-minor, R2-m6]`
- **N8 — "Learning pattern" definition.** The distinction "defined by the aggregation operator, not by the number of interactions" is subtle — add a one-line worked example. Define `d` (embedding dim) and give the shapes/roles of `z′` and the "post-alignment object" even with equations deferred. `[R1-minor]`

---

## 6. Revision checklist (copy into your tracker)

**Blocking (MAJOR):**
- [ ] **J1** Concrete (schematic/equation) instantiation of pedagogical alignment for ≥1 principle; state differentiability + how it avoids conflicting with the prediction loss.
- [ ] **J1b** Relate distributional guess/slip to BKT guess/slip and IRT 3PL/4PL — state what it adds.
- [ ] **J2** Side-by-side vs an entangled baseline; sharpen PLKT distinction (different-prediction example); justify distribution-*before*-pooling vs UKT/KeenKT/S2KT.
- [ ] **J3** Reframe monotonicity as hypothesis; add debiasing-vs-suppression ablation; reconcile with LPKT forgetting in text.
- [ ] **J4** Argue or downgrade cross-learner consistency; propose a consistency metric.
- [ ] **J5a** Add graph-fixed vs alignment-removed ablation (isolate graph confound).
- [ ] **J5b** Define success thresholds for H1; protocol for H2 scarce regimes; graph-noise model for H4.
- [ ] **J5c** Commit to multiple seeds + variance + significance testing.
- [ ] **J5d** Acknowledge reproduction burden for non-pyKT 2024–2026 baselines; guarantee identical splits.
- [ ] **J6** Complexity vs AKT/DKVMN; per-step pooling cost at scale; cold-start behavior.
- [ ] **(Optional/strongly recommended)** Minimal proof-of-concept on one graph-native dataset.

**Polish (MINOR):**
- [ ] **N1** Unify notation; disambiguate "evidence."
- [ ] **N2** Distinguish the two interpretability outputs; align with the Sec. V caveat.
- [ ] **N3** Soften abstract/intro contribution claims to match specification level.
- [ ] **N4** Name one design decision ECD *forces*.
- [ ] **N5** Scope or enumerate the "pedagogical principles" set.
- [ ] **N6** Flag preprint-based positioning; verify anonymity.
- [ ] **N7** Figure 1 ordering cue + grayscale legibility + color-coding caption.
- [ ] **N8** Worked example for "learning pattern"; define `d`, `z′`, "post-alignment object."

---

## 7. Consolidated questions to the authors (for the rebuttal)

1. **Mechanism.** Concretely, is pedagogical alignment a loss term, a projection, or a constraint layer? Through what is it differentiable, and how is it kept from conflicting with the prediction objective? `[R1, R2-Q1]`
2. **Classical formalism.** How does the distributional treatment of guessing/slipping (adjusting expectation/variance) differ theoretically from BKT guess/slip probabilities or IRT 3PL/4PL asymptotes? `[R1, R2-Q4]`
3. **Forgetting.** Could the monotonicity principle remove the model's ability to capture genuine forgetting? Which ablation separates "debiasing" from "signal suppression," and how do you reconcile it with LPKT's forgetting gate? `[R1, R2-Q3]`
4. **Consistency.** By what mechanism do patterns from the same operator become *semantically comparable* across learners, and how would you measure that consistency? `[R1, R2-Q2]`
5. **Confound isolation.** In the planned ablations, how will you separate the contribution of the (sometimes constructed) knowledge graph from the alignment mechanism? `[R1]`
6. **ECD.** Name one design decision that ECD *forces* and that a non-ECD model would get wrong. `[R1]`
7. **Cost.** What is the computational cost of the three pooling operators at ASSISTments/Algebra scale, and how does the framework behave on very short histories? `[R1, R2-Q5]`

---

## 8. Ethics & reproducibility

No ethical concerns. As a position paper, no code or data is expected at this stage; the planned use of the public pyKT benchmark and standard datasets is appropriate and supports future reproducibility. The main reproducibility risk is the reliance on several very recent (2024–2026) baselines not yet in pyKT — see J5d. `[R2-§7, R1-M6d]`

---

## 9. One-paragraph summary comment (visible to authors)

A clearly written, theoretically well-grounded vision paper with a genuinely interesting reframing of knowledge tracing as *evidence interpretation*, grounded in Evidence-Centered Design. The main gap, on which both reviews agree, is that the **pedagogical alignment — the paper's named core contribution — is currently its least-specified component**, and its novelty relative to existing memory/attention/distributional/gate-based KT models is not yet sharply drawn. Providing one concrete instantiation of the alignment (J1), sharpening the novelty delta (J2), reframing the monotonicity principle as a testable hypothesis (J3), establishing or downgrading cross-learner consistency (J4), and tightening the evaluation plan against the graph confound and statistical rigor (J5) would substantially strengthen the work. Recommended for acceptance as a position paper conditional on these revisions.
