# Dàn ý chi tiết — Short Paper

**Đề tài:** Distributional Pedagogical Alignment — một framework Knowledge Tracing lấy *evidence* làm trung tâm

**Loại bài:** short paper trình bày lập luận (position / framework paper của ongoing research). **Không có phần thực nghiệm** — deliverable là *motivation → reasoning → overall proposal*.

**Nguồn:** `proposal_final.md` (đề xuất gốc) + `review_KT_in_scope.md` (quyết định team, mã hiệu S1–S5 / A1–A4 / B1–B3 / C1–C2–C6).

> **Cách đọc dàn ý:** mỗi mục là một đề mục trong bài; mỗi bullet là một ý cần viết. Các nhãn dạng `[A2]`, `[C2]`, `[S4]`… trỏ về đúng quyết định trong file review để team truy vết và chỉnh sửa. Bullet mở đầu `↪ REASONING:` là lý do/định hướng viết, không phải câu để đưa vào bài.

---

## 0. Đề xuất thay đổi cấu trúc so với hình dung ban đầu của team

> Team có toàn quyền giữ/bỏ. Đây là các thay đổi mình đề xuất kèm lý do; phần dàn ý bên dưới đã áp dụng chúng.

1. **Thêm một mục ngắn "Related Work & Positioning" (Mục 4), tách khỏi Introduction.**
   ↪ REASONING: A2 (định vị đóng góp) là việc rủi ro cao nhất còn lại và đòi hỏi phân biệt *cơ chế* với PLKT và S²KT. Nhét toàn bộ vào Introduction sẽ làm Intro nặng và làm loãng câu "đóng góp khẳng định". Một mục positioning riêng, gọn (nửa cột), cho phép nói rõ điểm phân biệt mà không phá nhịp Intro. Nếu bị giới hạn độ dài gắt, có thể gập lại thành một đoạn trong Intro — nhưng nội dung phải còn nguyên.

2. **Thêm một tiểu mục "Intrinsic Interpretability / Attribution Trace" trong Proposed Method (6.7), thay vì rải rác trong từng module.**
   ↪ REASONING: interpretability là một trong hai trục novelty (evidence-centered + pedagogical-aligned) nhưng đang bị over-claim [A3]. Gom vào một chỗ giúp (a) hạ giọng nhất quán một lần, (b) trình bày attribution trace xuyên hai module (Pattern→KC ở Module 3, KC→Prediction ở Module 4) như một mạch liền, đúng như sơ đồ cuối proposal.

3. **Đổi "Proposed Method §1 = động lực" thành hai tiểu mục tách bạch: 6.1 Design Rationale & Perspective + 6.2 Framework Overview (kèm Fig 1).**
   ↪ REASONING: động lực/triết lý thiết kế (evidence-centered, tách 4 vai trò) là *lập luận*; overview + sơ đồ tổng thể là *mô tả kiến trúc*. Tách ra giúp người đọc thấy rõ "vì sao" trước "cái gì".

4. **Hình vẽ: đề xuất 2 hình (thay vì tối thiểu 1).**
   ↪ REASONING: xem Mục 9. Fig 1 = sơ đồ tổng thể (bắt buộc, team đã định). Fig 2 = hình khái niệm đối chiếu pipeline chuẩn vs pipeline có tầng trung gian — hình này gánh trực tiếp lập luận A1/A2 và rất đáng giá cho một position paper.

5. **Problem Formulation giữ ở mức tối thiểu (định nghĩa bài toán + ký hiệu 4 không gian), không dẫn công thức biến đổi.**
   ↪ REASONING: theo S3/S4 — thay công thức chi tiết bằng sơ đồ/khái niệm. Notation vẫn cần để Discussion và Method nói cho gọn, nhưng không sa vào đạo hàm phép cập nhật.

6. **[REFRAME — quyết định vòng review sau literature] Bỏ hẳn framing "hầu hết KT: interaction representation → cập nhật state trực tiếp".**
   ↪ REASONING: đó là claim phổ quát, gãy ngay khi reviewer chỉ ra phản ví dụ (PSI-KT/NSKT/S²KT/PLKT). Thay bằng framing **theory-led + comprehensive**: mở bằng lý thuyết đo lường (evidence-centered), trình bày các hướng prior như những **đóng góp bổ trợ** (mỗi hướng làm mạnh một khía cạnh), rồi định vị bài là **một khung comprehensive tổ chức các khía cạnh đó quanh một nguyên lý sư phạm tường minh**.
   ↪ KỶ LUẬT: (1) KHÔNG nói "hầu hết mô hình cập nhật trực tiếp" (claim phổ quát). (2) KHÔNG nói "nguyên liệu rời rạc / chưa ai kết hợp" (cũng là claim phổ quát ngầm — bẫy song song). (3) "comprehensive/combined" KHÔNG hứa đầy đủ/toàn diện; phát biểu là "khung mạch lạc neo vào lý thuyết sư phạm". (4) Novelty thật vẫn là **A2** (soft-rule trên cross-student pattern) — KHÔNG phải bản thân việc kết hợp.
   ↪ Ảnh hưởng: §3.1, §3.2, Abstract, §3.5-C1, Fig 2 (đổi sang design-space), checklist A1.

---

## 1. Title

- ↪ REASONING: tránh over-claim [A3]; nêu bật hai trục novelty *khẳng định* [A2] (evidence-centered + pedagogical alignment) và không nhét tên kỹ thuật cài đặt (Mamba/Beta) [S1, S2].
- Các phương án ứng viên (team chọn/ghép):
  - **PA1 (nhấn framework):** *"Distributional Pedagogical Alignment: An Evidence-Centered Framework for Knowledge Tracing."*
  - **PA2 (nhấn góc nhìn — hợp position paper):** *"Rethinking Knowledge Update in Knowledge Tracing: Aligning Learning Patterns in a Distributional Space."*
  - **PA3 (nhấn cơ chế lõi):** *"Pedagogically-Aligned Learning Patterns for Interpretable Knowledge Tracing."*
- Lưu ý đặt tên: không dùng "intrinsically interpretable"/"by construction" trong title [A3]; nếu dùng "interpretable" thì để ở nghĩa mở khả năng, không hứa faithfulness.
- ↪ [REFRAME] PA2 ("Rethinking Knowledge *Update*") nghiêng về framing cũ (ngụ ý phê phán khâu update). Sau reframe, ưu tiên PA1 hoặc PA3 (nhấn framework/cơ chế + evidence-centered); nếu vẫn thích "rethinking" thì đổi trọng tâm sang "evidence" (vd *"Knowledge Tracing as Evidence Interpretation…"*).

---

## 2. Abstract

> Cỡ ~150–200 từ. Một câu cho mỗi ý; giữ đúng thứ tự lập luận của bài.

- **Bối cảnh + khoảng trống (1–2 câu):** neo lý thuyết đo lường (kết quả học *suy ra từ bằng chứng không đầy đủ*); KT gần đây tiến bộ theo các hướng bổ trợ (biểu diễn interaction / mastery dạng phân phối / reasoning), mỗi hướng làm mạnh một khía cạnh. Bài đề xuất một khung **comprehensive** tổ chức các khía cạnh này quanh một tầng *diễn giải bằng chứng* tường minh, neo vào nguyên lý sư phạm.
  - ↪ [A1-reframe] KHÔNG dùng "vẫn cập nhật state trực tiếp từ interaction representation" (claim phổ quát). KHÔNG dùng "nguyên liệu rời rạc / chưa ai kết hợp" (claim phổ quát ngầm). Định vị bằng *nguyên lý tổ chức* của riêng mình.
- **Góc nhìn đề xuất (1 câu):** dựa trên evidence-centered design, một tương tác trước hết sinh *bằng chứng học tập*; chỉ sau khi bằng chứng được tổng hợp và diễn giải theo góc nhìn sư phạm thì mới cập nhật tri thức → cần một *tầng trung gian*. [A4, B1]
- **Đề xuất (2 câu):** framework 4 module qua 4 không gian biểu diễn (Interaction → Distribution → Knowledge → Prediction); tầng lõi *Distributional Pedagogical Alignment* xây learning pattern liên-học-sinh trong không gian phân phối và điều chỉnh chúng bằng *soft-rule* sư phạm trước khi cập nhật mastery.
- **Đóng góp khẳng định (1 câu):** đóng góp trung tâm là *áp soft-rule sư phạm lên learning pattern liên-học-sinh trong không gian phân phối* — kèm khả năng trích xuất attribution nội tại (pattern→KC, KC→prediction). [A2, A3]
- **Định vị bài (1 câu):** đây là bài trình bày framework/định hướng của một nghiên cứu đang tiến hành; thực nghiệm là bước kế tiếp.
  - ↪ REASONING: nói thẳng "position/ongoing" để reviewer không kỳ vọng bảng số — vừa đúng thực tế vừa phòng thủ được.

---

## 3. Introduction

> Mục tiêu: dẫn từ bối cảnh → cơ hội (không phải flaw của người khác) → góc nhìn evidence-centered → ý tưởng tầng trung gian → đóng góp. Kết thúc bằng danh sách contribution.

- **3.1 Bối cảnh KT và các hướng cải tiến gần đây (framing tích cực)**
  - Định nghĩa nhiệm vụ KT: mô hình hóa trạng thái tri thức tiến hóa để dự đoán kết quả tương tác kế tiếp.
  - **Mở bằng neo lý thuyết đo lường (câu đầu):** kết quả học không quan sát trực tiếp được, phải *suy ra từ bằng chứng không đầy đủ* qua một mô hình cân nhắc "weight and direction of evidence" (ECD). Dựng lăng kính evidence-centered ngay từ đầu. [A4]
  - Ba hướng cải tiến gần đây, phát biểu như các **đóng góp bổ trợ** (KHÔNG phải setup cho strawman): (i) biểu diễn interaction — content/topology/semantic-aware; (ii) biểu diễn mastery dạng phân phối/uncertainty-aware; (iii) reasoning/diễn giải — graph/neural-symbolic/probabilistic. Mỗi hướng làm mạnh MỘT khía cạnh của việc biến tương tác thành ước lượng tri thức.
  - ↪ [A1-reframe] BỎ mô tả "pipeline 4 khâu chung" + câu "đây là điểm bài xem xét lại". Không còn claim phổ quát về cách KT cập nhật state.
- **3.2 Khoảng trống: bước diễn giải bằng chứng chưa được làm thành một tầng tường minh**
  - Phát biểu khoảng trống như một *nguyên lý tổ chức chưa được khai thác đúng mức*, KHÔNG phải khuyết điểm của ai: các hướng trên làm mạnh hai đầu mút (biểu diễn interaction & dự đoán) và thêm cơ chế reasoning; còn bước *diễn giải bằng chứng* — tổng hợp, tổ chức, và điều chỉnh sư phạm — ít được trang bị như một tầng riêng, tường minh.
  - Định vị bài: đề xuất một hướng **comprehensive**, tổ chức các khía cạnh này quanh một nguyên lý evidence-centered. Một tầng trung gian tường minh mở ra:
    - (a) cấu trúc tương tác thành learning pattern có ý nghĩa sư phạm;
    - (b) thực hiện các phép *điều chỉnh sư phạm* trên bằng chứng trước khi chạm mastery;
    - (c) chuẩn hóa biểu diễn giữa các học sinh theo quy chuẩn sư phạm.
  - ↪ [kỷ luật comprehensive] KHÔNG hứa "đầy đủ/toàn diện" (reviewer sẽ hỏi "sao thiếu forgetting?"). Phát biểu là "khung mạch lạc neo vào lý thuyết sư phạm". Novelty thật = **A2** (soft-rule trên cross-student pattern), KHÔNG phải việc kết hợp.
  - ↪ REASONING: (a)(b)(c) phát biểu trung lập như năng lực một tầng tường minh mở ra — KHÔNG ngụ ý mô hình cũ "không làm được" (chúng làm được ngầm).
- **3.3 Góc nhìn evidence-centered (neo lý thuyết)**
  - Học tập: một tương tác không *trực tiếp* đổi trạng thái tri thức; nó sinh ra *bằng chứng*, và bằng chứng cần được tổng hợp + diễn giải mới thành cơ sở cập nhật. [A4]
  - Neo vào **ECD (Mislevy, Steinberg & Almond 2003):** proficiency "cannot be measured directly; must be inferred from incomplete evidence"; Measurement Model = "weight and direction of evidence". [A4, B1]
  - Neo vào **formative assessment (Black & Wiliam 2009):** học = "closing the gap" giữa trạng thái hiện tại và mục tiêu; feedback theo ngữ cảnh — động cơ sư phạm cho việc *alignment*. [A4]
  - ↪ [A4] "Theo dõi theo *nhóm mẫu hình* liên-học-sinh" là **lựa chọn thiết kế của team**, KHÔNG gán cho ECD/formative (hai nguồn đó nói về *cá thể* learner). Phát biểu rõ đây là chỗ ta mở rộng khung.
- **3.4 Ý tưởng lõi: tầng Distributional Pedagogical Alignment**
  - Đề xuất chèn một tầng trung gian giữa "biểu diễn interaction" và "cập nhật mastery": tổng hợp evidence → tổ chức thành pattern → điều chỉnh theo luật sư phạm → lượng hóa đóng góp.
  - Nguyên lý tách vai trò: bốn không gian, mỗi không gian một mục tiêu duy nhất (Interaction/Distribution/Knowledge/Prediction). [C1 — nêu là *aim*]
  - Chỉ báo hình: trỏ tới Fig 2 (design-space — các hướng prior tác động ở khâu nào; tầng đề xuất nằm ở khâu *diễn giải bằng chứng*).
- **3.5 Đóng góp (bullet, phát biểu khẳng định) [A2, A3]**
  - **C-1 (góc nhìn):** reframing KT theo evidence-centered — cập nhật tri thức là quá trình *diễn giải bằng chứng*, hiện thực bằng một tầng trung gian tường minh. [A4]
  - **C-2 (cơ chế lõi):** *soft-rule sư phạm áp lên learning pattern liên-học-sinh trong không gian phân phối* — phần mà các mô hình phân phối/pattern trước đó (PLKT, S²KT) không có. [A2]
  - **C-3 (kiến trúc):** framework 4 module / 4 không gian tách vai trò, với learning pattern định nghĩa bởi phép tổng hợp cố định (temporal / same-KC / prerequisite / neighbor) → cấu trúc pattern nhất quán giữa mọi học sinh.
  - **C-4 (diễn giải, hạ giọng):** framework *cho phép trích xuất* attribution nội tại (pattern→KC, KC→prediction), *mở khả năng* diễn giải ở mức model-level — không tuyên bố giải thích faithful. [A3]

---

## 4. Related Work & Positioning  *(mục mới đề xuất — xem Mục 0.1)*

> Gọn (~nửa cột). Định vị bằng *khẳng định* điểm phân biệt cơ chế; TUYỆT ĐỐI không nói bài khác "chồng lấn"/"chiếm chỗ". [A2]

- **4.1 Ba lát cắt của KT gần đây (mỗi lát 1 câu + 1–2 ví dụ đã xác minh):**
  - Biểu diễn interaction: MCKT (semantic + difficulty embedding), CMDKT (topology KC).
  - Biểu diễn mastery dạng phân phối/uncertainty: UKT (stochastic embeddings + Wasserstein), KeenKT (NIG distribution), S²KT (KC là Gaussian).
  - Reasoning/diễn giải: GKT (graph), PSI-KT / NSKT / PLKT (probabilistic / neural-symbolic / pattern-based reasoning).
- **4.2 Định vị vs PLKT — khác ở *nguồn* của phân phối [A2]**
  - PLKT: biểu diễn phân phối được **tra trực tiếp từ lịch sử tương tác** (evidence + pattern theo hướng look-up).
  - Đề xuất: phân phối được **diễn giải từ biểu diễn ẩn của bước-học đã mô hình hóa lịch sử** → phân phối mang *giá trị tri thức cao hơn* (sản phẩm của quá trình mô hình hóa, không phải bảng tra từ interaction thô). [gắn C2]
- **4.3 Định vị vs S²KT — khác ở *đối tượng* được phân phối + *tầng luật* [A2]**
  - ⚠️ KHÔNG công kích cách S²KT bắc cầu hai không gian: S²KT xử lý điều này *tường minh* bằng cross-attention học được (query = hidden state, train end-to-end), không phải giả định ngầm. Đọc kỹ bản gốc §3.4.
  - Điểm phân biệt đúng: S²KT phân phối trên **KC như thực thể ngữ nghĩa** (Gaussian centroid + uncertainty, pretrained triplet loss); đề xuất phân phối trên **learning pattern/evidence của bước học** rồi áp **soft-rule sư phạm**. S²KT không có tầng điều chỉnh sư phạm bằng luật (nó dùng diffusion + attention học được).
- **4.4 Câu định vị tổng (một câu khẳng định):** *"Đóng góp trung tâm — áp soft-rule sư phạm lên learning pattern liên-học-sinh trong không gian phân phối — là phần mà cả các mô hình phân phối (UKT/KeenKT/S²KT) lẫn các mô hình pattern-based (PLKT) đều không có."* [A2]

---

## 5. Problem Formulation and Notations

> Tối thiểu, đủ để Method/Discussion nói gọn; không dẫn công thức biến đổi. [S3, S4]

- **5.1 Định nghĩa bài toán KT**
  - Chuỗi tương tác của một học sinh: $\{(q_t, r_t)\}_{t=1}^{T}$ với $q_t$ = câu hỏi/mục, $r_t\in\{0,1\}$ = đúng/sai.
  - Nhiệm vụ: dự đoán $P(r_{t+1}=1\mid q_{t+1}, \text{lịch sử})$.
  - Cấu trúc tri thức: tập Knowledge Component $\mathcal{C}=\{c_1,\dots,c_C\}$; ánh xạ q→KC; đồ thị tri thức $\mathcal{G}$ (quan hệ tiên quyết/láng giềng) — *xem là đầu vào cho trước trong scope* [C6-a].
- **5.2 Ký hiệu bốn không gian biểu diễn** (bảng notation)
  - $z_t$ — Interaction Representation (Interaction Space).
  - $h_t$ — Distribution Representation (Distribution Space), tham số hóa bằng statistical descriptor (giữ *khái niệm phân phối*, không nêu dạng cụ thể) [S2, B2].
  - $P_i, \tilde P_i$ — learning pattern trước/sau alignment; $z'$ — structured pattern representation (Pattern Readout).
  - $M_t$ — Knowledge/Mastery State (Knowledge Space); $\hat y$ — dự đoán (Prediction Space).
  - $A_i$ — đóng góp pattern→KC (gating); $W$ — đóng góp KC→prediction.
- **5.3 Quy ước thuật ngữ (chốt để nhất quán toàn bài) [B1]**
  - "**evidence**" = biểu diễn phân phối ở Module 2 *sau khi* đã gộp thành pattern và điều chỉnh sư phạm (hậu-alignment). KHÔNG gọi $\Phi(z_t)$ (tiền-alignment) là evidence ở bất kỳ đâu.
  - "**learning pattern**" = kết quả của một Pattern Operator cố định (không định nghĩa bởi số lượng interaction).
  - ↪ REASONING: đóng khung thuật ngữ ngay tại notation giúp reviewer bắt được mọi chỗ dùng sai từ về sau.

---

## 6. Proposed Method

> Cấu trúc: 6.1 lý do thiết kế → 6.2 tổng quan (Fig 1) → 6.3–6.6 bốn module → 6.7 attribution trace. Mô tả ở mức khái niệm/sơ đồ; bỏ tên cài đặt và công thức nặng [S1–S5].

- **6.1 Design Rationale & Perspective** *(phần "§1 động lực" của team)*
  - Nhắc lại góc nhìn evidence-centered ở mức thao tác: interaction → evidence → (tổng hợp/điều chỉnh) → update.
  - Nguyên lý tách vai trò bốn không gian: mỗi tầng một mục tiêu; giảm pha trộn giữa biểu diễn ngữ cảnh / tri thức / dự đoán.
  - ↪ [C1] Phát biểu tách vai trò là **mục tiêu thiết kế (aim)**, KHÔNG phải bảo đảm được cưỡng chế; nêu *hướng khả thi* để ép vai trò: per-module contrastive learning (có tiền lệ KeenKT) hoặc auxiliary loss / train theo giai đoạn. Chi tiết cơ chế để Discussion/future work.
- **6.2 Framework Overview** — **Fig 1** (sơ đồ tổng thể)
  - Bốn module nối tiếp: Interaction Representation Learning → Distributional Pedagogical Alignment → Mastery State Tracking → Prediction Aggregation.
  - Ánh xạ module ↔ không gian; chỉ rõ evidence "sinh" ở Module 2, mastery "cập nhật" ở Module 3, dự đoán ở Module 4.
- **6.3 Module 1 — Interaction Representation Learning**
  - Hai nhóm đầu vào, hai loại dynamics:
    - *Interaction Context:* question semantic + student response + question difficulty.
    - *Knowledge Context:* localized mastery (truy xuất Mastery Memory theo ngữ cảnh câu hỏi qua Knowledge Graph) + concept difficulty.
  - Hai nhánh mô hình hóa chuỗi *song song* → hai luồng dynamics riêng (chuỗi tương tác vs trạng thái tri thức cục bộ); hợp nhất thành $z_t$.
  - ↪ [S1] KHÔNG nêu tên "Mamba"; gọi là "nhánh mô hình hóa chuỗi" — tên kiến trúc là lựa chọn cài đặt.
- **6.4 Module 2 — Distributional Pedagogical Alignment** *(tầng lõi)*
  - ↪ [S5] Có thể trình bày 4 bước như một *đoạn ý niệm* + sơ đồ, thay vì bốn tiểu mục nặng; nhưng vẫn nêu đủ 4 vai trò:
    1. **Distribution Projection:** $z_t \to h_t$ qua statistical descriptor decoder; phân phối là *không gian biểu diễn trung gian của interaction pattern*, KHÔNG phải mastery/learner belief [B2, khác PLKT].
    2. **Pattern Construction:** các Pattern Operator cố định — temporal / same-KC / prerequisite / neighbor; pattern định nghĩa bởi *phép tổng hợp*, nên cấu trúc pattern nhất quán giữa mọi học sinh.
    3. **Pedagogical Alignment (soft-rule):** điều chỉnh pattern bằng tập luật sư phạm; luật KHÔNG suy luận mastery trực tiếp mà *đưa pattern gần hơn nguyên lý sư phạm* — như một regularization trên không gian phân phối [khác NSKT].
    4. **Pattern Readout:** $\tilde P_i \to z'$, structured pattern representation, mỗi block ứng một loại pattern xác định.
  - ↪ [C2] Đây là chỗ trả lời "vì sao cần không gian phân phối": nêu **một tính chất cụ thể** phân phối cho mà vector-pooling không cho — ví dụ phép gộp giữ được *cả vị trí lẫn độ phân tán/độ tin cậy* như đại lượng có đơn vị; khoảng cách sư phạm giữa pattern đo bằng divergence có nghĩa. TUYỆT ĐỐI không ngụ ý "dùng phân phối = novelty" (UKT/KeenKT/S²KT đã dùng phép toán phân phối). Gắn với 4.2 (phân phối *diễn giải từ biểu diễn ẩn* → giá trị tri thức cao hơn).
  - ↪ [B3] Khi liệt kê "nguyên lý sư phạm" dự kiến, dùng dạng mở "for example / including but not limited to": singleton của mastery; topology khái niệm (đồng khái niệm / tiên quyết / ứng dụng / nâng cao); guessing/slipping; quan hệ temporal. Riêng "cần ≥n tương tác mới tự tin ước lượng mastery" phát biểu như **giả định về độ tin cậy thống kê của ước lượng**, KHÔNG gán tên "định luật sư phạm" (không có citation).
- **6.5 Module 3 — Mastery State Tracking**
  - Knowledge State $M_t$ là trạng thái *tường minh*, cập nhật trực tiếp từ structured pattern representation.
  - Học thêm **gating pattern→KC** $A_i=G(P_i)$: lượng hóa mức đóng góp của từng learning pattern lên từng KC; cập nhật $M_{t+1}=M_t+\Delta M$.
  - ↪ [S4] BỎ công thức cập nhật; thêm **một câu** rằng modeling forgetting/decay là *hướng để mở* (biến điểm yếu thành scope statement, tránh bị bắt vì lặng lẽ bỏ forgetting).
  - Vai trò diễn giải: gating trả lời "pattern nào tăng mastery của KC nào" — nhưng để dành phát biểu faithfulness cho 6.7.
- **6.6 Module 4 — Prediction Aggregation**
  - Đầu vào: $M_{t+1}$ + biểu diễn câu hỏi mục tiêu $q_{t+1}$.
  - Sinh **KC contribution** $W=C(M_{t+1},q_{t+1})$ và dự đoán $\hat y=f(M_{t+1},q_{t+1},W)$.
  - ↪ [S3] Giữ sơ đồ + ký hiệu tối thiểu, không dẫn công thức đầy đủ.
- **6.7 Intrinsic Interpretability — Attribution Trace** *(tiểu mục mới đề xuất — xem Mục 0.2)*
  - Mạch attribution xuyên hai module: Historical Interactions → Learning Patterns → (Pattern→Knowledge Contribution, $A_i$) → Updated Knowledge State → (Knowledge→Prediction Contribution, $W$) → Prediction. (Có thể là inset của Fig 1 hoặc Fig 2.)
  - ↪ [A3] Hạ giọng nhất quán MỘT lần: framework *cho phép trích xuất* attribution nội tại, *mở khả năng* phát triển cơ chế interpretable, **tối thiểu ở mức diễn giải dự đoán (model-level)**. Nêu rõ giới hạn: trọng số attention/gating ≠ giải thích faithful (dẫn Jain & Wallace 2019, "Attention is not Explanation"). Attribution là *by design của kiến trúc*, không phải module giải thích gắn ngoài — nhưng không tuyên bố đúng đắn về mặt faithfulness.

---

## 7. Discussion

> Nơi bàn sâu các lựa chọn thiết kế đã bị "hạ chi tiết" ở Method, và khoanh scope. Mỗi tiểu mục = một quyết định review.

- **7.1 Vì sao cần một không gian phân phối? [C2]**
  - Lập luận tính chất: gộp-thành-pattern + alignment *well-defined* trên phân phối (giữ vị trí + độ phân tán; divergence có nghĩa) hơn là trên vector.
  - Thành thật nêu tiền lệ: UKT (Wasserstein), KeenKT (NIG-distance), S²KT (Bhattacharyya + GMM) đã dùng phép toán phân phối → "lên phân phối" tự nó không phải novelty; novelty ở *dùng nó cho gộp-pattern-sư-phạm + soft-rule alignment* và ở *nguồn diễn giải từ biểu diễn ẩn* (khác PLKT look-up).
- **7.2 Tách bốn vai trò: aim, không phải bảo đảm [C1]**
  - Thừa nhận train end-to-end không tự cưỡng chế mỗi không gian giữ đúng vai trò.
  - Hướng cưỡng chế khả thi: per-module contrastive (tiền lệ KeenKT), auxiliary loss, hoặc train theo giai đoạn → để dành cho full paper.
- **7.3 Phụ thuộc Knowledge Graph & phạm vi [C6]**
  - **Mặc định (a):** xem KG là đầu vào cho trước & đã kiểm định; tiền lệ CMDKT (sinh quan hệ tiên quyết bằng LLM + chuyên gia kiểm định) → giữ proposal gọn.
  - Kỷ luật phát biểu: KHÔNG nói "quan hệ sư phạm là well-formed" như chân lý phổ quát; nói "với KG dựng & kiểm định như CMDKT, ta xem topology đáng tin *trong phạm vi bài*".
  - **Compromise (b), chỉ rút khi bị ép:** thừa nhận KG học được (S²KT) nhưng khoanh rõ *ngoài scope* + future work; không để (b) thành nghĩa vụ phải-học-KG.
- **7.4 Soft-rule vs hard-rule vs tự học [A4-liên quan]**
  - Vì sao soft-rule (điều chỉnh/regularize) thay vì hard-rule suy diễn (khác NSKT) hoặc để mô hình tự học hoàn toàn: giữ ưu tiên sư phạm mà không cứng nhắc; có tiền lệ per-module contrastive (KeenKT) cho việc định hình biểu diễn bằng tín hiệu phụ.
- **7.5 Scope & Limitations (gộp các "để mở")**
  - Forgetting/decay chưa mô hình hóa — future work [S4].
  - "≥n tương tác" = giả định độ tin cậy thống kê, cần kiểm chứng thực nghiệm [B3].
  - Chưa có thực nghiệm — đây là framework paper; validation là bước kế tiếp.
  - Diễn giải mới ở mức model-level, faithfulness cần đánh giá riêng [A3].

---

## 8. Conclusion

- Tóm tắt góc nhìn: KT như *diễn giải bằng chứng*, hiện thực bằng tầng trung gian Distributional Pedagogical Alignment.
- Nhắc lại đóng góp khẳng định (một câu, không lặp lại y nguyên Intro): soft-rule sư phạm trên learning pattern liên-học-sinh trong không gian phân phối + khả năng trích xuất attribution.
- Định hướng tiếp: hiện thực + thực nghiệm trên benchmark; cơ chế cưỡng chế tách vai trò (contrastive); modeling forgetting; đánh giá faithfulness của attribution; KG learnable (nếu mở rộng scope).
- ↪ REASONING: Conclusion của position paper nên đóng bằng *cam kết nghiên cứu tiếp*, không tuyên bố kết quả — nhất quán với A3 và tính chất ongoing.

---

## 9. Kế hoạch hình vẽ

- **Fig 1 — Framework Overview (BẮT BUỘC, team đã định).**
  - Sơ đồ ngang 4 module / 4 không gian; nhãn rõ evidence sinh ở Module 2, mastery cập nhật ở Module 3.
  - Có thể lồng inset attribution trace (pattern→KC→prediction) hoặc tách sang Fig 2.
- **Fig 2 — Design-Space Map (CHỐT — thay cho conceptual-contrast).**
  - Trục pipeline: Interaction Representation → **Evidence Interpretation** → Knowledge State → Prediction.
  - Map các hướng prior vào khâu chúng *chủ yếu* tác động: biểu diễn (MCKT, CMDKT) ở Interaction Representation; phân phối/uncertainty (UKT, KeenKT, S²KT) ở Knowledge State; reasoning/diễn giải (GKT, PSI-KT, NSKT, PLKT) là overlay state→prediction.
  - Tầng đề xuất (Distributional Pedagogical Alignment) highlight ở khâu **Evidence Interpretation** — trục "diễn giải bằng chứng" được làm tường minh, có cấu trúc sư phạm.
  - ↪ REASONING: thay cho hình đối chiếu "direct update" (vốn tái nhập chính strawman vừa bỏ). Hình này khớp framing comprehensive: biến đa dạng literature thành *bản đồ định vị*, KHÔNG phải claim direct-update. "Chủ yếu tác động ở khâu X" là mô tả, phòng thủ được.
- Lưu ý: các sơ đồ mermaid trong proposal gốc là bản nháp ý niệm; khi lên hình cuối cần vẽ lại publication-grade (dùng skill `figure-style`).

---

## 10. Checklist truy vết review (để team rà trước khi viết)

| Item | Trạng thái | Nằm ở mục nào trong dàn ý |
|---|---|---|
| S1 (bỏ Mamba) | 🟢 | 6.3 |
| S2 (bỏ dạng phân phối) | 🟢 | 5.2, 6.4 |
| S3 (sơ đồ thay công thức) | 🟢 | 5, 6.6 |
| S4 (bỏ công thức mastery + câu forgetting) | 🟡 | 6.5, 7.5 |
| S5 (gộp 4 bước Module 2) | 🟢 | 6.4 |
| A1 (tránh strawman) | 🟢 | 3.2 |
| A2 (định vị khẳng định) | 🟡 | 3.5, Mục 4 (4.2–4.4) |
| A3 (hạ giọng interpretability) | 🟢 | 3.5-C4, 6.7, 7.5 |
| A4 (neo ECD/formative) | 🟢 | 3.3, 7.4 |
| B1 (định nghĩa evidence) | 🟢 | 5.3, 6.4 |
| B2 (ngữ nghĩa distribution) | 🟢 | 6.4 |
| B3 (nguyên lý sư phạm mở) | 🟡 | 6.4 |
| C1 (tách vai trò = aim) | 🟢 | 6.1, 7.2 |
| C2 (không gian phân phối không thừa) | 🟡 | 6.4, 7.1 |
| C6 (phụ thuộc KG) | 🟢 | 5.1, 7.3 |

**Ba việc rủi ro cao nhất (viết trước, theo review):** A2 (4.2–4.4) → C2 (7.1) → giữ kỷ luật C6 (7.3).
