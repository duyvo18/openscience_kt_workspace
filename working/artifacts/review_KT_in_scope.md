# Review KT — IN-SCOPE: Lịch sử cải tiến & hướng giải quyết

**Tài liệu nguồn:** `proposal_final.md` — *Distributional Pedagogical Alignment framework*
**Phạm vi:** các vấn đề IN-SCOPE cho short paper (deliverable = *lập luận*: motivation → reasoning → overall proposal).
**File đi kèm:** `review_KT_defer.md` (vấn đề để dành cho cài đặt / full paper).
**Cách đọc bảng:** mỗi vấn đề đi theo 4 cột — **(1) Vấn đề gốc** trong proposal → **(2) Quyết định của team** → **(3) Phản biện / kiểm chứng** → **(4) Hướng giải quyết chốt lại**. Cột (3)–(4) là phần cần theo dõi qua các vòng cải tiến.

**Trạng thái:** 🟢 chốt · 🟡 chốt nhưng còn rủi ro cần canh · 🔴 cần đổi so với quyết định ban đầu

---

## 0. Tài liệu tham chiếu đã kiểm chứng (dùng khi viết)

Các citation dưới đây đã xác minh (đọc trực tiếp abstract/nội dung gốc):

**KT — arXiv:**

| Nhãn | arXiv | Tên đầy đủ | Điểm liên quan đến đề xuất |
|---|---|---|---|
| **PLKT** | 2605.09369 | Explainable KT via Probabilistic Embeddings and Pattern-based Reasoning | Dùng **Beta-distributed embeddings** + phát biểu dự đoán là **"goal-conditioned evidence reasoning"** + **pattern-based reasoning**. Trùng 3 từ khóa trung tâm (Beta + evidence + pattern) → mối đe dọa novelty gần nhất. |
| **UKT** | 2501.05415 | Uncertainty-aware Knowledge Tracing | Dùng **stochastic distribution embeddings** + **Wasserstein self-attention** → tiền lệ cho "phép toán trên phân phối là well-defined". |
| **KeenKT** | 2512.18709 | Knowledge Mastery-State Disambiguation for KT | Biểu diễn state bằng **NIG distribution** + **NIG-distance attention** + **distributional contrastive loss** → tiền lệ cho per-module contrastive (C1). |
| **Dynamic LENS** | 2407.17427 | Uncertainty-preserving deep KT with state-space models | VAE + Bayesian state-space, Gaussian; **không có attention**. Bài riêng — KHÔNG phải UKT. |
| **MemoryKT** | 2508.08122 | Integrative Memory-and-Forgetting Method for KT | Mô hình hóa forgetting qua temporal VAE → tham chiếu cho decay (defer, C4). |

**KT — hội nghị (đã đọc bản gốc PDF):**

| Nhãn | Nguồn | Điểm liên quan |
|---|---|---|
| **S²KT** (Wu et al.) | WWW'26, DOI 10.1145/3774904.3793039 | KC là **Gaussian Embedding**; **semantic propagation bằng Bhattacharyya distance**; **Dynamic GMM Attention** tích hợp IRT; mô hình **heterogeneous forgetting**; so với **22 baseline** trên 6 dataset. Là **đại diện trường phái KG learnable** (C6-b): tự học quan hệ KC vì "explicit dependency annotations are unavailable". Điểm phân biệt A2: phân phối trên **KC như thực thể ngữ nghĩa** (centroid+uncertainty, pretrained triplet loss), bắc cầu hai không gian bằng **cross-attention học được** (§3.4.2, có chủ đích — KHÔNG phải giả định ngầm). Đề xuất khác ở *đối tượng phân phối* (learning pattern) + *tầng soft-rule sư phạm*. |
| **CMDKT** (Xia et al.) | Array 28 (2025) 100523, DOI 10.1016/j.array.2025.100523; ASSIST09 + DBE-KT22 | Là **đại diện trường phái KG well-formed / cho trước** (C6-a): sinh quan hệ tiên quyết bằng **ChatGPT-4 rồi 2+2 chuyên gia kiểm định**; topology embedding qua GRU theo prerequisite; difficulty 2 mức (QS + KC). Chỗ dựa cho việc dựng-và-kiểm-định KG. |

**Nền lý thuyết giáo dục (đã đọc bản gốc PDF):**

| Nhãn | Nguồn | Điểm neo được |
|---|---|---|
| **ECD** | Mislevy, Steinberg & Almond (2003), *evidence-centered design* | 4 model: **Student** (phân phối xác suất trên student-model variables), **Evidence** (Measurement Model = "**weight and direction of evidence** that Observable Variables convey about Student Model Variables"), **Task**, **Assembly** ("**How Much Do We Need to Measure It**"). Neo trực tiếp: A4 (reframing evidence-centered), B1 (định nghĩa evidence), B3 (Assembly Model = "cần bao nhiêu evidence"). Câu lõi: proficiency "**cannot be measured directly; must be inferred from incomplete evidence**". |
| **Formative assessment** | Black & Wiliam (2009) | Ramaprasad 3 quá trình (learner đang ở đâu / đi đâu / cách tới); học = "**closing the gap** giữa present state và learning aim" (Sadler 1989); feedback contingent. Neo: A4 (động cơ sư phạm cho alignment/feedback). |

> Cả ba PDF (S²KT, ECD, formative) do team cung cấp và đã đọc trực tiếp — dùng được trong bài.

---

## 1. Flaw phạm vi — phần cần cắt / hạ chi tiết  🟢

| # | Vấn đề gốc | Quyết định của team | Phản biện / kiểm chứng | Hướng giải quyết chốt |
|---|---|---|---|---|
| S1 | Chi tiết kiến trúc Mamba song song | Rút thành "hai nhánh mô hình hóa chuỗi song song, hai luồng dynamics riêng biệt" | Đồng ý — tên kiến trúc là lựa chọn cài đặt | 🟢 Bỏ tên Mamba, giữ ý niệm hai dynamics |
| S2 | Nêu dạng phân phối cụ thể (Beta) | Chỉ giữ khái niệm "không gian/biểu diễn phân phối" | Đồng ý; **thêm lý do:** PLKT cũng dùng Beta → nêu Beta sớm càng làm mờ ranh giới với PLKT | 🟢 Bỏ tên dạng phân phối |
| S3 | Chuỗi công thức chi tiết | Giữ sơ đồ tổng thể; trọng tâm là luồng tri thức qua các space, không phải phép biến đổi | Đồng ý | 🟢 Thay công thức bằng sơ đồ khái niệm |
| S4 | Công thức cập nhật mastery `M+ΔM` | Bỏ công thức; tạm không nhắc decay | Đồng ý bỏ công thức. **Rủi ro:** bán "hiện thực sư phạm" mà lặng lẽ bỏ forgetting → dễ bị bắt | 🟡 Bỏ công thức; **thêm 1 câu** "modeling forgetting là hướng để mở" (biến điểm yếu thành scope statement) |
| S5 | Module 2 mô tả chi tiết từng bước | Abstract thành mô tả ý niệm | Đồng ý | 🟢 Gộp 4 tiểu bước thành một đoạn ý niệm |

---

## 2. Nhóm A — Flaw lập luận / diễn giải (ưu tiên cao nhất)

### A1. Luận điểm "gánh nặng kép" có nguy cơ là strawman  🟢
- **Vấn đề gốc:** claim phần lớn KT dùng interaction representation *trực tiếp* để cập nhật state → biểu diễn gánh hai vai trò.
- **Quyết định của team:** bỏ khẳng định về literature trước; định vị lại điểm đề xuất là tầng trung gian mở ra khả năng (a) cấu trúc tương tác thành pattern sư phạm, (b) thực hiện các phép điều chỉnh sư phạm, (c) chuẩn hóa biểu diễn giữa các học sinh theo quy chuẩn sư phạm.
- **Phản biện:** hướng đúng. Chỉ cần đảm bảo mỗi trong (a)(b)(c) là thứ pipeline cũ *khó* làm — nếu không sẽ rơi lại về A1 dạng nhẹ.
- **Hướng giải quyết chốt:** 🟢 Định vị bằng năng lực mới của tầng trung gian (a/b/c), không bằng khiếm khuyết của người khác.

### A2. Định vị đóng góp bằng phủ định  🔴 (cần đổi trọng tâm)
- **Vấn đề gốc:** tự phân biệt bằng "khác PLKT / khác NSKT / khác UKT".
- **Quyết định của team:** xác định đóng góp chính là **evidence-centered** và **pedagogical-aligned**.
- **Phản biện / kiểm chứng (đã tinh chỉnh theo team):** trọng tâm novelty là **pedagogical-aligned soft-rule** — đúng. Nhưng **KHÔNG được nói PLKT/S²KT "chồng lấn" hay "chiếm" phần của đề xuất** (framing bi quan ở vòng trước là sai). Có điểm phân biệt cụ thể với *từng* bài:
  - **vs PLKT — khác ở *nguồn* của phân phối.** PLKT dùng evidence+pattern theo hướng **look-up**: biểu diễn phân phối được tra ra *trực tiếp từ lịch sử tương tác*. Đề xuất **diễn giải phân phối từ biểu diễn ẩn của bước-học đã mô hình hóa lịch sử** → phân phối của đề xuất mang **giá trị tri thức cao hơn** (là sản phẩm của quá trình mô hình hóa, không phải bảng tra từ interaction thô).
  - **vs S²KT — khác ở *đối tượng* được phân phối (KHÔNG phải "assumption chưa bảo chứng").** ⚠️ Đọc kỹ bản gốc: S²KT *không* ngầm giả định hai không gian tương đồng — §3.4.1 nêu thẳng "assumption of independence impedes the model" như **vấn đề cần giải**, và §3.4.2 (eq 17–18) dựng **cross-attention học được** (W_Q, W_K, query bằng hidden state h_t, train end-to-end) làm **cầu nối có chủ đích** giữa hai không gian. Đừng công kích điểm này — nó được xử lý tường minh. Khác biệt *đúng* nằm ở **cái gì được phân phối**: S²KT phân phối trên **KC như thực thể ngữ nghĩa** (Gaussian centroid + uncertainty của khái niệm, pretrained bằng triplet loss trên semantic manifold); đề xuất phân phối trên **learning pattern/evidence của bước học** rồi áp **soft-rule sư phạm** lên đó. S²KT không có tầng điều chỉnh sư phạm bằng luật — nó dùng diffusion + attention học được. Đó mới là chỗ đề xuất khác.
- **Hướng giải quyết chốt:** 🔴→🟡 Định vị bằng câu **khẳng định**, nhấn vào điểm phân biệt cơ chế (không phải "khác về tên"):
  - vs PLKT: *"PLKT tra phân phối trực tiếp từ lịch sử tương tác; đề xuất diễn giải phân phối từ biểu diễn ẩn của bước học → phân phối mang giá trị tri thức cao hơn."*
  - vs S²KT: *"S²KT phân phối trên KC như thực thể ngữ nghĩa (centroid + uncertainty của khái niệm) và bắc cầu giữa các không gian bằng cross-attention học được; đề xuất phân phối trên learning pattern của bước học và điều chỉnh bằng soft-rule sư phạm — khác ở đối tượng phân phối và ở tầng luật, không phải ở cách bắc cầu."*
  - Đóng góp trung tâm khẳng định: **áp soft-rule sư phạm lên learning pattern liên-học-sinh trong không gian phân phối** — phần cả hai bài đều không có.

### A3. Over-claim "intrinsic interpretability by construction"  🟢
- **Vấn đề gốc:** claim diễn giải có được *tự nhiên* và *chắc chắn*.
- **Quyết định của team:** hạ thành "*có thể trích xuất attribution nội tại → mở khả năng phát triển cơ chế interpretable, tối thiểu ở mức diễn giải dự đoán (model-level)*".
- **Phản biện:** đúng mức phòng thủ được (trọng số ≠ giải thích faithful — dòng "Attention is not Explanation", Jain & Wallace 2019).
- **Hướng giải quyết chốt:** 🟢 Giữ nguyên cách hạ giọng của team.

### A4. Neo vào khung lý thuyết sẵn có  🟡
- **Vấn đề gốc:** trình bày góc nhìn "interaction → evidence → update" như trực giác riêng.
- **Quyết định của team:** lập luận theo ECD + formative assessment; mở rộng khung lý thuyết "theo dõi theo nhóm mẫu hình" thay vì cá thể đơn lẻ; liên kết giá trị của soft-rule so với hard-rule / tự học.
- **Phản biện / kiểm chứng:** ✅ **Đã đọc bản gốc ECD (Mislevy et al. 2003) và formative (Black & Wiliam 2009) — neo được, mạnh hơn dự kiến.** ECD cho đúng bộ khung: proficiency "*cannot be measured directly; must be inferred from incomplete evidence*"; Measurement Model = "*weight and direction of evidence*"; Assembly Model = "*How Much Do We Need to Measure It*". Formative cho động cơ feedback/alignment (Ramaprasad 3 quá trình; học = "closing the gap"). **Nhưng "theo dõi theo nhóm mẫu hình": không tìm thấy trong cả hai nguồn** — ECD/formative đều nói về *cá thể* learner. Đừng gán "nhóm mẫu hình" cho ECD/formative.
- **Hướng giải quyết chốt:** 🟢 (đã có citation) Neo A4 vào ECD (evidence-centered, Student/Evidence/Assembly model) + formative (gap-closing). "Nhóm mẫu hình liên-học-sinh" phát biểu như **lựa chọn thiết kế có động cơ của team**, KHÔNG gán cho ECD/formative (hai nguồn này là về cá thể). Soft-rule vs hard-rule: có tiền lệ (KeenKT contrastive).

---

## 3. Nhóm C-khái niệm — kiến trúc nhưng IN-SCOPE (phát biểu cho đúng)

### C1. Tách 4 không gian không được cưỡng chế bởi cơ chế nào  🟢
- **Vấn đề gốc:** "single responsibility" của 4 space là danh nghĩa; train end-to-end không đảm bảo mỗi space giữ đúng vai trò.
- **Quyết định của team:** hạn chế claim 4 space *chắc chắn* mang đúng 4 vai trò; phát biểu framework là *aim* đến điều này. Compromise: nếu cần nêu cơ chế → hướng auxiliary loss / train theo giai đoạn; khả thi nhất cho team là **per-module contrastive learning**.
- **Phản biện / kiểm chứng:** đúng scope. **Per-module contrastive có tiền lệ:** KeenKT (2512.18709) dùng distributional contrastive loss trong KT.
- **Hướng giải quyết chốt:** 🟢 Phát biểu tách vai trò là *aim*, không phải bảo đảm; nêu per-module contrastive như hướng khả thi (có precedent để dẫn).

### C2. Distribution Space có thể là tầng thừa  🟡
- **Vấn đề gốc:** nếu không có phép toán nào *chỉ* phân phối làm được, tầng này = vòng "vector → tham số → vector", cắt được.
- **Quyết định của team:** không gian phân phối cho phép **phép gộp phần tử** (xây pattern, Module 2) và **phép điều chỉnh sư phạm** (soft-rule, Module 2) *well-defined*. Argument: trên vector (a) phép gộp không mang giá trị toán học rõ ràng, (b) alignment không well-defined bằng việc đổi statistics descriptor của phân phối.
- **Phản biện / kiểm chứng:** argument bán được. **Nhưng ĐÃ có ba tiền lệ dùng phép toán phân phối trong KT:** UKT (Wasserstein), KeenKT (NIG-distance), và **S²KT (Bhattacharyya distance để lan truyền + GMM)**. → "lên không gian phân phối" **tự nó chắc chắn không phải novelty**. Đặc biệt S²KT đã dùng đúng lý lẽ "closed-form properties of Gaussians" cho các phép toán → team không thể lấy chính lý lẽ đó làm điểm mới. Sẽ bị hỏi: *"pooling/attention trên vector cũng định nghĩa được, mà phép toán phân phối thì S²KT/UKT đã làm — cái mới của bạn ở đâu?"*
- **Hướng giải quyết chốt:** 🟡 (1) Câu chữ **không được ngụ ý "phân phối = mới"** — novelty là *dùng nó cho gộp-thành-pattern-sư-phạm + soft-rule alignment*. (2) Chuẩn bị nêu **một tính chất cụ thể** phân phối cho mà vector-pooling không cho (ví dụ: phép gộp giữ được cả vị trí lẫn độ phân tán/độ tin cậy như đại lượng có đơn vị; khoảng cách sư phạm giữa pattern đo bằng divergence có nghĩa). (3) **Gắn với A2-vs-PLKT:** điểm mạnh của tầng phân phối ở đây là nó *diễn giải từ biểu diễn ẩn của bước học* (mang giá trị tri thức) chứ không tra thẳng từ lịch sử như PLKT — đây là câu trả lời tự nhiên cho "vì sao cần tầng phân phối".

### C6 (mặt giả định). Phụ thuộc knowledge graph  🟢 (có hai trường phái, cả hai đều có chỗ dựa đã xác minh)
- **Vấn đề gốc:** Prerequisite/Neighbor Operator đòi hỏi KG tin cậy; benchmark phổ biến chỉ có nhãn KC.
- **Định vị lại (theo team):** đây KHÔNG phải chuyện "trích S²KT để chống lưng cho giả định KG". Có **hai trường phái**, team chọn một tùy mức độ muốn phức tạp hóa proposal:
  - **(a) Topology KG là well-formed / cho trước.** Chỗ dựa đã xác minh: **CMDKT (Xia et al., Array 2025, ASSIST09 + DBE-KT22).** ✅ Đã đọc bản gốc: CMDKT sinh quan hệ tiên quyết bằng **ChatGPT-4 rồi cho chuyên gia (2 toán + 2 CSDL) kiểm định** → tồn tại quy trình xây KG được kiểm chứng, hợp lệ về sư phạm. Team đi hướng này nếu **không muốn phức tạp hóa** proposal: KG là đầu vào cho trước, có tiền lệ về cách dựng.
  - **(b) Topology KG là learnable.** Chỗ dựa: **S²KT (WWW'26).** ✅ Đã đọc bản gốc: vì *"explicit dependency annotations are unavailable"*, S²KT tự học quan hệ KC từ performance consistency + graph-walking. Đây là **compromise citation** — chỉ nhắc đến *nếu cần*, để lập luận "tồn tại giải pháp learnable" rồi **đặt concrete rằng nó nằm ngoài research scope của bài** và dẫn vào future work.
- **Phản biện / kiểm chứng:** ✅ Cả hai trường phái đều có precedent gốc đã đọc. Rủi ro 🔴 ở vòng trước (S²KT là bằng chứng ngược) **được hóa giải** khi framing lại đúng: S²KT không phải chỗ dựa cho (a), nó là *đại diện của trường phái (b)*. Chỉ còn một điều phải giữ kỷ luật: **không** phát biểu "quan hệ sư phạm là well-formed" như một chân lý phổ quát — mà là "với KG được dựng và kiểm định như CMDKT, ta xem topology là đáng tin trong phạm vi bài".
- **Hướng giải quyết chốt:** 🟢
  - **Mặc định (a):** trích CMDKT làm tiền lệ cho KG cho-trước-và-kiểm-định; phát biểu KG là đầu vào trong scope, giữ proposal gọn.
  - **Nếu reviewer ép về tính learnable:** rút compromise (b) — thừa nhận S²KT chứng minh KG học được, nhưng khoanh rõ ngoài scope + future work. Không để (b) biến thành nghĩa vụ phải-học-KG.

---

## 4. Nhóm B — Quyết định chưa chốt, trình bày ở mức KHÁI NIỆM

### B1. Định nghĩa thao tác của "evidence"  🟢
- **Vấn đề gốc:** khi `h_t = Φ(z_t)`, "evidence" chỉ là tên khác của vector đã chiếu.
- **Quyết định của team:** "evidence" = biểu diễn phân phối ở Module 2 *sau khi* đã gộp thành pattern và điều chỉnh sư phạm — mang tính đơn vị sư phạm cụ thể, aligned với quy tắc sư phạm, đồng bộ ngữ nghĩa giữa học sinh.
- **Phản biện:** 🟢 **Đây là câu trả lời tốt cho B1** và còn tách bạn khỏi PLKT (evidence của bạn là *sản phẩm hậu-alignment*, không phải embedding thô). **Neo được vào ECD:** ECD phân biệt *Observable Variables* (dữ liệu thô từ work product) với *Measurement Model* (tổng hợp evidence thành weight/direction về Student Model) — định nghĩa "evidence hậu-alignment" của team ánh xạ tự nhiên vào tầng Measurement Model, không phải tầng observable thô. Dùng được để chống lưng.
- **Hướng giải quyết chốt:** giữ định nghĩa hậu-alignment, neo vào tầng Measurement Model của ECD. **Nhất quán thuật ngữ:** không gọi biểu diễn tiền-alignment (`Φ(z_t)`) là "evidence" ở chỗ khác.

### B2. Ngữ nghĩa distribution space (mức khái niệm)  🟢
- **Quyết định của team:** không nêu dạng phân phối cài đặt; chỉ giải thích mục tiêu/động lực dùng không gian phân phối.
- **Hướng giải quyết chốt:** 🟢 Đúng scope. (Gắn với C2: động lực phải chỉ ra được tính chất phân phối mang lại.)

### B3. Liệt kê "nguyên lý sư phạm" dự kiến  🟡
- **Quyết định của team:** liệt kê dạng "for example / including but not limited to": tính singleton của mastery; topology khái niệm (đồng khái niệm, tiên quyết, ứng dụng, nâng cao); guessing/slipping; quan hệ temporal (kèm ý "cần ≥n tương tác mới tự tin ước lượng mastery").
- **Phản biện / kiểm chứng:** phần lớn có nền tảng. **"Cần ≥n tương tác": không tìm ra "định luật sư phạm" có tên;** gần với **độ tin cậy đo lường (measurement reliability) / số item tối thiểu** trong psychometrics và ngưỡng mastery của BKT hơn.
- **Hướng giải quyết chốt:** 🟡 Giữ danh sách dạng mở. **Phát biểu "≥n tương tác" như giả định về độ tin cậy thống kê của ước lượng**, KHÔNG gán tên "lý thuyết sư phạm" khi chưa có citation.

---

## 5. Bảng tổng — trạng thái & việc còn lại

| # | Vấn đề | Trạng thái | Việc còn lại trước khi viết |
|---|---|---|---|
| S1–S3, S5 | Cắt chi tiết cài đặt | 🟢 | — |
| S4 | Bỏ công thức mastery | 🟡 | Thêm 1 câu "forgetting để mở" |
| A1 | Strawman gánh nặng kép | 🟢 | Đảm bảo (a/b/c) là năng lực pipeline cũ khó có |
| **A2** | Định vị đóng góp | 🟡 | Novelty = pedagogical-aligned soft-rule. **KHÔNG nói PLKT/S²KT chồng lấn** — điểm phân biệt cơ chế: vs PLKT (phân phối *diễn giải từ biểu diễn ẩn*, không look-up từ lịch sử) · vs S²KT (khác *đối tượng phân phối*: KC-thực-thể vs learning-pattern, + tầng soft-rule; **KHÔNG** công kích cách bắc cầu — S²KT dùng cross-attention học được có chủ đích) |
| A3 | Over-claim interpretability | 🟢 | — |
| A4 | Neo lý thuyết | 🟢 | Neo ECD/formative (đã có bản gốc); "nhóm mẫu hình" = design choice của team, KHÔNG gán cho ECD/formative |
| C1 | Tách vai trò | 🟢 | Nêu per-module contrastive (dẫn KeenKT) |
| **C2** | Distribution space thừa? | 🟡 | **Nêu 1 tính chất cụ thể phân phối cho mà vector không; 3 tiền lệ (UKT/KeenKT/S²KT) đã dùng phép toán phân phối → không ngụ ý "phân phối = mới"** |
| **C6** | Phụ thuộc KG | 🟢 | Hai trường phái, đều có precedent gốc: **(a) KG well-formed → CMDKT** (LLM + chuyên gia kiểm định; mặc định, giữ gọn) · **(b) KG learnable → S²KT** (compromise, khoanh ngoài scope + future work) |
| B1 | Định nghĩa evidence | 🟢 | Neo vào Measurement Model của ECD; nhất quán thuật ngữ |
| B2 | Ngữ nghĩa distribution | 🟢 | — |
| B3 | Nguyên lý sư phạm | 🟡 | "≥n tương tác" = giả định độ tin cậy, không gán tên lý thuyết |

**Việc rủi ro cao nhất còn lại (làm trước):**
1. **A2 (🟡):** viết cho đúng điểm phân biệt *cơ chế* với PLKT (phân phối diễn giải từ biểu diễn ẩn, không look-up) và S²KT (khác *đối tượng phân phối*: KC-thực-thể vs learning-pattern, + tầng soft-rule). ⚠️ **Đừng công kích cách S²KT bắc cầu hai không gian** — nó dùng cross-attention học được có chủ đích (§3.4.2), không phải giả định ngầm; công kích điểm này là sai với bản gốc và dễ bị reviewer bắt.
2. **C2 (🟡):** ba tiền lệ (UKT/KeenKT/S²KT) đã dùng phép toán phân phối → nêu tính chất cụ thể phân phối mang lại, không lấy "dùng phân phối" làm điểm mới. Gắn với A2-vs-PLKT (giá trị tri thức của phân phối cao hơn).
3. **C6 đã hạ về 🟢** sau khi có CMDKT (trường phái a) — chỉ cần giữ kỷ luật: chọn (a) mặc định, (b) chỉ rút khi bị ép, và không phát biểu "quan hệ sư phạm well-formed" như chân lý phổ quát.

*Nguồn: `review_proposal_KT.md` v2 + vòng phản biện quyết định team (`pasted-text-2026-07-10`) + đọc bản gốc 4 PDF (S²KT WWW'26, CMDKT Array 2025, ECD Mislevy 2003, formative Black & Wiliam 2009) + tinh chỉnh định vị A2/C6 theo phản hồi team. Citation KT khác đã xác minh trên arXiv.*
