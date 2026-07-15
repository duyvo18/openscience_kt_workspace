# Review KT — Phần DEFER (lưu trữ, phân tích sâu khi vào cài đặt / full paper)

**Tài liệu nguồn:** `proposal_final.md` — *Distributional Pedagogical Alignment framework*
**Phạm vi file này:** các vấn đề **chưa cần xử lý cho short paper**, chỉ xuất hiện khi đi vào cài đặt. Lưu lại để phân tích thêm khi nghiên cứu tiến vào giai đoạn hiện thực hóa.
**File đi kèm:** `review_KT_in_scope.md` (phần làm ngay cho short paper).

> **Lưu ý quản lý:** mỗi mục ở đây là một *rủi ro thiết kế mở* hoặc *flaw kiến trúc* cần một **quyết định thiết kế cụ thể** để giải quyết — không thể xử lý bằng cách viết lại. Với short paper, nguyên tắc là **không phát biểu như đã giải quyết** và **không vẽ chi tiết cài đặt phơi ra các mâu thuẫn này** (đặc biệt C4).

---

## Tài liệu tham chiếu đã kiểm chứng (arXiv, đọc abstract gốc)

| Nhãn | arXiv | Tên đầy đủ | Liên quan đến mục nào |
|---|---|---|---|
| PLKT | 2605.09369 | Explainable KT via Probabilistic Embeddings and Pattern-based Reasoning | Beta + evidence + pattern (novelty A2/C2) |
| UKT | 2501.05415 | Uncertainty-aware Knowledge Tracing | Wasserstein self-attention trên phân phối (tiền lệ C2) |
| KeenKT | 2512.18709 | Knowledge Mastery-State Disambiguation for KT | NIG + distributional contrastive loss (tiền lệ C1/C3) |
| Dynamic LENS | 2407.17427 | Uncertainty-preserving deep KT with state-space models | VAE + Bayesian state-space, KHÔNG có attention — bài riêng, không phải UKT |
| MemoryKT | 2508.08122 | Integrative Memory-and-Forgetting Method for KT | Forgetting qua temporal VAE (tham chiếu C4/decay) |
| **S²KT** | WWW'26, DOI 10.1145/3774904.3793039 | Wu et al., Semantic-aware Structured Gaussian Distributions | ✅ **Đã đọc bản gốc.** Gaussian embedding + Bhattacharyya propagation + GMM attention + heterogeneous forgetting; **tự học quan hệ KC vì "explicit dependency annotations are unavailable"** → bằng chứng NGƯỢC cho giả định KG của team (C6). Cũng là tiền lệ decay (C4). |

| **CMDKT** | Array 28 (2025) 100523 | Xia et al., Course Map + Difficulty enhanced KT | ✅ **Đã đọc bản gốc.** KG well-formed: ChatGPT-4 sinh quan hệ tiên quyết + 2+2 chuyên gia kiểm định; GRU topology embedding; difficulty 2 mức. Đại diện trường phái KG-cho-trước (C6-a). |

**Nền lý thuyết giáo dục (đã đọc bản gốc):** ECD (Mislevy, Steinberg & Almond 2003) — Student/Evidence/Task/Assembly model; formative assessment (Black & Wiliam 2009) — gap-closing, Ramaprasad 3 quá trình. Chi tiết trong `review_KT_in_scope.md §0`.

---

## C3. Alignment mềm bị tối ưu hóa "vô hiệu hóa" — *động lực huấn luyện*

**Vấn đề:** nếu "pedagogical alignment" là số hạng regularization mềm với trọng số học được, tối ưu hóa end-to-end có động cơ đẩy trọng số đó về 0 để tự do giảm loss dự đoán. Khi đó "alignment" chỉ còn trên tên gọi.

**Vì sao reframing không cứu được:** đây là hành vi của quá trình huấn luyện, không phải cách viết. Muốn luật thực sự **ràng buộc** (binding) phải có thiết kế loss/constraint cứng hơn.

**Cần quyết khi vào cài đặt:**
- Align là *prior cứng* (projection lên tập khả thi, constraint) hay *hint mềm* (regularizer)?
- Nếu mềm: cơ chế nào đảm bảo nó không bị học bỏ qua (annealing, trọng số cố định, curriculum, tách pha train)?
- Đo lường: có cách kiểm tra hậu nghiệm rằng luật *thực sự* ảnh hưởng đến biểu diễn (ablation bật/tắt align, đo dịch chuyển phân phối)?

---

## C4. Cập nhật mastery thuần cộng, không decay, không chặn — *mâu thuẫn với chính claim sư phạm*

**Vấn đề:** `M_{t+1} = M_t + ΔM` không có số hạng quên (forgetting), mastery không bị chặn, có thể trôi vô hạn. Điều này **trái với tham vọng "hiện thực sư phạm"** của chính đề xuất (forgetting là nguyên lý sư phạm cốt lõi).

**Hệ quả cho short paper:** **không vẽ công thức này ra** — vẽ ra là tự phơi mâu thuẫn. Đã ghi trong `review_KT_in_scope.md §1`.

**Cần quyết khi vào cài đặt:**
- Dạng cập nhật có decay/gating: `M_{t+1} = g_t ⊙ M_t + ΔM` (với `g_t` là cổng quên phụ thuộc thời gian/ngữ cảnh).
- Hoặc chuẩn hóa/chặn mastery về `[0,1]` (sigmoid, clamp, hoặc tham số hóa qua không gian bị chặn).
- Cân nhắc mô hình quên có cơ sở (time-since-last-practice, đường cong quên Ebbinghaus) để nhất quán với trục evidence-centered.
- **Tiền lệ:** S²KT (WWW'26) mô hình **heterogeneous forgetting trajectories** gắn với KC complexity; MemoryKT (2508.08122) mô hình forgetting cá nhân hóa qua temporal VAE. Khi thiết kế decay, đối chiếu hai bài này để không lặp lại và để định vị khác biệt.

---

## C5. Rủi ro leakage trong vòng phụ thuộc Mastery Memory — *correctness*

**Vấn đề:** Mastery Memory là *input* của Module 1 (qua localized mastery) rồi lại được *cập nhật* ở Module 3. Nếu chỉ số thời gian sai — dùng mastery *đã cập nhật bằng response tại `t`* để dự đoán chính response `t` — đó là **data leakage**, làm hỏng toàn bộ kết quả.

**Vì sao không thuộc phạm vi lập luận:** đây là ràng buộc luồng dữ liệu / tính đúng, không liên quan diễn giải. Nhưng **phải đúng tuyệt đối** ở full paper, nếu không mọi con số AUC đều vô nghĩa.

**Cần đảm bảo khi vào cài đặt:**
- Ghi rõ timeline: đọc `M_t` (trạng thái *trước* khi thấy response `t`) để dự đoán `y_t`; chỉ cập nhật lên `M_{t+1}` *sau* khi đã dùng cho dự đoán.
- Localized mastery cho câu mục tiêu phải lấy từ trạng thái *trước* interaction mục tiêu.
- Kiểm thử leakage: shuffle nhãn tương lai / kiểm tra mô hình không thể "nhìn trước" — AUC không được cao bất thường.

---

## C6 (mặt cơ chế). Xử lý khi knowledge graph thiếu / nhiễu — *giả định dữ liệu*

**Đã thừa nhận giả định trong short paper** (xem in-scope C6). Phần cơ chế để defer:

**Vấn đề:** Prerequisite Operator và Neighbor Operator là thành phần kiến trúc *đòi hỏi* một KG có quan hệ tiên quyết/láng giềng đáng tin. Phần lớn benchmark KT (ASSISTments, EdNet, Junyi) chỉ có nhãn KC, không có prerequisite graph chuẩn. Nếu tự suy ra, graph đó là một nguồn nhiễu/giả định nữa và có thể quyết định luôn kết quả.

**Cần quyết khi vào cài đặt:**
- Nguồn graph: có sẵn từ dataset? tự học (từ co-occurrence, từ transition)? do chuyên gia? 
- Chế độ suy biến: khi không có graph, hai operator này tắt hay thay bằng gì? Framework có còn chạy được không?
- Độ nhạy: đánh giá kết quả nhạy thế nào với chất lượng graph (ablation graph thật vs. graph nhiễu vs. không graph).

---

## Chi tiết Nhóm B khi vào cài đặt (mở rộng từ mức khái niệm)

Short paper chỉ cần nêu ý niệm; đây là các quyết định thao tác cần chốt khi hiện thực hóa.

### B1-impl. Định nghĩa thao tác của "evidence"
- Tiêu chí phân biệt `h_t` (evidence) với `z_t` (interaction representation) ở mức đo được — evidence có tổng hợp được dạng đóng không? có bất biến theo học sinh không?
- Nếu `h_t` chỉ là projection tuyến tính của `z_t`, cần lý do vì sao không gộp hai bước.

### B2-impl. Ngữ nghĩa và dạng của distribution space
- Chọn dạng phân phối (Beta? Dirichlet? Gaussian?) và **ý nghĩa của tham số** khi *không* biểu diễn belief/uncertainty.
- Gắn chặt với C2: phép toán nào trên phân phối là lý do tồn tại của tầng (product/mixture/moment-matching/KL-alignment).

### B3-impl. Đặc tả các "nguyên lý sư phạm"
- Với mỗi nguyên lý đã liệt kê (forgetting, đơn điệu, tiên quyết, spacing, guessing/slipping): dạng toán của luật, áp lên đại lượng nào, cứng hay mềm.
- Xử lý xung đột prior ↔ dữ liệu: nhiều nguyên lý bị dữ liệu thật vi phạm (mastery *có* giảm; có guessing/slipping) → luật là ràng buộc cứng hay chỉ hint, và điều gì xảy ra khi dữ liệu mâu thuẫn.

---

## Kế hoạch đánh giá interpretability (mở rộng từ A3, defer chi tiết)

Short paper chỉ nêu *ý định*. Khi vào thực nghiệm cần một khung validate thật, vì KT gần như không có ground-truth cho lời giải thích:

- **Synthetic data với generative process đã biết** — cách mạnh nhất để có ground-truth cho attribution; kiểm tra Pattern→KC và KC→Prediction có khớp cơ chế sinh dữ liệu.
- **Khớp cấu trúc prerequisite đã biết** trên dataset có graph tin cậy.
- **Faithfulness test bằng perturbation** — nhiễu loạn input/trọng số và đo lời giải thích có phản ánh đúng thay đổi output không.
- **Tính hợp thành (composition):** attribution cục bộ ở mỗi mũi tên (Historical → Patterns → Pattern-KC → State → KC-Prediction) không tự động ghép thành giải thích end-to-end trung thực — cần kiểm chứng riêng.

---

## Đánh đổi cần theo dõi (không phải flaw, nhưng ghi để không quên)

- **Modularity ↔ hiệu năng:** chuỗi module tuần tự tạo bottleneck; mỗi lần bàn giao giữa các không gian có thể mất thông tin và *làm giảm* AUC so với end-to-end. Cần giả thuyết về vị trí đánh đổi và ablation tương ứng.
- **Cấu trúc cố định ↔ expressiveness:** operator cố định (nhất là Same-KC) dễ làm mất thứ tự/recency mà mô hình chuỗi tự do giữ được. Đo mất mát này khi có dữ liệu.

---

*Tổng hợp từ `review_proposal_KT.md` v2. Các mục ở đây là rủi ro thiết kế mở — mỗi mục cần một quyết định kiến trúc/cài đặt cụ thể, không giải được bằng viết lại.*
