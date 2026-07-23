# Danh sách nội dung cần revision — Paper DPA (ACOMPA 2026)

> **Loại paper:** Short paper / Position — Vision (chưa có thực nghiệm).
>
> **Kết luận review:** Weak Accept — **có điều kiện**, phải hoàn thành 6 mục MAJOR trước camera-ready.
>
> **Nguyên tắc chung:** 6 mục MAJOR (J1–J6) là **blocking**. 8 mục MINOR (N1–N8) là polish cho camera-ready. Phần PoC là tùy chọn nhưng **rất nên làm**.

---

## A. MAJOR — Blocking (bắt buộc, ưu tiên cao nhất)

### J1 — Đặc tả cụ thể cơ chế pedagogical alignment (ít nhất ở mức sơ đồ)
Đây là **lỗ hổng trung tâm** của paper: alignment chính là đóng góp cốt lõi nhưng chưa nói rõ nó là *gì* (loss regularization? projection lên constraint set? auxiliary objective? constraint layer?).

- [ ] Đưa **ít nhất 1 instantiation cụ thể** (phương trình hoặc sơ đồ chính xác) cho **tối thiểu 1 principle**: tính đại lượng nào, nó chèn vào chỗ nào trong computation graph, và differentiable qua đâu.
- [ ] Nêu rõ cách **ngăn alignment xung đột với prediction objective** (ví dụ: weighting, stop-gradient, curriculum, hoặc nêu rõ trade-off).
- [ ] Với **guess/slip**: định nghĩa differentiable cách phát hiện "outlier response" và cách điều chỉnh expectation/variance của distribution để "hấp thụ" nó. Sau đó **so sánh với formalism cổ điển** — distributional treatment thêm được gì so với guess/slip probabilities của BKT hay asymptote 3PL/4PL của IRT?
- [ ] Với **monotonicity**: chỉ ra cách 2 rule viết tay trở thành soft/differentiable constraint mà không sụp về threshold/hyperparameter cứng.

### J2 — Làm sắc nét novelty của "explicit evidence-interpretation stage"
Vấn đề: Các model memory- (DKVMN), attention- (AKT), gate-based (LPKT) đã có intermediate representation có thể đọc như evidence interpretation; model distributional (UKT/KeenKT/S2KT) đã làm uncertainty-weighted update. Nói "tách nó thành explicit alignment stage" là chưa đủ dày.

- [ ] Đưa bảng **side-by-side**: chỉ ra chỗ một model tiêu biểu **trộn lẫn (entangle)** evidence interpretation vào mastery update, so với chỗ DPA **tách** nó ra — và quan trọng nhất là **sự tách đó thay đổi gì về hành vi/prediction**.
- [ ] Làm rõ điểm khác biệt với **PLKT** cụ thể: sliding window theo thời gian của PLKT bản thân đã là một pooling operator, nên "chúng tôi dùng pooling" **không** phải điểm phân biệt. Nêu khác biệt (tổ chức theo pedagogical operator + align với principle) dưới dạng **một case mà DPA và PLKT cho prediction khác nhau**.
- [ ] Làm rõ việc đặt **distribution *trước* pooling và alignment** mang lại gì so với UKT/KeenKT/S2KT (vốn đã giả định dispersion/uncertainty).

### J3 — Diễn đạt lại monotonicity thành hypothesis kiểm chứng được + hòa giải với forgetting
Vấn đề: Gọi "time-gap = forgetting" là "bias theo dataset" đi ngược lại literature về forgetting-curve / spacing-effect và ngược với forgetting gate tường minh của LPKT (một baseline). Một prior monotonicity phủ khắp có nguy cơ **triệt tiêu tín hiệu forgetting thật** — đổi bias này lấy bias khác.

- [ ] Trình bày monotonicity như **một hypothesis cần validate**, không phải axiom thiết kế.
- [ ] Thêm **ablation tách "debiasing" khỏi "signal suppression"** (ví dụ: so sánh có/không prior monotonicity trên dataset có spacing mạnh vs yếu).
- [ ] **Hòa giải tường minh với cơ chế forgetting của LPKT** ngay trong phần text, không chỉ liệt kê LPKT như baseline.

### J4 — Chứng minh (hoặc hạ cấp) "structural consistency across learners"
Vấn đề: Một operator cố định áp lên các history khác nhau sẽ ra distribution khác nhau; dùng chung operator **không** đảm bảo pattern **so sánh được về mặt ngữ nghĩa** giữa các learner — đây chính là điều làm cho "một principle cho mọi history" trở nên well-posed.

- [ ] Hoặc **lập luận lý thuyết** cho tính consistency này, hoặc **hạ cấp rõ ràng** thành working assumption/hypothesis cần test.
- [ ] Đề xuất một **consistency metric cụ thể** (ví dụ: độ khớp của geometry trong pattern-space giữa các cohort learner) để implementation paper có thể report.

### J5 — Bịt các lỗ hổng của evaluation plan
- [ ] **J5a — Graph confound.** Với dataset flat-tag, graph `G` được *construct* (CMDKT), nên gain có thể đến từ graph được inject chứ không phải DPA. Thêm ablation **giữ graph cố định + bỏ alignment (và ngược lại)** để tách riêng 2 đóng góp.
- [ ] **J5b — Falsifiability của hypothesis.** H1 gần như không thể bác bỏ như đang viết — **định nghĩa success threshold**. Với **H2** (gain khi data khan hiếm): đưa lập luận vượt quá "prior đóng vai regularizer" + nêu protocol dựng chế độ scarce (subsample learner? cold-start?). Với **H4**: đặc tả graph-noise model dùng để test graceful degradation.
- [ ] **J5c — Statistical rigor.** Cam kết **multiple seeds + report variance + significance testing** trên toàn bộ baseline set (chuẩn kỳ vọng của cộng đồng pyKT để đỡ cho claim "competitive").
- [ ] **J5d — Reproduction burden.** Nhiều baseline 2024–2026 (UKT, KeenKT, S2KT, NSKT, PLKT, MemoryKT) **có thể chưa có trong pyKT**; thừa nhận công sức tái hiện và nêu cách đảm bảo **split/preprocessing đồng nhất**.

### J6 — Bổ sung thảo luận về scalability/complexity và cold-start
Vấn đề: Nhiều pooling operator + distributional projection + per-KC recurrent memory `M ∈ R^{C×d}` có thể tốn kém trên dataset nhiều KC (ví dụ XES3G5M); pattern construction cũng phụ thuộc độ dài history.

- [ ] Thêm một đoạn về **complexity kỳ vọng so với AKT/DKVMN**, chi phí per-step pooling trên toàn history ở quy mô ASSISTments/Algebra, và **hành vi cold-start / short-history** (điều mà pattern construction phụ thuộc vào).

---

## B. Tùy chọn nhưng RẤT NÊN làm

- [ ] **Proof-of-concept tối thiểu** trên 1 dataset graph-native (ví dụ XES3G5M hoặc Junyi), chỉ implement riêng lớp alignment. Một kết quả duy nhất cũng đồng thời giải quyết J1, J2, J3 và đẩy paper từ "borderline" lên gần "clear accept". **Nếu venue kỳ vọng có kết quả sơ bộ thì mục này gần như bắt buộc.**

---

## C. MINOR — Polish cho camera-ready

- [ ] **N1 — Notation.** Thống nhất `r_i ∈ {0,1}` vs `r_t` và `q_i` vs `q_t`; phân biệt rõ chữ **"evidence"** (dùng theo nghĩa thông thường ở Introduction vs nghĩa đặc biệt *post-alignment* ở Sec. III). Định nghĩa term sớm và một lần duy nhất.
- [ ] **N2 — Trùng lặp output interpretability.** Phân biệt **pattern-to-KC gating** (Module 3) với **KC-to-prediction contribution** (Module 4); và lưu ý caveat ở Section V làm giảm nhẹ cách trình bày trước đó về chúng như "architectural feature" — chỉnh cho nhất quán.
- [ ] **N3 — Cách diễn đạt contribution.** "Perspective"/"architecture" phần lớn là ECD áp vào KT, và "core mechanism" chưa được đặc tả — **làm mềm abstract/introduction** để khớp mức độ đặc tả hiện tại.
- [ ] **N4 — ECD như ràng buộc thực chất.** Nêu **ít nhất 1 design decision xuất phát *tất yếu* từ ECD** và sẽ "sai" trong model không-ECD; nếu không, mapping chỉ như một cách đổi tên hấp dẫn.
- [ ] **N5 — Phạm vi "pedagogical principles".** Mới có 2 principle (guess/slip, monotonicity). Hoặc liệt kê không gian principle dự kiến, hoặc thu hẹp claim thành "một tập nhỏ, mở rộng được".
- [ ] **N6 — Positioning dựa trên preprint & anonymity.** Nhiều reference then chốt, gồm **closest prior work PLKT [11]**, là preprint 2026 chưa được cộng đồng kiểm chứng — nêu rõ điều này để reader tự hiệu chỉnh, và verify positioning vẫn đứng vững nếu preprint thay đổi. Đồng thời xác nhận các arXiv preprint được trích **không làm lộ danh tính (de-anonymize)** tác giả theo policy blind-review (placeholder "Author One / University Name" cho thấy double-blind).
- [ ] **N7 — Figure 1.** Layout U-shape 1→2→3→4 hơi phản trực giác — thêm gợi ý thứ tự rõ ràng. Đảm bảo đọc được ở **grayscale** và color-coding của 4 space được định nghĩa trong caption.
- [ ] **N8 — Định nghĩa "learning pattern".** Phân biệt "định nghĩa bởi aggregation operator, không phải số lượng interaction" khá tinh tế — thêm 1 ví dụ minh họa 1 dòng. Định nghĩa `d` (embedding dim) và nêu shape/vai trò của `z′` và "post-alignment object" (dù equation được để lại sau).

---

## D. Câu hỏi cần chuẩn bị trả lời (rebuttal)

1. **Cơ chế:** Alignment cụ thể là loss term, projection, hay constraint layer? Differentiable qua đâu, và giữ nó không xung đột với prediction objective bằng cách nào?
2. **Formalism cổ điển:** Xử lý distributional cho guess/slip (điều chỉnh expectation/variance) khác gì về lý thuyết so với guess/slip của BKT hay asymptote 3PL/4PL của IRT?
3. **Forgetting:** Monotonicity có thể triệt tiêu khả năng bắt forgetting thật không? Ablation nào tách "debiasing" khỏi "signal suppression", và hòa giải với forgetting gate của LPKT ra sao?
4. **Consistency:** Bằng cơ chế nào các pattern từ cùng operator trở nên *so sánh được về ngữ nghĩa* giữa các learner, và đo consistency đó thế nào?
5. **Tách confound:** Trong ablation, tách đóng góp của knowledge graph (đôi khi được construct) khỏi cơ chế alignment bằng cách nào?
6. **ECD:** Nêu 1 design decision mà ECD *bắt buộc* và model không-ECD sẽ làm sai.
7. **Chi phí:** Chi phí tính toán của 3 pooling operator ở quy mô ASSISTments/Algebra là bao nhiêu, và framework hành xử ra sao trên history rất ngắn?

---

## E. Lưu ý cần giữ nguyên (ĐỪNG làm loãng khi revise)

Đây là các điểm mạnh cả 2 reviewer đều công nhận — khi sửa cần bảo toàn:

- Theory grounding: mapping ECD (student/evidence/task) lên các không gian Knowledge/Distribution/Interaction.
- Positioning trung thực, tích hợp: taxonomy 3 theme, đối chiếu trực tiếp với PLKT và LPKT.
- Interpretability stance chín chắn: trích cả *Attention is not Explanation* lẫn *Attention is not not Explanation*, hạ attribution xuống mức behavioral consistency.
- Evaluation plan đáng tin: partition dataset theo graph availability, baseline phủ 3 theme + BKT, metric calibration-aware (ECE, NLL), ablation map 1-1 vào precondition chain, 4 hypothesis falsifiable.
- Clarity & structure: tổ chức 4-module/4-space và precondition-chain dễ theo dõi.
