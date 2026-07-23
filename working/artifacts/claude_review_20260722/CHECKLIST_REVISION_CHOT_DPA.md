# Checklist revision đã CHỐT — Paper DPA (ACOMPA 2026)
### Dùng để chỉnh `main.tex`

> **Phạm vi:** đây là bản tổng hợp **cuối cùng** sau 3 vòng review–phản hồi. Chỉ liệt kê những gì team **đã đồng ý làm**. Các mục team quyết định bỏ/omit được ghi ở **Mục F** để tránh nhầm là còn nợ.
>
> **Mục tiêu chiến lược (ràng buộc khi thực hiện):** short paper / vision, giới hạn trang + thời gian + nhân lực. Đích là **weak accept**. Nguyên tắc: **không overclaim, không tự tạo thêm vấn đề, không mở mặt trận mới.** Nếu một chỉnh sửa làm phát sinh việc lớn hơn giá trị nó mang lại → không làm.
>
> **Ký hiệu ưu tiên:** ⭐ = bắt buộc (đóng lỗ hổng review) · ✎ = polish rẻ, nên làm · 💬 = chuẩn bị cho rebuttal (không nhất thiết vào bài).

---

## A. Cơ chế cốt lõi — Alignment & Module 2 (DPA)

**A1. ⭐ Viết dạng hàm *rule-shaped* cho alignment (chốt: rule-shaped functional form, train end-to-end).**
→ *Vị trí:* Sec IV-D (Module 2), bước "Pedagogical alignment".
- Trình bày alignment là **một hàm biến đổi có dạng cố định theo luật**, chỉ **biên độ điều chỉnh (delta/β) là learnable**; ràng buộc sư phạm nằm ở *dạng hàm*, còn prediction loss chỉ fit các tham số tự do. Train **end-to-end**, **không** optimizer riêng, **không** loss phụ.
- Đưa **ít nhất 1 instantiation cụ thể** (1 phương trình + 1–2 câu) cho **một** principle — khuyến nghị dùng **difficulty–response consistency** (grounding IRT vững) hoặc **guess/slip**: nêu rõ (i) đại lượng tính từ pattern, (ii) chèn vào chỗ nào trong forward pass, (iii) tác động lên descriptor (kỳ vọng/variance) của phân phối.
- **Bỏ hoàn toàn** cách diễn đạt "tách optimizer khỏi prediction loss" (không phòng thủ được, không cần thiết).

**A2. ⭐ Cập nhật danh sách principle (thay monotonicity).**
→ *Vị trí:* Sec IV-D, danh mục principle.
- Gỡ **monotonicity** khỏi bài (xóa cả 2 rule time-gap).
- Thay bằng: **(i) Difficulty–response consistency** (điều chỉnh trọng số item trong pattern theo difficulty, nền IRT); **(ii) KC-relation consistency** (điều chỉnh phân phối tổng thể theo quan hệ giữa các KC, ví dụ prerequisite).
- ✎ Ghi rõ đây là **"một tập nhỏ, mở rộng được"** (đóng N5), tránh claim "aligns with pedagogical principles" một cách bao quát.

**A3. ⭐ Giữ guess/slip *soft*, không thành trigger cứng.**
→ *Vị trí:* Sec IV-D, principle guess/slip.
- Định nghĩa: pattern cùng KC + difficulty tương đồng nhưng có **outlier** (đang-đúng-tự-nhiên-sai hoặc ngược lại) ⇒ coi là tín hiệu guess/slip, cho mô hình **học delta phân phối mềm** khi gặp trường hợp này.
- β / điều kiện kích hoạt phải là **cổng mềm liên tục (differentiable)**, **không** phải công tắc nhị phân if-then (nếu cứng sẽ tái sinh đúng vấn đề "sụp về threshold cứng").

**A4. ✎ Đưa lập luận distributional guess/slip vào bài + tương phản với BKT.**
→ *Vị trí:* Sec IV-D hoặc Related Work.
- Lý lẽ (từ rebuttal D2): guess/slip **không phải nhãn trắng/đen** quan sát được từ dataset ⇒ biểu diễn bằng phân phối để mang **possibility/độ bất định**; **variance = độ phủ**, **kỳ vọng = giá trị mong đợi**.
- Nêu tương phản rõ: **BKT tham số hóa guess/slip là xác suất điểm cố định**; DPA để chúng là điều chỉnh *trên phân phối*, nên độ bất định của chính tín hiệu guess/slip được mang theo vào mastery update.

---

## B. Định vị & Novelty

**B1. ⭐ Argument novelty của "explicit stage" (chốt: KHÔNG làm bảng side-by-side).**
→ *Vị trí:* Sec II (Positioning) + Sec IV-A (Design Rationale).
- Trình bày giá trị của việc tách DPA thành stage explicit ở **2 ý**: (i) có thể **thêm/bớt lý thuyết sư phạm** vào model (qua pattern constructor + pedagogical alignment) thay vì chỉ để learnable; (ii) các thành phần soft-symbolic **có *tiềm năng* được trích xuất** để hiểu prediction.
- ⚠️ Với ý (ii): giữ đúng từ **"tiềm năng / potential"**. **Không** phát biểu "model của chúng tôi interpretable" — phải nhất quán với caveat Sec V (behavioral consistency, trích *Attention is not Explanation*).

**B2. ⭐ Làm sắc khác biệt DPA vs PLKT.**
→ *Vị trí:* Sec II (đoạn về closest prior work).
- (i) DPA dùng **distributional encoder từ history representation** (mang tri thức cao hơn), còn PLKT dùng **lookup table từ interaction**; (ii) **pooling + alignment mang giá trị sư phạm** cao hơn sliding-window theo thời gian.

**B3. ⭐ Lý do đặt distribution *trước* pooling/alignment.**
→ *Vị trí:* Sec IV-A hoặc IV-D bước "Distribution projection".
- Nêu: đưa representation về **không gian phân phối có nền toán** để làm **cơ sở** cho các toán tử pooling và hàm alignment (thao tác trực tiếp lên descriptor thống kê: kỳ vọng, variance). Đây là giá trị so với UKT/KeenKT/S2KT (vốn chỉ giả định dispersion/uncertainty).

---

## C. Framing ECD & Consistency

**C1. ⭐ Thêm design-decision *bắt buộc* từ ECD (đóng N4).**
→ *Vị trí:* Sec IV-A (Design Rationale).
- Câu chốt: **ECD tách "evidence model" khỏi "student model" ⇒ mastery memory `M` chỉ được cập nhật từ evidence hậu-alignment `z′`, KHÔNG bao giờ cập nhật trực tiếp từ interaction representation.** Đây là ràng buộc kiến trúc cụ thể + có thể sai (DKT/DKVMN/AKT gộp hai bước này) ⇒ chính là ý nghĩa vận hành của "explicit stage", phát sinh *tất yếu* từ ECD.
- Có thể mô tả evidence như thành phần điều chỉnh có dạng theo luật với **β learnable** (kích hoạt *mềm*). ⚠️ **Tránh gắn nhãn "neural-symbolic" như một đóng góp riêng** — NSKT [10] đã chiếm từ khóa này và nằm trong baseline set; dùng nhãn đó sẽ kéo lại gánh phân biệt DPA vs NSKT. Mô tả bằng từ **chức năng** ("soft/rule-shaped adjustment") là đủ.

**C2. ⭐ Reframe "structural consistency across learners" (đóng J4).**
→ *Vị trí:* Sec IV-A (precondition chain).
- Diễn giải lại: operator cố định áp lên history khác nhau ⇒ **phân phối khác nhau**, nhưng chúng **chia sẻ ngữ nghĩa sư phạm chung** (ví dụ: operator same-KC luôn mang cùng một ý nghĩa sư phạm dù phân phối khác).
- Phân tách rõ: **shared *coordinate*** = không gian descriptor phân phối (mean/variance); **shared *semantics*** = operator cố định.
- ⚠️ **KHÔNG bỏ rời rạc claim này** — precondition-chain ở Sec IV-A dựa vào nó; chỉ *định nghĩa lại cho gọn* để chain vẫn đứng vững.

---

## D. Evaluation Plan (Sec V)

**D1. ⭐ Hạ H1 về định tính (chốt).**
- Bỏ mọi ngưỡng định lượng (không hứa "0.01 AUC"). Phát biểu: **"competitive với baseline mạnh, cải thiện rõ nhất nơi cấu trúc KC đáng tin cậy."**
- ✎ Vẫn **report bình thường** AUC/ACC/RMSE + ECE/NLL — định tính chỉ áp cho *câu hứa*, không phải cho việc báo số.

**D2. ⭐ Thêm ablation tách graph confound (đóng J5a).**
- Với dataset có KC-topology **được construct** (`G` dựng theo CMDKT cho flat-tag): thêm ablation **giữ `G` cố định + bỏ alignment (và ngược lại)** để tách đóng góp của graph khỏi đóng góp của alignment.
- ⚠️ Vì principle **KC-relation consistency (A2-ii)** cũng phụ thuộc `G`, ablation phải bao phủ luôn phần principle dựa trên `G`.

**D3. ⭐ Đặc tả protocol cho H2 & H4.**
- **H2 (data-scarce):** dựng chế độ khan hiếm bằng **subsampling** + **cold-start** (lấy k interaction đầu).
- **H4 (graceful degradation):** graph-noise ở **mức cạnh** — edge rewrite, edge add/drop.

**D4. ⭐ Xử lý comparability mà KHÔNG cần nêu pyKT.**
- Bỏ mọi tham chiếu pyKT như một mục tiêu/nền tảng (pyKT chỉ là công cụ hỗ trợ implement baseline, không phải research goal).
- **Bắt buộc thêm 1 câu** trong Sec V: **mọi model (baseline + đề xuất) dùng chung train/valid/test split và cùng pipeline tiền xử lý** — để giữ tính so sánh công bằng.

**D5. ✎ Cập nhật danh mục ablation cho khớp principle mới.**
- Ba ablation gốc giữ nguyên (point-vector vs distribution; bỏ alignment; bật operator từng cái), nhưng cập nhật mô tả để phản ánh tập principle mới (guess/slip, difficulty-response, KC-relation) thay cho monotonicity.

---

## E. Minor / Camera-ready

- **E1. ✎ N1 — Notation:** đã sửa; rà lại lần cuối cho thống nhất `r_i`/`r_t`, `q_i`/`q_t`; dùng chữ **"evidence"** nhất quán (nghĩa post-alignment ở Sec III vs nghĩa thông thường ở Intro — định nghĩa 1 lần).
- **E2. ✎ N3 — Làm mềm abstract/intro:** vì cơ chế lõi vẫn ở mức sơ đồ, hạ giọng từ "a core mechanism" → "a proposed mechanism/design" cho khớp mức đặc tả. (Rẻ, giảm rủi ro overclaim.)
- **E3. ✎ PoC là phiên bản rút gọn:** thêm **1 câu** nói rõ kết quả sơ khởi hiện thực một **phiên bản đơn giản hóa** của framework, và liệt kê ngắn gọn phần nào được giảm bớt (gỡ mâu thuẫn "framework đầy đủ" vs "PoC cơ bản").
- **E4. ✎ N6 — Preprint & anonymity:** chấp nhận trích preprint (PLKT [11] v.v.); anonymity là chủ ý omit — giữ nguyên, chỉ đảm bảo các arXiv preprint được trích không lộ danh tính theo policy double-blind.

---

## F. Đã CHỐT BỎ / KHÔNG đưa vào bài (để không nhầm là còn nợ)

- **Monotonicity principle** — bỏ hoàn toàn (kéo theo bỏ mọi bàn luận **forgetting** trong bài).
- **J5c — multiple seeds + significance testing** — **không** đưa vào bài (hệ quả trực tiếp: H1 đã hạ về định tính ở D1; không được đặt lại ngưỡng định lượng).
- **J6 — scalability/complexity + cold-start cost** — **không** đưa vào bài.
- **Bảng side-by-side entangle-vs-explicit (J2)** — **không** làm (thay bằng argument B1).
- **N7 — Figure 1** — không chỉnh (đã chốt giữ cấu trúc; grayscale đã đảm bảo).
- **N2 — tách rõ 2 output interpretability** — không bắt buộc, vì B1 đã hạ interpretability xuống mức "tiềm năng" nên rủi ro mâu thuẫn không còn.

---

## G. 💬 Chuẩn bị cho rebuttal (không cần vào bài)

- **G1. Forgetting:** nếu bị hỏi (LPKT/MemoryKT là baseline có forgetting tường minh) → trả lời: forgetting **ngoài phạm vi** bản này, *hoặc* được **recurrent memory + uncertainty-weighting xử lý ngầm**.
- **G2. Chi phí tính toán:** nếu bị hỏi về `M ∈ R^{C×d}` trên dataset nhiều KC (XES3G5M) → chuẩn bị ước lượng complexity so với AKT/DKVMN (đã chốt không viết vào bài, chỉ thủ sẵn).
- **G3. Guess/slip vs BKT/IRT:** dùng lập luận A4 (possibility/độ bất định; variance = độ phủ, kỳ vọng = expected value).
- **G4. DPA vs NSKT:** nếu reviewer đòi phân biệt với neural-symbolic KT → nhấn: DPA đặt điều chỉnh có-dạng-theo-luật **trên không gian phân phối** của learning pattern, không phải suy luận symbolic trên nhãn.

---

## H. Thứ tự thực hiện đề xuất (tối ưu công sức)

1. **A1, A2, A3** — cốt lõi Module 2 (khóa J1/B1). Viết 1 phương trình + cập nhật danh sách principle.
2. **C1, C2** — 2 câu vào Design Rationale (đóng N4, J4; gia cố B1).
3. **B1, B2, B3** — tinh chỉnh Positioning (phần lớn là viết lại đoạn văn, không thêm thí nghiệm).
4. **D1, D2, D3, D4** — cập nhật Sec V (hạ H1, thêm ablation graph, protocol H2/H4, câu split chung).
5. **A4, E1–E4** — polish.
6. Rà **Mục F** để chắc chắn không sót phần đã bỏ; thủ sẵn **Mục G** cho rebuttal.
