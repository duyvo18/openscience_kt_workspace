# Review thiết kế — DPA Framework (bản chi tiết cho full paper)

> **Nguồn:** `proposal_final.md` (bản implementation-level) + `proposal_architecture.png`.
>
> **Góc review:** mức cài đặt — tập trung vào những quyết định thiết kế còn thiếu, mâu thuẫn đặc tả, và rủi ro kỹ thuật sẽ lộ ra khi thực sự dựng model. Không lặp lại các mục đã chốt ở short paper trừ khi bản chi tiết này *đi ngược* lại chúng.
>
> **Quy ước mức độ:** 🔴 blocking (phải xử lý trước khi code/nộp full paper) · 🟠 quan trọng · 🟡 nên làm rõ · 🟢 điểm mạnh cần giữ.

---

## 0. Kết luận nhanh

Ý tưởng cốt lõi — tách bước *diễn giải bằng chứng* thành một stage tường minh giữa interaction và mastery, và đặt nó trên một không gian phân phối có thể áp toán tử + luật sư phạm — vẫn là một inductive bias có giá trị và định vị tốt. Bản chi tiết này mạnh hơn short paper ở chỗ đã có sơ đồ dữ liệu chảy rõ ràng (dual-Mamba, Beta descriptor, 4 operator, gating, KC contribution).

Tuy nhiên, ở mức cài đặt bản này có **ba vấn đề blocking** cần xử lý trước khi viết full paper:

1. **Mâu thuẫn đặc tả Module 2 $\to$ 3 (🔴):** Module 3 lấy đầu vào là $P_i$ (pattern *trước* alignment), trong khi toàn bộ đóng góp của framework nằm ở $\tilde P_i$/$z'$ (*sau* alignment). Theo đúng công thức đang viết, **Pedagogical Alignment không tác động gì lên mastery update** — tức là bộ phận lõi bị vô hiệu trong luồng dữ liệu.
2. **Alignment bị đặc tả *lùi* so với short paper (🔴):** short paper đã chốt "rule-shaped functional form, biên độ $\beta$ learnable, không loss phụ" kèm instantiation cụ thể. Bản full-paper này lại quay về mô tả mơ hồ $\mathrm{Align}(P_i, R_i)$ + gọi nó là "regularization" — vừa ngược quyết định đã chốt, vừa thiếu đúng phần mà full paper *bắt buộc* phải cụ thể hóa.
3. **"Distribution space" chưa có đại số (🔴):** chưa định nghĩa Beta là phân phối *trên cái gì*, pooling các Beta thế nào (Beta không đóng dưới phép cộng), và luồng gradient dùng moment hay sampling. Đây là điểm phân biệt novelty với UKT/KeenKT/PLKT nên không thể để trống ở full paper.

Phần còn lại là các vấn đề quan trọng/nên-làm-rõ, chi tiết bên dưới.

---

## 1. Đánh giá thiết kế tổng thể

### 1.1 Điểm mạnh cần giữ 🟢
- **Phân tách vai trò 4 không gian** là một khung tổ chức sạch và dễ kiểm chứng bằng ablation (mỗi tầng tương ứng một giả thuyết đóng góp). Giữ mapping ECD (student/evidence/task) làm nền lý thuyết.
- **Diễn giải nội tại có đường truy vết end-to-end** (Pattern $\to$ KC ở M3 nối với KC $\to$ Prediction ở M4) là một câu chuyện mạch lạc, khác biệt với các giải thích post-hoc.
- **Operator cố định + readout có cấu trúc** ($z'$ chia block theo loại pattern) là inductive bias thật, không phải chỉ đặt tên.

### 1.2 Rủi ro ở tầng khái niệm

**🟠 Nhãn ngữ nghĩa không được cơ chế nào cưỡng chế.** Luận điểm trung tâm "distribution space mang *evidence* chứ không phải *mastery*" hiện chỉ được bảo đảm bằng *vị trí đặt module*. Vì model train end-to-end chỉ bằng prediction loss, các biểu diễn trung gian sẽ tự tổ chức theo hướng tối thiểu hóa loss — không có gì buộc chúng mang đúng ngữ nghĩa "evidence" hay "pedagogical pattern". Thứ *thực sự* tạo ra cấu trúc là (a) operator cố định và (b) dạng hàm alignment bị ràng buộc. Do đó toàn bộ tính "diễn giải được" của framework **dựa hết vào việc hai thành phần này đủ hạn chế**. Hệ quả cho full paper: phải có ablation chứng minh *bỏ alignment $\to$ khả năng diễn giải/tính nhất quán giảm trong khi accuracy gần như không đổi* — đó mới là bằng chứng cấu trúc có ý nghĩa chứ không phải trang trí.

**🟠 Rò rỉ vai trò giữa Interaction Space và các tầng sau.** Module 1 đọc Mastery Memory $M_t$ (qua KG) để tạo Knowledge Context, rồi fuse vào $z_t$. Nghĩa là "interaction representation" **đã chứa thông tin mastery** trước khi vào tầng evidence. Điều này mâu thuẫn với phát biểu "Interaction Space: không quan tâm reasoning" và làm mờ ranh giới evidence-vs-mastery mà cả framework dựa vào. Cần hoặc (i) thừa nhận rõ và định nghĩa lại ranh giới, hoặc (ii) chuyển việc đọc mastery sang sau tầng distribution.

**🟡 Ba cơ chế mang lịch sử chồng lấn.** Model đồng thời có: hidden state của Mamba (mang lịch sử chuỗi), bộ nhớ tường minh $M$ (mang mastery tích lũy), và pattern operator (tổng hợp lại toàn history). Ba đường cùng "nhớ quá khứ" dễ bị reviewer đánh giá là dư thừa/over-engineered. Cần lập luận vai trò *không thay thế nhau* của từng cái, và có ablation (vd bỏ nhánh knowledge của Mamba, hoặc bỏ $M$ tường minh) để biện minh.

---

## 2. Review từng module

### Module 1 — Interaction Representation Learning

- **🟠 Cơ chế "Localized Mastery read qua KG" chưa đặc tả.** Đây là điểm nối quan trọng nhưng chỉ nói "truy xuất theo ngữ cảnh câu hỏi qua Knowledge Graph". Cần rõ: gather k-hop neighborhood của KC câu hỏi? graph attention? trọng số theo cạnh nào? Chi phí ra sao khi số KC lớn? Đây cũng là chỗ knowledge graph đi vào model lần đầu $\to$ liên quan trực tiếp graph-confound (xem M2).
- **🟡 Nhánh "knowledge dynamics" của Mamba chạy trên chuỗi nào?** Mamba là sequence model; cần nêu rõ nó chạy trên chuỗi *localized-mastery-reads theo thời gian*. Nếu đúng vậy thì OK, nhưng phải viết tường minh vì hiện dễ hiểu nhầm.
- **🟡 Chỉ số thời gian và rò rỉ nhãn.** Xác nhận $M_t$ được đọc *trước* cập nhật (tức $M_{t-1}$ sau bước trước), và dự đoán tại bước $t+1$ chỉ dùng thông tin tới $t$. Nên ghi rõ quy ước chỉ số trong bài để reviewer khỏi nghi ngờ leakage.
- **🟡 Biện minh Mamba vs attention (AKT).** Nêu lý do chọn Mamba (chi phí tuyến tính theo độ dài) và có/không so sánh với biến thể attention trong ablation.
- **Cold start:** history rất ngắn $\to$ hidden state Mamba + localized read gần như rỗng. Nêu hành vi mặc định.

### Module 2 — Distributional Pedagogical Alignment (lõi)

**2.1 Distribution Projection**
- **🔴 Beta *trên cái gì*?** Cần nói rõ: mỗi chiều latent một Beta độc lập ($d$ Beta), hay một Beta chung? Beta có support $[0,1]$ — nếu chỉ là "không gian biểu diễn trung gian" (không phải mastery) thì *vì sao lại là Beta* thay vì Gaussian? Lựa chọn Beta rất tự nhiên cho đại lượng thuộc $[0,1]$ (xác suất đúng/mastery); dùng nó cho một latent code trừu tượng cần lý do mạnh hơn "vì nó là một phân phối". Nếu không, reviewer sẽ đọc đây là mastery-belief trá hình $\to$ va đúng vào PLKT.
- **🟡** Ánh xạ $z_t \to (\alpha_t, \beta_t)$: nêu rõ dùng softplus + hằng số ổn định ($\alpha, \beta > 0$), và xử lý khi $\alpha, \beta$ rất nhỏ (phân phối gần suy biến).

**2.2 Pattern Construction**
- **🔴 Chưa có đại số phân phối.** "Pooling các Beta" không đóng: tổng/trộn các Beta không còn là Beta. Phải định nghĩa phép tổng hợp: (a) moment-matching (gộp theo kỳ vọng/variance rồi khớp lại $(\alpha,\beta)$), (b) product-of-experts, hay (c) mixture? Mỗi lựa chọn có hệ quả khác nhau về differentiability và ngữ nghĩa. Đây là phần *cốt lõi* phân biệt với UKT (Wasserstein trên Gaussian) và KeenKT (NIG) — không thể để mở.
- **🟠 Tập rỗng / quá nhỏ.** Early sequence: operator Same-KC / Prerequisite / Neighbor có thể trả về tập rỗng $\to$ pattern không xác định. Cần default/masking tường minh, và nêu ảnh hưởng lên $z'$ (block tương ứng bằng giá trị gì?).
- **🟠 Graph confound (đã biết từ short paper, J5a).** Prerequisite & Neighbor operator cần đồ thị KC. Trên dataset flat-tag đồ thị được *dựng* (theo CMDKT) $\to$ gain có thể đến từ graph được inject chứ không từ cơ chế DPA. Ablation "giữ $G$ cố định + bỏ alignment / và ngược lại" là bắt buộc cho full paper.
- **🟡 "Structural consistency across learners" là giả định, chưa phải định lý (J4).** Cùng một operator áp lên history khác nhau cho phân phối khác nhau; việc chúng "so sánh được về ngữ nghĩa" giữa learner cần hoặc lập luận, hoặc một *consistency metric* đo được để full paper report (vd độ khớp hình học của pattern-space giữa các cohort).
- **🟠 Chi phí.** Với mỗi bước $t$, dựng 4 pattern trên history tới $t$; Same-KC/Prereq cần gather $\to$ ngây thơ là $O(t)$ mỗi bước, tức $O(T^2)$ mỗi chuỗi. Cần phân tích complexity và cách amortize (vd cache theo KC).

**2.3 Pedagogical Alignment**
- **🔴 Đặc tả bị lùi so với short paper.** Bản chốt short paper: alignment là **hàm biến đổi dạng cố định theo luật, chỉ biên độ ($\beta$) learnable, train end-to-end, KHÔNG loss phụ**, kèm ít nhất 1 instantiation (difficulty-response consistency / guess-slip soft-gate / KC-relation consistency). Bản full-paper này chỉ còn $\tilde P_i = \mathrm{Align}(P_i, R_i)$ + note gọi nó là **"regularization trên không gian phân phối"**. Hai vấn đề:
  1. **Mâu thuẫn framing:** "regularization" (một loss/ràng buộc mềm) ngược với quyết định đã chốt "functional form trong forward pass, không loss phụ". Phải chọn một và nhất quán. Khuyến nghị giữ đúng bản chốt: alignment là *phép biến đổi tường minh* tác động lên descriptor $(\alpha,\beta)$ trong forward pass.
  2. **Thiếu instantiation:** full paper *phải* có ít nhất 1 luật viết đầy đủ: (i) đại lượng tính từ pattern, (ii) chèn vào đâu trong forward pass, (iii) tác động cụ thể lên $(\alpha,\beta)$ — kỳ vọng/variance dịch chuyển ra sao, và (iv) $R_i$ được tham số hóa thế nào ($\beta$ là scalar/vector, khởi tạo, ràng buộc dấu). Đây chính là phần short paper đã làm được; bản chi tiết không được để trắng.
- **🟡** Ký hiệu $R_i$ (tập luật cho pattern $i$) nên nêu rõ ánh xạ 1-1 giữa loại operator và loại luật, hay luật dùng chung.

**2.4 Pattern Readout**
- **🟡 Readout dùng moment hay sample?** Nếu đọc thẳng $(\alpha,\beta) \to$ vector (moment readout) thì differentiable dễ nhưng "distribution" thực chất chỉ là embedding 2 lần có cấu trúc $\to$ phải thừa nhận và biện minh novelty không nằm ở "lấy mẫu ngẫu nhiên". Nếu *sample* từ Beta thì cần reparameterization: Beta không có reparam đơn giản $\to$ phải dùng implicit reparam gradient hoặc surrogate Kumaraswamy. Full paper phải chọn và nêu rõ; đây là câu hỏi reviewer chắc chắn hỏi.

### Module 3 — Mastery State Tracking

- **🔴 Dùng sai đầu vào ($P_i$ thay vì $z'$/$\tilde P_i$).** Như mục 0.1: $A_i = G(P_i)$ và $U_i(M_t, P_i)$ dùng pattern *trước* alignment. Nếu đúng chữ, alignment (lõi framework) không ảnh hưởng gì tới mastery. Sửa: Module 3 phải nhận **$z'$ (readout của $\tilde P_i$)** hoặc trực tiếp $\tilde P_i$. Đồng bộ ký hiệu trên cả bài + sơ đồ.
- **🟠 Đụng độ ký hiệu $A_i$.** Mục 2.2 dùng $\mathcal A_i$ (operator) cho $P_i = \mathcal A_i(H)$; Module 3 dùng $A_i = G(P_i)$ (gating contribution). Hai ký hiệu $A$ khác nghĩa $\to$ đổi tên một cái (vd gating dùng $g_i$ hoặc $\Gamma_i$).
- **🟠 Ổn định của cập nhật cộng dồn.** $M_{t+1} = M_t + \Delta M$ là residual không chặn, không có forgetting/normalize (forgetting đã chủ động bỏ) $\to$ trên chuỗi dài $M$ có thể trôi/nổ. DKVMN dùng erase-add (chặn). Cân nhắc: cổng cập nhật chặn biên, hoặc layernorm trên $M$, hoặc tanh-bounded $\Delta M$. Nêu rõ trong bài.
- **🟡 Shape.** $A_i$ (Pattern $\to$ KC) chắc là vector $C$ chiều broadcast trên $d$; $U_i$ là cập nhật $C \times d$. Ghi rõ shape để $A_i \odot U_i$ well-defined.
- **🟠 Tính trung thực (faithfulness) của gating như "giải thích".** Trọng số $A_i$ được diễn giải là "pattern nào nâng KC nào" — nhưng đây đúng là cảnh báo *Attention is not Explanation*: trọng số không đảm bảo trung thực. Short paper đã hạ xuống mức "tiềm năng / behavioral consistency" — full paper phải giữ caveat đó và tốt hơn nữa là *kiểm chứng* (vd can thiệp: bỏ pattern có gating cao $\to$ mastery/prediction đổi đúng hướng gating dự báo). Đây là cơ hội biến điểm yếu thành đóng góp.
- **🟠 Không có tín hiệu giám sát cho các đại lượng diễn giải.** Chỉ prediction loss $\to$ gating & Pattern-to-KC contribution hoàn toàn unsupervised; không gì buộc chúng khớp ý nghĩa sư phạm thật. Hoặc thêm weak supervision, hoặc để faithfulness test, hoặc phát biểu giới hạn rõ ràng.

### Module 4 — Prediction Aggregation

- **🟡 $W$ có thể trùng vai với attention trong $f$.** $W = C(M_{t+1}, q_{t+1})$ (đóng góp per-KC) rồi $\hat y = f(M_{t+1}, q_{t+1}, W)$. Nếu $f$ cũng attend trên KC thì $W$ dư thừa/double-count. Làm rõ $W$ là *thành phần tính ra* $\hat y$ (vd $\hat y = \sigma\left(\sum_c w_c \cdot \mathrm{score}_c\right)$) chứ không phải nhánh song song chỉ để "giải thích".
- **🟡 Ánh xạ câu hỏi mục tiêu $\to$ KC.** $q_{t+1}$ liên hệ KC nào (qua cùng KG?) cần nêu, vì $W$ là per-KC.
- **🟠 Faithfulness (như M3).** $W$ cũng là trọng số diễn giải — cùng caveat và cùng khuyến nghị kiểm chứng.

---

## 3. Vấn đề xuyên suốt (cross-cutting)

1. **🔴 Nhất quán ký hiệu Module 2 $\to$ 3 $\to$ 4.** Thống nhất một bộ: $H$ (chuỗi distribution), $P_i$/$\tilde P_i$ (pattern trước/sau align), $z'$ (readout), gating $g_i$, contribution $W$. Sơ đồ và công thức hiện đang lệch nhau ($P_i$ so với $z'$).
2. **🔴 Chuỗi khả vi end-to-end.** Vẽ rõ đường gradient: $(\alpha,\beta)$ (softplus) $\to$ pooling (đại số đã chọn) $\to$ alignment (functional, khả vi) $\to$ readout (moment/implicit-reparam) $\to$ update $\to$ prediction. Chỉ ra chỗ nào có sampling và cách lấy gradient qua đó.
3. **🟠 Complexity & bộ nhớ.** $M \in \mathbb R^{C \times d}$ với $C$ lớn (vd XES3G5M hàng nghìn KC) + 4 operator mỗi bước + dual Mamba. Cho bảng complexity so với AKT/DKVMN và chi phí per-step pooling ở quy mô ASSISTments/Algebra.
4. **🟠 Định danh (identifiability) của cấu trúc.** Nêu tường minh: tính diễn giải phụ thuộc operator cố định + dạng alignment bị ràng buộc; và thiết kế ablation để *chứng minh* điều đó (bỏ align/đổi operator tự do $\to$ cấu trúc/consistency sập).
5. **🟡 Đọc được ở grayscale + màu 4 space định nghĩa trong caption** (giữ từ N7 short paper) — sơ đồ full paper cũng nên tuân.

---

## 4. Định vị novelty (nhất quán với literature đã kiểm chứng trong project)

- **vs PLKT (mối đe dọa gần nhất — Beta + evidence + pattern):** phải chỉ ra khác biệt *hành vi*, không chỉ khác biệt mô tả. Điểm phân biệt khả dĩ: (i) distribution là *evidence trung gian* dựng từ history representation, không phải belief tra bảng từ interaction; (ii) pooling theo *operator sư phạm* + alignment dạng-luật, thay vì sliding-window thời gian. Nên có 1 ca cụ thể mà DPA và PLKT cho prediction khác nhau.
- **vs UKT / KeenKT / S2KT:** họ đặt phân phối lên *state/mastery*; DPA đặt sớm hơn ở *evidence* và thao tác pooling+alignment tường minh trên descriptor. Lập luận "distribution *trước* pooling/alignment" là điểm bán.
- **vs NSKT:** luật *điều chỉnh phân phối* chứ không suy luận symbolic trên nhãn. Tránh nhận nhãn "neural-symbolic" như đóng góp riêng (NSKT đã chiếm).

---

## 5. Khuyến nghị ưu tiên cho full paper

1. **Sửa luồng Module 3 dùng $z'$/$\tilde P_i$ (hậu-alignment)** — blocking; hiện alignment đang bị bypass trên giấy.
2. **Port lại + hoàn thiện đặc tả Alignment** (functional-form, ít nhất 1 instantiation đầy đủ, reconcile "regularization" vs forward-pass transform) — blocking, đây là phần full paper *bắt buộc* phải cụ thể.
3. **Định nghĩa đại số phân phối** (Beta trên gì, pooling đóng thế nào, moment vs sampling + reparam) — blocking, là chỗ phân biệt novelty.
4. **Thống nhất ký hiệu + vẽ chuỗi khả vi** end-to-end.
5. **Xử lý tập rỗng/cold-start cho operator** và **ổn định của $M$** (bounded/normalized update).
6. **Faithfulness cho gating & KC contribution** (test can thiệp hoặc phát biểu giới hạn).
7. **Ablation tách graph confound** + **consistency metric** cho claim structural consistency.
8. **Bảng complexity/bộ nhớ** cho dataset nhiều KC.
9. **Làm rõ rò rỉ mastery $\to$ interaction rep** ở Module 1 so với claim tách vai trò.
10. **Ablation "bỏ alignment"** để chứng minh cấu trúc mang ý nghĩa chứ không trang trí.
