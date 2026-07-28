# Đối chiếu review thiết kế với implementation hiện tại

> **Nguồn:** `review_DPA_design_fullpaper.md` (review mức thiết kế) đối chiếu với `DPA_KT_implementation_synthesis.md` + đọc lại code trong `code/dpa_kt/models/` + kết quả ablation trong `code/runs-50-epochs/`.
>
> **Ký hiệu trạng thái:** ✅ đã giải quyết · ◐ giải quyết một phần · ❌ còn tồn đọng · ✱ vấn đề MỚI phát sinh (không có trong review gốc).

---

## 0. Tóm tắt

Trên **24 mục** review nêu, implementation đã **giải quyết 13 mục** (3 mục blocking ở Mục 1 gồm 2 mục đã giải quyết, cộng 11 mục kỹ thuật ở Mục 2: đại số phân phối, tập rỗng, ổn định cập nhật, leakage, shape, chuỗi khả vi), **giải quyết một phần 4 mục** (Mục 3), và **còn tồn đọng 7 mục** (Mục 5 — chủ yếu thuộc tầng framing/lý thuyết mà code không thể tự giải quyết; hàng đầu của Mục 5 chính là Blocking #2, được nhắc lại từ Mục 1).

Điểm quan trọng nhất: **trong ba vấn đề blocking của review, hai cái được giải quyết theo cách "thay đổi thiết kế" chứ không phải "bổ sung đặc tả"**, và cái thứ ba thì **xấu hơn** — code chọn đúng nhánh mà short paper đã bác. Đồng thời phát sinh **9 vấn đề mới**, trong đó có 3 cái nghiêm trọng (mâu thuẫn paper-code trên chính thành phần hiệu quả nhất; một loss có dấu đáng ngờ và thực nghiệm cho thấy có hại; thiếu multi-seed trong khi ablation đã trở thành bằng chứng trung tâm).

---

## 1. Ba vấn đề BLOCKING của review

### Blocking #1 — Module 3 nhận $P_i$ (tiền alignment) thay vì $\tilde P_i$/$z'$ · ✅ (nhưng do vấn đề tự tiêu biến)

Trong code, `MasteryState.update()` nhận `patterns[name]["v"]`, tức là **readout $v_i$ của phân phối đã pool** — đúng là $z'$ block, không phải $P_i$ thô. Nên về mặt luồng dữ liệu code làm đúng.

Tuy nhiên cần nói chính xác: vấn đề này **tự tiêu biến** chứ không phải được sửa. Vì code không có bước biến đổi $\tilde P_i = \mathrm{Align}(P_i, R_i)$ nào cả, nên **không tồn tại $\tilde P_i$** để mà nhận sai. Câu hỏi "trước hay sau alignment" trở nên vô nghĩa trong cài đặt hiện tại.

Hệ quả cho bài: sơ đồ và công thức trong proposal vẫn phải sửa (Module 3 nhận $z'$), nhưng lời giải thích không thể nói "alignment đã tác động lên $z'$" — vì nó không tác động ở forward pass.

### Blocking #2 — Đặc tả Alignment bị lùi · ❌ **XẤU HƠN**

Review cảnh báo bản proposal quay về mô tả mơ hồ và gọi alignment là "regularization", ngược với bản chốt short paper ("rule-shaped functional form, biên độ learnable, KHÔNG loss phụ").

Code **đi hẳn theo nhánh regularization**: alignment là 3 loss phụ ($\mathrm{mono}$, $\mathrm{gs}$, $\mathrm{KL}$) cộng vào objective. Không có transform nào trong forward pass. Không có file `alignment.py` (README liệt kê nhầm).

Nghĩa là: **cả proposal-chi-tiết và code đều chọn nhánh mà short paper đã công khai bác bỏ.** Đây không còn là vấn đề "thiếu đặc tả" mà là **mâu thuẫn ba chiều thực sự** giữa short paper (đã nộp/đã chốt), proposal, và code đang chạy. Không thể để nguyên khi viết full paper.

### Blocking #3 — "Distribution space" chưa có đại số · ✅ **giải quyết trọn vẹn**

Code trả lời đầy đủ cả ba câu hỏi review nêu, nhưng bằng cách **đổi họ phân phối**:

| Câu hỏi review | Câu trả lời trong code |
|---|---|
| Beta trên cái gì? | Không dùng Beta. Dùng **Gaussian chéo** $\mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ trên latent $d_z = 64$ chiều; hai Linear head từ $z_t$, $\mathrm{logvar}$ clamp $[-6, 2]$ |
| Pooling thế nào (Beta không đóng)? | **Moment matching** — đóng dưới Gaussian: $\mu_P = \sum_j w_j \mu_j$, $\mathrm{var}_P = \sum_j w_j(\mathrm{var}_j + \mu_j^2) - \mu_P^2$, floor $10^{-3}$ |
| Moment hay sampling? | **Moment readout**, không sampling: $v_i = \mathrm{Linear}([\mu_P \oplus \log \mathrm{var}_P])$. Không cần reparameterization |

Đây là cách giải quyết sạch về mặt kỹ thuật. Nhưng nó sinh ra vấn đề mới về novelty (xem ✱1).

---

## 2. Các mục review ĐÃ được giải quyết

| Mục review | Cách code giải quyết |
|---|---|
| ✅ **M2.2 Tập rỗng / quá nhỏ** (operator Same-KC/Prereq/Neighbor có thể rỗng ở early sequence) | `masked_softmax` trả về zeros cho hàng bị mask hoàn toàn, kèm cờ `has_valid`; `_pool` khi đó fallback về **null distribution học được per-operator** (`null_mu`, `null_logvar`). Mỗi operator có prior riêng thay vì giá trị hằng tùy ý |
| ✅ **M3 Ổn định cập nhật cộng dồn** ($M_{t+1} = M_t + \Delta M$ không chặn) | Thay bằng **DKVMN erase-add có chặn**: $\mathrm{new} = \mathrm{rows} \cdot \mathrm{keep} + \mathrm{add}$ với $\mathrm{keep} = \prod_i (1 - A_i e_i) \in (0,1)$, $a_i = \tanh(\cdot) \in (-1,1)$. Thêm **hai** cơ chế review không nghĩ tới: LayerNorm trên mastery read, và **grad-clip-through-time** qua `register_hook` (clamp $\pm 1.0$) — comment ghi rõ Jacobian nhân dồn đạt $\sim 10^{11}$ ở $\mathrm{tbptt}=25$ và bf16 tràn trước optimizer |
| ✅ **M1 Chỉ số thời gian / rò rỉ nhãn** | Code tường minh predict-then-update: Module 4 tính $\hat y_t$ từ $M$ (dựng từ interactions $< t$) **trước** khi bất kỳ nhánh nào nhìn $r_t$. Không leakage |
| ✅ **M1 Cơ chế "Localized Mastery read qua KG" chưa đặc tả** | `MasteryState.read()`: masked-softmax attention trên đúng $K_{rel} = 20$ KC liên quan, query $= W_{read}(e_q)$, key $= \mathrm{kc\_key}$. Tập liên quan $= $ KC riêng của câu $\oplus$ graph-expansion (`q_rel`) cắt còn $k_{rel}$. Chi phí bị chặn cứng bởi budget $k_{rel}$, không phụ thuộc $C$ |
| ✅ **M2.4 Readout moment hay sample** | Moment readout, không sampling — nên toàn bộ pipeline khả vi trực tiếp, không cần implicit reparam/Kumaraswamy |
| ✅ **M2.1 Ổn định tham số phân phối** | Clamp $\mathrm{logvar} \in [-6,2]$ + floor $\mathrm{var}_P \ge 10^{-3}$ (comment ghi rõ floor để chặn $\partial \log / \partial \mathrm{var}$) |
| ✅ **M3 Shape của $A_i \odot U_i$** | $A_i$ là $(B, K_{rel})$, unsqueeze broadcast trên $d_v$; erase/add là $(B, 1, d_v)$. Well-defined |
| ✅ **M4 $W$ trùng vai với attention trong $f$** | Không double-count: $\beta$ **chính là** trọng số tổng hợp — $u = \mathrm{LayerNorm}(\sum_k \beta_k \cdot \mathrm{rows}_k)$ rồi $p_{master} = \sigma(\mathrm{MLP}([u \oplus e_q \oplus e_{dq}]))$. Không có attention song song nào khác trên KC |
| ✅ **M4 Ánh xạ câu hỏi mục tiêu $\to$ KC** | `_related()`: KC riêng của câu, rồi graph-expand qua `q_rel`, truncate $k_{rel}$ |
| ✅ **Cross-cutting #2 Chuỗi khả vi end-to-end** | Khả vi toàn tuyến, không có sampling ở bất kỳ đâu. Chỉ có 2 điểm cắt gradient có chủ ý: biên tBPTT (detach $M$, $h_b$, $\mu/\mathrm{var}$ prefix mỗi 5 bước) và `surprise.detach()` trong loss guess/slip |
| ✅ **1.2 Cần ablation chứng minh cấu trúc có ý nghĩa** | Đã có 9 preset ablation **và đã chạy xong** trên assist09 + xes3g5m (xem mục 4) |

---

## 3. Các mục giải quyết MỘT PHẦN

| Mục review | Đã có gì | Còn thiếu gì |
|---|---|---|
| ◐ **M2.2 Graph confound** | Đã chạy `no_prereq` / `no_neighbor` (assist09: $-0.0091$ / $-0.0090$; xes3g5m: $-0.0088$ / $-0.0056$) | Ablation này **bỏ operator**, không **bỏ graph**. Nó không tách được "gain do thông tin graph được inject" khỏi "gain do cơ chế DPA". Cần thêm nhánh: giữ operator nhưng thay $G$ bằng graph ngẫu nhiên/shuffle cùng mật độ. Đồ thị vẫn **ước lượng từ train split** (proposal giả định cho trước) |
| ◐ **M3/M4 Faithfulness của gating và $\beta$** | `return_trace=True` xuất đầy đủ `pattern_w`, `gates`, `beta`, `alpha`, `mastery` evolution; notebook có attribution case study | Vẫn chỉ là **trọng số attention** — chưa có intervention test (bỏ pattern có gating cao rồi kiểm tra mastery/prediction đổi đúng hướng gating dự báo). Caveat *Attention is not Explanation* vẫn nguyên |
| ◐ **Cross-cutting #3 Complexity và bộ nhớ** | Có giảm thiểu thực tế: mask $(B,L,L)$ precompute một lần/batch, budget $k_{rel} = 20$ chặn cứng chi phí read/update, `d_v=16` cho dataset nhiều KC (xes3g5m, junyi), memmap sequences, logging throughput + peak memory mỗi epoch | Chưa có **bảng complexity** so với AKT/DKVMN, chưa phân tích $O(T^2)$ của pattern pooling (vẫn stack toàn prefix mỗi bước: `torch.stack(mu_list, dim=1)`) |
| ◐ **Cross-cutting #4 Định danh cấu trúc** | Ablation đã chạy, cho thấy mỗi operator đóng góp dương | Ablation chỉ đo **AUC**, không đo **tính diễn giải/consistency**. Review yêu cầu đúng chiều ngược lại: bỏ alignment thì *diễn giải sập trong khi accuracy gần như không đổi* — hiện chưa có metric nào để phát biểu điều đó |

---

## 4. Kết quả ablation (bằng chứng mới, `runs-50-epochs/`, single-seed, 50 epoch)

| Ablation | assist09 AUC | $\Delta$ | xes3g5m AUC | $\Delta$ |
|---|---|---|---|---|
| **full** | **0.7170** | — | **0.8105** | — |
| no_distributional | 0.6931 | $-0.0239$ | 0.8030 | $-0.0075$ |
| no_samekc | 0.7043 | $-0.0127$ | 0.8068 | $-0.0037$ |
| single_branch | 0.7049 | $-0.0121$ | 0.8075 | $-0.0030$ |
| no_mono | 0.7056 | $-0.0114$ | 0.8072 | $-0.0033$ |
| no_prereq | 0.7079 | $-0.0091$ | 0.8016 | $-0.0088$ |
| no_neighbor | 0.7080 | $-0.0090$ | 0.8049 | $-0.0056$ |
| no_temporal | 0.7085 | $-0.0085$ | 0.8056 | $-0.0049$ |
| **no_gs** | **0.7208** | **$+0.0038$** | 0.8075 | $-0.0030$ |

Đọc kết quả:
- **Thành phần distributional là đóng góp lớn nhất** ($-0.0239$ trên assist09). Đây là tin tốt cho story trung tâm. Cần nói chính xác ablation này làm gì: khi `use_distributional=False` thì $\mathrm{logvar} \equiv 0$ nên $\mathrm{var}_j = 1$, và $\mathrm{var}_P = 1 + \mathrm{Var}_w(\mu)$ — tức là vẫn giữ **độ phân tán giữa các bước được pool**, chỉ bỏ **uncertainty học được theo từng input**. Ablation này sạch hơn tưởng, nhưng phải phát biểu đúng.
- **Cả 4 operator đều đóng góp dương** trên cả hai dataset — ủng hộ thiết kế operator cố định.
- **`no_gs` cho AUC CAO HƠN full trên assist09.** Loss guess/slip đang **làm hại** ở đó và gần như vô hại ở xes3g5m ($-0.0030$, cùng cỡ nhiễu). Xem ✱3.
- **Hiệu ứng trên xes3g5m nhỏ hơn hẳn** ($-0.003$ đến $-0.009$) so với assist09. Cần giải thích, không nên báo cáo gộp.

---

## 5. Các mục CÒN TỒN ĐỌNG

| Mục review | Trạng thái sau khi đọc code |
|---|---|
| ❌ **Blocking #2 — Alignment là loss hay transform** | Xấu hơn: code chốt hẳn là loss. Mâu thuẫn ba chiều với short paper |
| ❌ **1.2 Rò rỉ vai trò: Interaction Space đọc mastery** | Xác nhận có trong code — `BranchBCell` nhận $m_{read}$ từ `MasteryState.read(M, ...)`, rồi fuse vào $z_t$. Nên $z_t$ mang thông tin mastery trước khi vào tầng evidence. Là lựa chọn có ý thức (nhánh B *là* knowledge context) nhưng claim "Interaction Space không quan tâm reasoning" trong bài vẫn sai |
| ❌ **1.2 Nhiều cơ chế mang lịch sử chồng lấn** | Thực tế code có **bốn** đường, không phải ba: (1) Transformer nhánh A trên toàn chuỗi, (2) hidden state GRU nhánh B, (3) bộ nhớ tường minh $M$, (4) pattern operator pool toàn prefix. Ablation `single_branch` chỉ tắt nhánh B ($-0.0121$ / $-0.0030$); chưa có ablation nào bỏ $M$ tường minh hay bỏ nhánh A |
| ❌ **M2.2 "Structural consistency across learners" chưa đo được** | Chưa có metric nào trong code. Vẫn là giả định |
| ❌ **M3 Không có tín hiệu giám sát cho đại lượng diễn giải** | `gates` và $\beta$ vẫn hoàn toàn unsupervised, chỉ chịu áp lực từ prediction loss |
| ❌ **Cross-cutting #1 Nhất quán ký hiệu** | Chưa làm, và **khó hơn trước**: giờ có ba bộ tên (proposal: $P_i$, $\tilde P_i$, $\mathcal A_i$, $A_i$, $U_i$, $W$; code: `v`, `w`, `gates`, `A`, `beta`, `alpha`; sơ đồ: lại khác). Đụng độ $\mathcal A_i$ (operator) với $A_i$ (gating) vẫn còn |
| ❌ **Cross-cutting #5 Sơ đồ đọc được ở grayscale** | Không liên quan code; chưa xử lý |

---

## 6. Vấn đề MỚI phát sinh (không có trong review gốc)

### ✱1 — Novelty vs UKT bị thu hẹp do đổi Beta thành Gaussian · nghiêm trọng
Review định vị điểm phân biệt với UKT là "UKT dùng Gaussian + Wasserstein, ta dùng Beta". Nhưng **code cũng dùng Gaussian**. Điểm phân biệt còn lại chỉ là: (a) phân phối đặt ở *evidence* thay vì *state*, và (b) pooling theo **operator sư phạm + moment matching** thay vì distance metric. Cả hai đều là lập luận về *vị trí và cách tổng hợp*, không phải về họ phân phối. Phần định vị novelty trong bài phải viết lại từ tiền đề này.

### ✱2 — Mâu thuẫn paper-code trên chính thành phần hiệu quả nhất · nghiêm trọng
Short paper đã **chủ động bỏ monotonicity** (thay bằng difficulty-response consistency và KC-relation consistency). Nhưng code **có `use_align_mono`** và ablation cho thấy nó **đóng góp dương rõ rệt** ($-0.0114$ trên assist09 khi bỏ — lớn thứ tư trong bảng, hơn cả ba operator graph). Đồng thời **hai principle mới mà short paper hứa thì không tồn tại trong code.**

Tức là: thành phần bài nói đã loại bỏ thì thực tế đang chạy và đang có tác dụng; thành phần bài nói đã thêm vào thì chưa được cài. Đây là mâu thuẫn khó biện minh nhất nếu reviewer đối chiếu code.

### ✱3 — Loss guess/slip: dấu đáng ngờ và thực nghiệm cho thấy có hại · nghiêm trọng
Hai vấn đề cộng dồn:
1. **Dấu:** code tính $\mathrm{gs}_t = \mathrm{surprise} \cdot \mathrm{delta}$ với $\mathrm{surprise}$ đã `detach()`, rồi **tối thiểu hoá**. Vì $\mathrm{surprise}$ là hằng số, việc này **giảm** $\mathrm{delta}$ mạnh hơn ở nơi surprise cao — tức **kìm** cập nhật mastery khi bất ngờ. README lại mô tả là muốn cập nhật *tỉ lệ thuận* với surprise, tức ngược dấu.
2. **Thực nghiệm:** `no_gs` cho AUC **cao hơn** full trên assist09 ($+0.0038$), và trên xes3g5m chênh lệch nằm trong nhiễu ($-0.0030$).

Kết hợp lại: đây là thành phần vừa mơ hồ về ý định, vừa không có bằng chứng hữu ích. Ba lựa chọn: sửa dấu rồi chạy lại, bỏ hẳn khỏi model, hoặc giữ và báo cáo thẳng như một negative result.

### ✱4 — Backbone không khớp bài
Mamba $\to$ causal Transformer (nhánh A) + GRUCell (nhánh B). Mọi lập luận "chi phí tuyến tính theo độ dài" của Mamba trong bài đều không áp dụng cho code hiện tại. Hơn nữa nhánh A là Transformer nên chi phí là $O(L^2)$ — **ngược** hướng lập luận ban đầu.

### ✱5 — Phân phối không tham gia vào việc *chọn* pattern
Attention logits của các operator là $\mu_t \cdot \mu_j$ (chỉ dùng mean, scale $1/\sqrt{d_z}$); phương sai **không** ảnh hưởng bước nào được pool, chỉ ảnh hưởng kết quả tổng hợp. Với một framework tên là "Distributional", reviewer sẽ hỏi vì sao uncertainty không được dùng ở chính chỗ nó có ý nghĩa nhất (chọn bằng chứng nào đáng tin). Đây vừa là điểm yếu vừa là hướng cải tiến rõ ràng.

### ✱6 — Mastery khởi tạo từ prior học được
`M0 = nn.Parameter(zeros(n_kcs, d_v))` — trạng thái ban đầu là **tham số học được toàn cục**, không phải zero/uninformed. Nghĩa là model học một "mastery prior" chung cho mọi học sinh. Hợp lý (giải quyết cold-start một cách tự nhiên) nhưng chưa được nêu trong proposal và cần phát biểu — nó ảnh hưởng cách đọc mọi biểu đồ mastery ở bước đầu chuỗi.

### ✱7 — Ablation chưa multi-seed trong khi đã thành bằng chứng trung tâm
Toàn bộ delta nằm trong khoảng $0.003$–$0.024$, single-seed, 50 epoch. Riêng nhóm xes3g5m ($0.003$–$0.009$) **không thể phân biệt với nhiễu seed** nếu không có nhiều seed. Short paper đã cố ý defer multi-seed + significance testing sang full paper — nhưng ở full paper, ablation chính là bằng chứng cho claim "cấu trúc có ý nghĩa", nên **multi-seed trở thành bắt buộc**, không còn là tùy chọn.

### ✱8 — Độ phủ cross-validation không đều
`runs-200-epochs/` có 5 fold cho algebra05, assist09, bridge06, xes3g5m; **assist12 chỉ 4 fold**; **eedi và junyi không có fold nào** dù có config và loader. Bảng kết quả chính của full paper sẽ khập khiễng nếu không hoàn tất hoặc không giải thích.

### ✱9 — Documentation drift giữa các nguồn cấu hình
Ba chỗ nói ba giá trị khác nhau cho cùng tham số:

| Tham số | `config.py` (dataclass) | `configs/base.yaml` (thực dùng) |
|---|---|---|
| `epochs` | 50 | 200 |
| `patience` | 5 | 10 |
| `tbptt` | 50 | 5 |
| `weight_decay` | $10^{-5}$ | 0.0 |

Vì `load_config` merge base.yaml lên sau, **giá trị base.yaml mới là giá trị chạy thật**. Ngoài ra README liệt kê `alignment.py` không tồn tại. Bất kỳ bảng hyperparameter nào trong full paper phải lấy từ base.yaml + per-dataset override, không lấy từ dataclass.

---

## 7. Thứ tự ưu tiên đề nghị

1. **Quyết định về alignment** (✱2, Blocking #2) — chọn một trong ba đường: (a) sửa bài theo code, thừa nhận alignment là auxiliary loss và giữ monotonicity; (b) cài đúng rule-shaped transform như short paper đã chốt rồi chạy lại; (c) giữ cả hai, trình bày như hai variant có so sánh. Mọi việc khác phụ thuộc quyết định này.
2. **Xử lý loss guess/slip** (✱3) — xác định ý định về dấu, rồi sửa/bỏ/báo cáo negative result.
3. **Chạy multi-seed cho ablation** (✱7) — điều kiện cần để bảng ablation nói được điều gì.
4. **Ablation graph-confound đúng nghĩa** (◐) — thêm nhánh graph shuffle giữ nguyên operator.
5. **Viết lại phần định vị novelty** trên tiền đề Gaussian (✱1) và backbone thực tế (✱4).
6. **Metric cho consistency + intervention test cho faithfulness** (◐, ❌) — biến hai điểm yếu thành đóng góp đo được.
7. **Hoàn tất CV coverage** (✱8) hoặc nêu rõ giới hạn phạm vi.
8. **Thống nhất ký hiệu ba chiều** và **sửa README** (❌, ✱9).
9. Cân nhắc **dùng variance trong attention logits** (✱5) — cải tiến nhỏ, hợp story, dễ ablate.
