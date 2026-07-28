# DPA-KT — Tổng hợp chi tiết kiến trúc cài đặt hiện tại

> **Nguồn:** `code/README.md` + toàn bộ `code/dpa_kt/models/*.py` + `config.py` + `configs/base.yaml` + baselines report + `runs-200-epochs/`.
>
> **Mục đích:** nắm chính xác *cài đặt thực tế* của mô hình (khác với bản conceptual proposal/short paper ở nhiều chỗ — được đánh dấu CẢNH BÁO).

---

## 0. Định vị chung

Đây là một **reference implementation** đầy đủ của framework 4 module, train full-scale trên 5 họ dataset (7 config), có visualization per-module, ablation study, bảng so literature, và checkpoint-resume. Điểm mấu chốt cần nhớ ngay:

**Bản cài đặt KHÔNG dùng đúng các thành phần trong sơ đồ/proposal.** README nói rõ đã thay bằng "trainable-at-this-scale equivalents":

| Trong proposal / sơ đồ | Trong code | Lý do (theo README/comment) |
|---|---|---|
| Mamba SSM (2 nhánh) | **causal Transformer** (nhánh A) + **GRU** (nhánh B) | trainable ở quy mô này |
| Phân phối **$Beta(\alpha,\beta)$** | **Gaussian** $\mathcal N(\mu, \mathrm{diag}\,\sigma^2)$ | dễ pooling bằng moment matching, khả vi |
| Cập nhật cộng $M_{t+1} = M_t + \Delta M$ | **DKVMN-style erase-add** (nhân) | ổn định số học |
| Alignment = "functional form trong forward pass" (short paper đã chốt) | **Alignment = các loss phụ** (mono + guess/slip + KL) | CẢNH BÁO: đi *ngược* hướng short paper đã chốt (xem mục 7) |

Module boundaries, data flow và attribution trace thì khớp sơ đồ. Mô hình khoảng **1.3 triệu** tham số.

---

## 1. Kích thước và cấu hình (từ `config.py` / `base.yaml`)

| Nhóm | Giá trị |
|---|---|
| `seq_len=200`, `k_max=4` (KC slots/interaction), `k_rel=20` (related-KC budget/câu), `min_interactions=3`, `n_difficulty_bins=20` |
| `d_emb=64`, `d_model=128`, `d_z=64` (distribution space), `d_v=32` (mastery value; $=16$ cho dataset nhiều KC), `d_key=32`, `n_heads=4`, `dropout=0.2`, `temporal_k=20` |
| Loss weights: `w_mono=0.1`, `w_gs=0.1`, `w_kl=1e-4`, `mono_margin=0.05` |
| Train: `batch=64/128`, `lr=1e-3`, `weight_decay=1e-5` (base.yaml để 0), `tbptt=5`, `mastery_grad_clip=1.0`, `grad_clip=5.0`, `amp=bf16`, `seed=42` |

Đồ thị KC (`P_rel` prerequisite, `N_rel` neighbor, `q_rel` graph-expanded) được **ước lượng từ train split**, không phải cho trước (README ghi rõ đây là một khác biệt so proposal; learnable graph là extension).

---

## 2. Module 1 — Interaction Representation (`interaction_encoder.py`, `embeddings.py`)

Dual-branch, nhưng **hai nhánh chạy ở hai chế độ khác nhau**:

- **`InteractionEmbeddings`**: bảng embedding chung cho question id (padding_idx=0), response (2), KC id, question-difficulty bin, KC-difficulty bin. KC được lấy **mean** trên các slot hợp lệ (`kc_mean`, `kc_diff_mean`, mask $-1$ padding).
- **Branch A (`BranchA`)** — *interaction context*: nối $[e_q \oplus e_r \oplus e_{dq}]$ $\to$ Linear $\to$ **1-layer causal TransformerEncoderLayer** (`norm_first`, causal mask + key_padding_mask). **Tính song song một lần trước time loop** vì không phụ thuộc mastery. Zero hàng NaN (hàng toàn pad).
- **Branch B (`BranchBCell`)** — *knowledge context*: là một **GRUCell** step *trong* time loop vì phụ thuộc $M_t$. Đầu vào là $[\mathrm{read\_proj}(m_{read}) \oplus e_{c,\mathrm{mean}} \oplus e_{dc,\mathrm{mean}}]$. **Có LayerNorm sau GRUCell** — comment nêu rõ: GRU thô có gain khoảng $4$ mỗi bước $\to$ nổ gradient qua tBPTT; LayerNorm ép hidden state để Jacobian xấp xỉ unit-gain. Carry $h_{prev}$ qua các bước pad.
- **`Fusion`**: $z_t = \mathrm{LayerNorm}\big(\mathrm{Linear}([h_a \oplus h_b])\big)$, kích thước $(B, d_{model})$.

**CẢNH BÁO** — đúng như review trước đã cảnh báo: $z_t$ **đã chứa thông tin mastery** (nhánh B đọc $M_t$) — nên "interaction representation" không thuần ngữ cảnh. Đây là lựa chọn cài đặt có ý thức (nhánh B *là* knowledge context), không phải bug.

---

## 3. Module 2 — Distributional Alignment (`distribution.py`, `patterns.py`)

### 2.1 Projection (`GaussianProjection`)
$z_t \to$ hai Linear head $\to \mu$ và $\mathrm{logvar}$ (clamp vào $[-6, 2]$). Khi ablation `use_distributional=False`: $\mathrm{logvar} \equiv 0 \to$ pattern suy biến thành pooling embedding điểm. **CẢNH BÁO: Gaussian, không phải Beta** — nên câu hỏi "Beta trên cái gì" trong review trước không còn áp dụng ở code; thay vào đó là Gaussian trên latent $d_z = 64$ chiều.

### 2.2 Pattern Operators (`PatternOperators`) — pooling bằng moment matching
Bốn operator, mỗi cái sinh trọng số $w$ trên prefix $j \le t$ rồi gộp Gaussian theo moment (giải quyết đúng vấn đề "đại số phân phối" mà review nêu):

$$
\mu_P = \sum_j w_j \mu_j
$$
$$
\mathrm{var}_P = \sum_j w_j \big(\mathrm{var}_j + \mu_j^2\big) - \mu_P^2 \quad \text{(floor } 10^{-3}\text{)}
$$

- **temporal**: exponential recency decay, rate **learnable** ($\mathrm{softplus}(\mathrm{decay})$), giới hạn cửa sổ `temporal_k=20`.
- **samekc / prereq / neighbor**: attention (dot-product $\mu_t \cdot \mu_j$, scale $1/\sqrt{d_z}$) **giới hạn trong support** theo mask quan hệ KC. Mask kích thước $(B,L,L)$ precompute một lần cho cả batch (`build_pattern_masks`), nhân với causality khi slice prefix.
- **Tập rỗng** $\to$ fallback về **null distribution học được** per-operator (`null_mu`, `null_logvar`). (Cũng là điểm review trước lo, đã xử lý.)

### 2.3 "Pedagogical Alignment" — CẢNH BÁO: hiện thực bằng LOSS, không phải transform
Trong code **không có bước $\tilde P_i = \mathrm{Align}(P_i, R_i)$ trong forward pass**. Không có file `alignment.py` (README liệt kê nhầm). Thay vào đó, "alignment" là **3 loss phụ** tính trong `dpa_kt.py`:
- **monotonicity** (`use_align_mono`): khi trả lời đúng, phạt phần mastery của KC liên quan bị *giảm* quá `mono_margin`: $\mathrm{ReLU}(m_{pre} - m_{post} - \mathrm{margin})$.
- **guess/slip** (`use_align_gs`): khuyến khích độ lớn cập nhật mastery tỉ lệ với "surprise" $(r - \hat y)^2$ (detach).
- **KL** (`use_distributional`): $\mathrm{KL}\big(\mathcal N(\mu,\sigma^2) \,\|\, \mathcal N(0,1)\big)$, trọng số rất nhỏ $10^{-4}$.

### 2.4 Readout
Mỗi operator: $v_i = \mathrm{Linear}([\mu_P \oplus \log \mathrm{var}_P])$, cho ra block chiều $d_z = 64$. $z' = [v_{temp}; v_{same}; v_{pre}; v_{nb}]$; operator bị tắt đóng góp toàn số không. Đây là **moment readout** (không sampling) $\to$ khả vi trực tiếp, không cần reparam.

---

## 4. Module 3 — Mastery State Tracking (`mastery.py`)

- **Bộ nhớ tường minh** $M \in \mathbb R^{B \times C \times d_v}$, khởi tạo từ tham số học $M_0$. `kc_key` embedding cho read/gating.
- **`read`**: localized attention **chỉ trên $K_{rel}$ KC liên quan** của câu hiện tại (query $= W_{read}(e_q)$), trả $m_{read}$ (đã LayerNorm) và $\alpha$. LayerNorm để kiểm soát vòng hồi tiếp $M \to \mathrm{read} \to \mathrm{GRU} \to \mathrm{patterns} \to M$.
- **`update`** — **DKVMN erase-add** trên các hàng KC liên quan:
  - Gating **Pattern $\to$ KC**: $A_i = \mathrm{softmax}_{\text{KC}}\big(\mathrm{keys} \cdot W_{gate}^{(i)}(v_i)\big)$ cho từng operator $\to$ đây chính là **khối diễn giải pattern-to-KC** của attribution trace.
  - erase $e = \sigma(W_{erase} v)$, add $a = \tanh(W_{add} v)$; $\mathrm{keep} \mathrel{*}= (1 - A \cdot e)$, $\mathrm{add} \mathrel{+}= A \cdot a$; $\mathrm{new} = \mathrm{rows} \cdot \mathrm{keep} + \mathrm{add}$. **CẢNH BÁO**: đây là phép nhân (bounded), **không phải cộng thuần** như proposal — giải quyết lo ngại "$M$ trôi/nổ" của review.
  - **Grad clipping THROUGH TIME**: `M_new.register_hook(g -> g.clamp(-c, c))`. Comment rất chi tiết: recurrence erase-add có Jacobian nhân dồn $\to$ ở $\mathrm{tbptt}=25$ grad norm khoảng $10^{11}$, bf16 overflow $\mathrm{inf}/\mathrm{nan}$ *trước khi* tới optimizer; clamp elementwise tại điểm gradient tái nhập recurrence giữ norm hữu hạn $\to$ cho phép chạy full sequence.
- `scalar_mastery`: $\sigma(\mathrm{Linear}(\mathrm{row})) \in (0,1)$ cho loss monotonicity + visualization.

---

## 5. Module 4 — Prediction (`predictor.py`)

- **KC $\to$ prediction contribution**: $\beta = \mathrm{softmax}_{\text{KC}}\big(\mathrm{keys} \cdot W_{pred}(e_q)\big)$ — khối diễn giải thứ hai của attribution trace.
- Read mastery theo $\beta$, LayerNorm, nối $[u \oplus e_q \oplus e_{dq}]$ $\to$ MLP $\to p_{master} \in (0,1)$.
- **Guess/slip head bounded**: $\hat y = (1-s) \cdot p_{master} + g \cdot (1 - p_{master})$, với $g = 0.35 \cdot \sigma(\theta_g)$ (khởi tạo $\approx 0.2$), $s = 0.30 \cdot \sigma(\theta_s)$ (khởi tạo $\approx 0.1$). Guess/slip là scalar học được, bị cap cứng.

---

## 6. Assembly và vòng thời gian (`dpa_kt.py`)

Causality mỗi bước $t$:
1. **Module 4 dự đoán $\hat y_t$** từ $M_t$ (dựng từ interactions $< t$) và identity/difficulty của $q_t$ — **chưa thấy $r_t$** (không leakage).
2. Sau đó Module 1(B) $\to$ 2 $\to$ 3 tiêu thụ interaction $t$ (gồm $r_t$) để cập nhật $M_t \to M_{t+1}$.

- `_related`: KC riêng của câu trước, rồi graph-expand (`q_rel`), cắt còn $k_{rel} = 20$.
- **tBPTT**: mỗi `tbptt` bước, detach $M$, $h_b$, và prefix $\mu/\mathrm{var}$. `base.yaml` chốt $\mathrm{tbptt} = 5$ với comment: full-BPTT ($200$) cộng clip *ổn định số* nhưng hội tụ *kém hơn* (assist09 AUC $0.677$ so với $0.717$); cửa sổ ngắn tuy noisy nhưng tới optimum tốt hơn.
- **Loss tổng**: $\mathcal L = \mathrm{BCE} + 0.1 \cdot \mathrm{mono} + 0.1 \cdot \mathrm{gs} + 10^{-4} \cdot \mathrm{kl}$. BCE tính tay ở fp32 (vì $\hat y$ là mixture guess/slip, không phải $\sigma(\mathrm{logits})$ $\to$ `F.binary_cross_entropy` không an toàn dưới autocast). Chỉ tính trên `selectmask & pad_mask`.
- `return_trace=True` xuất **attribution trace** đầy đủ: `pattern_w`, `gates` (pattern $\to$ KC), `beta` (KC $\to$ pred), `rel`, `alpha`, `mastery` evolution, `guess/slip`.

Trainer: AdamW, bf16 AMP, `ReduceLROnPlateau` trên val AUC, early stopping; checkpoint lưu model + optimizer + scheduler + epoch + RNG.

---

## 7. Khác biệt quan trọng so với proposal và short paper

1. **Backbone**: Transformer+GRU thay Mamba. Câu chuyện "Mamba tuyến tính" trong bài không khớp code — cần thống nhất framing (dùng "sequence encoder" trung tính, hoặc bổ sung Mamba thật).
2. **Gaussian thay Beta**: toàn bộ lập luận Beta trên $[0,1]$ trong proposal không đúng với code. Novelty vs UKT (Gaussian + Wasserstein) *thu hẹp lại* vì code cũng là Gaussian — điểm phân biệt còn lại là **pooling theo operator sư phạm + moment matching**, không phải "phân phối lạ".
3. **Alignment là regularization (loss), KHÔNG phải functional transform**: đây là điểm **ngược hẳn** với quyết định đã chốt của short paper ("rule-shaped functional form, không loss phụ"). Code đi theo nhánh "regularization" mà chính short paper đã bác. Phải reconcile: hoặc sửa bài theo code (thừa nhận alignment là auxiliary loss), hoặc sửa code theo bài.
4. **Monotonicity quay lại**: short paper đã *bỏ* monotonicity (thay bằng difficulty-response + KC-relation consistency). Code lại **có `use_align_mono`** (và không thấy hai principle mới kia). Mâu thuẫn trực tiếp với bản chốt; cần quyết lại.
5. **Đồ thị KC ước lượng từ data** (proposal giả định cho trước) — đúng với lo ngại graph-confound; cần ablation tách.
6. **Attribution là trọng số attention/gating**: vẫn là attention weights — giữ caveat faithfulness; chưa có intervention test.

---

## 8. Kết quả sẵn có (`runs-200-epochs/`, 200 epoch + 5-fold CV)

Chỉ full model có 5-fold (chưa thấy các fold ablation trong 200-epoch; ablation nằm ở runs-50-epochs). Ví dụ điểm test:
- **xes3g5m** fold0: AUC $0.802$, ACC $0.823$, RMSE $0.357$.
- **assist09** fold1/2: AUC $0.717$ / $0.721$, ACC xấp xỉ $0.70$.

Đã có fold cho: algebra05, assist09, assist12 (4 fold), bridge06, xes3g5m. **CẢNH BÁO**: so sánh literature chỉ là *chỉ dấu*, không head-to-head (README nói rõ: preprocessing/splits khác nhau; chỉ hàng "DPA-KT (ours)" là trên split của mình).

---

## 9. Ablation switches có sẵn (`config.ABLATIONS`)

`full`, `no_temporal`, `no_samekc`, `no_prereq`, `no_neighbor`, `no_mono`, `no_gs`, `no_distributional`, `single_branch` — đủ để chạy đúng các ablation review trước yêu cầu (tách từng operator, tách alignment, tách distributional, tách dual-branch).

---

## 10. Việc cần chú ý cho full paper

- **Reconcile mạnh giữa 3 tài liệu** (proposal / short paper / code): backbone, Beta-vs-Gaussian, alignment-là-loss-hay-transform, monotonicity còn hay bỏ. Hiện **ba tài liệu đang nói ba phiên bản khác nhau** — đây là rủi ro lớn nhất khi reviewer đối chiếu.
- `alignment.py` được README liệt kê nhưng **không tồn tại** — sửa README/kiến trúc mô tả cho khớp (alignment nằm rải trong `dpa_kt.py` dưới dạng loss).
- Bổ sung ablation graph-confound + consistency metric + faithfulness/intervention (đã có switch, chỉ cần chạy + báo cáo).
- Nếu giữ story "distributional pedagogical", cần lập luận vì sao Gaussian moment-matched pooling + null-fallback vẫn mang đúng ngữ nghĩa "evidence/pattern" — nối lại với các note lý thuyết trong proposal.
