# DPA-KT — Chi tiết DKVMN erase-add & các loss alignment

> Trích trực tiếp từ `dpa_kt/models/mastery.py`, `dpa_kt/models/dpa_kt.py`, `dpa_kt/training/trainer.py`.

## 1. Luồng đề xuất so với luồng code (time loop, thứ tự 4 -> 1 -> 2 -> 3)

Trước vòng lặp: Branch A (Transformer) chạy 1 lần cho cả chuỗi (không phụ thuộc $M$).

Mỗi bước $t$:
1. (Module 4) $\hat y_t, \beta_t = \mathrm{PredictionHead}(M, \mathrm{rel}_t, e_q, e_{dq})$ — dự đoán từ $M_t$, CHƯA thấy $r_t$ (không leakage)
2. (Module 1) $m_{read} = \mathrm{Mastery.read}(M, \mathrm{rel}_t, e_q)$; $h_b = \mathrm{BranchBCell}(\ldots)$; $z_t = \mathrm{Fusion}(h_a[t], h_b)$ — biểu diễn lịch sử
3. (Module 2) $\mu_t, \mathrm{logvar}_t = \mathrm{GaussianProjection}(z_t)$; $\mathrm{pats} = \mathrm{PatternOperators}(\mathrm{prefix})$ — mẫu hình sư phạm
4. (Module 3) $M_{new}, \mathrm{gates} = \mathrm{Mastery.update}(M, \mathrm{rel}_t, \mathrm{pats})$ — cập nhật mastery; tính $\mathrm{mono}_t$, $\mathrm{gs}_t$
5. $M \leftarrow M_{new}$; mỗi $\mathrm{tbptt}=5$ bước detach $M$, $h_b$, $\mu_{prefix}$, $\mathrm{var}_{prefix}$

Quy ước "predict-then-update" chuẩn KT: $\hat y_t$ dùng $M$ tích lũy từ các bước $< t$. $\mu_{prefix}$/$\mathrm{var}_{prefix}$ bị detach mỗi 5 bước, nên pattern pooling chỉ nhận gradient trong cửa sổ tBPTT hiện tại.

## 2. DKVMN erase-add (`mastery.py :: update`)

Chỉ $K_{rel}=20$ hàng KC liên quan được gather và cập nhật; phần còn lại đi qua `scatter` không đổi.

Với mỗi operator $i \in \{\mathrm{temporal}, \mathrm{samekc}, \mathrm{prereq}, \mathrm{neighbor}\}$, từ readout $v_i$ (chiều $d_z=64$):

$$
A_i = \mathrm{softmax}_{\text{related KC}}\big( \mathrm{keys} \cdot W_{gate}^{(i)}(v_i) \big) \quad \text{— (B,K\_rel), Pattern} \to \text{KC (attribution)}
$$
$$
e_i = \sigma\big( W_{erase}^{(i)}(v_i) \big) \in (0,1)^{d_v}
$$
$$
a_i = \tanh\big( W_{add}^{(i)}(v_i) \big) \in (-1,1)^{d_v}
$$

Gộp qua 4 operator:

$$
\mathrm{keep} = \prod_i \big( 1 - A_i \cdot e_i \big) \quad \text{— erase nhân dồn, bounded trong } (0,1)
$$
$$
\mathrm{add} = \sum_i \big( A_i \cdot a_i \big)
$$
$$
\mathrm{new\_rows} = \mathrm{rows} \cdot \mathrm{keep} + \mathrm{add} \quad \text{— DKVMN erase-add (có chặn, khác } M_{t+1}=M_t+\Delta M \text{ cộng thuần)}
$$
$$
M_{new} = M.\mathrm{scatter}(\mathrm{related} \leftarrow \mathrm{new\_rows})
$$

- $A_i$ được tính cho cả operator bị tắt (phục vụ trace), nhưng operator tắt ($v_i$ toàn số không) bị bỏ qua trong $\mathrm{keep}$/$\mathrm{add}$.
- Gradient: chảy qua $M$ cũ (recurrence + identity của `scatter`) và qua $v_i$ (patterns $\to$ gauss $\to$ fusion $\to$ encoder $\to$ embeddings) thông qua $W_{erase}, W_{add}, W_{gate}$, và $\mathrm{keys} = \mathrm{kc\_key}$.
- **Grad-clip-through-time**: `M_new.register_hook(lambda g: g.clamp(-1.0, +1.0))`. Chặn elementwise ngay tại điểm gradient tái nhập recurrence: Jacobian của erase-add nhân dồn qua nhiều bước — ở $\mathrm{tbptt}=25$ grad norm khoảng $10^{11}$, dưới bf16 tràn thành $\mathrm{inf}/\mathrm{nan}$ trước khi tới optimizer, nên `clip_grad_norm_` toàn cục không cứu được. Clamp per-step giữ norm tổng hữu hạn — đây là điều cho phép tBPTT chạy full sequence (dù `base.yaml` vẫn chốt $\mathrm{tbptt}=5$ vì hội tụ tốt hơn).

## 3. Ba loss alignment (`dpa_kt.py`) — KHÔNG phải transform, mà là loss phụ

### (a) Monotonicity — $w_{mono}=0.1$

$$
m_{pre}  = \mathrm{scalar\_mastery}\big(M.\mathrm{gather}(\text{own KC})\big) \in (0,1) \quad \text{— trước update}
$$
$$
m_{post} = \mathrm{scalar\_mastery}\big(M_{new}.\mathrm{gather}(\text{own KC})\big) \quad \text{— sau update}
$$
$$
\mathrm{correct\_mask} = \mathrm{own\_valid} \;\land\; (r_t = 1) \quad \text{— chỉ interaction ĐÚNG}
$$
$$
\mathrm{drop} = \mathrm{ReLU}\big( m_{pre} - m_{post} - 0.05 \big) \quad \text{(} \mathrm{mono\_margin}=0.05 \text{)}
$$
$$
\mathrm{mono}_t = \frac{\sum (\mathrm{drop} \cdot \mathrm{correct\_mask})}{\sum \mathrm{correct\_mask}}
$$

Hinge một phía: khi trả lời đúng, phạt nếu scalar-mastery của chính KC đó giảm quá $0.05$.

Gradient: chảy vào $m_{post}$ (mọi tham số update: $W_{erase}$/$W_{add}$/$W_{gate}$, patterns, gauss, encoder, embeddings), vào $m_{pre}$ ($M$ cũ, qua recurrence tới điểm detach), và vào $\mathrm{scalar\_u}$ (head của $\mathrm{scalar\_mastery}$). Không có gì bị detach ở đây.

### (b) Guess/Slip — $w_{gs}=0.1$

$$
\mathrm{surprise} = \big( (r_t - \hat y_t)^2 \big)_{\mathrm{detach}} \quad \text{— DETACH: chỉ là trọng số per-sample}
$$
$$
\mathrm{delta} = \mathrm{mean}_{d_v}\big( (\mathrm{own\_rows\_post} - \mathrm{own\_rows\_pre})^2 \big)
$$
$$
\mathrm{delta} = \mathrm{masked\_mean}_{\text{KC}}(\mathrm{delta})
$$
$$
\mathrm{gs}_t = \frac{\sum \big( \mathrm{surprise} \cdot \mathrm{delta} \cdot \mathrm{step\_valid} \big)}{\sum \mathrm{step\_valid}}
$$

Gradient: $\mathrm{surprise}$ bị `detach()`, nên không có gradient qua $\hat y$ — nó chỉ đóng vai trò hệ số trọng số per-sample. Gradient chỉ chảy qua $\mathrm{delta}$ (tới $M_{new}$, $M$, và các tham số update).

**Điểm cần xác nhận về mặt ngữ nghĩa:** $\mathrm{gs}_t = \mathrm{surprise} \cdot \mathrm{delta}$ được cộng vào loss và tối thiểu hoá. Vì $\mathrm{surprise}$ là hằng số (detached), tối thiểu hoá $\mathrm{surprise} \cdot \mathrm{delta}$ sẽ làm giảm $\mathrm{delta}$, và giảm mạnh hơn ở những bước $\mathrm{surprise}$ cao — tức là code đang kìm biên độ cập nhật mastery ở đúng các interaction gây bất ngờ nhiều. Có hai cách đọc:
- **Đọc thuận với tên "guess/slip":** interaction gây bất ngờ lớn có thể do đoán mò (guess) hoặc sơ suất (slip), nên đừng cập nhật mastery mạnh theo nhiễu đó. Cách đọc này hợp lý.
- **Nhưng README mô tả** là khuyến khích biên độ cập nhật "scale với" surprise, tức muốn cập nhật nhiều hơn khi bất ngờ. Nếu đó là ý định thật thì dấu của loss đang ngược (cần $-\mathrm{surprise} \cdot \mathrm{delta}$ hoặc $(\mathrm{target\_delta} - \mathrm{delta})^2$ thay vì công thức hiện tại).

Đây là mâu thuẫn giữa code và mô tả — nên chốt lại ý định trước khi viết full paper.

### (c) KL — $w_{kl}=10^{-4}$

$$
l_{kl} = \mathrm{mean}_{\text{valid}}\left( \frac{1}{2} \sum_{d_z} \big( \sigma^2 + \mu^2 - 1 - \log \sigma^2 \big) \right) \quad \text{— } \mathrm{KL}\big(\mathcal N(\mu,\sigma^2) \,\|\, \mathcal N(0,1)\big)
$$

Kéo phân phối projection về gần chuẩn tắc (chống $\mu, \mathrm{logvar}$ trôi). Trọng số cực nhỏ nên chủ yếu là ổn định số. Gradient chỉ tới hai head $\mu$/$\mathrm{logvar}$ của `GaussianProjection` (và ngược lên encoder).

## 4. Loss tổng và cơ chế cập nhật gradient (`trainer.py`)

$$
\mathcal L = \mathrm{BCE} + 0.1 \cdot \mathrm{mean}(\mathrm{mono}) + 0.1 \cdot \mathrm{mean}(\mathrm{gs}) + 10^{-4} \cdot \mathrm{KL}
$$

- **BCE**: tính tay ở fp32 trên $\mathrm{selectmask} \land \mathrm{pad\_mask}$, với $\hat y$ được clamp vào $[10^{-6}, 1-10^{-6}]$. Lý do tính tay: $\hat y$ là hỗn hợp guess/slip $\hat y = (1-s)\,p + g\,(1-p)$, không phải $\sigma(\mathrm{logits})$, nên `F.binary_cross_entropy` không an toàn dưới autocast.
- Cả 4 số hạng chia sẻ một lần `loss.backward()`. BCE chảy ngược qua prediction head, mastery read, và toàn bộ upstream (recurrence, patterns, encoder, embeddings); mono/gs chảy qua đường update mastery; KL chỉ tới gauss projection.

Vòng cập nhật:
```
with autocast(bf16):                              # AMP bf16, không dùng GradScaler
    out = model(batch)
optimizer.zero_grad(set_to_none=True)
loss.backward()                                    # hook clamp +-1.0 trên M kích hoạt tại đây
gnorm = clip_grad_norm_(params, grad_clip=5.0)     # clip norm toàn cục
if not isfinite(gnorm): continue                   # bỏ qua step hỏng, không nhiễm NaN vào weight
optimizer.step()                                   # AdamW, lr=1e-3, weight_decay=1e-5
```

Có hai tầng chặn gradient: (1) elementwise clamp $\pm 1.0$ trên $M$ ngay trong recurrence (per-step, chống overflow bf16); (2) `clip_grad_norm_` $= 5.0$ trên toàn bộ tham số sau backward. Optimizer là AdamW; scheduler `ReduceLROnPlateau` (factor $0.5$, patience $2$) trên val AUC; early stopping patience $5$–$10$.
