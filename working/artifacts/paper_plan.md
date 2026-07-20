Mình nghĩ đây là thời điểm thích hợp để chuyển từ giai đoạn **thiết kế framework** sang **đóng gói research narrative**.

Sau nhiều vòng trao đổi, proposal hiện tại (mình sẽ gọi là **proposal_final.md / proposal_final.pdf**) đã không còn là một bản ghi ý tưởng rời rạc nữa, mà đã hình thành một câu chuyện nghiên cứu khá hoàn chỉnh:

> Existing KT directly maps interactions to mastery representations, whereas our framework explicitly introduces an intermediate pedagogical reasoning layer that transforms interaction evidence into structured learning patterns before updating knowledge states.

Đó mới là "main story" của short paper.

---

# Mục tiêu của short paper

Theo mình, short paper **không nên cố chứng minh toàn bộ mô hình**.

Nó nên trả lời được 4 câu hỏi.

1. Existing KT đang thiếu điều gì?
2. Vì sao thiếu điều đó?
3. Chúng tôi đề xuất góc nhìn mới nào?
4. Framework hiện thực hóa góc nhìn đó ra sao?

Toàn bộ paper nên xoay quanh 4 câu hỏi này.

---

# Dàn ý đề xuất

Mình đề xuất cấu trúc khoảng **4.5–5 trang IEEE 2 cột** như sau.

---

# Title

Không cần nhấn mạnh Mamba hay Beta.

Nên nhấn mạnh idea.

Ví dụ:

> **From Learning Interactions to Pedagogically Structured Evidence: A Distributional Framework for Knowledge Tracing**

hoặc

> **Towards Pedagogically Aligned Knowledge Tracing via Distributional Learning Pattern Modeling**

---

# Abstract (~180–220 words)

## Paragraph 1

Giới thiệu KT.

Có thể tham khảo cách mở đầu của PSI-KT và S²KT:

* KT dự đoán trạng thái tri thức từ lịch sử tương tác.
* Deep KT đạt accuracy cao nhưng còn hạn chế về khả năng mô hình hóa cấu trúc tri thức và tính sư phạm.  

---

## Paragraph 2

Research gap.

Không dùng kiểu

> Existing models are black-box.

Thay vào đó.

> Existing KT models update mastery directly from interaction representations. Consequently, pedagogical evidence contained in historical interactions is only implicitly encoded inside latent representations, making it difficult to organize, align, and explain the knowledge updating process.

Đây là research gap của chính proposal_final. 

---

## Paragraph 3

Research idea.

Một câu duy nhất.

> We propose inserting an intermediate distributional pedagogical reasoning stage between interaction encoding and mastery updating.

Sau đó liệt kê 4 module.

Reference trực tiếp đến framework tổng thể trong proposal_final. 

---

## Paragraph 4

Contribution.

Không nói performance.

Chỉ nói

* new representation
* pedagogical reasoning
* interpretable update
* extensible framework

---

# I. Introduction (~1 trang)

Theo mình phần này nên được đầu tư nhất.

---

## 1. KT background

Ngắn.

Reference

* DKT
* AKT
* MambaKT
* PSI-KT

Mục tiêu:

KT ngày càng mạnh về sequence modeling.

---

## 2. Existing research directions

Có thể chia thành bảng hoặc paragraph.

| Direction           | Representative papers |
| ------------------- | --------------------- |
| Sequential modeling | DKT, AKT, MCKT        |
| Structure-aware KT  | GKT, CMDKT, S²KT      |
| Interpretable KT    | PSI-KT, NSKT, KeenKT  |

Reference:

* GKT: graph prior và quan hệ KC. 
* PSI-KT: interpretable latent traits và prerequisite structure. 
* S²KT: Gaussian semantic representation và semantic propagation. 
* KeenKT: distributional mastery representation. 

---

## 3. Research gap

Đây là trọng tâm.

Không nên liệt kê thiếu sót từng paper.

Nên gom thành 3 mức abstraction.

### Gap 1

Interaction representation

↓

Mastery

quá trực tiếp.

---

### Gap 2

Không tồn tại tầng

Pedagogical Evidence.

---

### Gap 3

Knowledge update

không được tổ chức theo learning patterns.

---

Có thể minh họa bằng một hình rất đơn giản.

Existing

```
Interaction

↓

Mastery

↓

Prediction
```

Ours

```
Interaction

↓

Pedagogical Evidence

↓

Knowledge

↓

Prediction
```

Theo mình đây sẽ là Figure 1 của paper.

---

## 4. Research hypothesis

Một paragraph.

Ví dụ.

> Student knowledge should be updated from structured pedagogical evidence rather than directly from isolated interaction representations.

Đây là "linh hồn" của paper.

---

## 5. Contributions

Ba ý.

Contribution 1

Framework.

Contribution 2

Distributional pedagogical alignment.

Contribution 3

Intrinsic interpretability.

---

# II. Proposed Framework (~2 trang)

Toàn bộ section này chỉ cần một hình lớn (framework tổng thể trong proposal_final, đơn giản hóa cho paper) và 4 subsection.

Reference trực tiếp đến proposal_final. 

---

## A. Overall Framework

Một hình.

Không đưa chi tiết decoder.

Không đưa operator.

Chỉ:

Interaction Space

↓

Distribution Space

↓

Knowledge Space

↓

Prediction Space

Đây chính là 4 tầng logic của proposal_final. 

---

## B. Interaction Representation Learning

Chỉ nói motivation.

Reference

* MCKT

parallel Mamba

* CMDKT

topology-aware

Không mô tả implementation.

Chỉ nói

Context-aware interaction representation.

---

## C. Distributional Pedagogical Alignment

Đây là phần mới nhất.

Nên dành khoảng 40% section.

Gồm đúng 4 bước.

* Distribution Projection
* Pattern Construction
* Pedagogical Alignment
* Pattern Readout

Reference proposal_final. 

Ở mỗi bước chỉ cần 3–5 câu.

Liên hệ công trình trước:

* PLKT → distribution space.
* NSKT → pedagogical rules.
* GKT/S²KT → structured relations.
* Nhưng nhấn mạnh: các ý tưởng này được tái cấu trúc để xây dựng **interaction evidence**, không phải trực tiếp biểu diễn mastery.

---

## D. Mastery State Tracking

Chỉ nói

Aligned Pattern

↓

Knowledge State

Nhấn mạnh:

pattern contribution

↓

KC update

Không đi vào attention.

---

## E. Prediction Aggregation

Mastery

*

Target Question

↓

Prediction

Thêm một câu:

KC contribution

↓

Prediction explanation

---

# III. Discussion (~0.7 trang)

Section này rất đáng giá trong short paper.

Mình sẽ chia thành 3 subsection.

---

## A. Why Distribution?

Liên hệ

PLKT

S²KT

KeenKT.

Khác biệt:

distribution ở đây

không biểu diễn mastery

mà biểu diễn interaction evidence.

---

## B. Why Pedagogical Alignment?

Liên hệ

NSKT.

Khác biệt:

rules

không update mastery

mà regularize pattern.

---

## C. Why Explainability?

Liên hệ

PSI-KT

Interpretable KT

Khác biệt:

không post-hoc

không attention visualization

mà giải thích được theo chuỗi:

Pattern → KC → Prediction.

---

# IV. Conclusion (~0.3 trang)

Một paragraph.

Không nói

Future work:

Benchmark.

Nên nói

Framework mở đường cho

* distributional reasoning
* pedagogically grounded KT
* interpretable knowledge evolution

---

# Hệ thống hình

Theo mình chỉ cần 3 hình.

---

## Figure 1

Research Motivation

Existing

vs

Ours

---

## Figure 2

Overall Framework

(lấy từ proposal_final, rút gọn)

---

## Figure 3

Distributional Pedagogical Alignment

(rút riêng module 2)

Không cần đưa hình chi tiết của module 1 hay module 3 trong short paper.

---

# Hệ thống công thức

Theo mình chỉ nên giữ **4–5 công thức**.

1. $\mathbf z_t = Fusion(...)$ (Module 1)

2. $h_t=\Phi(z_t)$ (Module 2)

3. $P_i=\mathcal A_i(H)$ (Pattern)

4. $M_t=f(\tilde P)$ (Mastery)

5. $\hat y=f(M_t,q_{t+1})$ (Prediction)

Toàn bộ các công thức chi tiết (tham số Beta, decoder, alignment, contribution score...) nên để dành cho phiên bản full paper.

---

## Vai trò của các công trình tham khảo trong paper

Nếu quy về narrative thay vì implementation, mình sẽ dùng chúng theo đúng "vai":

| Vai trò                             | Công trình nên nhắc                                                                                                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sequential interaction modeling     | DKT, AKT, MCKT                                                                                                                                                       |
| Structure-aware KT                  | GKT, CMDKT, S²KT                                                                                                                                                     |
| Distributional representation       | PLKT, S²KT, KeenKT                                                                                                                                                   |
| Pedagogical / rule-guided reasoning | NSKT                                                                                                                                                                 |
| Interpretable KT                    | PSI-KT, KeenKT                                                                                                                                                       |
| Khởi nguồn framework đề xuất        | proposal_final.md / proposal_final.pdf (4 module, 4 representation spaces, Distributional Pedagogical Alignment, Mastery State Tracking và Prediction Aggregation)   |

Theo mình, với cấu trúc này, short paper sẽ không giống một bản rút gọn của proposal mà trở thành một **idea paper** có câu chuyện rõ ràng: vấn đề → giả thuyết → framework → ý nghĩa khoa học. Điều đó phù hợp hơn với giới hạn 4–5 trang và tạo nền tảng tốt để mở rộng thành full paper sau này.
