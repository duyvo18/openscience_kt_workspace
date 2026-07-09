# Mô hình tổng thể

Đề xuất xây dựng một framework KT gồm bốn module chính tương ứng với bốn giai đoạn của quá trình theo dõi trạng thái tri thức.

* **Học biểu diễn vector ẩn của tương tác lịch sử - Interaction Representation Learning**

  * Học biểu diễn ngữ cảnh của từng tương tác từ lịch sử học tập.

* **Điều chỉnh sư phạm trên biểu diễn phân phối - Distributional Pedagogical Alignment**

  * Chuyển đổi biểu diễn tương tác sang không gian phân phối.
  * Xây dựng các learning pattern theo nhiều góc nhìn sư phạm.
  * Điều chỉnh các pattern theo các nguyên lý sư phạm trước khi cập nhật mastery.

* **Theo dõi trạng thái tri thức - Mastery State Tracking**

  * Tổng hợp các pattern đã được điều chỉnh để cập nhật Knowledge State của học sinh.
  * Đồng thời mô hình hóa mức độ đóng góp của từng learning pattern lên từng KC nhằm phục vụ khả năng diễn giải nội tại của quá trình cập nhật tri thức.

* **Prediction Aggregation**

  * Tổng hợp Knowledge State và thông tin câu hỏi mục tiêu để dự đoán kết quả của tương tác tiếp theo.
  * Đồng thời mô hình hóa mức độ đóng góp của từng KC lên kết quả dự đoán nhằm phục vụ khả năng diễn giải của mô hình.

Toàn bộ framework được tổ chức thành bốn tầng biểu diễn liên tiếp:

* Interaction Space
* Distribution Space
* Knowledge Space
* Prediction Space

Cơ bản:

```mermaid
flowchart LR

B[Interaction Representation Learning]

B-->C[Distributional Pedagogical Alignment]

C-->D[Mastery State Tracking]

D-->E[Prediction Aggregation]
```

---

# Mô hình chi tiết

## 1. Interaction Representation Learning

Module đầu tiên chịu trách nhiệm học biểu diễn ngữ cảnh của từng interaction từ lịch sử học tập.

Đầu vào của module được chia thành hai nhóm thông tin với hai loại động lực học khác nhau.

* **Nhóm tương tác - Interaction Context**

  * Question Semantic
  * Student Response
  * Question Difficulty

* **Nhóm tri thức - Knowledge Context**

  * Mastery Memory
  * Knowledge Graph
  * Concept Difficulty

Trong đó, Mastery Memory không được sử dụng trực tiếp mà trước tiên được truy xuất theo ngữ cảnh của câu hỏi hiện tại thông qua Knowledge Graph để tạo thành biểu diễn mastery cục bộ.

Hai nhóm thông tin này được đưa qua hai phân nhánh Mamba song song nhằm học hai loại dynamics khác nhau:

* Dynamics của chuỗi tương tác.
* Dynamics của trạng thái tri thức cục bộ.

Sau đó, hai biểu diễn được hợp nhất để tạo thành biểu diễn interaction cuối cùng.

```mermaid
flowchart LR

Q[Question]

R[Response]

QD[Question Difficulty]

KG[Knowledge Graph]

M[Mastery Memory]

CD[Concept Difficulty]

Q-->OBS

R-->OBS

QD-->OBS

OBS[Interaction Context]

OBS-->MB1[Mamba]

KG-->READ

M-->READ

READ[Localized Mastery]

READ-->KN

CD-->KN

KN[Knowledge Context]

KN-->MB2[Mamba]

MB1-->F

MB2-->F

F[Fusion]

F-->Z[Interaction Representation]
```

Cơ bản:

$$
\mathbf z_t = Fusion ( Mamba_{interaction}(I_t), Mamba_{knowledge}(K_t) )
$$

Trong đó

* $I_t$ là Interaction Context.
* $K_t$ là Knowledge Context.

### Note

Khác với proposal v0.1, semantic embedding không chỉ biểu diễn nội dung câu hỏi mà còn mang nhận thức về trạng thái tri thức hiện tại của học sinh thông qua Mastery Memory và Knowledge Graph.

Hai nhánh Mamba không còn được tách theo semantic và difficulty mà theo hai loại dynamics khác nhau của quá trình học tập.

---

## 2. Distributional Pedagogical Alignment

Module thứ hai thực hiện xây dựng và điều chỉnh learning pattern trên không gian phân phối.

Khác với PLKT, phân phối trong đề xuất không đại diện cho mastery belief hay learner belief.

Thay vào đó, phân phối đóng vai trò là **một không gian biểu diễn trung gian của interaction pattern**, cho phép thực hiện các phép tổng hợp và điều chỉnh theo lý thuyết sư phạm trước khi cập nhật Knowledge State.

Module này gồm bốn bước:

* Biến đổi biểu diễn phân phối - Distribution Projection
* Tổng hợp mẫu hình - Pattern Construction
* Tinh chỉnh sư phạm - Pedagogical Alignment
* Biến đổi vector - Pattern Readout

### 2.1 Distribution Projection

Đầu tiên, biểu diễn interaction được chuyển sang biểu diễn phân phối.

```mermaid
flowchart LR

Z[Interaction Representation]

-->Decoder[Statistical Descriptor Decoder]

-->H[Distribution Representation]
```

Sơ bộ,

$$
h_t = \Phi(z_t)
$$

Trong đó

$$ h_t=(\alpha_t,\beta_t) $$

đối với Beta Distribution.

#### Note

Statistical Descriptor Decoder học phép ánh xạ từ interaction representation sang các tham số của phân phối.

Không gian phân phối không biểu diễn mastery mà chỉ đóng vai trò là không gian biểu diễn của interaction pattern.

### 2.2 Pattern Construction

Sau khi thu được chuỗi phân phối, mô hình xây dựng nhiều learning pattern thông qua các Pattern Operator.

Khác với proposal trước, pattern không được định nghĩa bởi số lượng interaction mà được định nghĩa bởi các phép tổng hợp cố định.

Đề xuất sử dụng:

* Theo thời gian - Temporal Pattern Operator
* Cùng KC - Same-Knowledge-Component Operator
* KC tiên quyết - Prerequisite Operator
* KC láng giềng - Neighbor Concept Operator

```mermaid
flowchart LR

Distribution

-->Temporal

Distribution

-->SameKC

Distribution

-->Prerequisite

Distribution

-->Neighbor

Temporal-->Patterns

SameKC-->Patterns

Prerequisite-->Patterns

Neighbor-->Patterns
```

Sơ bộ,

$$
P_i = \mathcal A_i(H)
$$

#### Note

Mỗi Pattern Operator luôn mang cùng một ý nghĩa sư phạm.

Do đó, cấu trúc Pattern Representation luôn nhất quán giữa mọi học sinh bất kể lịch sử học tập có độ dài hay phân bố khác nhau.

### 2.3 Pedagogical Alignment

Sau khi xây dựng các learning pattern, mô hình thực hiện điều chỉnh các pattern bằng tập luật sư phạm.

Khác với NSKT, các luật không trực tiếp suy luận mastery mà chỉ điều chỉnh biểu diễn phân phối nhằm đưa các learning pattern tiến gần hơn với các nguyên lý sư phạm.

```mermaid
flowchart LR

Patterns

-->Rules

Rules

-->AlignedPatterns
```

Sơ bộ,

$$
\tilde P_i = Align(P_i,R_i)
$$

Trong đó

* $P_i$ là learning pattern ban đầu.
* $R_i$ là tập luật sư phạm.
* $\tilde P_i$ là pattern sau điều chỉnh.

#### Note

Pedagogical Alignment đóng vai trò như một cơ chế regularization trên không gian phân phối thay vì một hệ luật suy diễn.

### 2.4 Pattern Readout

Sau khi hoàn thành quá trình điều chỉnh, mô hình ánh xạ các learning pattern trở lại không gian vector.

```mermaid
flowchart LR

AlignedPatterns

-->Readout

-->PatternVector
```

Sơ bộ,

$$
z'_i = Readout(\tilde P_i)
$$

Trong đó

$$
z' = [ z^{temp}, z^{same}, z^{pre}, z^{neighbor} ]
$$

là structured pattern representation.

#### Note

Pattern Vector không phải latent vector ngẫu nhiên mà là biểu diễn có cấu trúc, trong đó mỗi block luôn tương ứng với một loại learning pattern xác định.

Điều này giúp đảm bảo tính nhất quán của đầu vào cho quá trình cập nhật mastery.

---

## 3. Mastery State Tracking

Module thứ ba chịu trách nhiệm duy trì và cập nhật trạng thái tri thức của học sinh.

Khác với proposal trước, Knowledge State trở thành trạng thái tường minh của mô hình và được cập nhật trực tiếp từ structured pattern representation.

Ngoài việc cập nhật mastery, module còn học **mức độ đóng góp của từng learning pattern lên giới hạn KC liên quan**, từ đó cho phép truy vết quá trình hình thành Knowledge State sau mỗi interaction.

```mermaid
flowchart LR

PatternVector

-->Contribution[Pattern-to-KC Gating Function]

Contribution

-->UpdateNetwork

UpdateNetwork

-->DeltaMastery

DeltaMastery

-->MasteryMemory
```

Sơ bộ,

$$
A_i = G(P_i)
$$

Trong đó

* $A_i$ biểu diễn mức độ ảnh hưởng của learning pattern thứ $i$ lên giới hạn các KC liên quan.

Sau đó,

$$
\Delta M = \sum_i A_i \odot U_i(M_t,P_i)
$$

Cuối cùng,

$$
M_{t+1} = M_t+\Delta M
$$

### Note

Thay vì chỉ học trực tiếp hàm cập nhật mastery, mô hình đồng thời học thêm **Pattern-to-Knowledge Contribution** nhằm lượng hóa ảnh hưởng của từng learning pattern lên từng KC.

Các trọng số đóng góp này không tham gia trực tiếp vào quá trình dự đoán mà đóng vai trò diễn giải nội tại cho quá trình cập nhật Knowledge State.

Nhờ đó, mô hình có thể trả lời các câu hỏi như:

* Learning pattern nào làm tăng mastery của một KC?
* KC nào chịu ảnh hưởng mạnh nhất từ một learning pattern cụ thể?

---

## 4. Prediction Aggregation

Module cuối cùng thực hiện dự đoán kết quả của tương tác tiếp theo.

Đầu vào của module gồm:

* Knowledge State sau cập nhật.
* Biểu diễn của câu hỏi mục tiêu.

Ngoài xác suất dự đoán, module đồng thời **học mức độ đóng góp của từng KC** lên kết quả dự đoán nhằm cung cấp khả năng diễn giải của mô hình.

```mermaid
flowchart LR

Mastery

-->KCContribution[KC Contribution]

KCContribution

-->Prediction

TargetQuestion

-->Prediction

Prediction

-->Probability
```

Sơ bộ,

$$
W = C(M_{t+1},q_{t+1})
$$

Trong đó

* $W=[w_1,w_2,\ldots,w_C]$ biểu diễn mức độ đóng góp của từng KC lên tương tác mục tiêu.

Sau đó,

$$
\hat y = f(M_{t+1},q_{t+1},W)
$$

### Note

Prediction Aggregation không chỉ sinh ra xác suất trả lời đúng mà còn đồng thời sinh ra **KC Contribution**.

Kết hợp với **Pattern-to-Knowledge Contribution** từ Module 3, mô hình có thể truy vết toàn bộ chuỗi hình thành kết quả dự đoán:

```text
Historical Interactions
        │
        ▼
Learning Patterns
        │
        ▼
Pattern → Knowledge Contribution
        │
        ▼
Updated Knowledge State
        │
        ▼
Knowledge → Prediction Contribution
        │
        ▼
Prediction
```

Do đó, khả năng diễn giải của mô hình được hình thành một cách **intrinsic** trong chính quá trình cập nhật tri thức và dự đoán, thay vì được bổ sung bởi một module giải thích độc lập sau khi mô hình đã hoàn thành dự đoán. 