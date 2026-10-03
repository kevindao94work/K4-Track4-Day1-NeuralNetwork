# Báo cáo Lab Day 1 — MSSV 2A202602793

> Báo cáo này được tạo từ kết quả notebook, các tệp JSON thí nghiệm và `eval_result.json`. Mọi lựa chọn cấu hình được quyết định bằng validation trước khi mở eval.

## 1. Thiết lập

- Môi trường: Python/PyTorch 2.6.0, thiết bị `cpu`. CUDA khả dụng: `False`. GPU: không có GPU CUDA.
- Dữ liệu: Forest CoverType; train 464.809 và eval 116.203 dòng theo metadata. Validation là 20% train, phân tầng với seed 42: train 371.847, val 92.962, eval 116.203; 54 đặc trưng, nhãn 0–6. Chuẩn hoá 10 cột số chỉ fit trên train; các cột nhị phân và `eval_row_id` được giữ nguyên.
- Mô hình: M-base, 54→256→128→7, 47.879 tham số, output là logits. Baseline dùng CE, SGD+momentum 0,9, learning rate 0.100, batch 512, 20 epoch, He, seed 42/123.
- Accuracy đoán lớp đa số trên val: 0.4876. Chủ đề đã thử: loss, optimizer, batch size, dropout, clipping, BF16 autocast trên CPU và khởi tạo.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47.879 / (B, 7) |
| Loss CE bước 0, so với ln(7)=1.9459 | 2.3776 |
| Quá khớp 20 mẫu train | loss 0.000302, accuracy 100.0%, 30 bước |
| Gradient mọi tensor của M-base | khác 0 sau backward |
| Baseline | 2 seed; `20261003-124413-base-s42`, `20261003-124413-base-s123` |
| Val accuracy TB ± độ lệch chuẩn | 0.9089 ± 0.0045 |
| Val macro-F1 TB ± độ lệch chuẩn | 0.8519 ± 0.0004 |

Dùng 2σ=0.0007 làm mốc tham khảo cho chênh lệch validation macro-F1. Ước lượng độ nhiễu còn yếu vì chỉ có hai seed baseline.

## 3. Kết quả theo chủ đề

### 3.1 Loss — CE và MSE

Dự đoán trước: MSE trên logits có thể hội tụ chậm hơn CE. Baseline `20261003-124413-base-s42` đạt macro-F1 0.8522; MSE `20261003-124413-loss-mse-s42` đạt 0.7374 (accuracy 0.8678). MSE thấp hơn; trị số loss CE và MSE không so trực tiếp do cách định nghĩa/thang đo khác nhau. [Hình so sánh](figures/compare_loss.png).

### 3.2 Bộ tối ưu hoá

Mỗi optimizer được thử hai learning rate; bảng ghi toàn bộ các lượt và best epoch trên val.

| Optimizer | lr | exp_id | Val macro-F1 | Best epoch | Hình |
|---|---:|---|---:|---:|---|
| sgd | 0.03 | `20261003-124413-opt-sgd-0p03-s42` | 0.6400 | 16 | [hình](figures/20261003-124413-opt-sgd-0p03-s42.png) |
| sgd | 0.1 | `20261003-124413-opt-sgd-0p1-s42` | 0.7446 | 18 | [hình](figures/20261003-124413-opt-sgd-0p1-s42.png) |
| sgd_momentum | 0.1 | `20261003-124413-base-s42` | 0.8522 | 18 | [hình](figures/20261003-124413-base-s42.png) |
| sgd_momentum | 0.03 | `20261003-124413-opt-sgd_momentum-0p03-s42` | 0.8044 | 16 | [hình](figures/20261003-124413-opt-sgd_momentum-0p03-s42.png) |
| adam | 0.001 | `20261003-124413-opt-adam-0p001-s42` | 0.8486 | 19 | [hình](figures/20261003-124413-opt-adam-0p001-s42.png) |
| adam | 0.003 | `20261003-124413-opt-adam-0p003-s42` | 0.8712 | 20 | [hình](figures/20261003-124413-opt-adam-0p003-s42.png) |
| adamw | 0.001 | `20261003-124413-opt-adamw-0p001-s42` | 0.8486 | 19 | [hình](figures/20261003-124413-opt-adamw-0p001-s42.png) |
| adamw | 0.003 | `20261003-124413-opt-adamw-0p003-s42` | 0.8712 | 20 | [hình](figures/20261003-124413-opt-adamw-0p003-s42.png) |

Trong lưới này, Adam và AdamW với lr=0,003 cao nhất ở 0.8712; chênh baseline seed 42 là 0.0190, lớn hơn mốc 2σ. AdamW đặt `weight_decay=0`, nên trùng Adam. SGD nhạy hơn trong hai learning rate đã thử; không suy rộng ra ngoài lưới nhỏ này. [Hình nhóm](figures/compare_optimizer.png).

### 3.3 Batch size

Chỉ đổi batch 512→1024: `20261003-124413-hparam-batch1024-s42` có macro-F1 0.8335, thời gian/epoch 0.398 giây. Batch 512/1024 cho lần lượt 727/364 batch mỗi epoch; do đó cùng 20 epoch vẫn có số bước cập nhật khác. [Hình](figures/compare_hparam.png).

### 3.4 Dropout

Dự đoán: dropout p=0,2 có thể giảm khoảng cách train–val nếu baseline quá khớp. Baseline có chênh loss cuối 0.0240; dropout `20261003-124413-dropout-p02-s42` có chênh 0.0082, nhưng macro-F1 0.8161 thấp hơn baseline 0.8522. Thu hẹp loss gap không đồng nghĩa với metric validation tốt hơn. [Hình](figures/compare_dropout.png).

### 3.5 Gradient clipping

Ngưỡng 0.5065 được đặt bằng 90% median grad norm trung bình epoch của baseline; stress test cùng lr=1. Không clip `20261003-124413-clip-none-highlr-s42` đạt macro-F1 0.6234; có clip `20261003-124413-clip-on-highlr-s42` đạt 0.7948. Tỷ lệ batch bị clip trong bản chạy có ghi instrumentation: 0.199%; cả hai lượt đều không diverge. Kết quả cho thấy metric cao hơn trong cặp stress này, dù kích hoạt clip hiếm và không chứng minh nguyên nhân duy nhất. Cấu hình cuối không clip. [Hình](figures/compare_clipping.png).

### 3.6 Mixed precision

CUDA không khả dụng nên chỉ thử BF16 autocast trên CPU. `20261003-124413-precision-bf16-cpu-s42` đạt macro-F1 0.8583, thời gian/epoch 7.238 giây; FP32 baseline là 0.522 giây (BF16 chậm hơn khoảng 13.9 lần). Không đo bộ nhớ GPU vì không chạy GPU; không kết luận BF16 tiết kiệm bộ nhớ trên thiết bị này. [Hình](figures/compare_amp.png).

### 3.7 Khởi tạo

Độ lệch chuẩn activation được đo sau từng Linear trên batch validation: He [0,7010; 0,6655; 0,6561], Xavier [0,2926; 0,2268; 0,2177], zero [0; 0; 0]. Loss bước 0 tương ứng 2,3776; 2,0429; 1,9459. Zero-init làm gradient weight/bias ở tầng ẩn bằng 0 sau backward; chỉ bias output có gradient khác 0. Macro-F1: He 0.8522, Xavier 0.8553, zero 0.0936. Đây là bằng chứng thực nghiệm cho vấn đề đối xứng ở zero-init. [Hình](figures/compare_init.png).

## 4. Đánh giá cuối trên eval

Cấu hình được khóa trước Part 4 theo validation: `20261003-124413-opt-adam-0p003-s42`, val macro-F1 0.8712, best epoch 20. Cấu hình cuối được huấn luyện với seed 42; checkpoint epoch có validation loss thấp nhất được dùng để dự đoán eval.

hidden=(256, 128); init=he; loss=ce; optimizer=adam; lr=0.003; momentum=0.9 (không dùng với Adam); betas=(0.9, 0.999); eps=1e-08; weight_decay=0.0; batch=512; epochs=20; dropout=0.0; clip_norm=None; precision=fp32; seed=42; best_epoch=20

| Cấu hình | Seed | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---:|---:|---:|---:|
| Baseline `20261003-124413-base-s42` | 42 | 0.8522 | 0.8539 | 0.9052 |
| Cuối `20261003-124413-final` | 42 | 0.8712 | 0.8725 | 0.9156 |

So sánh eval là mô tả một seed cho mỗi cấu hình; không có ước lượng nhiễu eval nhiều seed. Mốc 2σ ở trên chỉ lấy từ validation. Mọi thí nghiệm chọn cấu hình chỉ dùng validation; eval chỉ được chạy cho baseline seed 42 và final.

### 4.1 Phân tích lỗi theo lớp

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42368 | 0.9069 | 0.9200 | 0.9134 |
| 1 | 56661 | 0.9306 | 0.9245 | 0.9275 |
| 2 | 7151 | 0.9382 | 0.8809 | 0.9086 |
| 3 | 549 | 0.7364 | 0.8907 | 0.8063 |
| 4 | 1899 | 0.7684 | 0.8036 | 0.7856 |
| 5 | 3473 | 0.8439 | 0.8359 | 0.8399 |
| 6 | 4102 | 0.9233 | 0.9298 | 0.9265 |

Ma trận nhầm lẫn (hàng là nhãn thật, cột là dự đoán):

| Thật \ Dự đoán | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38978 | 3037 | 0 | 0 | 63 | 6 | 284 |
| 1 | 3703 | 52382 | 86 | 1 | 357 | 99 | 33 |
| 2 | 6 | 271 | 6299 | 133 | 30 | 412 | 0 |
| 3 | 0 | 0 | 43 | 489 | 0 | 17 | 0 |
| 4 | 28 | 323 | 19 | 0 | 1526 | 3 | 0 |
| 5 | 18 | 235 | 267 | 41 | 9 | 2903 | 0 |
| 6 | 247 | 40 | 0 | 0 | 1 | 0 | 3814 |

**Quan sát:** lớp 4 có F1 thấp nhất (0.7856, support 1899); lỗi phổ biến nhất của lớp này là dự đoán thành lớp 1 (323 mẫu). **Giả thuyết cần kiểm chứng:** đặc trưng của hai lớp có thể chồng lấn; ngoài ra phân bố lớp không cân bằng có thể làm mô hình ưu tiên lớp phổ biến. Báo cáo này chưa có thí nghiệm riêng để xác nhận nguyên nhân.

## 5. Trả lời câu hỏi dẫn dắt

1. **Optimizer:** trong các learning rate đã thử, Adam/AdamW ở 0,003 cao nhất; với lr=0,1 của SGD+momentum, kết quả là 0.8522. Khi giữ một lr chung, so sánh có thể thiên lệch vì các optimizer phản ứng với thang lr khác nhau. Kết quả còn giới hạn bởi hai lr cho mỗi optimizer và một seed cho các lượt so sánh.
2. **Dropout:** ở bài này p=0,2 giảm macro-F1 dù làm nhỏ train–val loss gap. Nên cân nhắc dropout khi đường train/val có bằng chứng quá khớp và xác minh bằng validation, không chỉ dựa vào train loss.
3. **Clipping:** clipping giới hạn norm gradient trước bước cập nhật, nhằm hạn chế bước quá lớn. Trong cặp stress lr=1, macro-F1 tăng từ 0.6234 lên 0.7948; 0,2% batch được clip và không lượt nào diverge. Đây là quan sát cho cặp cấu hình cụ thể.
4. **Mixed precision:** BF16 trên CPU này chậm hơn FP32 khoảng 13.9 lần. CUDA không khả dụng và máy không có AVX, vì vậy kết quả không đại diện cho GPU hoặc CPU có vector instruction phù hợp.
5. **Khởi tạo:** zero-init gán cùng một giá trị cho mọi neuron, tạo đối xứng; ReLU(0)=0 làm gradient tầng ẩn bằng 0 trong phép đo. He giữ phương sai cho mạng ReLU; Xavier giảm độ lệch chuẩn activation trong các phép đo này. Xavier/He khác nhau có thể quan trọng tùy phi tuyến và độ sâu, nhưng ở đây chỉ có một seed cho so sánh.
6. **Nếu loss không giảm sau 2.000 bước, ba kiểm tra đầu tiên:**
   - **Dữ liệu và giao diện loss:** kiểm tra shape/dtype/range của X, y; xác nhận logits có shape `(B, 7)`, là số hữu hạn và truyền logits thô cùng nhãn int64 vào CE. Phần chuẩn hóa chỉ fit trên train; xem một batch và kiểm tra loss ban đầu.
   - **Gradient và khởi tạo:** chạy `zero_grad → forward → loss → backward`; kiểm tra từng tham số có gradient, giá trị hữu hạn và norm khác 0. Xem activation sau Linear/ReLU để phát hiện zero-init hoặc neuron chết; kiểm tra tham số thực sự nhận gradient.
   - **Vòng cập nhật và learning rate:** xác nhận thứ tự `zero_grad → backward → (unscale/clip) → optimizer.step`, theo dõi norm trước clipping và xác minh tham số đổi sau một bước. Thử overfit một batch nhỏ cố định như 20 mẫu; nếu vẫn không giảm, kiểm tra lr quá nhỏ/lớn và so sánh train/val.

## 6. Hạn chế và điều bất ngờ

- Adam/AdamW vượt baseline validation nhưng chênh lệch nhỏ giữa một số cấu hình cần được diễn giải cùng mức 2σ; chỉ hai seed baseline khiến ước lượng nhiễu thiếu chắc chắn.
- Các experiment Part 3 đa số có một seed; lưới optimizer chỉ có hai learning rate. Cùng số epoch giữa batch size khác nhau không giữ số bước cập nhật.
- BF16 chạy trên CPU không có AVX và chậm đáng kể; không thể suy luận tốc độ GPU từ kết quả này.
- Zero-init có loss bước 0 gần ln(7) nhưng vẫn học rất kém do các tầng ẩn không nhận gradient. Đây là ví dụ cho thấy loss khởi tạo gần mốc đều không bảo đảm gradient hữu ích.
- Nếu có thêm thời gian, chạy nhiều seed cho optimizer cuối, thử learning-rate sweep hẹp quanh 0,003 và dùng validation độc lập lặp lại để lượng hóa nhiễu.

## 7. Tệp nộp

Notebook `code/lab.ipynb`; các module trong `code/`; `47` JSON kết quả và hình theo từng `exp_id`; hình so sánh; `experiments.xlsx`; `predictions_eval.csv`; `eval_result.json`; báo cáo này. Tổng thời gian epoch đo được của các thí nghiệm trong lượt hiện tại: khoảng 5.8 phút (không tính thời gian chuẩn bị dữ liệu và evaluator).
