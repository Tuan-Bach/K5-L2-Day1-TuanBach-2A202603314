# REPORT

# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
họ tên/MSSV chỉ nằm trong tên repository.
> 

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.

Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ;`taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    - Record này nói về một vật thể riêng trong ảnh. Vì trong record có chia lớp vật thể class_id, class_name.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    - Class list thường đi kèm với bộ data, taxonomy hoặc checkpoint của mô hình
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    - class_id: mã định danh lớp, class_name: tên lớp, taxonomy_name: hệ thống phân loại. Trong một tập dữ liệu training có thể có nhiều bộ dataset được phân loại theo từng đặc trưng của dataset. Nếu chỉ ghi mỗi tên lớp khi kết hợp dữ liệu từ nhiều nguồn có nguy cơ gộp nhầm nhãn có cùng tên nhưng khác phân loại.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    - Guideline cần quy định quy tắc xác định đối tượng hoặc chủ đề chính. VD: trong ảnh có nhiều loại xe như car, truck, bike, … Cần xác định đối tượng chính như truck để người gãn nhãn thực hiện
6. Vì sao không được lấy `score` của model làm ground truth?
    - Ground truth là nhãn chuẩn được xác định từ dữ liệu thực tế thông qua quy trình annotation.
    - Model score là điểm số hoặc xác suất được đưa model đưa ra qua quá trình dự đoán. Model score phán ảnh mức độ tự tin theo cách model được huấn luyện và hiệu chỉnh, không đưa ra dự đoán chắc chắn rằng nhãn đó là đúng.
    
    Một ảnh thực tế chứa một chiếc xe đạp. Mô hình đưa ra dự đoán với Motorbike với score: 0.65, Bycicle với score: 0.30, Car với score: 0.15. Với kết quả này thì model đã đưa ra dự đoán sai. Nếu lấy score làm ground truth đồng nghĩa với ghi nhận một nhãn sai. 
    

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= “oven” ; `score`= 0.686791; `bbox_xyxy`=[0.11, 187.54, 195.96, 292.81] ; `bbox_width`=195.85 ; `bbox_height`=105.26
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? 
    - Góc trên trái ở vị trí cách mép trái ảnh 0.11 pixel và cách mép trên ảnh 187.54 pixel
    - Góc dưới phải ở vị trí cách mép trái ảnh 195.96 pixel và cách mép trên ảnh 292.81 pixel
    - Chiều rộng 195.85 pixel và chiều cao 105.26 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 
    - Ngưỡng 0.20 có 17 prediction
    - Ngưỡng 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    - Khi hạ confidence threshold thấp, reviewer phải thực hiện kiểm tra, loại bỏ những bounding bõ không chính xác.
        
        VD: Bounding box không bao quanh đối tương, phân loại sai đối tượng, …
        
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    - Người gán nhãn phải vẽ bounding bõ bao quanh toàn bộ phần nhìn thấy của đối tượng cần nhận diện. Box phải sát với phần viền đối tượng nhất có thể. Nếu đối tượng bị che khuất chỉ vẽ box quanh phần nhìn thấy. Nếu không thể xác định chắc chắn đối tượng, đánh dấu để reviewer kiểm tra.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Person ở mép trái ảnh. Đối tượng này bị cắt khỏi khung hình, chỉ nhìn thấy một phần cánh tay.
    - Tôi sẽ viết thành quy tắc trong guideline: Khi đối tượng bị cắt bởi mép ảnh, người gán nhãn chỉ vẽ bounding bõ bao quanh phần đối tượng nhìn thấy. Không tự suy đoán hoặc mở rộng box ra ngoài phạm vi ảnh. Nếu phần đối tượng còn nhìn thấy quá ít, không thể xác định chắc chắn người gãn nhãn nên đánh dấu để có thể reviewer xem lại.
    - Lí do viết thành quy tắc trong guidline vì đây có thể là một tình huống thường gặp trong object dectetion. Có thể đưa ra quy tắc chung để mọi người gán nhãn xử lý đồng nhất.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.

Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= 

[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0]

1. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    - Box chỉ là hình chữ nhật bao quanh vật thể bao gồm background, polygon là các điểm tọa độ [x, y] bám biên vật thể. Từ đó cho ra hình dạng vật thể tách khỏi background.
2. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - instance_id dùng để cho biết đó là đối tượng thứ mấy trong ảnh. class_id chỉ dùng để phân biệt loại đối tượng, còn tracking_id dùng để theo dõi cùng một đối tượng qua nhiều frame ảnh/video.
3. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    - Vẽ mask sát theo đường viền thực tế của đối tượng, bao phủ toàn bộ phần nhìn thấy và không lấn sang vùng background hoặc đối tượng khác. Với đối tượng bị che khuất hoặc cắt mép ảnh, chỉ vẽ phần có thể quan sát được, không tự suy đoán phần bị khuất. Nếu ranh giới không thể xác định rõ, đánh dấu để review.
4. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Tôi chọn 2 chiếc bowl nằm sát nhau trên mặt bàn ở góc dưới bên trái ảnh. Tôi sẽ gán nhãn với guideline như đã viết ở trên. Tuy nhiên trong trường hợp khó xác định đường viền giữa hai vật tôi sẽ escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | class của vật thể trong ảnh, cab, minibus, … | Ảnh `traffic`: model dự đoán `cab` (0.511), `minibus` (0.164), `police_van` (0.086). | Quan sát toàn bộ ảnh, chọn class đúng theo taxonomy và guideline; không lấy prediction làm ground truth. | Kiểm tra class có đúng nội dung ảnh, đúng taxonomy và có nhất quán giữa các annotator không. |
| Phát hiện vật thể | Mỗi vật thể là một nhãn gồm class_id + bounding box | Ảnh `kitchen` có các box bowl gần nhau, cần tránh bỏ sót hoặc gán trùng. | Vẽ bounding box bao quanh từng vật thể, gán class chính xác; kiểm tra vật thể nhỏ, bị che khuất và sát mép ảnh. | Kiểm tra class, số lượng vật thể, độ chính xác của box, box bị thiếu hoặc trùng. |
| Instance segmentation | Mỗi vật thể có class_id, instance_id, polygon mask xác định vùng của vật thể | Ảnh `kitchen`: instance `kitchen-002`, class `bowl`, có `polygon_point_count: 67`. Biên mask cần kiểm tra tại vùng vật thể tiếp xúc hoặc bị che khuất. | Vẽ mask sát biên từng vật thể, tách riêng từng instance cùng class và tuân thủ guideline khi gặp vùng bị che. | Kiểm tra mask có khớp vật thể, polygon có lỗi, instance có bị gộp/tách sai hoặc thiếu không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    - Chỉ sử dụng dữ liệu trong phạm vi được lab cho phép, không tự ý tải xuống, sao chép, chia sẻ công khai ảnh và thông tin của bài lab ra bên ngoài
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    - Tôi sẽ dừng xử lý, giữ nguyên trạng thái và báo cho Lab Coach hoặc người phụ trách.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x]  `REPORT.md` (file này đã điền)
- [x]  `day1_lab_outputs/classification_predictions.json`
- [x]  `day1_lab_outputs/detection_predictions.json`
- [x]  `day1_lab_outputs/segmentation_predictions.json`
- [x]  `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x]  `day1_lab_outputs/visuals/classification_top5.png`
- [x]  `day1_lab_outputs/visuals/detection_predictions.png`
- [x]  `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x]  Ô validation cuối notebook báo `PASS`
- [x]  Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x]  Đã nộp **link repository** trên LMS