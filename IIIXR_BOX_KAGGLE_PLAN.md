# Prompt giao cho coding agent — Tổ chức repo QuantiPhy và chạy IIIXR box trên Kaggle 2×T4

Hãy triển khai trong repo leetranwyng/quantiphy_trannl một thí nghiệm có thể chạy trực tiếp trên Kaggle GPU T4 x2. Mục tiêu là lấy video và câu hỏi QuantiPhy validation làm đầu vào, chạy lại nhánh đo chuyển động tạo ra g_box của IIIXR, lưu kết quả và mọi đầu vào/đầu ra quan sát được của các bước thực sự chạy. Tôi muốn mở output offline và lần từ một đáp án sai ngược về frame, track, vật chuẩn và phép tính đã tạo nó. Tôi sẽ tự chạy Kaggle. Hãy hoàn thiện code, notebook và kiểm tra phần có thể chạy trong môi trường của bạn. Chỉ báo cáo kết quả GPU nếu đã đo thực tế.

## 0. Tổ chức repo để thử nhiều lời giải

Repo đích là leetranwyng/quantiphy_trannl. Kaggle chỉ cần clone repo này ở một commit. Chủ repo có thể sửa ngay source IIIXR, commit cùng notebook và chạy lại. Dùng Git subtree cho source upstream: mỗi thư mục chứa file thật, clone thông thường có đủ code. GitHub chỉ gắn một quan hệ fork ở cấp repository. Nếu muốn có nút Sync fork hoặc gửi PR upstream, tạo fork IIIXR riêng dưới tài khoản chủ repo rồi dùng fork đó làm remote nhập subtree. Ghi cả URL repo gốc lẫn URL fork nếu có; chỉ ghi fork đã thực sự tạo. Có thể bắt đầu bằng subtree trực tiếp từ repo gốc.

### Bố cục và ranh giới chỉnh sửa

| Đường dẫn trong repo tổng | Nội dung |
|---|---|
| code_as_world/, notebooks/quantiphy_caw_*.ipynb | Baseline CAW hiện có, tiếp tục chạy theo entry point cũ. Các thư mục assets/, examples/, requirements/ và paper ở root đang phục vụ CAW. |
| solutions/iiixr/upstream/ | Source tree IIIXR ở commit đã pin, có thể chỉnh sửa trực tiếp. Bảo toàn cấu trúc tương đối, thông tin tác giả và giấy phép đi kèm. |
| solutions/iiixr/kaggle/ | Launcher, instrumentation, lock dependencies, tests và notebook sanity/full cho IIIXR. Gọi upstream với cwd trỏ đúng solutions/iiixr/upstream/ khi script dùng đường dẫn tương đối. |
| solutions/iiixr/README.md | URL, commit nguồn, lệnh chạy, thay đổi so với upstream, chế độ fresh/replay/reconstruction và danh mục artifact. |
| solutions/jx/, solutions/cookie/ | Quy ước cho lần nhập tiếp theo. Chỉ tạo thư mục khi có source hoặc adapter thực sự dùng. |
| README.md, visualization/ | README root là mục lục cả project với lệnh chạy từng solution. Giữ viewer CAW ở vị trí hiện tại; viewer mới dùng đường dẫn riêng theo solution và run. |

### Thứ tự triển khai và kiểm tra

1. Chốt commit CAW trước refactor; kiểm tra working tree và liệt kê đường dẫn hardcode, import, file lock, snapshot_source, notebook bootstrap và verifier trong archive. Bổ sung mục lục root, giữ lại hướng dẫn CAW và LICENSE hiện có. Giai đoạn đầu chỉ thêm solution mới: việc dời toàn bộ CAW đem lại ít lợi ích nhưng có thể làm hỏng notebook cũ và nguồn được đóng gói trong TAR. Dùng archive CAW đã có để kiểm tra cùng 158 ID và macro-MRA 0.4838907058654399.
2. Nhập IIIXR một lần. Kiểm tra license/điều khoản phân phối và kích thước từng blob, bao gồm archive cache, Git LFS và trọng số, trước khi đẩy lên repo tổng. Nếu upstream thiếu quyền phân phối rõ ràng, chuẩn bị checkout local tại commit và ghi bước lấy source trong launcher; nêu rõ phần nào chưa thể đưa vào repo public. Với asset vượt giới hạn lưu trữ, ghi revision, đường lấy và checksum để tải hoặc mount riêng. Kiểm tra nội dung cần cho nhánh box vẫn sẵn có.
3. Trên working tree sạch, thêm remote trỏ về IIIXR gốc hoặc fork cá nhân đã xác thực. Ví dụ nhập lần đầu dưới đây giả định prefix chưa tồn tại:

   ```bash
   git remote add iiixr-source https://github.com/seoneun/NeurIPS-2026-QuantiPhy-Challenge-IIIXR.git
   git fetch iiixr-source main
   git subtree add --prefix=solutions/iiixr/upstream iiixr-source ec4bfc15e9d5d1372bea0ecafdadf54c6e38b378 --squash
   ```

   Tách commit import nguyên bản khỏi commit sửa code. Khi cập nhật, fetch ref mới, xem diff giữa hai upstream SHA, rồi chạy git subtree pull với cùng prefix, remote và SHA mới đã chọn, kèm --squash. Test lại sau khi merge. Clone mới từ repo tổng không kế thừa remote phụ của máy phát triển; URL trong README đủ để khai báo lại. Nếu fork GitHub riêng đã tồn tại, dùng remote fork đó cho các lần đồng bộ tiếp theo.
4. Giữ môi trường Python và process riêng cho từng solution để tránh xung đột Torch, Transformers và thư viện tracking. Chỉ chia sẻ phần đã chứng minh tương đương: video manifest, ánh xạ câu hỏi và evaluator. Giữ parser, preprocessing, FPS, prompts, routing và công thức vật lý riêng của mỗi đội. Chỉ trích helper chung khi cả CAW và IIIXR có test thể hiện cùng hành vi. Kết quả ghi theo outputs/<solution>/<run_id>/, bên trong Kaggle đặt dưới /kaggle/working; Git theo dõi code và metadata tái lập, các archive chạy lớn để ở máy hoặc Kaggle Output.
5. Hoàn thành IIIXR sanity và full theo các phần bên dưới trước khi thêm đội khác. Với JX: pin code và adapter LFS, ghi rõ dùng final LoRA, kiểm tra model/adapter và yêu cầu bộ nhớ trước khi chọn GPU. Với Cookie: pin track_b, tách replay session có sẵn khỏi gọi agent/tool mới, dùng secrets cho endpoint nếu cần. Mỗi solution có CLI sanity/full hoặc replay/fresh phù hợp, README nguồn và phép thử một câu thật.
6. Nghiệm thu cấu trúc bằng một clone sạch của repo tổng ở commit đã chốt: xác nhận file upstream IIIXR có sẵn (hoặc materialize được đúng SHA nếu phải dùng phương án tải riêng), notebook tìm đúng source khi mở từ Kaggle, chạy lại CLI/kiểm tra CPU của CAW, rồi chạy smoke test IIIXR. Lưu source SHA, hash input, model revision và backend trong manifest từng lượt. Khi so nhiều lời giải, ghép theo video/câu hỏi/prior/FPS/depth, báo độ phủ và tập giao; tránh so hai số MRA trên hai tập khác nhau như cùng benchmark.

## Phạm vi và nguồn chuẩn

- Repo IIIXR: https://github.com/seoneun/NeurIPS-2026-QuantiPhy-Challenge-IIIXR . Pin ec4bfc15e9d5d1372bea0ecafdadf54c6e38b378 cho lượt tái hiện đầu tiên. Ghi nguồn, giấy phép và diff cho mọi đoạn upstream được sử dụng.
- Điểm bắt đầu: team/member1/trackB/package/z1s2c/2_c2_g3a/root/analysis/v3/ag/r10/MOTION/. File rules.py ở đây là consumer: đọc cache, so với s1_g và tính biến thể từ tỉ lệ g_pt/g_box. ACCEPTED = []. Chạy file này hay đọc mot_val_401.json chỉ là replay. Tôi cần fresh inference và trace.
- Tập đối chiếu chuẩn có 18 câu / 6 video: 142, 143, 398–401, 427–434, 1896, 2153, 2233, 2234. Dùng ID gốc trong CSV validation, xác nhận thêm tên video. Các bản ghi đều có mode_box=2d_ratio; 17 bản ghi có tv=v2, riêng #142 có tv=v3nt. Cần xác minh ý nghĩa và nguồn thực tế của tag trong code.
- Hai mốc debug trên simulation_0163: #399, g_box=0.38209679865232415; #401, g_box=0.3420306158490744. Đây là output tham chiếu của đội. Ground truth tương ứng là 0.359 và 0.339 m/s². Chỉ cho evaluator đọc các giá trị này sau khi predictions đã ghi xong.
- Hai nền để so là 18 g_box từ cache và 18 dự đoán CAW đúng ID/video từ run/predictions.csv trong quantiphy_full.tar. Tách ba cột cached_box, fresh_box, caw. Archive CAW có thể được gắn làm Kaggle Input hoặc dùng sau lượt chạy để ghép điểm. Đọc đúng member cần thiết trực tiếp từ TAR.
- 158 câu validation là phạm vi tùy chọn để khảo sát độ phủ sau khi hoàn thành thí nghiệm 18 câu. Đánh dấu unsupported với loại bài ngoài phạm vi. Không trộn điểm subset với macro-MRA toàn validation hoặc điểm leaderboard của pipeline fusion.

## 1. Truy nguồn phép đo trước khi viết lại thuật toán

Tại checkout đã pin, tìm g_box, s1_g, 2d_ratio, mot_val_, các tag v2/v3nt, tên track và ID ví dụ. Kiểm tra cả code lồng trong team/, script tạo cache, manifest và archive intermediates. Đi từ giá trị cuối ngược qua nguồn track, chọn đối tượng, xử lý thời gian, ước lượng chuyển động, prior và quy đổi đơn vị. Các đầu mối gồm scripts/pipeline/{00_data,01_parse,02_track,04_geo_chain}.sh, geo/measure.py, geo/measure_v3.py và geo/track*.py. Tên file chỉ là đầu mối, chưa phải bằng chứng rằng mọi stage đều tham gia g_box. Đọc cờ thực tế, metadata và đường dẫn input/output.

Bàn giao bảng lineage theo từng biến thể, mỗi hàng có stage, file:hàm ở commit đã pin, input artifact, output artifact, consumer, model/revision nếu có và bằng chứng áp dụng cho các ID nào. Xác nhận cụ thể box được tạo trực tiếp hay suy từ mask, điểm đại diện là gì, tham số làm mượt và phép tính đạo hàm nào được dùng. Nếu thiếu generator hoặc checkpoint, ghi rõ đoạn cuối cùng truy được. Vẫn làm phần có thể: (a) replay cache để kiểm tra evaluator, (b) chạy lại từ track upstream nếu có, (c) nếu phải dựng thuật toán thay thế thì xuất mode reconstruction riêng. Không trình bày một kết quả đọc cache hoặc một thuật toán tự thiết kế như tái hiện nguyên bản. Không thêm CAW, Kimi, depth hay SAM vào luồng chỉ vì chúng xuất hiện trong MODELS.md.

## 2. Đóng gói và chạy trên Kaggle

Đặt hai notebook iiixr_box_sanity_t4.ipynb và iiixr_box_full_t4.ipynb trong solutions/iiixr/kaggle/. Giữ notebook ngắn: config → clone repo của tôi tại ref được chọn → gọi CLI → hiển thị kết quả. Ghi commit đã resolve. Có chế độ online tải snapshot đã pin và offline dùng Kaggle Inputs. Kiểm tra revision, hash CSV, video, trọng số và file trung gian. Lưu cả ánh xạ ID; so video của IIIXR với input thực tế về hash, kích thước, FPS, số frame, timestamp và resize/480p. Ghi rõ nếu chỉ tương đương về nội dung.

- Sanity bắt đầu bằng #399, #401 (chung video) và #142 (khác tag); thêm #2234 để kiểm tra vận tốc ngang. Kiểm tra đường đi của một raw video tới giá trị vật lý. In ngay lỗi đầu tiên, sample, stage, traceback và vị trí artifact khi worker thất bại.
- Full chạy 18 câu. Khi model/chuỗi cần thiết vừa một T4, chia theo video giữa hai process, một GPU/process; tận dụng track dùng chung nhưng giữ parse, prior và câu hỏi riêng. Khóa cache bằng hash của toàn bộ inputs ảnh hưởng đến kết quả. Phân biệt một lần model call với nhiều câu đọc lại output đó.
- Đo VRAM lúc load và peak thực tế, latency, disk và kích thước trace từ sanity. Ưu tiên cấu hình tương thích T4, batch 1; FP16 chỉ cho stage hỗ trợ. Nếu OOM, giải phóng model giữa các stage hoặc tách stage theo GPU. Chỉ shard khi backend hỗ trợ và đã smoke-test. Dtype, resize, lượng tử hóa, giảm số frame hay thay checkpoint đều là thay đổi có thể ảnh hưởng phép đo: ghi profile riêng và so sai lệch. Không tự bỏ frame/artifact khi thiếu quota; chia phiên theo video và hỗ trợ resume có xác thực hash.
- Ghi root run bên dưới /kaggle/working. Mỗi attempt ghi atomic, có trạng thái complete/error/deferred, exit code và stderr. Cuối cùng xuất iiixr_box_full.tar, summary.json và trang xem offline. Khi Kaggle giới hạn output, dùng các TAR theo shard với manifest chung.

## 3. Hợp đồng trace: ghi tại nơi hàm/model trả kết quả

Mọi event có run_id, video_id, sample_id hoặc danh sách consumer, attempt, event_id, parent_id, stage, input/output refs, backend, device, timestamp, duration và trạng thái. Phân biệt model/tool đã gọi, cache đã đọc, bước bỏ qua, retry và fallback. Lưu dữ liệu số dạng lossless, dùng hash để deduplicate; preview có thể nén. Với API trả tuple/dict nhiều tensor, lưu schema và tất cả trường output công khai được return. Nếu cần instrumentation cấp thấp, ghi riêng những layer/hook được chọn; không hứa truy hồi trạng thái nội bộ mà backend không xuất.

| Stage thực sự chạy | Bằng chứng phải truy xuất được |
|---|---|
| Source/decode/preprocess | Video hoặc bản sao chia sẻ theo hash, FPS/PTS thật, indices được đọc, frame trước/sau resize/crop/pad, kích thước, phép biến đổi tọa độ, processor config và tensor/ảnh thực sự đưa vào model. |
| Parse/grounding | Câu hỏi, prior, depth_info, đơn vị, timestamp, đối tượng đích và vật chuẩn. Với parser bằng rule, lưu rule và kết quả. Với model, lưu prompt/messages chính xác, model/revision, input refs, config, output thô, token IDs nếu có, kết quả parse, lỗi và retry. |
| Detect/segment/track | Toàn bộ candidates trả về, class, confidence, logits/masks/boxes/points/track IDs nếu model xuất. Ghi trước/sau lọc, quy ước tọa độ và đơn vị, candidate bị loại, rule chọn target/reference. Mỗi frame đã xử lý có output gốc cùng ảnh overlay. Đánh dấu occlusion, gap, interpolation, smoothing nếu pipeline dùng. |
| Đo hình học và động học | Theo từng timestamp: box, điểm/tâm thực sự dùng, tọa độ trước/sau lọc, pixel span/displacement của target và prior, FPS/Δt, scale với đơn vị, cửa sổ fit và hệ số, trục/dấu, số trung gian để tái tính đại lượng cuối. Nếu có fallback, lưu cả kết quả trước fallback và lý do. Dùng công thức thật từ code. |
| Finalize/evaluate | Giá trị trước/sau đổi đơn vị, raw/final prediction, đơn vị, status, ground truth chỉ sau inference, sai số tương đối, MRA từng câu và delta so với cached_box và CAW. Với stage không sinh ảnh, lưu số/tensor thật; mọi ảnh đồ thị được đánh dấu derived. |

## 4. Viewer phục vụ phân tích failure cases

Tạo HTML mới, không sửa visualization/QuantiPhy_Review.html. Hiển thị liên tục cả 18 câu, có tìm theo ID/video, lọc theo stage lỗi, lệch cache, tốt/xấu hơn CAW. Mỗi câu hiện chính video, scrub theo frame và PTS, đồng bộ ảnh gốc với box/mask/point overlay, legend target và reference. Có công tắc xem raw/smoothed track và biểu đồ x(t), y(t), kích thước box, v(t), a(t) theo những đại lượng có thật; đánh dấu thời điểm câu hỏi yêu cầu và các frame thực sự tham gia phép đo. Bên cạnh là công thức với giá trị thay số, luồng model/tool calls theo thứ tự và link tải raw artifact. Mở được offline sau giải nén. Thể hiện rõ vị trí thiếu trace khi chỉ có cache/reconstruction.

Ở #401, so fresh_box và cached_box với cached_point nếu có; ghi rõ point là output cache trừ khi thực sự chạy nhánh point. Để người xem tự kiểm tra sự nhạy của gia tốc với quỹ đạo và cửa sổ thời gian, không suy đoán nguyên nhân chỉ từ đáp số. Ở #142 kiểm tra khác biệt tag, ở #2234 kiểm tra thành phần vận tốc ngang.

## 5. Kiểm thử và nghiệm thu

1. Test resolver 18 ID, 6 video, không join nhầm index; test hình học với track tổng hợp tĩnh, vận tốc đều và gia tốc đều, bất biến khi resize đồng thời vật chuẩn và target, xử lý frame thiếu và Δt thật. Chỉ áp dụng bất biến phù hợp với công thức upstream. Test chống đọc cache đáp án trong fresh mode bằng cách bỏ các file mot_val_*.json khỏi đường inference. Chạy đối chiếu trong process riêng sau khi chốt prediction.
2. Replay chỉ từ artifact số đã ghi, tái tính fresh_box trên CPU không tải model. Verify mọi ref/checksum và mọi call có output hoặc exception. Trên sanity bật/tắt trace với cùng seed để xác định instrumentation có đổi kết quả không.
3. Xuất per_question.csv đầy đủ 18 dòng kể cả lỗi, cột coverage và paired delta. Báo cả sai số tuyệt đối/tương đối fresh đối với cache và đối với nhãn, số khớp ở atol=1e-6/rtol=1e-4 như mốc chẩn đoán đề xuất, cùng danh sách ngoại lệ; không bắt pipeline phải khớp số bằng cách điều chỉnh theo nhãn. Replay subset cũ phải cho mean per-question MRA xấp xỉ 0.8389 cho cached_box và 0.4278 cho CAW. Với fresh mode tự tính điểm; ghi mẫu số và cách tính cho câu lỗi, tránh bỏ âm thầm. Đây là trung bình theo 18 câu chuyển động 2D, không phải điểm toàn challenge.
4. Bàn giao source, hai notebook, dependency lock, lệnh chạy, sơ đồ lineage, danh sách input Kaggle, checklist mở output và báo cáo test đã thực sự chạy. Nếu chưa có GPU Kaggle, hoàn thành test CPU và ghi các kiểm tra GPU còn mở. Nếu có GPU, ghi cả peak VRAM từng card, runtime từng stage, kích thước archive và coverage của 18 câu. Khi kết quả lệch, dùng trace tìm stage đầu tiên có input hoặc output khác upstream; chỉ kết luận nguyên nhân khi có bằng chứng.
