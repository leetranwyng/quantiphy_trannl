# Giao việc cho Sol — Code-as-World-VL-4B trên QuantiPhy với Kaggle 2×T4

Hãy tạo và chạy notebook quantiphy_caw_4b_t4.ipynb. Mục tiêu là inference checkpoint gốc MirroS-Lab/Code-as-World-VL-4B trên toàn bộ validation có nhãn. Dùng output của một Kaggle notebook version để người khác tra lại được từng phép dự đoán, kể cả khi không có GPU hoặc Internet. Bàn giao notebook, output version và kết quả kiểm tra. Đây là kế hoạch thực hiện, chưa có kết quả chạy.

## 1. Chuẩn bị code, model và dữ liệu

Trong môi trường có mạng, chạy:

```bash
git clone https://github.com/MirroS-Lab/Code-as-World.git
git clone https://github.com/Paulineli/QuantiPhy.git
git -C Code-as-World rev-parse HEAD
git -C QuantiPhy rev-parse HEAD
hf download MirroS-Lab/Code-as-World-VL-4B --local-dir weights/4b
hf download PaulineLi/QuantiPhy-validation --repo-type dataset --local-dir QuantiPhy-validation
```

Chốt SHA thực tế của hai snapshot HF trước khi tải bằng --revision. Trên Kaggle có thể gắn source, weights, videos và wheelhouse đã chuẩn bị dưới /kaggle/input; mọi kết quả ghi vào /kaggle/working/run/. Ghi commit, diff instrumentation, templates, command line, pip freeze, image/Python, driver/CUDA, nvidia-smi, dung lượng trống và sha256 của CSV, video, cấu hình, weights. Sao chép source dùng để chạy và các video validation đã dùng vào output một lần mỗi video cho mục đích review cá nhân. Tránh đóng gói trùng weights. Dùng quantiphy_validation.csv từ repo benchmark và validation_videos/ từ dataset. Đếm số dòng thực tế, video, bốn nhóm và file thiếu thay vì hardcode số câu từ dataset card. Tạo sample_id ổn định từ SHA CSV + số dòng gốc. Tách ground truth ở loader, chỉ cho evaluator và viewer đọc nhãn.

## 2. Xác nhận cấu hình trên T4 trước khi chạy cả tập

Entry gốc là python -m code_as_world.evaluation 4b --model-path weights/4b --input-csv QuantiPhy/quantiphy_validation.csv --video-dir QuantiPhy-validation/validation_videos. Đọc code ở SHA vừa pin. Evaluator được kiểm tra khi viết kế hoạch dùng vLLM, BF16, tensor_parallel_size=1 và gpu_memory_utilization=0.2 cho 4B. Đừng chạy nguyên cấu hình đó rồi kết luận model không vừa T4. Hai card có bộ nhớ riêng. Ghi tên, compute capability, free VRAM và hỗ trợ dtype thực tế cho từng card.

Ưu tiên backend Transformers eager/SDPA FP16 cho lượt trace vì có thể hook đúng module. Tách các helper prompt/video/parser khỏi import vLLM nếu cần; kiểm tra API Qwen3.5 trong bản Transformers tương thích model card (5.11.0). Batch 1, eval + inference_mode, giữ nguyên template, dropout lúc eval, sampling, stop, mask/logit bias và giới hạn gốc trong chừng mực backend hỗ trợ. Kiểm tra token đặc biệt, placeholder video sau expansion, số visual tokens và timestamp; không tokenize hai lần hoặc sample video lần hai. So sánh token IDs trước expansion với _build_prompt_ids cho một clip; mọi khác biệt backend/dtype/decoding phải xuất vào config_diff.json. Nếu chạy được upstream vLLM trên một host tương thích, đối chiếu cùng fixture; nếu không, ghi parity chưa kiểm chứng. Không gọi kết quả FP16 là điểm tái lập BF16.

Thử một replica trên một T4, 16 frame, max_pixels=262144, batch 1. Đo load peak, prefill/decode peak, latency, output hữu hạn và trace trên cả video ngắn lẫn video có visual-token count cao. Nếu đạt, khởi chạy hai process độc lập, mỗi process chỉ thấy một GPU, chia theo video_id để giữ cache, ghép lại theo số dòng gốc. Nếu một replica OOM, chạy một process phân lớp model trên cả hai GPU với device map và ngân sách riêng có chừa VRAM cho activation; lưu map và đo lại. Nếu kernel thiếu hỗ trợ, chuyển sang implementation PyTorch tương thích hoặc ghi lỗi cụ thể. Chỉ sau đó thử 4-bit hoặc giảm visual budget dưới run_id riêng. Không trộn kết quả các profile; lưu mọi attempt và lý do retry. Nếu deadline Kaggle sắp hết, checkpoint theo sample và resume từ version output trước.

## 3. Instrument ngay trên đường dữ liệu thực tế

| Điểm gắn trace | Nội dung phải lưu |
|---|---|
| _load_csv, _content_from_record, _message_content | Dòng đầu vào, video_id, câu hỏi nguyên văn, prior, depth_info, FPS trong CSV, đơn vị, nhóm; lưu bản chuẩn hóa riêng. Tất cả system/user messages theo đúng thứ tự. |
| _build_prompt_ids và chat template | Hai file Jinja gốc, system prompt, prompt sau render nguyên văn với mọi special token, tokenizer revision, token IDs thực tế trước/sau multimodal expansion. Ghi từng request kể cả retry. |
| _process_video, _video_input | Chụp trực tiếp output fetch_video. Lưu pixel tensor trước processor dạng lossless cùng dtype/layout, từng frame PNG có thứ tự và contact sheet. Lưu raw metadata trước khi evaluator ghi đè FPS, FPS nguồn, sample_fps, processor_fps, indices gốc nếu có, indices thực tế truyền model, PTS nếu decoder cung cấp. Khi upstream fallback range(num_frames), đánh dấu synthetic; chỉ xuất timestamp suy ra kèm công thức và nguồn. Ghi trường hợp clip ngắn làm giảm số frame. |
| Processor → vision encoder → merger/projector → language model | Ghi module thực sự gọi được từ model.named_modules(), thứ tự, parent call ID, shape, dtype, device, thời gian và trạng thái. Lưu input_ids, attention mask, video_grid_thw và pixel_values_videos thực tế (hoặc tên field của revision). Lưu output tensor của vision encoder và merger dạng safetensors/NPY lossless có layout. Với visual input trùng nhau, dedup theo hash nội dung sau xử lý. Chỉ lưu khi module tồn tại; test hook bằng một forward thật. |
| Mọi model/tool trung gian được gọi | Lưu tên/revision, input refs, prompt nếu có, output gốc. Với ảnh/mask/depth/video, lưu mọi frame hoặc map gốc kèm dtype/scale và PNG preview. Embedding là tensor: có thể thêm heatmap ghi rõ phép chiếu, vẫn giữ tensor nguồn. Đường evaluation công khai gọi một VLM; không tự thêm SAM/DA3/VGGT/MuJoCo. Liệt kê thành phần chỉ xuất hiện trong paper/demo ở mục “không chạy”. |
| generate → _parse_prediction → evaluator | Generation config đã resolve, seed, logit processors, output token IDs, văn bản thô trước mọi thao tác strip, finish reason nếu backend có hoặc căn cứ suy ra, số và đơn vị parse được, exceptions, thời gian, VRAM mỗi GPU. Lưu cả câu lỗi. Ghép nhãn và điểm sau khi inference hoàn tất. |

Một production trace cho từng câu cần tái dựng được inputs, outputs và thứ tự stage. Không bật output_attentions trên mọi layer cho toàn bộ tập. Với ít nhất một câu mỗi nhóm, thêm deep trace: hook block đầu/giữa/cuối của visual và language stack, lưu activation prefill và ít nhất một decode step, kèm token/step index và mapping tên module. Tách file chẩn đoán này khỏi output ảnh do model sinh. Đồng bộ CUDA khi đo timing mẫu; nêu rõ timing có overhead của hook. Trên fixture cố định chạy trace off/on cùng seed, đối chiếu token IDs và parsed answer để phát hiện hook làm đổi kết quả.

## 4. Output để quan sát ngay sau khi tải xuống

Tổ chức run/ gồm manifest.json, source/, configs/, videos/, blobs/, samples/<sample_id>/attempt-N/, predictions.csv, failures.jsonl, metrics.json, trace_index.jsonl, index.html, verify.py. Mỗi attempt chứa messages.json, rendered_prompt.txt, events.jsonl, generation.json và artifact_refs.json; blob dùng đường dẫn tương đối + SHA256, shape/dtype khi thích hợp. Event phải nối được run → sample → attempt → stage → input/output. Ghi file tạm rồi rename; chỉ đánh dấu complete sau khi xác nhận đủ artifact. Khi resume, so run fingerprint và checksum. Mỗi GPU ghi shard riêng. Merge theo sample_id, phát hiện thiếu/trùng.

index.html tĩnh có bảng theo nhóm, video, lỗi, dự đoán và nhãn; mỗi dòng mở được video, filmstrip có thời gian, prompt, trace, output gốc và các tensor tải được. Không cần CDN, server hay đọc lại /kaggle/input để xem. Xuất notebook đã chạy với bảng tổng hợp và ít nhất bốn ví dụ có liên kết tương đối. Chạy pilot 1 câu/nhóm, đo byte/video, byte/câu, headroom ổ đĩa và ETA. Nếu quota nhỏ, chia nhiều notebook version có index và manifest hợp nhất, tuyệt đối không âm thầm bỏ ảnh hoặc trace rồi báo đủ.

## 5. Điều kiện nghiệm thu

1. Cả hai T4 được nhận và có device assignment thực tế trong manifest. Đã chạy pilot rồi mới chạy full validation. Ghi rõ backend, dtype, hiệu năng, peak VRAM từng GPU và cấu hình nào chạy thành công.
2. Mỗi dòng CSV đầu vào có đúng một trạng thái cuối, mỗi attempt có prompt, artifact và trace hoặc lỗi tại stage cụ thể. Lưu coverage, parse failures và MRA theo từng nhóm + macro từ evaluator chính thức. Không loại câu lỗi khỏi mẫu số một cách âm thầm.
3. verify.py chạy CPU offline từ một thư mục sạch chỉ chứa output, kiểm tra toàn bộ checksum, số frame lưu đúng tensor, links, coverage, prompt/token mapping và replay parser + evaluator từ raw generations. Thử một câu ngắn, một câu 3D có depth_info, hai câu chung video và đường resume.
4. Báo cáo phân biệt rõ kết quả thật, trường tùy chọn không quan sát được và thí nghiệm chưa chạy. Đưa đường dẫn output Kaggle version sau khi đã lưu thành công. Nếu Sol không có Kaggle runtime, bàn giao notebook chạy được và ghi đúng bước còn thiếu; không tạo số MRA, VRAM hay runtime giả.

Nguồn đối chiếu: https://github.com/MirroS-Lab/Code-as-World ; https://github.com/MirroS-Lab/Code-as-World/blob/main/code_as_world/evaluation.py ; https://huggingface.co/MirroS-Lab/Code-as-World-VL-4B ; https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation ; https://github.com/Paulineli/QuantiPhy .
