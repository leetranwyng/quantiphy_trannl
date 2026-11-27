# IIIXR validation 3D · geo chain + fallback

Full gồm **90 câu trên 12 video**, chia thành 43 S3, 21 V3 và 26 A3. Sanity xuất ID `1090`, `2345`, `2604` và `2607`. Hai câu cuối cùng dùng clip `captured_0027s`. Mỗi worker xử lý toàn bộ câu và prior thuộc clip được chọn để giữ ngữ cảnh dùng chung của upstream.

## Chạy trên Kaggle

1. Push toàn bộ thay đổi IIIXR 3D lên `main`, rồi import [sanity](../iiixr_3d_sanity_t4.ipynb) hoặc [full](../iiixr_3d_full_t4.ipynb). Notebook lấy checkout mới từ GitHub mỗi lần chạy. `REPO_REVISION` mặc định là `main` và có thể đặt thành SHA để tái lập.
2. Bật **Internet** và để trống **Kaggle Inputs**. Notebook tải metadata và các clip cần thiết từ snapshot Hugging Face đã pin. Sanity mặc định chạy `replay` trên CPU. Full mặc định chạy `fresh`, tải model và dùng **GPU T4 x2**. Nếu repo GitHub private, thêm token có quyền đọc repo vào Kaggle Secrets và điền tên secret vào `GITHUB_SECRET`.
3. Tải `iiixr_3d.zip` theo đường dẫn notebook in ra. Giải nén, rồi mở `run/index.html`. Khi có track, viewer vẽ bbox màu, track ID và timestamp trên video. Mục **Decision and stage artifacts** chứa candidate, lý do từ chối, prompt, phản hồi và liên kết mảng. Replay có video và dữ liệu theo câu đã phát hành. Bản upstream thiếu track, MoGe và trace model của các clip validation này.
4. Khi tiếp tục phiên trước, đặt `RESUME` đến `iiixr_3d_checkpoint.zip` hoặc archive fresh trước đó. Giữ nguyên code, phạm vi, shard và profile. Checkpoint mặc định cập nhật sau mỗi 600 giây. Chia theo video bằng `SHARDS` và `SHARD` nếu chạy qua nhiều phiên. Khi chạy lại notebook với cùng `OUTPUT`, launcher cấp thư mục con riêng và in đường dẫn mới.

```bash
python -m solutions.iiixr.kaggle.three_d.workflow --scope full --mode replay \
  --download --output outputs/iiixr_3d/replay
python -m solutions.iiixr.kaggle.three_d.workflow --scope full --mode fresh \
  --download --output outputs/iiixr_3d/fresh
python -m solutions.iiixr.kaggle.three_d.workflow --scope full --mode fresh \
  --download --resume iiixr_3d_checkpoint.zip --output outputs/iiixr_3d/resumed
python -m solutions.iiixr.kaggle.three_d.workflow \
  --merge shard0/iiixr_3d.zip shard1/iiixr_3d.zip \
  --output outputs/iiixr_3d/merged
python -m solutions.iiixr.kaggle.three_d.verify outputs/iiixr_3d/merged/iiixr_3d.zip
```

## Đường đi của dữ liệu

Khi đọc được revision `74aec82473912ccab6a589645a39fcd0e78ee23c` của `PaulineLi/QuantiPhy-validation`, launcher đối chiếu tập ID 3D và từng trường đầu vào với bảng upstream. Manifest kèm repo giữ nguyên ID và dữ liệu nhiều dòng của `depth_info`. `ground_truth_prior` là đại lượng chuẩn do đề cấp và `depth_info` là khoảng cách camera đến vật thể do đề cấp. Track, MoGe, focal và các candidate đo là đầu ra của pipeline.

| Stage | Producer tại upstream `ec4bfc15e9d5` | Input → output | Điều kiện hoặc model |
|---|---|---|---|
| Parse | `geo/parse.py:main`, `geo/parse_v2.py:main` | Câu hỏi, prior, depth → đối tượng, loại phép đo, đơn vị, thời điểm | Qwen3-VL-8B sinh JSON. Regex và các quy tắc upstream dùng JSON theo điều kiện. |
| Track v1, v2 | `geo/track.py:main`, `geo/measure.py:main`, `geo/affected_v2.py`, `geo/track_v2.py:main` | Query, clip → bbox, mask, tâm, độ tin cậy, camera shift | GroundingDINO và SAM2.1, thêm OWLv2 khi retrack. Rule xét thay đổi parse, detection yếu, va chạm hoặc câu khoảng cách. |
| Track VLM | `geo/vlm_ground.py:main`, `geo/track_v3.py:main` | Parse v2, clip → bbox `_notext` và track `v3nt` | Qwen3-VL-8B trên tối đa bốn ảnh từ 16 detection frames, SAM2.1 propagate. |
| Độ sâu | `geo/moge_run.py` | Tối đa 8 frame đều trên trục track → depth, valid mask, intrinsics | MoGe-2 ViT-L. Dùng median focal khi focal đã fit bị bác bỏ theo `fbrej`. |
| Point và mover | `geo/molmo_ground.py:main`, `geo/sam2_points.py`, `geo/mover_scan.py`, `geo/sam2_movers.py`, `geo/patch_movers.py` | Bootstrap failures, track và ảnh → point prompt, moving blobs, track `v2mv` và `v3ntmv` | Molmo2-8B, SAM2.1, OpenCV. Mover scan xét detection/track yếu và loại camera chuyển động mạnh. |
| Scene | `geo/sa2va_track.py:main`, `geo/patch_sa2va.py`, `geo/scene_ruler.py` | Clip và mover tracks → scene masks, `v2sc`, scene ruler | Sa2VA Qwen3-VL-4B SAM3 dùng nhóm frame đầu. OpenCV tìm mẫu đường hoặc vòng tròn cho prior phù hợp. |
| Đo và fusion | `geo/measure_v3.py:measure_video`, `geo/arbitrate_v2.py:main`, `analysis/v3/fuse_v3.py:merge`, `analysis/v3/xground.py` | Track `v2sc/v3ntmv`, prior, depth, focal → SI, candidate và route | `--fix abcdef --moge --moge_opts fbrej`. Fusion `d_vw` ưu tiên phép đo detector hợp lệ. Xground đổi sang nhánh VLM-box khi có ít nhất hai đối chiếu đều dưới IoU 0,5 và nhánh kia đo được. |
| Ruler và focal người | `analysis/v3/ruler_build.py`, `analysis/v3/sruler_apply.py`, `geo/person_focal.py`, `geo/personf_merge.py` | Candidate sau xground → rule ruler và merge hai biến thể focal | `stage_q` chỉ sinh câu ruler cho video 2D, nên g16 có zero requests trên tập này. Person focal dùng GroundingDINO, depth người từ đề và prior chiều cao 1,70 m. |
| Fallback | `tools/run_vlm.py`, `tools/heuristic.py`, `geo/arbitrate_v2.py` | Tối đa 64 frame lấy đều, prompt có prior/depth → giá trị Qwen và prior-copy H1 | `vlm_value` gốc do **Qwen3-VL-8B-Instruct qua vLLM** sinh. Arbitration giới hạn sai khác so với H1 trong hai bậc log10. |

`measure_v3` biến khoảng cách camera–vật thể `D` thành `Z = D / sqrt(1 + ((u-cx)/f)^2 + ((v-cy)/f)^2)`, rồi dựng điểm `[(u-cx)Z/f, (v-cy)Z/f, Z]` theo m. Stride track là `max(1, ceil(N/160))`. Mốc thời gian dùng chỉ số frame gốc chia FPS metadata. Focal fit prior trên lưới 120 giá trị từ `0.25W` đến `12W`. Nhánh gốc có thể dùng prior-depth khi thiếu target-depth. Ngưỡng OWLv2 là giá trị pin hiệu chỉnh từ **test tracks**, độc lập với việc lọc validation.

| Câu hỏi | Hàm đo, lấy mẫu và đơn vị |
|---|---|
| S3 kích thước | `Track.ext_px`, `agg3`, `known_size` dùng extent × `Z/f` trên các frame đạt tiêu chí, rồi aggregate theo loại extent. |
| S3 khoảng cách/độ cao | `dist2_px` dùng trajectory 3D hoặc điểm contour theo ngữ nghĩa câu hỏi. `hfloor_px` và guard scene/part xử lý các trường hợp đặc biệt. |
| V3 tốc độ, dịch chuyển, quãng đường | `px_quantity`, `vel_at_c`, `avg_speed_c`, `pos_at_c` dùng trajectory 3D, mốc trong câu và fit cục bộ. Dịch chuyển là chênh vị trí hai mốc. Tốc độ lấy chuẩn đạo hàm, hoặc quãng đường chia thời gian cho tốc độ trung bình. |
| A3 gia tốc | `px_quantity` chọn `acc_at_c`, `acc_interval_c` hoặc `acc_whole_c` theo thời điểm hoặc đoạn hỏi. Prior trọng lực có thể qua `freefall`. Kết quả SI tính bằng m/s² trước khi đổi đơn vị. |

## Phạm vi đã kiểm chứng

- **Replay** đọc cột `parsed_value` từ `preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv` trong `upstream/intermediates/intermediates.tar.xz`. CPU tái hiện **81 meas / 9 vlm**. ID 2607 bị từ chối với `part_target`, rồi dùng VLM **44 cm**. Archive cũng có bảng Q3n cho validation. Summary tính điểm của bảng này riêng vì chưa xác định được generator. Các lớp sau trong `run.sh` sử dụng submission/test và bảng đã lưu.
- **Fresh** dùng Qwen3-VL-8B với Transformers NF4/FP16 và SDPA trên từng T4. SAM2 và Sa2VA dùng FP16 thay BF16. Video được chia giữa hai worker GPU. Mỗi producer có process riêng. Receipt ghi model revision, cờ, timing, peak allocated/reserved khi PyTorch báo được, stderr và OOM.
- Upstream thiếu generator và input validation cho hai bootstrap `geo_v4ntrj5pbsrng_d_vw` và `_v3m_rrj5plm` mà Molmo/mover đọc. Worker tạo bootstrap bằng **measure v3 + arbitrate** từ track fresh. Do thay đổi này có thể ảnh hưởng câu được patch, mọi kết quả fresh đang mang trạng thái `provisional`. Molmo mặc định tắt. `MOLMO="vllm-bnb"` yêu cầu vLLM hỗ trợ Molmo2 và bitsandbytes trên T4. Khi Molmo thiếu, lỗi chứa video/phrase vừa được rule chọn. Lane ghi thêm lỗi per-query của Sa2VA nếu có.
- `predictions_locked.json` được hash trước khi evaluator đọc GT. Các lỗi vẫn có dòng và prediction null. MRA là trung bình theo câu trên denominator trong summary. Run fresh zero coverage để MRA null. File `per_question.csv` lưu từng ID, điểm và chênh lệch route/value. Phép so CAW-4B dùng cùng giao ID cho hai phía.
- Gói mã nguồn có bảng CAW-4B gồm ID và giá trị dự đoán cho 89 câu. Evaluator dùng bảng này sau bước khóa dự đoán và ghi checksum của bảng cùng nguồn `run/predictions.csv` trong `quantiphy_full.tar`.

| Đối chiếu CPU | Số câu | MRA |
|---|---:|---:|
| Replay 3D | 90 | 64,89% |
| S3 | 43 | 70,47% |
| V3 | 21 | 54,29% |
| A3 | 26 | 64,23% |
| Replay IIIXR trên giao CAW | 89 | 64,61% |
| CAW-4B trên cùng giao | 89 | 49,44% |

Kiểm tra CPU đã giải mã 12 clip từ `quantiphy_full.tar`. Metadata đến từ manifest input upstream vì phiên local chưa truy cập được Hugging Face. Chưa đo GPU T4, VRAM, thời gian inference hoặc tính tương đương fresh.
