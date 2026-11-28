"""Instrumentation imported by the isolated upstream subprocesses."""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CURRENT_VIDEO = None


def identify(video_id):
    global CURRENT_VIDEO
    CURRENT_VIDEO = str(video_id)


def start():
    random.seed(0)
    np.random.seed(0)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        import torch

        torch.cuda.init()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()


def write_usage():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        import torch

        print("IIIXR_GPU_USAGE", json.dumps({
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        }), flush=True)


def value(obj):
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu()
        if str(obj.dtype) == "torch.bfloat16":
            obj = obj.float()
        return obj.numpy()
    return np.asarray(obj)


def blob(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **{key: value(v) for key, v in arrays.items() if v is not None})
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = Path(os.environ["IIIXR_TRACE_DIR"]) / "arrays" / f"{digest}.npz"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path.relative_to(Path(os.environ["IIIXR_TRACE_DIR"])).as_posix()


def log(kind, context=None, **fields):
    scope = context or {}
    location = {key: val for key in ("v", "j", "fi", "k", "oid", "desc", "text", "phrase", "key", "vi")
                if isinstance((val := scope.get(key)), (int, float, str, np.integer))}
    record = {"event_id": uuid.uuid4().hex, "parent_id": os.environ["IIIXR_STAGE_EVENT"], "video_id": CURRENT_VIDEO,
              "kind": kind, "timestamp": time.time(), "location": location, **fields}
    with (Path(os.environ["IIIXR_TRACE_DIR"]) / "calls.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, default=str, ensure_ascii=False, allow_nan=False) + "\n")


def arrays(kind, context, **values):
    log(kind, context, artifact=blob(**values))


def geometry(row, track, **values):
    log("measurement", question_id=int(row["id"]), video_id=row["video_id"],
        artifact=blob(**values), frame_indices=track.okc.tolist(),
        track_key=track.q.get("key"), note="See the trace array and corresponding measurement CSV")


class SamplingParams(SimpleNamespace):
    pass


class Qwen:
    """Small adapter preserving upstream prompts, media order and retry decisions."""

    def __init__(self, model, **config):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.config = config
        self._video_blob_cache = {}
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, local_files_only=True, device_map={"": 0}, dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.float16),
            attn_implementation="sdpa",
        ).eval()

    def generate(self, requests, sampling, use_tqdm=False):
        from dataclasses import fields
        import torch
        from transformers.video_utils import VideoMetadata

        allowed_video_meta = {f.name for f in fields(VideoMetadata)}
        results = []
        for request in requests:
            started = time.time()
            prompt = request if isinstance(request, str) else request["prompt"]
            media = {} if isinstance(request, str) else request.get("multi_modal_data", {})
            context = {} if isinstance(request, str) else request.get("trace_context", {})
            kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
            metadata = None
            frames = None
            if media.get("image") is not None:
                kwargs["images"] = [media["image"]]
            if media.get("video"):
                frames, metadata = media["video"][0]
                if isinstance(metadata, dict):
                    do_sample_frames = bool(metadata.get("do_sample_frames", False))
                    clean_metadata = {k: v for k, v in metadata.items() if k in allowed_video_meta}
                else:
                    do_sample_frames = False
                    clean_metadata = metadata
                kwargs.update(videos=[frames], video_metadata=[clean_metadata], do_sample_frames=do_sample_frames)
            input_artifact = None
            media_artifact = None
            try:
                batch = self.processor(**kwargs)
                token_tensors = {k: batch[k] for k in batch if k not in ("pixel_values", "pixel_values_videos")}
                input_artifact = blob(**token_tensors) if token_tensors else None
                if "pixel_values_videos" in batch and frames is not None:
                    cache_key = (id(frames), getattr(frames, "shape", None))
                    if cache_key not in self._video_blob_cache:
                        self._video_blob_cache = {cache_key: blob(pixel_values_videos=batch["pixel_values_videos"])}
                    media_artifact = self._video_blob_cache[cache_key]
                elif "pixel_values" in batch:
                    media_artifact = blob(pixel_values=batch["pixel_values"])
                target_device = getattr(self.model, "device", "cuda:0")
                batch = batch.to(target_device)
                limits = {"max_new_tokens": sampling.max_tokens, "do_sample": sampling.temperature > 0}
                if limits["do_sample"]:
                    limits.update(temperature=sampling.temperature, top_k=sampling.top_k, top_p=sampling.top_p)
                    torch.manual_seed(sampling.seed)
                with torch.inference_mode():
                    generated = self.model.generate(**batch, **limits)
                ids = generated[0][batch.input_ids.shape[1]:]
                response = self.processor.decode(ids, skip_special_tokens=True)
                log("generation", prompt=prompt, response=response, artifact=input_artifact,
                    media_artifact=media_artifact, media_metadata=metadata, generation=vars(sampling),
                    start=started, **context)
                results.append(SimpleNamespace(outputs=[SimpleNamespace(text=response)],
                                               prompt_token_ids=batch.input_ids[0].tolist()))
            except Exception as exc:
                log("generation_error", prompt=prompt, artifact=input_artifact, media_artifact=media_artifact,
                    media_metadata=metadata, generation=vars(sampling), start=started, error=repr(exc), **context)
                raise
        return results
