"""Instrument the Transformers generation path actually run on the T4s."""
from __future__ import annotations

import argparse
import math
import os
import re
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import decord
import numpy as np
import torch
from jinja2 import Template
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, LogitsProcessor, LogitsProcessorList, set_seed

from code_as_world import evaluation as upstream
from code_as_world.kaggle.artifacts import (
    blob, latest_attempt, parse_answer, read_json, reference, relative, require,
    resumable, write_json, write_jsonl, write_text,
)

HOT = {"vision", "merger"}


def array_blob(run, tensor, layout):
    array = tensor.detach().cpu().contiguous().numpy()
    return blob(run, ".npy", lambda stream: np.save(stream, array, allow_pickle=False),
                shape=list(array.shape), dtype=str(array.dtype), layout=layout)


def tensor_leaves(value, name):
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{name}.{index}")


def tensor_info(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def synchronize():
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


def peaks(reset=False):
    values = []
    for idx in range(torch.cuda.device_count()):
        values.append({
            "local_device": idx, "allocated": torch.cuda.memory_allocated(idx),
            "peak_allocated": torch.cuda.max_memory_allocated(idx),
            "peak_reserved": torch.cuda.max_memory_reserved(idx),
        })
        if reset:
            torch.cuda.reset_peak_memory_stats(idx)
    return values


def resolve_device_map(model):
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict) and hf_map:
        return {k: str(v) for k, v in hf_map.items()}
    by_module = {}
    for name, module in model.named_modules():
        devices = sorted({str(p.device) for p in module.parameters(recurse=False)})
        if len(devices) == 1:
            by_module[name or "<root>"] = devices[0]
        elif len(devices) > 1:
            by_module[name or "<root>"] = devices
    unique = {v for v in by_module.values() if isinstance(v, str)}
    if len(unique) == 1 and all(isinstance(v, str) for v in by_module.values()):
        return {"": next(iter(unique))}
    return by_module


class Trace:
    def __init__(self, run, folder, spec, worker, deep):
        self.run, self.folder, self.spec, self.worker, self.deep = run, folder, spec, worker, deep
        self.stage = "start"
        self.refs = {}
        self.events = []
        self.stack = []
        self.steps = -1
        self.handles = []
        self.prefill_peaks = None
        self.forward_finished_at = None

    def event(self, stage, phase, **fields):
        index = len(self.events)
        self.events.append({
            "id": index, "stage": stage, "phase": phase,
            "sample_id": self.spec["id"], "attempt": relative(self.run, self.folder),
            "worker": self.worker, "parent_call_id": self.stack[-1][0] if self.stack else None,
            "time_ns": time.time_ns(), **fields,
        })
        return index

    def begin(self, stage, **fields):
        self.stage = stage
        self.stack.append((self.event(stage, "begin", **fields), time.perf_counter()))

    def end(self, stage, **fields):
        call, start = self.stack.pop()
        self.event(stage, "end", call_id=call, wall_seconds=time.perf_counter() - start, **fields)
        self.stage = self.events[self.stack[-1][0]]["stage"] if self.stack else stage

    def save(self, name, content, *, plain=False):
        path = self.folder / name
        (write_text if plain else write_json)(path, content)
        self.refs[name] = reference(self.run, path)
        return name

    def add(self, name, data):
        self.refs[name] = data
        return name

    def hook_modules(self, model):
        modules = dict(model.named_modules())
        targets = {
            "": "model_forward", "model.visual": "vision",
            "model.visual.merger": "merger", "model.language_model": "language_model",
            "lm_head": "output_head",
        }
        if self.deep:
            for prefix in ("model.visual.blocks", "model.language_model.layers"):
                children = list(modules[prefix].named_children())
                for index in sorted({0, len(children) // 2, len(children) - 1}):
                    targets[f"{prefix}.{children[index][0]}"] = "diagnostic_activation"
        for name, stage in targets.items():
            module = modules[name]
            self.handles.extend([
                module.register_forward_pre_hook(self.before_module(name, stage), with_kwargs=True),
                module.register_forward_hook(self.after_module(name, stage), with_kwargs=True),
            ])
        self.save("hook_modules.json", {
            name or "<root>": {"class": type(modules[name]).__name__, "stage": stage}
            for name, stage in targets.items()
        })

    def before_module(self, name, stage):
        def before(_module, args, kwargs):
            if stage == "model_forward":
                self.steps += 1
                if self.steps == 0:
                    self.forward_finished_at = None
            inputs = list(tensor_leaves({"args": args, "kwargs": kwargs}, ""))
            artifacts = []
            if stage == "vision" and args and torch.is_tensor(args[0]):
                artifacts.append(self.add("vision.effective_pixels", array_blob(self.run, args[0], "patch, pixel channel")))
            if stage == "language_model" and self.steps <= 1:
                for field in ("position_ids", "attention_mask"):
                    if torch.is_tensor(kwargs.get(field)):
                        key = f"language_model.step{self.steps}.{field}"
                        artifacts.append(self.add(key, array_blob(self.run, kwargs[field], field)))
            self.begin(stage, module=name or "<root>", step=self.steps,
                       inputs={n: tensor_info(t) for n, t in inputs}, input_refs=artifacts)
        return before

    def after_module(self, name, stage):
        def after(_module, _args, _kwargs, output):
            outputs = list(tensor_leaves(output, "output"))
            artifact_names = []
            if stage in HOT or (stage == "diagnostic_activation" and self.steps <= 1):
                for output_name, tensor in outputs:
                    key = f"{stage}.{name}.step{self.steps}.{output_name}"
                    layout = "batch, token, hidden" if tensor.ndim == 3 else "token, hidden"
                    artifact_names.append(self.add(key, array_blob(self.run, tensor, layout)))
            if stage == "model_forward" and self.steps == 0:
                if self.deep:
                    synchronize()
                self.prefill_peaks = peaks(reset=True)
                self.forward_finished_at = time.perf_counter()
            self.end(stage, module=name or "<root>", step=self.steps,
                     outputs={n: tensor_info(t) for n, t in outputs}, output_refs=artifact_names)
        return after

    def close(self):
        for handle in self.handles:
            handle.remove()
        write_jsonl(self.folder / "events.jsonl", self.events)
        self.refs["events.jsonl"] = reference(self.run, self.folder / "events.jsonl")
        write_json(self.folder / "artifact_refs.json", self.refs)


def filmstrip(run, pixels, frames):
    images, refs = [], []
    for frame_index, values in enumerate(pixels):
        rgb = values.detach().cpu().round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        image = Image.fromarray(rgb)
        refs.append(blob(run, ".png", lambda stream: image.save(stream, format="PNG"),
                         frame_index=frame_index, width=image.width, height=image.height))
        preview = image.copy()
        preview.thumbnail((190, 120))
        images.append(preview)
    columns, cell_w, cell_h = 4, 198, 142
    canvas = Image.new("RGB", (columns * cell_w, math.ceil(len(images) / columns) * cell_h), "#20252c")
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x, y = i % columns * cell_w, i // columns * cell_h
        canvas.paste(image, (x, y))
        draw.text((x + 2, y + 122), f"{i:02d}  {frames[i]['processor_time_seconds']:.3f}s", fill="white")
    return refs, blob(run, ".png", lambda stream: canvas.save(stream, format="PNG"))


def observed_video(run, record):
    captures, requests, pts = [], [], []
    original_reader = decord.VideoReader

    class ObservedReader(original_reader):
        def get_batch(self, indices):
            frames = super().get_batch(indices)
            try:
                pts.append({"indices": [int(i) for i in indices], "seconds": self.get_frame_timestamp(indices).tolist()})
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
            return frames

    def observe(pixels, raw, sample_fps, metadata, synthetic):
        captures.append((pixels, json_value(raw), float(sample_fps), json_value(metadata), synthetic))

    with patch.object(decord, "VideoReader", ObservedReader):
        multimodal, processor_kwargs = upstream._video_input(record, observe, requests.append)
    require(len(captures) == 1, "Expected one video in the QuantiPhy request")
    pixels, raw, sample_fps, metadata, synthetic = captures[0]
    raw = raw or {}
    raw_indices = raw.get("frames_indices")
    times = next((p["seconds"] for p in pts if p["indices"] == raw_indices), None)
    indices = metadata["frames_indices"]
    frames = [
        {"frame_index": idx, "model_index": source_index,
         "processor_time_seconds": source_index / metadata["fps"],
         "decord_pts_seconds": times[idx] if times is not None and raw.get("video_backend") == "decord" else None}
        for idx, source_index in enumerate(indices)
    ]
    pngs, sheet = filmstrip(run, pixels, frames)
    video = {
        "original_fetch_metadata": raw, "processor_metadata": metadata, "sample_fps": sample_fps,
        "source_fps": raw.get("fps"), "processor_fps": metadata["fps"],
        "requested_nframes": upstream.VIDEO_NFRAMES, "effective_nframes": len(indices),
        "synthetic_indices": synthetic, "frame_index_origin": "synthetic range" if synthetic else "decoder",
        "processor_time_formula": "processor frame index / processor FPS",
        "decoder_pts_source": "Decord get_frame_timestamp on the same VideoReader and indices" if times is not None
        and raw.get("video_backend") == "decord" else None,
        "requests": json_value(requests), "frames": frames,
        "preview_conversion": "float pixels rounded and clipped to RGB uint8; NPY contains exact fetched pixels",
    }
    artifacts = {"fetch_pixels.npy": array_blob(run, pixels, "T,C,H,W"), "filmstrip.png": sheet}
    artifacts.update({f"frame_{i:03d}.png": ref for i, ref in enumerate(pngs)})
    return multimodal, processor_kwargs, video, artifacts


class RecordLogits(LogitsProcessor):
    def __init__(self, bias):
        self.bias = bias or {}
        self.finite = []

    def __call__(self, input_ids, scores):
        self.finite.append(torch.isfinite(scores).all())
        for token_id, bias in self.bias.items():
            scores[:, token_id] += bias
        return scores


def model_inputs(trace, tokenizer, processor, rendered, pre_ids, video_data, upstream_kwargs):
    wrapper = f"{processor.vision_start_token}{processor.video_token}{processor.vision_end_token}"
    require(rendered.count(wrapper) == 1, "Expected one wrapped video placeholder")
    processor_text = rendered.replace(wrapper, processor.video_token, 1)
    video_tensor, metadata = video_data["video"][0]
    inputs = processor(
        text=[processor_text], videos=[video_tensor],
        video_metadata=[dict(metadata)],
        return_tensors="pt",
        text_kwargs={"return_text_replacement_offsets": True, "add_special_tokens": False},
        **upstream_kwargs,
    )
    replacements = inputs.pop("text_replacement_offsets")
    require(len(replacements) == 1 and len(replacements[0]) == 1, "Missing processor replacement")
    change = replacements[0][0]
    left, right = change["span"]
    require(processor_text[left:right] == processor.video_token, "Unexpected processor replacement span")
    expanded = processor_text[:left] + change["replacement"] + processor_text[right:]
    expanded_ids = inputs["input_ids"][0].tolist()
    grid = inputs["video_grid_thw"][0].tolist()
    merge = processor.video_processor.merge_size
    visual_count = math.prod(grid) // merge**2
    require(expanded_ids.count(processor.video_token_id) == visual_count, "Incorrect visual token count")
    require(pre_ids.count(processor.video_token_id) == 1, "Incorrect pre-expansion token count")
    require(len(expanded_ids) < upstream.MAX_MODEL_LEN, "Expanded prompt exceeds model context")
    trace.save("processor_text.txt", processor_text, plain=True)
    trace.save("expanded_prompt.txt", expanded, plain=True)
    trace.save("token_mapping.json", {
        "pre_input_ids": pre_ids, "post_input_ids": expanded_ids, "special_ids": {
            "video": processor.video_token_id, "image": processor.image_token_id,
            "vision_start": processor.vision_start_token_id, "vision_end": processor.vision_end_token_id,
            "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id,
        },
        "replacement": json_value(change),
        "vision_prefix": processor.vision_start_token, "vision_suffix": processor.vision_end_token,
        "grid_thw": grid, "merge_size": merge, "visual_token_count": visual_count,
        "temporal_patch_size": processor.video_processor.temporal_patch_size,
        "processor_timestamps_text": re.findall(r"<([0-9.]+) seconds>", change["replacement"]),
        "processor_kwargs": upstream_kwargs,
    })
    for name, tensor in inputs.items():
        layout = "patch, pixels" if name == "pixel_values_videos" else name
        trace.add(f"processor.{name}", array_blob(trace.run, tensor, layout))
    return inputs


def generate(model, tokenizer, processor, inputs, trace, use_hooks):
    synchronize()
    peaks(reset=True)
    set_seed(upstream.SEED)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    eos = list(dict.fromkeys(token for token in [tokenizer.eos_token_id, *eos] if token is not None))
    target = model.get_input_embeddings().weight.device
    bound = {key: val.to(target) for key, val in inputs.items()}
    limit = min(upstream.MAX_RESPONSE_LENGTH, upstream.MAX_MODEL_LEN - bound["input_ids"].shape[-1])
    knobs = {
        "eos_token_id": eos, "pad_token_id": tokenizer.pad_token_id, "do_sample": True,
        "temperature": upstream.SAMPLING_CONFIG["temperature"], "top_p": upstream.SAMPLING_CONFIG["top_p"],
        "top_k": 0, "max_new_tokens": limit, "num_beams": 1, "num_return_sequences": 1,
        "repetition_penalty": upstream.SAMPLING_CONFIG["repetition_penalty"], "use_cache": True,
        "output_attentions": False, "output_hidden_states": False, "return_dict_in_generate": False,
    }
    from copy import deepcopy
    config = deepcopy(model.generation_config)
    config.update(**knobs)
    observer = RecordLogits(upstream._logit_bias(processor))
    trace.save("generation_config.json", {
        "generation_config": config.to_dict(), "seed": upstream.SEED,
        "logit_processors": [{"type": "additive_token_bias_and_finite_check", "bias": observer.bias}],
        "actual_input_shapes": {key: tensor_info(value) for key, value in bound.items()},
    })
    trace.begin("generate", input_refs=["token_mapping.json", "generation_config.json"])
    if use_hooks:
        trace.hook_modules(model)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = model.generate(
                **bound, generation_config=config, logits_processor=LogitsProcessorList([observer])
            )
        synchronize()
        passed = bool(torch.stack(observer.finite).all().item()) if observer.finite else False
        require(passed, "Generation contains non-finite logits or no forward")
        require(torch.equal(result[0, :bound["input_ids"].shape[-1]], bound["input_ids"][0]),
                "generate changed the input token prefix")
        output_ids = result[0, bound["input_ids"].shape[-1]:].tolist()
        require(output_ids, "Generation returned no token")
        raw = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = {
            "output_token_ids": output_ids, "raw_text": raw,
            "raw_text_with_special_tokens": tokenizer.decode(
                output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "finish_reason": "eos" if output_ids[-1] in eos else ("length" if len(output_ids) >= limit else "unknown"),
            "finish_reason_source": "last token, resolved EOS IDs and max_new_tokens",
            "finite_logits": passed, "generated_token_count": len(output_ids),
            "wall_seconds_cuda_synchronized": time.perf_counter() - started,
            "prefill_peak": trace.prefill_peaks,
            "decode_peak_since_end_of_prefill": peaks() if use_hooks else None,
            "whole_generate_peak": peaks() if not use_hooks else None,
            "step_count_observed": trace.steps + 1,
        }
        trace.save("generation.json", result)
        trace.end("generate", output_refs=["generation.json"])
        return result
    finally:
        for handle in trace.handles:
            handle.remove()
        trace.handles.clear()


def run_sample(root, spec, options, model, tokenizer, processor, template, video_cache, run, use_hooks, tag):
    key = spec["id"]
    _, count = latest_attempt(root, key)
    folder = root / "samples" / key / f"attempt-{count + 1}"
    folder.mkdir(parents=True)
    trace = Trace(run, folder, spec, options["worker"], use_hooks and key in options["deep_examples"])
    started = time.perf_counter()
    result = {
        "sample_id": key, "profile": options["profile"], "role": tag, "trace_enabled": use_hooks,
        "deep_trace": trace.deep, "fingerprint": options["fingerprint"], "worker": options["worker"],
    }
    try:
        trace.begin("prompt")
        trace.save("input_row.json", spec)
        messages = upstream._messages(spec["record"], template)
        trace.save("messages.json", messages)
        rendered = upstream._render_prompt(messages, processor)
        trace.save("rendered_prompt.txt", rendered, plain=True)
        pre_ids = tokenizer.encode(rendered, add_special_tokens=False)
        require(len(pre_ids) <= upstream.MAX_PROMPT_LENGTH, "Prompt exceeds upstream unexpanded token limit")
        if tag.startswith("smoke"):
            reference_ids = upstream._build_prompt_ids(spec["record"], tokenizer, processor, template)
            require(pre_ids == reference_ids, "Upstream pre-expansion token IDs differ")
            trace.save("upstream_prompt_check.json", {"pre_input_ids_match": True, "token_count": len(pre_ids)})
        trace.end("prompt", output_refs=["messages.json", "rendered_prompt.txt"])
        trace.begin("fetch_video", input_refs=["source_video"])
        trace.add("source_video", spec["video"])
        cache_key = (spec["record"]["video_id"], spec["record"]["fps"])
        cache_hit = cache_key in video_cache
        if not cache_hit:
            record = {**spec["record"], "video": str(run / spec["video"]["path"])}
            video_cache[cache_key] = observed_video(run, record)
        video_cache.move_to_end(cache_key)
        while len(video_cache) > 2:
            video_cache.popitem(last=False)
        video_data, processor_kwargs, video, artifacts = video_cache[cache_key]
        trace.save("video.json", {**video, "fetch_cache_hit": cache_hit})
        trace.refs.update(artifacts)
        trace.end("fetch_video", output_refs=["video.json", *artifacts])
        trace.begin("processor", input_refs=["rendered_prompt.txt", "fetch_pixels.npy", "video.json"])
        inputs = model_inputs(trace, tokenizer, processor, rendered, pre_ids, video_data, processor_kwargs)
        trace.end("processor", output_refs=["token_mapping.json", *[n for n in trace.refs if n.startswith("processor.")]])
        decoded = generate(model, tokenizer, processor, inputs, trace, use_hooks)
        trace.begin("parse", input_refs=["generation.json"])
        value, unit = parse_answer(decoded["raw_text"])
        trace.save("parsed.json", {"value": value, "unit_text": unit})
        trace.end("parse", output_refs=["parsed.json"])
        result.update(status="ok" if value is not None else "parse_error", finish_reason=decoded["finish_reason"])
    except Exception as exc:
        result.update(
            status="error", stage=trace.stage, error_type=type(exc).__name__,
            oom=isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower(),
        )
        trace.save("exception.txt", traceback.format_exc(), plain=True)
        trace.event(trace.stage, "error", error_type=type(exc).__name__, output_refs=["exception.txt"])
    finally:
        trace.close()
        result["wall_seconds"] = time.perf_counter() - started
        write_json(folder / "attempt.json", result)
    return result


def run_worker(options):
    run = Path(options["run"])
    root = run / options["subdirectory"] if options["subdirectory"] else run
    worker_output = root / "shards" / options["worker"]
    model_path = Path(options["model"])
    report = {
        "worker": options["worker"], "profile": options["profile"],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assigned_sample_ids": [s["id"] for s in options["samples"]],
        "model_loaded": False, "results": [], "resumed": [], "deferred": [],
    }
    try:
        pending = list(options["samples"])
        if options["phase"] == "evaluation":
            pending = []
            for spec in options["samples"]:
                if resumable(run, root, spec, options["fingerprint"]):
                    report["resumed"].append(spec["id"])
                else:
                    pending.append(spec)
        if not pending:
            write_json(worker_output / "worker.json", report)
            return
        count = torch.cuda.device_count()
        required = 2 if options["profile"] == "split_fp16" else 1
        require(count == required, f"Worker expected {required} visible GPU(s), got {count}")
        report["devices"] = [
            {"local_device": i, "name": torch.cuda.get_device_name(i),
             "capability": list(torch.cuda.get_device_capability(i)),
             "free_total_bytes": list(torch.cuda.mem_get_info(i))}
            for i in range(count)
        ]
        report["torch_cuda_version"] = torch.version.cuda
        for device in report["devices"]:
            with torch.cuda.device(device["local_device"]):
                device["native_bfloat16"] = torch.cuda.is_bf16_supported(including_emulation=False)
        require(all("T4" in d["name"] and d["capability"] == [7, 5] for d in report["devices"]),
                f"Expected Nvidia T4 devices: {report['devices']}")
        tokenizer, processor = upstream._load_model_tools(model_path)
        template = Template(upstream.FORMAT_PROMPT.read_text(encoding="utf-8").strip())
        use_split = required == 2
        load_options = {"device_map": "balanced" if use_split else {"": "cuda:0"}}
        if use_split:
            load_options["max_memory"] = {i: min(12 * 1024**3, torch.cuda.mem_get_info(i)[0] - 3 * 1024**3)
                                          for i in range(count)}
        peaks(reset=True)
        started = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.float16, attn_implementation="eager", local_files_only=True, **load_options,
        ).eval()
        if not use_split and not all(p.is_cuda for p in model.parameters()):
            model = model.to("cuda:0")
        synchronize()
        device_map = resolve_device_map(model)
        realized = {p.device.index for p in model.parameters() if p.is_cuda}
        require(realized == set(range(required)) and all(p.is_cuda for p in model.parameters()),
                f"Model placement must use the assigned GPUs without CPU offload: {device_map}")
        report.update(model_loaded=True, device_map=device_map, model_class=type(model).__name__,
                      model_dtype=str(model.dtype), load_peak=peaks(), load_seconds=time.perf_counter() - started)
        write_json(worker_output / "worker.json", report)
        cache = OrderedDict()
        for index, spec in enumerate(pending):
            if time.time() > options["deadline"] or file_budget_exhausted(run, options):
                report["deferred"].extend(s["id"] for s in pending[index:])
                break
            if options["phase"] == "smoke" and index == 0:
                off = run_sample(root, spec, options, model, tokenizer, processor, template, cache, run, False, "smoke_off")
                report["results"].append(off)
            tag = "smoke_on" if options["phase"] == "smoke" else "evaluation"
            result = run_sample(root, spec, options, model, tokenizer, processor, template, cache, run, True, tag)
            report["results"].append(result)
            print(f"{options['worker']} {spec['id']} {result['status']} {result['wall_seconds']:.1f}s", flush=True)
            write_json(worker_output / "worker.json", report)
            if result.get("oom"):
                torch.cuda.empty_cache()
                if options["phase"] == "smoke":
                    break
    except Exception as exc:
        report.update(worker_error=type(exc).__name__, oom=isinstance(exc, torch.OutOfMemoryError)
                      or "out of memory" in str(exc).lower())
        write_text(worker_output / "worker_exception.txt", traceback.format_exc())
    write_json(worker_output / "worker.json", report)


def file_budget_exhausted(run, options):
    import shutil
    from code_as_world.kaggle.artifacts import file_bytes

    reserve = 600 * 1024**2
    return shutil.disk_usage(run).free < reserve or file_bytes(run) > options["output_budget"] - reserve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_item", type=Path)
    run_worker(read_json(parser.parse_args().work_item))
