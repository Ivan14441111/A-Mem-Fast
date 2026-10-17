"""Measured OpenAI-compatible inference with a persistent, exact-request cache."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from operator import index
from pathlib import Path
import re
import sqlite3
import threading
import time

import requests

SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."


def bounded_tags_instruction(budget):
    return (f"Return no more than {budget} items in each TAGS/tags list, including lists for updated neighbors. "
            "Follow the requested format and stop after completing its required fields.")


def stage_for(prompt: str) -> str:
    if prompt.startswith("Analyze the following content"):
        return "analysis"
    if prompt.startswith("Create and evolve one memory note in one operation."):
        return "joint_note"
    if prompt.startswith("Evolve this new memory with its retrieved neighbors in one operation."):
        return "evolution_fused"
    if prompt.startswith("You are an AI memory evolution agent"):
        return "evolution_decision"
    if "CONNECTIONS:" in prompt:
        return "strengthen"
    if prompt.startswith("Given the new memory and its neighbor memories"):
        return "update_neighbors"
    if "generate several keywords" in prompt:
        return "query_keywords"
    if prompt.startswith("Based on the context:"):
        return "answer"
    return "other"


class SQLiteResponseCache:
    """WAL storage: each thread owns its connection; only successful responses persist."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS responses "
                         "(key TEXT PRIMARY KEY, payload TEXT NOT NULL, response TEXT NOT NULL)")

    def connection(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.path, timeout=30)
        return self._local.conn

    def get(self, key):
        row = self.connection().execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload, response):
        conn = self.connection()
        with conn:
            conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?)",
                         (key, payload, json.dumps(response, ensure_ascii=False)))


class MeasuredClient:
    """A drop-in get_completion controller. Cache is deterministic-only by default.

    `write` records a cold baseline without reading; `readwrite` permits reuse;
    `replay` never falls back to inference, making semantic comparisons auditable.
    """

    def __init__(self, base_url, model, *, cache=None, cache_mode="off", revision="",
                 temperature=0.0, seed=42, max_tokens=1000, timeout=180,
                 repetition_penalty=None, tag_budget=None):
        if cache_mode not in {"off", "write", "readwrite", "replay"}:
            raise ValueError("Invalid cache mode")
        if cache_mode != "off" and cache is None:
            raise ValueError("Cache storage is required")
        if cache_mode != "off" and (temperature != 0 or not revision):
            raise ValueError("Caching requires temperature=0 and an explicit model revision")
        if repetition_penalty is not None:
            if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
                raise ValueError("repetition_penalty must be finite and positive")
            repetition_penalty = float(repetition_penalty)
        if tag_budget is not None:
            tag_budget = index(tag_budget)
            if tag_budget <= 0:
                raise ValueError("tag_budget must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache = cache
        self.cache_mode = cache_mode
        self.revision = revision
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Omit the field by default so historical request identities remain valid.
        # Explicit penalties affect decoding and belong in every arm's protocol.
        self.repetition_penalty = repetition_penalty
        self.tag_budget = tag_budget
        self._local = threading.local()
        self._lock = threading.Lock()
        self._inflight = {}
        self.events = []

    def _session(self):
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    @staticmethod
    def _validate_response(result):
        """Apply the same checks to live and persisted responses."""
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response must contain a text completion") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned no text")
        if choice.get("finish_reason") == "length":
            raise ValueError("LLM reached max_tokens; completion is truncated")
        if choice.get("finish_reason") in {"content_filter", "tool_calls", "function_call"}:
            raise ValueError("LLM did not finish a text completion")
        usage = result.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("LLM response usage must be an object")
            for field in ("prompt_tokens", "completion_tokens"):
                value = usage.get(field, 0)
                if type(value) is not int or value < 0:
                    raise ValueError("LLM token usage must be a nonnegative integer")
        return content

    def _validate_tag_budget(self, content, stage, prompt):
        """Check the tags that the known successful parser would use; never clip.

        This opt-in generation profile changes the common system instruction.
        It does not interpret QA prose or repair malformed policy JSON.
        """
        if self.tag_budget is None or stage not in {
            "analysis", "strengthen", "update_neighbors", "evolution_fused", "joint_note"
        }:
            return
        from llm_text_parsers import (
            _parse_list_items, parse_analyze_content, parse_strengthen_details,
            parse_update_neighbors, strip_markdown_fences,
        )
        if stage == "analysis":
            tag_lists = [parse_analyze_content(content, "").get("tags", [])]
        elif stage == "strengthen":
            tag_lists = [parse_strengthen_details(content).get("tags", [])]
        elif stage == "update_neighbors":
            counts = re.findall(r"continue for all (\d+) neighbors", prompt)
            if not counts:
                raise ValueError("tag budget requires the requested update-neighbor count")
            tag_lists = [update["tags"] for update in parse_update_neighbors(content, int(counts[-1]))]
        else:
            try:
                data = json.loads(strip_markdown_fences(content))
            except (json.JSONDecodeError, ValueError):
                # The policy parser owns JSON validity. This guard must not
                # fabricate a successful fused/joint response from plain text.
                return

            def nested_tags(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == "tags":
                            yield child
                        elif key in {"metadata", "evolution", "neighbor_updates"}:
                            yield from nested_tags(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from nested_tags(child)

            tag_lists = nested_tags(data)
        for tags in tag_lists:
            if isinstance(tags, str):
                tags = _parse_list_items(tags)
            if isinstance(tags, list) and len(tags) > self.tag_budget:
                raise ValueError(f"LLM tag list exceeds tag_budget={self.tag_budget}: {len(tags)} items")

    def _validated_completion(self, result, event, prompt):
        """Retain bounded failure evidence and usage before rejecting output.

        Successful responses retain no text preview in events. Failed responses
        never enter the cache; the diagnostic is also attached to the exception
        so benchmark failure reports can preserve it without the full prompt.
        """
        choice = None
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                choice = choices[0]
        usage = result.get("usage") if isinstance(result, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        # Preserve valid live token counts even for a rejected/truncated output.
        event["usage"] = {field: value for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                          if type(value := usage.get(field)) is int and value >= 0}
        event["finish_reason"] = choice.get("finish_reason") if choice is not None else None
        try:
            content = self._validate_response(result)
            self._validate_tag_budget(content, event["stage"], prompt)
            return content
        except Exception as exc:
            message = choice.get("message") if choice is not None else None
            content = message.get("content") if isinstance(message, dict) else None
            content = content if isinstance(content, str) else ""
            diagnostic = {"stage": event["stage"], "request_key": event["key"],
                          "finish_reason": event["finish_reason"], "usage": event["usage"],
                          "content_chars": len(content),
                          "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                          "content_prefix": content[:512],
                          "content_suffix": content[-512:] if len(content) > 512 else ""}
            event["llm_diagnostic"] = diagnostic
            exc.llm_diagnostic = diagnostic
            raise

    def get_completion(self, prompt, temperature=0.7, *, _prefetch=False):
        # Both experiment arms deliberately use the same deterministic decoding.
        system_message = SYSTEM_MESSAGE
        if self.tag_budget is not None:
            system_message += "\n\n" + bounded_tags_instruction(self.tag_budget)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_message},
                         {"role": "user", "content": prompt}],
            "temperature": self.temperature, "seed": self.seed,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        identity = json.dumps({"revision": self.revision, "base_url": self.base_url,
                               "request": payload},
                              sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        key = hashlib.sha256(identity.encode()).hexdigest()
        event = {"stage": stage_for(prompt), "key": key, "prefetch": _prefetch, "cache_hit": False,
                 "http_requests": 0, "error": None, "http_seconds": 0.0,
                 "cache_read_seconds": 0.0, "cache_write_seconds": 0.0}
        started = time.perf_counter()
        flight = None
        if self.cache_mode in {"readwrite", "replay"}:
            with self._lock:
                flight = self._inflight.setdefault(key, [threading.Lock(), 0])
                flight[1] += 1
            flight[0].acquire()
        try:
            cache_started = time.perf_counter()
            result = self.cache.get(key) if self.cache_mode in {"readwrite", "replay"} else None
            event["cache_read_seconds"] = time.perf_counter() - cache_started
            if result is not None:
                event["cache_hit"] = True
            else:
                if self.cache_mode == "replay":
                    raise RuntimeError(f"Replay cache miss: {event['stage']} {key}")
                for attempt in range(3):
                    event["http_requests"] += 1
                    try:
                        http_started = time.perf_counter()
                        try:
                            response = self._session().post(self.base_url + "/chat/completions",
                                                            json=payload, timeout=self.timeout)
                        finally:
                            event["http_seconds"] += time.perf_counter() - http_started
                        if response.status_code == 429 or response.status_code >= 500:
                            response.raise_for_status()
                        response.raise_for_status()
                        result = response.json()
                        # Truncation and malformed responses never enter cache.
                        self._validated_completion(result, event, prompt)
                        break
                    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                        status = getattr(exc.response, "status_code", None) if isinstance(exc, requests.HTTPError) else None
                        retryable = (not isinstance(exc, requests.HTTPError) or
                                     status == 429 or (status is not None and status >= 500))
                        if not retryable or attempt == 2:
                            raise
                        time.sleep(0.5 * 2 ** attempt)
                if self.cache_mode in {"write", "readwrite"}:
                    cache_started = time.perf_counter()
                    self.cache.put(key, identity, result)
                    event["cache_write_seconds"] = time.perf_counter() - cache_started
            content = self._validated_completion(result, event, prompt)
            return content
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["seconds"] = time.perf_counter() - started
            with self._lock:
                self.events.append(event)
            if flight is not None:
                flight[0].release()
                with self._lock:
                    flight[1] -= 1
                    if flight[1] == 0:
                        self._inflight.pop(key, None)

    def summary(self):
        by_stage = defaultdict(lambda: {"logical_calls": 0, "http_requests": 0,
                                       "cache_hits": 0, "seconds_sum": 0.0,
                                       "prompt_tokens_live": 0, "completion_tokens_live": 0})
        with self._lock:
            events = tuple(self.events)
        for e in events:
            s = by_stage[e["stage"]]
            s["logical_calls"] += 1
            s["http_requests"] += e["http_requests"]
            s["cache_hits"] += int(e["cache_hit"])
            s["seconds_sum"] += e["seconds"]
            if not e["cache_hit"]:
                for kind in ("prompt", "completion"):
                    s[kind + "_tokens_live"] += e.get("usage", {}).get(kind + "_tokens", 0)
        # Quantiles describe logical-call durations (including cache/singleflight
        # waits). Summed concurrent durations are not elapsed wall-clock time.
        def latency(values):
            ordered = sorted(values)
            if not ordered:
                return {"latency_seconds_p50": 0.0, "latency_seconds_p95": 0.0,
                        "latency_seconds_max": 0.0}
            def percentile(q):
                position = (len(ordered) - 1) * q
                lower = int(position)
                upper = min(lower + 1, len(ordered) - 1)
                return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
            return {"latency_seconds_p50": percentile(0.5),
                    "latency_seconds_p95": percentile(0.95),
                    "latency_seconds_max": ordered[-1]}
        for stage, summary in by_stage.items():
            summary.update(latency(e["seconds"] for e in events if e["stage"] == stage))
        return {"logical_calls": len(events),
                "repetition_penalty": self.repetition_penalty,
                "tag_budget": self.tag_budget,
                "generation_profile": "bounded-tags-v1" if self.tag_budget is not None else None,
                "upstream_calls": sum(not e["prefetch"] for e in events),
                "prefetch_calls": sum(e["prefetch"] for e in events),
                "http_requests": sum(e["http_requests"] for e in events),
                "cache_hits": sum(e["cache_hit"] for e in events),
                "errors": sum(e["error"] is not None for e in events),
                "http_seconds_sum": sum(e["http_seconds"] for e in events),
                "cache_read_seconds_sum": sum(e["cache_read_seconds"] for e in events),
                "cache_write_seconds_sum": sum(e["cache_write_seconds"] for e in events),
                "prompt_tokens_live": sum(s["prompt_tokens_live"] for s in by_stage.values()),
                "completion_tokens_live": sum(s["completion_tokens_live"] for s in by_stage.values()),
                **latency(e["seconds"] for e in events),
                "by_stage": dict(by_stage)}
