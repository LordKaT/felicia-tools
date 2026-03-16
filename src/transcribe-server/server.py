import json
import logging
import os
import re
import shutil
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from whisper_lib import WhisperConfig, WhisperTranscriber

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("transcribe-server")

app = FastAPI()

# Config

WHISPER_BIN = "/Users/feliciamirabel/whisper.cpp/build/bin/whisper-cli"
MODEL = "/Users/feliciamirabel/whisper.cpp/models/ggml-base-q8_0.bin"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm.local:7777").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "180"))

# Optional diarization config
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN = os.environ.get("HF_WHISPER_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# Segment shaping
MIN_SEGMENT_MS = 250
MERGE_GAP_MS = 450
MAX_MERGED_SEGMENT_MS = 15_000
MAX_MERGED_CHARS = 320

# Anti-repetition
MIN_LOOP_PHRASE_WORDS = 4
MAX_LOOP_PHRASE_WORDS = 8
MIN_LOOP_REPEATS = 3

# Lazy-loaded diarization pipeline
_DIARIZATION_PIPELINE = None
_LLM_MODEL_CACHE: Optional[str] = None
WHISPER = WhisperTranscriber(
    WhisperConfig(
        whisper_bin=WHISPER_BIN,
        model_path=MODEL,
        min_segment_ms=MIN_SEGMENT_MS,
    )
)

def now() -> float:
    return time.perf_counter()

def elapsed(start: float) -> float:
    return time.perf_counter() - start


TOPICS_PROMPT = """Take the following transcript and provide topics for different segments/topics/arguments, along with the starting timestamp of each topic/argument/segment.

Return only JSON.
Return a JSON array, and each item must use this format:
{
  "topic": "Topic title",
  "start": "HH:MM:SS",
  "end": "HH:MM:SS",
  "transcript": "Provide the exact text of the segment."
}

Do not provide commentary outside the JSON.

Transcript:
"""


class TopicsRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Completed timestamped transcript.")


def llm_request(
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    method: str = "POST",
) -> Dict[str, Any]:
    url = f"{LLM_BASE_URL}{path}"
    headers = {
        "Accept": "application/json",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    request_obj = urllib_request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib_request.urlopen(request_obj, timeout=LLM_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM endpoint returned non-JSON response: {exc}") from exc


def get_llm_model() -> str:
    global _LLM_MODEL_CACHE

    if LLM_MODEL:
        return LLM_MODEL
    if _LLM_MODEL_CACHE:
        return _LLM_MODEL_CACHE

    models_response = llm_request("/v1/models", method="GET")
    models = models_response.get("data") or []
    if not models:
        raise RuntimeError("LLM endpoint returned no models and LLM_MODEL is not configured.")

    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError("LLM endpoint returned a model entry without an id.")

    _LLM_MODEL_CACHE = model_id
    return model_id


def extract_json_text(text: str) -> Any:
    candidate = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        starts = [idx for idx in (candidate.find("["), candidate.find("{")) if idx != -1]
        end = max(candidate.rfind("]"), candidate.rfind("}"))
        if not starts or end == -1:
            raise RuntimeError("LLM response did not contain JSON.")

        candidate = candidate[min(starts):end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse JSON from LLM response: {exc}") from exc


def normalize_topics_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list) or not payload:
        raise RuntimeError("LLM response must be a JSON object or a non-empty JSON array.")

    required_keys = {"topic", "start", "end", "transcript"}
    normalized: List[Dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            raise RuntimeError("Each topic entry must be a JSON object.")
        missing = required_keys - item.keys()
        if missing:
            raise RuntimeError(f"Topic entry missing required keys: {sorted(missing)}")
        normalized.append({
            "topic": str(item["topic"]).strip(),
            "start": str(item["start"]).strip(),
            "end": str(item["end"]).strip(),
            "transcript": str(item["transcript"]).strip(),
        })

    return normalized


def get_chat_completion_text(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response did not contain any choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError("LLM response did not contain message content.")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        joined = "".join(parts).strip()
        if joined:
            return joined

    raise RuntimeError("LLM response content format was not recognized.")


def generate_topics(transcript: str) -> List[Dict[str, Any]]:
    model = get_llm_model()
    logger.info("Generating topics with model=%s transcript_chars=%s", model, len(transcript))
    t0 = now()

    response_payload = llm_request(
        "/v1/chat/completions",
        payload={
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You segment timestamped transcripts into topical JSON. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": f"{TOPICS_PROMPT}{transcript}",
                },
            ],
        },
    )

    content = get_chat_completion_text(response_payload)
    topics = normalize_topics_payload(extract_json_text(content))
    logger.info("Generated %s topics in %.3fs", len(topics), elapsed(t0))
    return topics

# Anti-Repetition

def normalize_sentence_key(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    parts = re.findall(r"[^.!?]+[.!?]*", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def collapse_repeated_sentences(text: str, max_repeats: int = 1) -> Tuple[str, bool]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text, False

    out: List[str] = []
    last_key = None
    repeat_count = 0
    changed = False

    for sentence in sentences:
        key = normalize_sentence_key(sentence)
        if not key:
            continue

        if key == last_key:
            repeat_count += 1
        else:
            last_key = key
            repeat_count = 1

        if repeat_count <= max_repeats:
            out.append(sentence)
        else:
            changed = True

    if not out:
        return text, False

    rebuilt = " ".join(out).strip()
    return rebuilt, changed

def collapse_repeated_phrases(
    text: str,
    min_words: int = MIN_LOOP_PHRASE_WORDS,
    max_words: int = MAX_LOOP_PHRASE_WORDS,
    min_repeats: int = MIN_LOOP_REPEATS,
) -> Tuple[str, bool]:
    tokens = text.split()
    if len(tokens) < min_words * min_repeats:
        return text, False

    changed = False
    out: List[str] = []
    i = 0

    while i < len(tokens):
        best_match = None

        max_n = min(max_words, (len(tokens) - i))
        for n in range(max_n, min_words - 1, -1):
            phrase = tokens[i:i + n]
            if len(phrase) < n:
                continue

            j = i + n
            repeats = 1
            while j + n <= len(tokens) and tokens[j:j + n] == phrase:
                repeats += 1
                j += n

            if repeats >= min_repeats:
                best_match = (n, repeats, phrase, j)
                break

        if best_match:
            _, repeats, phrase, new_i = best_match
            logger.info(
                "Collapsed repeated phrase loop: phrase='%s' repeats=%s",
                " ".join(phrase[:8]),
                repeats,
            )
            out.extend(phrase)
            i = new_i
            changed = True
        else:
            out.append(tokens[i])
            i += 1

    return " ".join(out).strip(), changed

def scrub_repetition(text: str) -> Tuple[str, bool]:
    changed = False

    text2, c1 = collapse_repeated_sentences(text, max_repeats=1)
    changed = changed or c1

    text3, c2 = collapse_repeated_phrases(text2)
    changed = changed or c2

    return text3, changed

def scrub_segments_for_repetition(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    logger.info("Running anti-repetition scrub on %s segments", len(segments))
    t0 = now()

    cleaned: List[Dict[str, Any]] = []
    changed_count = 0

    for seg in segments:
        new_text, changed = scrub_repetition(seg["text"])
        item = seg.copy()
        item["text"] = new_text
        if changed:
            item["repeat_scrubbed"] = True
            changed_count += 1
        cleaned.append(item)

    logger.info(
        "Anti-repetition scrub complete in %.3fs changed=%s",
        elapsed(t0),
        changed_count,
    )
    return cleaned, changed_count

def get_diarization_pipeline():
    global _DIARIZATION_PIPELINE

    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE
    if not HF_TOKEN:
        raise RuntimeError(
            "Diarization requested but HF_WHISPER_TOKEN/HUGGINGFACE_TOKEN is not set."
        )

    logger.info("Lazy-loading pyannote diarization pipeline: %s", PYANNOTE_MODEL)
    t0 = now()

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(
            f"Diarization requested but pyannote.audio could not be imported: {e}"
        )

    try:
        try:
            pipeline = Pipeline.from_pretrained(
                PYANNOTE_MODEL,
                use_auth_token=HF_TOKEN,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                PYANNOTE_MODEL,
                token=HF_TOKEN,
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load pyannote pipeline: {e}")

    if pipeline is None:
        raise RuntimeError(
            "Failed to load pyannote pipeline. "
            "Check HF token and access to gated pyannote models."
        )

    _DIARIZATION_PIPELINE = pipeline
    logger.info("Loaded pyannote diarization pipeline in %.3fs", elapsed(t0))
    return _DIARIZATION_PIPELINE

def diarize_audio(
    wav_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    pipeline = get_diarization_pipeline()

    kwargs: Dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    logger.info("Starting diarization on %s with args=%s", wav_path, kwargs)
    t0 = now()
    diarization = pipeline(wav_path, **kwargs)
    logger.info("Diarization model returned in %.3fs", elapsed(t0))

    turns: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(float(turn.start) * 1000)
        end_ms = int(float(turn.end) * 1000)

        if end_ms - start_ms < MIN_SEGMENT_MS:
            continue

        turns.append({
            "speaker": str(speaker),
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    turns.sort(key=lambda x: x["start_ms"])
    logger.info("Collected %s diarization turns", len(turns))
    return turns

def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def nearest_speaker(seg_start: int, seg_end: int, turns: List[Dict[str, Any]]) -> str:
    seg_mid = (seg_start + seg_end) / 2
    best_speaker = "UNKNOWN"
    best_distance = float("inf")
    for turn in turns:
        turn_mid = (turn["start_ms"] + turn["end_ms"]) / 2
        distance = abs(seg_mid - turn_mid)
        if distance < best_distance:
            best_distance = distance
            best_speaker = turn["speaker"]

    return best_speaker

def assign_speakers(
    whisper_segments: List[Dict[str, Any]],
    diarization_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    logger.info(
        "Assigning speakers: %s whisper segments, %s diarization turns",
        len(whisper_segments),
        len(diarization_turns),
    )
    t0 = now()

    if not diarization_turns:
        out = [
            {
                "speaker": "UNKNOWN",
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg["text"],
                **({"repeat_scrubbed": True} if seg.get("repeat_scrubbed") else {}),
            }
            for seg in whisper_segments
        ]
        logger.info("No diarization turns available; assigned UNKNOWN in %.3fs", elapsed(t0))
        return out

    assigned: List[Dict[str, Any]] = []

    for seg in whisper_segments:
        best_overlap = -1
        best_speaker = None

        for turn in diarization_turns:
            ov = overlap_ms(
                seg["start_ms"], seg["end_ms"],
                turn["start_ms"], turn["end_ms"],
            )
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]

        if not best_speaker or best_overlap <= 0:
            best_speaker = nearest_speaker(seg["start_ms"], seg["end_ms"], diarization_turns)

        item = {
            "speaker": best_speaker,
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "text": seg["text"],
        }
        if seg.get("repeat_scrubbed"):
            item["repeat_scrubbed"] = True
        assigned.append(item)

    logger.info("Speaker assignment complete in %.3fs", elapsed(t0))
    return assigned

def can_merge(prev: Dict[str, Any], cur: Dict[str, Any], require_same_speaker: bool) -> bool:
    if require_same_speaker and prev.get("speaker") != cur.get("speaker"):
        return False

    gap_ok = (cur["start_ms"] - prev["end_ms"]) <= MERGE_GAP_MS
    duration_ok = (cur["end_ms"] - prev["start_ms"]) <= MAX_MERGED_SEGMENT_MS
    chars_ok = (len(prev["text"]) + 1 + len(cur["text"])) <= MAX_MERGED_CHARS

    return gap_ok and duration_ok and chars_ok


def merge_adjacent_segments(
    segments: List[Dict[str, Any]],
    require_same_speaker: bool = False,
) -> List[Dict[str, Any]]:
    logger.info(
        "Merging adjacent segments: input=%s require_same_speaker=%s",
        len(segments),
        require_same_speaker,
    )
    t0 = now()

    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]

        if can_merge(prev, seg, require_same_speaker=require_same_speaker):
            prev["end_ms"] = seg["end_ms"]
            prev["text"] = (prev["text"].rstrip() + " " + seg["text"].lstrip()).strip()
            if seg.get("repeat_scrubbed"):
                prev["repeat_scrubbed"] = True
        else:
            merged.append(seg.copy())

    logger.info("Merge complete in %.3fs output=%s", elapsed(t0), len(merged))
    return merged


def ms_to_timestamp(ms: int) -> str:
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_transcript_no_speakers(segments: List[Dict[str, Any]]) -> str:
    return "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())


def format_transcript_with_speakers(segments: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[{seg['speaker']}] {seg['text'].strip()}"
        for seg in segments
        if seg["text"].strip()
    )

# Init
WHISPER.ensure_startup()

# Routes
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/topics")
async def topics(payload: TopicsRequest):
    transcript = payload.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript must not be empty.")

    try:
        return JSONResponse(generate_topics(transcript))
    except Exception as exc:
        logger.error("TOPICS FAIL error=%s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Query(False, description="Enable slower speaker diarization"),
    num_speakers: Optional[int] = Query(None),
    min_speakers: Optional[int] = Query(None),
    max_speakers: Optional[int] = Query(None),
    strip_silence: bool = Query(True, description="Remove long silences before transcription"),
    silence_threshold_db: float = Query(-45.0),
    min_silence_duration: float = Query(0.8),
    keep_silence: float = Query(0.25),
    anti_repeat: bool = Query(True, description="Collapse obvious repeated phrase loops"),
):
    request_t0 = now()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    logger.info(
        "REQUEST START file=%s diarize=%s strip_silence=%s anti_repeat=%s "
        "num_speakers=%s min_speakers=%s max_speakers=%s",
        file.filename,
        diarize,
        strip_silence,
        anti_repeat,
        num_speakers,
        min_speakers,
        max_speakers,
    )

    with tempfile.TemporaryDirectory() as tmp:
        input_media = os.path.join(tmp, file.filename)
        normalized_wav = os.path.join(tmp, "normalized.wav")
        whisper_input_wav = os.path.join(tmp, "whisper_input.wav")
        whisper_base = os.path.join(tmp, "whisper")

        try:
            t0 = now()
            with open(input_media, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info("Saved upload in %.3fs -> %s", elapsed(t0), input_media)

            whisper_result = WHISPER.transcribe_file(
                input_path=input_media,
                normalized_wav_path=normalized_wav,
                whisper_input_wav_path=whisper_input_wav,
                output_base=whisper_base,
                strip_silence=strip_silence,
                silence_threshold_db=silence_threshold_db,
                min_silence_duration=min_silence_duration,
                keep_silence=keep_silence,
            )
            media_info = whisper_result.media_info
            whisper_segments = whisper_result.segments

            scrubbed_segment_count = 0
            if anti_repeat:
                whisper_segments, scrubbed_segment_count = scrub_segments_for_repetition(whisper_segments)

            if diarize:
                diarization_turns = diarize_audio(
                    whisper_input_wav,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                assigned = assign_speakers(whisper_segments, diarization_turns)
                final_segments = merge_adjacent_segments(assigned, require_same_speaker=True)
                transcript = format_transcript_with_speakers(final_segments)
            else:
                diarization_turns = []
                final_segments = merge_adjacent_segments(whisper_segments, require_same_speaker=False)
                transcript = format_transcript_no_speakers(final_segments)

            total_dt = elapsed(request_t0)
            logger.info(
                "REQUEST END file=%s total=%.3fs diarize=%s strip_silence=%s anti_repeat=%s "
                "whisper_segments=%s final_segments=%s scrubbed_segments=%s",
                file.filename,
                total_dt,
                diarize,
                strip_silence,
                anti_repeat,
                len(whisper_segments),
                len(final_segments),
                scrubbed_segment_count,
            )

            response_segments = []
            for seg in final_segments:
                item = {
                    "start_ms": seg["start_ms"],
                    "end_ms": seg["end_ms"],
                    "start": ms_to_timestamp(seg["start_ms"]),
                    "end": ms_to_timestamp(seg["end_ms"]),
                    "text": seg["text"],
                }
                if "speaker" in seg:
                    item["speaker"] = seg["speaker"]
                if seg.get("repeat_scrubbed"):
                    item["repeat_scrubbed"] = True
                response_segments.append(item)

            return JSONResponse({
                "uploaded_filename": file.filename,
                "input_was_video": any(s.get("codec_type") == "video" for s in media_info.get("streams", [])),
                "input_audio_stream_count": sum(1 for s in media_info.get("streams", []) if s.get("codec_type") == "audio"),
                "diarization_enabled": diarize,
                "silence_stripping_enabled": strip_silence,
                "anti_repeat_enabled": anti_repeat,
                "timestamps_relative_to_processed_audio": strip_silence,
                "text": transcript,
                "segments": response_segments,
                "diarization_turns": diarization_turns,
                "repeat_scrubbed_segment_count": scrubbed_segment_count,
            })

        except Exception as e:
            logger.error("REQUEST FAIL file=%s error=%s", file.filename, e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
