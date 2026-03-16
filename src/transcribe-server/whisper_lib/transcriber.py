import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("whisper-lib")


def now() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def run_cmd(cmd: List[str], label: Optional[str] = None) -> subprocess.CompletedProcess:
    tag = label or Path(cmd[0]).name
    logger.info("CMD START [%s]: %s", tag, " ".join(cmd))
    t0 = now()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    dt = elapsed(t0)
    logger.info("CMD END   [%s]: %.3fs rc=%s", tag, dt, result.returncode)

    if result.returncode != 0:
        logger.error("CMD FAIL  [%s] stdout:\n%s", tag, result.stdout)
        logger.error("CMD FAIL  [%s] stderr:\n%s", tag, result.stderr)
        raise RuntimeError(
            f"Command failed ({result.returncode}) [{tag}]: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    if result.stderr.strip():
        logger.debug("CMD STDERR [%s]:\n%s", tag, result.stderr)

    return result


def run_cmd_json(cmd: List[str], label: Optional[str] = None) -> Dict[str, Any]:
    result = run_cmd(cmd, label=label)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON output for {label or cmd[0]}: {exc}")


@dataclass(frozen=True)
class WhisperConfig:
    whisper_bin: str
    model_path: str
    min_segment_ms: int = 250


@dataclass(frozen=True)
class WhisperPipelineResult:
    media_info: Dict[str, Any]
    whisper_json: Dict[str, Any]
    segments: List[Dict[str, Any]]


class WhisperTranscriber:
    def __init__(self, config: WhisperConfig):
        self.config = config

    def ensure_startup(self) -> None:
        logger.info("Startup checks beginning")
        if not Path(self.config.whisper_bin).exists():
            raise RuntimeError(f"whisper-cli not found: {self.config.whisper_bin}")
        if not Path(self.config.model_path).exists():
            raise RuntimeError(f"Whisper model not found: {self.config.model_path}")
        logger.info("Startup checks passed")

    def probe_media(self, input_path: str) -> Dict[str, Any]:
        logger.info("Probing media file: %s", input_path)
        probe = run_cmd_json(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                input_path,
            ],
            label="ffprobe",
        )

        audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
        video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]

        logger.info(
            "Media probe: audio_streams=%s video_streams=%s duration=%s format=%s",
            len(audio_streams),
            len(video_streams),
            probe.get("format", {}).get("duration"),
            probe.get("format", {}).get("format_name"),
        )

        if not audio_streams:
            raise RuntimeError("Uploaded media contains no audio stream.")

        return probe

    def transcode_media_to_wav(self, input_path: str, output_path: str) -> None:
        logger.info("Transcoding media to 16k mono WAV: %s -> %s", input_path, output_path)
        t0 = now()

        run_cmd([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-map", "0:a:0",
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path,
        ], label="ffmpeg-transcode-media")

        logger.info("Media transcode complete in %.3fs", elapsed(t0))

    def strip_silence_audio(
        self,
        input_path: str,
        output_path: str,
        threshold_db: float = -45.0,
        min_silence_duration: float = 0.8,
        keep_silence: float = 0.25,
    ) -> None:
        filter_str = (
            f"silenceremove="
            f"stop_periods=-1:"
            f"stop_duration={min_silence_duration}:"
            f"stop_threshold={threshold_db}dB:"
            f"stop_silence={keep_silence}"
        )

        logger.info(
            "Stripping silence: threshold=%sdB min_silence=%ss keep_silence=%ss",
            threshold_db,
            min_silence_duration,
            keep_silence,
        )

        t0 = now()
        run_cmd([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", filter_str,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path,
        ], label="ffmpeg-strip-silence")
        logger.info("Silence stripping complete in %.3fs", elapsed(t0))

    def prepare_audio(
        self,
        input_path: str,
        normalized_wav_path: str,
        whisper_input_wav_path: str,
        strip_silence: bool = True,
        silence_threshold_db: float = -45.0,
        min_silence_duration: float = 0.8,
        keep_silence: float = 0.25,
    ) -> Dict[str, Any]:
        media_info = self.probe_media(input_path)
        self.transcode_media_to_wav(input_path, normalized_wav_path)

        if strip_silence:
            self.strip_silence_audio(
                normalized_wav_path,
                whisper_input_wav_path,
                threshold_db=silence_threshold_db,
                min_silence_duration=min_silence_duration,
                keep_silence=keep_silence,
            )
        else:
            shutil.copy2(normalized_wav_path, whisper_input_wav_path)

        return media_info

    def run_whisper_once(self, wav_path: str, output_base: str) -> Dict[str, Any]:
        logger.info("Running whisper.cpp on full file: %s", wav_path)
        t0 = now()

        result = subprocess.run(
            [
                self.config.whisper_bin,
                "-m", self.config.model_path,
                "-f", wav_path,
                "-of", output_base,
                "--output-json-full",
                "-np",
            ],
            capture_output=True,
            text=True,
        )

        dt = elapsed(t0)
        logger.info("whisper.cpp finished in %.3fs rc=%s", dt, result.returncode)

        if result.returncode != 0:
            logger.error("whisper.cpp stdout:\n%s", result.stdout)
            logger.error("whisper.cpp stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"whisper.cpp failed\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        json_path = output_base + ".json"
        if not os.path.exists(json_path):
            logger.error("Expected whisper output file missing: %s", json_path)
            raise RuntimeError("whisper.cpp finished but did not write the JSON output file.")

        with open(json_path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        logger.info("Loaded whisper JSON from %s", json_path)
        return data

    def extract_segments(self, whisper_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Extracting whisper segments from JSON")
        t0 = now()

        raw = whisper_json.get("transcription")
        if raw is None:
            raw = whisper_json.get("segments", [])

        segments: List[Dict[str, Any]] = []

        for seg in raw:
            offsets = seg.get("offsets", {})
            start = int(offsets.get("from", seg.get("start", 0)))
            end = int(offsets.get("to", seg.get("end", 0)))
            text = (seg.get("text") or "").strip()

            if not text:
                continue
            if end - start < self.config.min_segment_ms:
                continue

            segments.append({
                "start_ms": start,
                "end_ms": end,
                "text": text,
            })

        logger.info(
            "Extracted %s whisper segments in %.3fs",
            len(segments),
            elapsed(t0),
        )
        return segments

    def transcribe_file(
        self,
        input_path: str,
        normalized_wav_path: str,
        whisper_input_wav_path: str,
        output_base: str,
        strip_silence: bool = True,
        silence_threshold_db: float = -45.0,
        min_silence_duration: float = 0.8,
        keep_silence: float = 0.25,
    ) -> WhisperPipelineResult:
        media_info = self.prepare_audio(
            input_path=input_path,
            normalized_wav_path=normalized_wav_path,
            whisper_input_wav_path=whisper_input_wav_path,
            strip_silence=strip_silence,
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_silence_duration,
            keep_silence=keep_silence,
        )
        whisper_json = self.run_whisper_once(whisper_input_wav_path, output_base)
        segments = self.extract_segments(whisper_json)
        return WhisperPipelineResult(
            media_info=media_info,
            whisper_json=whisper_json,
            segments=segments,
        )
