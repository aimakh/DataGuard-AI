import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional


@dataclass
class ChunkInfo:
    id: int
    tokens: int
    carryover: str
    content: str
    tokens_saved: int


@dataclass
class MarkovianRun:
    id: str
    prompt: str
    mode: str
    chunks: List[ChunkInfo]
    total_tokens_used: int
    total_tokens_saved: int
    efficiency_percent: float
    duration_ms: int
    completed: bool


class MarkovianEngine:
    """
    Meterless Markovian Engine: O(1) Context per Step Chunked Reasoning Runtime.
    Eliminates quadratic token accumulation by carrying over strictly bounded state across chunks.
    """

    STATE_RE = re.compile(r"\[(?:STATE_CHECKPOINT|STATE)\][\s\S]*$")
    FINAL_RE = re.compile(r"\[(?:TASK_COMPLETE|FINAL)\][\s\S]*$")

    def __init__(
        self,
        chunk_size: int = 8000,
        max_chunks: int = 6,
        carryover_tokens: int = 256
    ):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.carryover_tokens = carryover_tokens

    @staticmethod
    def clean_display(text: str) -> str:
        text = MarkovianEngine.FINAL_RE.sub("", text)
        text = MarkovianEngine.STATE_RE.sub("", text)
        return text.strip()

    def run_chain(
        self,
        task_prompt: str,
        generator_fn: Callable[[str, str], str],
        mode: str = "ARCHITECT"
    ) -> MarkovianRun:
        start_time = time.time()
        chunks: List[ChunkInfo] = []
        carryover = ""
        completed = False

        total_tokens_used = 0
        total_tokens_saved = 0

        for chunk_id in range(1, self.max_chunks + 1):
            # Input to chunk is: Prompt + bounded carryover state (NOT previous full outputs)
            chunk_input = f"{task_prompt}\n\n[PRIOR_STATE_CARRYOVER]\n{carryover}" if carryover else task_prompt
            input_tokens = max(1, len(chunk_input) // 4)

            # Generate step reasoning
            raw_output = generator_fn(chunk_input, carryover)
            output_tokens = max(1, len(raw_output) // 4)
            step_tokens = input_tokens + output_tokens
            total_tokens_used += step_tokens

            # Compute quadratic savings vs standard cumulative history
            # In standard cumulative: step i pays for all steps 1..i-1
            cumulative_baseline = (chunk_id * self.chunk_size)
            markovian_actual = (self.carryover_tokens + output_tokens)
            step_saved = max(0, cumulative_baseline - markovian_actual)
            total_tokens_saved += step_saved

            # Check termination
            if "[TASK_COMPLETE]" in raw_output or chunk_id == self.max_chunks:
                completed = True
                clean_content = self.clean_display(raw_output)
                chunks.append(ChunkInfo(
                    id=chunk_id,
                    tokens=step_tokens,
                    carryover=carryover[:200],
                    content=clean_content,
                    tokens_saved=step_saved
                ))
                break

            # Carryover compression cascade
            marker_match = re.search(r"\[STATE_CHECKPOINT\]([\s\S]*?)(?:\[|$)", raw_output)
            if marker_match and len(marker_match.group(1).strip()) > 10:
                carryover = marker_match.group(1).strip()
            else:
                # Heuristic keypoint extraction
                lines = [line.strip() for line in raw_output.split("\n") if line.strip().startswith(("-", "*", "Insight:", "Root cause:"))]
                if lines:
                    carryover = " | ".join(lines[:4])
                else:
                    carryover = raw_output[-400:].strip()

            clean_content = self.clean_display(raw_output)
            chunks.append(ChunkInfo(
                id=chunk_id,
                tokens=step_tokens,
                carryover=carryover[:200],
                content=clean_content,
                tokens_saved=step_saved
            ))

        duration_ms = int((time.time() - start_time) * 1000)
        efficiency = (total_tokens_saved / (total_tokens_used + total_tokens_saved) * 100) if (total_tokens_used + total_tokens_saved) > 0 else 0.0

        return MarkovianRun(
            id=f"MRUN-{int(time.time()*1000)}",
            prompt=task_prompt[:120],
            mode=mode,
            chunks=chunks,
            total_tokens_used=total_tokens_used,
            total_tokens_saved=total_tokens_saved,
            efficiency_percent=round(efficiency, 1),
            duration_ms=duration_ms,
            completed=completed
        )
