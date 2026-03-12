from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import torch




STORY_OUTLINE = [
    (
        "Meet the Friend",
        "A curious main character (pet/toy/creature) wonders about the world and sets a small, friendly goal.",
    ),
    (
        "First Try",
        "A gentle attempt with a small setback; a kind helper appears (child, bird, or breeze).",
    ),
    (
        "Learning Together",
        "Practice through simple activities (counting steps, calm breathing, following shapes).",
    ),
    (
        "Small Win",
        "The character succeeds at a tiny version of the goal; notices cozy details (light, sounds, colors).",
    ),
    (
        "Warm Finish",
        "Gratitude, sharing with a friend, and a simple lesson (patience, kindness, trying again).",
    ),
]


@dataclass
class GenerationConfig:
    model: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 160
    seed: int = 42


def _pipeline_device_kwargs() -> dict:
    import torch

    # Prefer CUDA GPUs and let Transformers/Accelerate shard automatically.
    if torch.cuda.is_available():
        return {"device_map": "auto", "torch_dtype": torch.float16}
    return {"device": -1}


def _clean_chapter_text(text: str) -> str:
    text = text.strip()
    # Remove accidental spillover headings from some generations.
    stop_markers = ["Chapter 2", "Chapter 3", "Chapter 4", "Chapter 5"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    return text


def _chapter_prompt(chapter_number: int, chapter_title: str, chapter_goal: str, prior_summary: str) -> str:
    return (
        "You are a gentle children's author writing for ages 5-8. "
        "Use simple words, warm tone, and short sentences. "
        "Write only the chapter body (no bullet points, no extra chapter headings).\n\n"
        f"Story continuity so far: {prior_summary}\n"
        f"Now write Chapter {chapter_number}: {chapter_title}.\n"
        f"Goal for this chapter: {chapter_goal}\n"
        "Constraints: 80-130 words, kind tone, sensory details suitable for children, clear ending line."
    )


def generate_book(cfg: GenerationConfig) -> str:
    import torch
    from transformers import pipeline, set_seed

    random.seed(cfg.seed)
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model_candidates = [cfg.model, "distilgpt2"]
    last_err: Exception | None = None
    generator = None
    pipe_device_kwargs = _pipeline_device_kwargs()
    for model_name in model_candidates:
        try:
            generator = pipeline(
                "text-generation",
                model=model_name,
                **pipe_device_kwargs,
            )
            break
        except Exception as exc:
            last_err = exc
            generator = None

    if generator is None:
        raise RuntimeError(
            "Could not load any text-generation model. "
            f"Tried: {model_candidates}"
        ) from last_err

    chapters: list[str] = []
    prior_summary = "A cozy story is beginning."

    for idx, (title, goal) in enumerate(STORY_OUTLINE, start=1):
        prompt = _chapter_prompt(
            chapter_number=idx,
            chapter_title=title,
            chapter_goal=goal,
            prior_summary=prior_summary,
        )

        output = generator(
            prompt,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            return_full_text=False,
        )[0]["generated_text"]

        chapter_body = _clean_chapter_text(output)
        chapter_text = f"Chapter {idx} - {title}\n{chapter_body}"
        chapters.append(chapter_text)

        prior_summary = (
            f"Chapter {idx} recap: {chapter_body[:220].replace(chr(10), ' ').strip()}"
        )

    book_title = "Milo and the Little Sky Steps"
    return f"{book_title}\n\n" + "\n\n".join(chapters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a brief 5-chapter children's book from a high-level outline."
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="Hugging Face model ID for text-generation pipeline.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the generated story as a text file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and outline without calling the model.",
    )
    return parser


def main() -> None:
    print("Welcome to the Children's Book Generator!")
    args = build_parser().parse_args()
    cfg = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    if args.dry_run:
        print("Dry run configuration:")
        print(cfg)
        for idx, (title, goal) in enumerate(STORY_OUTLINE, start=1):
            print(f"{idx}. {title}: {goal}")
        return

    book = generate_book(cfg)
    print(book)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(book, encoding="utf-8")
        print(f"\nSaved story to: {out_path}")


if __name__ == "__main__":
    main()
