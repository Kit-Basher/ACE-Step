import argparse
import os
import time

from loguru import logger

from acestep.pipeline_ace_step import ACEStepPipeline


DEFAULT_PROMPT = (
    "funk, pop, soul, rock, melodic, guitar, drums, bass, keyboard, percussion, "
    "105 BPM, energetic, upbeat, groovy, vibrant, dynamic"
)

DEFAULT_LYRICS = """[verse]
Neon lights they flicker bright
City hums in dead of night
Rhythms pulse through concrete veins
Lost in echoes of refrains

[chorus]
Turn it up and let it flow
Feel the fire let it grow
In this rhythm we belong
Hear the night sing out our song
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate ACE-Step tracks in a loop (CPU-only by default)",
    )
    parser.add_argument(
        "--num_tracks",
        type=int,
        default=1,
        help="Number of tracks to generate in this run.",
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=30.0,
        help="Target duration in seconds for each track (approximate).",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=60,
        help="Number of diffusion inference steps (lower = faster, lower quality).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated WAV files.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Tag-style prompt for the music (comma-separated).",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default=DEFAULT_LYRICS,
        help="Lyrics text. If you want to use a file, pass @path/to/file.txt.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help=(
            "Optional explicit checkpoint directory. If empty, uses the cached "
            "models under ~/.cache/ace-step/checkpoints."
        ),
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=-1,
        help=(
            "Device ID for the pipeline. Use -1 for CPU-only (recommended on 6GB GPUs)."
        ),
    )
    return parser.parse_args()


def load_lyrics_text(arg_lyrics: str) -> str:
    """Allow passing a lyrics file with @path/to/file.txt syntax."""
    if arg_lyrics.startswith("@"):
        path = arg_lyrics[1:]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lyrics file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return arg_lyrics


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lyrics_text = load_lyrics_text(args.lyrics)

    logger.info("Initialising ACE-Step pipeline (this may take a while on first run)...")
    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path or None,
        device_id=args.device_id,
        dtype="float32",  # CPU-friendly default
        torch_compile=False,
        cpu_offload=False,
        quantized=False,
        overlapped_decode=False,
    )

    # Lazy checkpoint loading happens on first call; we just log here.
    logger.info(
        "Starting batch generation: num_tracks=%d, duration=%.1fs, steps=%d",
        args.num_tracks,
        args.audio_duration,
        args.infer_steps,
    )

    for i in range(args.num_tracks):
        logger.info("===== Generating track %d / %d =====", i + 1, args.num_tracks)
        start = time.time()

        # Use a deterministic-ish seed per track based on time + index.
        seed = int(time.time()) + i

        try:
            output_paths, input_params = pipeline(
                format="wav",
                audio_duration=args.audio_duration,
                prompt=args.prompt,
                lyrics=lyrics_text,
                infer_step=args.infer_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=str(seed),
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps=None,
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                audio2audio_enable=False,
                ref_audio_strength=0.5,
                ref_audio_input=None,
                lora_name_or_path="none",
                lora_weight=1.0,
                retake_seeds=None,
                retake_variance=0.5,
                task="text2music",
                repaint_start=0,
                repaint_end=0,
                src_audio_path=None,
                edit_target_prompt=None,
                edit_target_lyrics=None,
                edit_n_min=0.0,
                edit_n_max=1.0,
                edit_n_avg=1,
                save_path=args.output_dir,
                batch_size=1,
                debug=False,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user during generation. Stopping batch loop.")
            break
        except Exception as e:  # noqa: BLE001
            logger.exception("Error while generating track %d: %s", i + 1, e)
            continue

        elapsed = time.time() - start
        logger.info("Track %d finished in %.1f seconds", i + 1, elapsed)
        if isinstance(output_paths, list) and output_paths:
            logger.info("Saved to: %s", output_paths[0])

    logger.info("Batch generation complete.")


if __name__ == "__main__":
    main()
