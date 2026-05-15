"""Command-line interface for EXPLORE.

Commands
--------
explore gui              Launch the browser-based graphical application.
explore run CONFIG       Run headless analysis from a YAML config file.
explore init             Interactively create a new config YAML.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
@click.version_option(package_name="explore")
def main() -> None:
    """EXPLORE — behavioral analysis for object recognition tests."""


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@main.command()
def gui() -> None:
    """Launch the browser-based graphical application."""
    try:
        from explore.gui.nicegui_app import launch
    except ImportError:
        click.echo(
            "NiceGUI is required for the GUI.  Install with:\n  pip install nicegui",
            err=True,
        )
        sys.exit(1)
    launch()


# ---------------------------------------------------------------------------
# Headless run
# ---------------------------------------------------------------------------


@main.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the results output directory.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def run(config: Path, output: Path | None, verbose: bool) -> None:
    """Run headless analysis from a YAML CONFIG file."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from explore.config import ExperimentConfig
    from explore.pipeline.prediction import ExplorationPipeline

    cfg = ExperimentConfig.from_yaml(config)
    if output:
        cfg.project_path = output

    pipeline = ExplorationPipeline(cfg, headless=True)
    results = pipeline.run()

    if results.empty:
        click.echo("No results produced — check your configuration.", err=True)
        sys.exit(1)

    click.echo(f"Done. {len(results)} rows saved to {cfg.project_dir}/results/")


# ---------------------------------------------------------------------------
# Interactive config init
# ---------------------------------------------------------------------------


@main.command(name="init")
@click.option("--output", "-o", default="experiment.yaml", help="Output YAML filename.")
def init_config(output: str) -> None:
    """Interactively create a new experiment config YAML."""
    click.echo("EXPLORE — new experiment setup\n")

    project_name = click.prompt("Project name")
    project_path = click.prompt("Project folder", default=str(Path.cwd()))
    duration = click.prompt("Video duration (minutes)", default=5, type=int)

    n_objects = click.prompt("Number of objects", default=2, type=int)
    objects = []
    for i in range(n_objects):
        name = click.prompt(f"  Object {i + 1} name (e.g. familiar / novel)")
        objects.append({"name": name, "bounding_box": None})

    pos_str = click.prompt("Exploration prompts (semicolon-separated)")
    neg_str = click.prompt("Non-exploration prompts (semicolon-separated)")
    pos = [p.strip() for p in pos_str.split(";") if p.strip()]
    neg = [p.strip() for p in neg_str.split(";") if p.strip()]

    config: dict = {
        "project_name": project_name,
        "project_path": project_path,
        "video_paths": [],
        "video_duration_minutes": duration,
        "objects": objects,
        "behavior": {
            "exploration_prompts": pos,
            "no_exploration_prompts": neg,
            "confidence_threshold": 0.5,
            "min_bout_seconds": 1.0,
        },
        "model": {
            "clip_model": "ViT-B-32",
            "clip_pretrained": "openai",
        },
        "analysis": {
            "bin_duration_minutes": 1,
            "compute_di": True,
        },
    }

    out_path = Path(output)
    with open(out_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    click.echo(f"\nConfig written to '{out_path}'.")
    click.echo(
        f"Open the GUI to draw bounding boxes, then run:\n  explore run {out_path}"
    )
