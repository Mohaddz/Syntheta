"""CLI generate command."""

from __future__ import annotations

import os

import click

from syntheta.config.loader import load_config


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file")
@click.option("--recipe", type=str, help="Built-in recipe name")
@click.option("--domain", type=str, help="Domain for generation")
@click.option("--n", type=int, help="Number of samples to generate")
@click.option("--method", type=str, help="Generation method")
@click.option("--output", type=str, help="Output JSONL file path")
@click.option("--seed", type=int, help="Random seed")
@click.option("--dry-run", is_flag=True, help="Estimate cost without generating")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.argument("overrides", nargs=-1, type=str)
def generate(
    config_path: str | None,
    recipe: str | None,
    domain: str | None,
    n: int | None,
    method: str | None,
    output: str | None,
    seed: int | None,
    dry_run: bool,
    resume: bool,
    overrides: tuple[str, ...],
) -> None:
    """Generate synthetic training data."""
    # Build CLI overrides dict
    cli_overrides: dict = {}
    if domain:
        cli_overrides["domain"] = domain
    if n:
        cli_overrides["n"] = n
    if method:
        cli_overrides["method"] = method
    if output:
        cli_overrides["output"] = output
    if seed:
        cli_overrides["seed"] = seed

    # Parse --key=value overrides (dot notation)
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            key = key.lstrip("-")
            # Try to parse as int/float/bool
            cli_overrides[key] = _parse_value(value)

    config = load_config(config_path, recipe, cli_overrides)

    if dry_run:
        _dry_run(config)
        return

    _run_pipeline(config, resume)


def _dry_run(config: dict) -> None:
    """Print cost estimate without executing."""
    import math

    n = config["n"]
    factor = config.get("over_generate_factor", 1.2)
    n_generate = math.ceil(n * factor)

    # Count pipeline stages
    stages = ["generation"]
    if config.get("evolve", {}).get("enabled"):
        stages.append("evol_instruct")
    if config.get("responses", {}).get("enabled"):
        stages.append("response_generator")
    if config.get("filters", {}).get("quality", {}).get("enabled"):
        stages.append("quality_filter")

    # Rough estimate: ~1000 tokens per LLM call on average
    calls_per_sample = len(stages)
    total_calls = n_generate * calls_per_sample
    est_tokens = total_calls * 1000

    click.echo("\nDry-run estimate (not a commitment -- actual costs will vary):")
    click.echo(f"  Planned samples:     {n_generate:,} ({n:,} target x {factor} over-generation)")
    click.echo(f"  Pipeline stages:     {len(stages)} ({' -> '.join(stages)})")
    click.echo(f"  Estimated LLM calls: ~{total_calls:,}")
    click.echo(f"  Estimated tokens:    ~{est_tokens:,}")
    click.echo(f"  Model:               {config['llm']['model']}")


def _run_pipeline(config: dict, resume: bool) -> None:
    """Build and run the pipeline from config."""
    from syntheta.filters.quality import QualityFilter
    from syntheta.filters.safety import SafetyFilter
    from syntheta.filters.template_collapse import TemplateCollapseDetector
    from syntheta.generators.persona import PersonaGenerator
    from syntheta.generators.seed_dataset import SeedDatasetGenerator
    from syntheta.generators.topic_tree import TopicTreeGenerator
    from syntheta.llm.backend import OpenAICompatibleLLM
    from syntheta.pipeline.pipeline import Pipeline
    from syntheta.transformers.evol_instruct import EvolInstruct
    from syntheta.transformers.response import ResponseGenerator

    # Build LLM
    llm_cfg = config["llm"]
    api_key = os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
    llm = OpenAICompatibleLLM(
        model=llm_cfg["model"],
        response_model=llm_cfg.get("response_model"),
        embedding_model=llm_cfg.get("embedding_model"),
        base_url=llm_cfg.get("base_url"),
        api_key=api_key,
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        max_concurrent=llm_cfg.get("max_concurrent", 10),
        max_retries=llm_cfg.get("max_retries", 3),
        timeout=llm_cfg.get("timeout", 60),
        pricing=llm_cfg.get("pricing"),
    )

    # Build generator
    method = config["method"]
    difficulty = config.get("difficulty", [1, 5])
    prompt_overrides = config.get("prompts", {})

    if method == "topic_tree":
        generator = TopicTreeGenerator(
            domain=config["domain"],
            task_types=config.get("task_types"),
            languages=config.get("languages"),
            difficulty_range=tuple(difficulty),
            topic_depth=config.get("topic_tree", {}).get("depth", 2),
            topic_breadth=config.get("topic_tree", {}).get("breadth", 5),
            seed=config.get("seed"),
            prompt_overrides=prompt_overrides,
            llm=llm,
        )
    elif method == "persona":
        generator = PersonaGenerator(
            domain=config["domain"],
            languages=config.get("languages"),
            seed=config.get("seed"),
            prompt_overrides=prompt_overrides,
            llm=llm,
        )
    elif method == "seed_dataset":
        generator = SeedDatasetGenerator(
            source=config.get("source", ""),
            seed=config.get("seed"),
            prompt_overrides=prompt_overrides,
            llm=llm,
        )
    else:
        click.echo(f"Unknown method: {method}", err=True)
        raise SystemExit(1)

    # Build transformers
    transformers = []
    if config.get("evolve", {}).get("enabled"):
        transformers.append(
            EvolInstruct(
                rounds=config["evolve"].get("rounds", 1),
                strategies=config["evolve"].get("strategies"),
                seed=config.get("seed"),
                prompt_overrides=prompt_overrides,
                llm=llm,
            )
        )
    if config.get("responses", {}).get("enabled"):
        transformers.append(
            ResponseGenerator(
                use_cot=config.get("responses", {}).get("use_cot", False),
                prompt_overrides=prompt_overrides,
                llm=llm,
            )
        )

    # Build filters
    filters = []
    filter_cfg = config.get("filters", {})
    if filter_cfg.get("safety", {}).get("enabled"):
        safety_cfg = filter_cfg["safety"]
        filters.append(
            SafetyFilter(
                cultural_context=safety_cfg.get("cultural_context"),
                blocklist_topics=safety_cfg.get("blocklist_topics"),
                blocklist_words=safety_cfg.get("blocklist_words"),
                custom_instruction=safety_cfg.get("custom_instruction"),
                prompt_overrides=prompt_overrides,
                llm=llm,
            )
        )
    if filter_cfg.get("quality", {}).get("enabled"):
        filters.append(
            QualityFilter(
                min_score=filter_cfg["quality"].get("min_score", 0.7),
                prompt_overrides=prompt_overrides,
                llm=llm,
            )
        )
    if filter_cfg.get("template_collapse", {}).get("enabled"):
        tc_cfg = filter_cfg["template_collapse"]
        filters.append(
            TemplateCollapseDetector(
                prefix_length=tc_cfg.get("prefix_length", 50),
                max_repeat=tc_cfg.get("max_repeat", 3),
                action=tc_cfg.get("action", "report"),
            )
        )

    # Build and run pipeline
    pipe = Pipeline(
        generator=generator,
        transformers=transformers,
        filters=filters,
        llm=llm,
        seed=config.get("seed"),
        over_generate_factor=config.get("over_generate_factor", 1.2),
        batch_size=config.get("batch_size", 100),
    )

    dataset = pipe.run(
        n=config["n"],
        output=config["output"],
        checkpoint_path=config.get("checkpoint_path", "./checkpoints"),
        resume=resume,
        config_dict=config,
    )

    click.echo(f"\nGenerated {len(dataset)} samples → {config['output']}")

    # Print cost report
    cost = llm.cost_tracker.get_cost(llm.model)
    if cost["total_cost"] is not None:
        click.echo(f"Total cost: ${cost['total_cost']:.4f}")
    click.echo(f"Total tokens: {cost['total_tokens']:,}")


def _parse_value(value: str):
    """Parse a CLI value string to appropriate Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
