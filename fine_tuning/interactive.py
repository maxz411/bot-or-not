from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

    from fine_tuning.constants import (
        DEFAULT_GEMINI_LOCATION,
        FINAL_RESULTS_DIR,
        RAW_DIR,
        RUNS_DIR,
    )
    from fine_tuning.dataset_inventory import (
        DatasetAsset,
        FinalDatasetAsset,
        discover_dataset_assets,
        discover_final_dataset_assets,
    )
    from fine_tuning.env import load_project_env
    from fine_tuning.model_catalog import (
        RunnableModel,
        RunnableModelCatalog,
        list_runnable_models,
    )
else:
    from .constants import DEFAULT_GEMINI_LOCATION, FINAL_RESULTS_DIR, RAW_DIR, RUNS_DIR
    from .dataset_inventory import (
        DatasetAsset,
        FinalDatasetAsset,
        discover_dataset_assets,
        discover_final_dataset_assets,
    )
    from .env import load_project_env
    from .model_catalog import RunnableModel, RunnableModelCatalog, list_runnable_models

def list_selectable_dataset_assets() -> tuple[DatasetAsset, ...]:
    return discover_dataset_assets()


def list_selectable_final_dataset_assets() -> tuple[FinalDatasetAsset, ...]:
    return discover_final_dataset_assets()


def choose_dataset_ids() -> tuple[int, ...]:
    import questionary

    assets = list_selectable_dataset_assets()
    choices = [
        questionary.Choice(
            title=f"{asset.dataset_id} ({asset.source})",
            value=asset.dataset_id,
            checked=True,
        )
        for asset in assets
    ]
    selected = questionary.checkbox(
        "Select full dataset files to evaluate",
        choices=choices,
        validate=lambda values: True if values else "Select at least one dataset.",
    ).ask()
    if not selected:
        raise KeyboardInterrupt("Interactive dataset selection cancelled.")
    return tuple(int(dataset_id) for dataset_id in selected)


def choose_final_dataset_ids() -> tuple[int, ...]:
    import questionary

    assets = list_selectable_final_dataset_assets()
    choices = [
        questionary.Choice(
            title=f"{asset.dataset_id}",
            value=asset.dataset_id,
            checked=True,
        )
        for asset in assets
    ]
    selected = questionary.checkbox(
        "Select final dataset files to generate submission detections for",
        choices=choices,
        validate=lambda values: True if values else "Select at least one dataset.",
    ).ask()
    if not selected:
        raise KeyboardInterrupt("Interactive final dataset selection cancelled.")
    return tuple(int(dataset_id) for dataset_id in selected)


def choose_workflow() -> str:
    import questionary

    selected = questionary.select(
        "Choose workflow",
        choices=[
            questionary.Choice(
                title="Evaluate on labeled full datasets",
                value="evaluate",
            ),
            questionary.Choice(
                title="Generate final submission detections from datasets/final/",
                value="submit-final",
            ),
        ],
        use_indicator=True,
    ).ask()
    if selected is None:
        raise KeyboardInterrupt("Interactive workflow selection cancelled.")
    return str(selected)


def _initial_google_context(
    *,
    google_project: str | None,
    google_location: str | None,
) -> tuple[str | None, str | None]:
    load_project_env()
    resolved_project = google_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    resolved_location = (
        google_location
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or DEFAULT_GEMINI_LOCATION
    )
    return resolved_project, resolved_location


def _prompt_google_context(
    *,
    current_project: str | None,
    current_location: str | None,
) -> tuple[str | None, str | None]:
    import questionary

    project_input = questionary.text(
        "Google Cloud project id for Vertex model refresh (leave blank to skip Gemini)",
        default=current_project or "",
    ).ask()
    if project_input is None:
        raise KeyboardInterrupt("Interactive Google project entry cancelled.")

    resolved_project = project_input.strip() or None
    if not resolved_project:
        return None, current_location or DEFAULT_GEMINI_LOCATION

    location_input = questionary.text(
        "Vertex AI location",
        default=current_location or DEFAULT_GEMINI_LOCATION,
    ).ask()
    if location_input is None:
        raise KeyboardInterrupt("Interactive Google location entry cancelled.")
    resolved_location = location_input.strip() or DEFAULT_GEMINI_LOCATION
    return resolved_project, resolved_location


def _print_catalog_errors(catalog: RunnableModelCatalog) -> None:
    if catalog.openai_error is not None:
        print(f"openai_model_refresh_error={catalog.openai_error}")
    if catalog.gemini_error is not None:
        print(f"gemini_model_refresh_error={catalog.gemini_error}")


def _choice_title(model: RunnableModel) -> str:
    return model.title


def choose_runnable_model(
    *,
    openai_api_key: str | None,
    google_project: str | None,
    google_location: str | None,
) -> tuple[RunnableModel, str | None, str | None]:
    import questionary

    resolved_google_project, resolved_google_location = _initial_google_context(
        google_project=google_project,
        google_location=google_location,
    )
    should_prompt_for_google_context = False

    while True:
        if should_prompt_for_google_context:
            (
                resolved_google_project,
                resolved_google_location,
            ) = _prompt_google_context(
                current_project=resolved_google_project,
                current_location=resolved_google_location,
            )
            should_prompt_for_google_context = False

        catalog = list_runnable_models(
            openai_api_key=openai_api_key,
            gemini_project=resolved_google_project,
            gemini_location=resolved_google_location,
        )
        _print_catalog_errors(catalog)

        choices: list[questionary.Choice | questionary.Separator] = [
            questionary.Choice(
                title="Refresh available models",
                value="__refresh__",
            ),
            questionary.Choice(
                title="Reconfigure Google Cloud project/location",
                value="__reconfigure_google__",
            ),
        ]

        openai_models = [
            model for model in catalog.models if model.provider == "openai"
        ]
        gemini_models = [
            model for model in catalog.models if model.provider == "gemini"
        ]

        if openai_models:
            choices.append(questionary.Separator(" OpenAI "))
            choices.extend(
                questionary.Choice(
                    title=_choice_title(model),
                    value=model,
                )
                for model in openai_models
            )

        if gemini_models:
            choices.append(questionary.Separator(" Gemini "))
            choices.extend(
                questionary.Choice(
                    title=_choice_title(model),
                    value=model,
                )
                for model in gemini_models
            )

        choices.append(
            questionary.Choice(
                title="Enter a model manually",
                value="__manual__",
            )
        )

        selected = questionary.select(
            "Select a fine-tuned model to evaluate",
            choices=choices,
            use_indicator=True,
        ).ask()
        if selected is None:
            raise KeyboardInterrupt("Interactive model selection cancelled.")
        if selected == "__refresh__":
            continue
        if selected == "__reconfigure_google__":
            should_prompt_for_google_context = True
            continue
        if selected == "__manual__":
            provider = questionary.select(
                "Select provider for manual model entry",
                choices=[
                    questionary.Choice(title="OpenAI", value="openai"),
                    questionary.Choice(title="Gemini / Vertex AI", value="gemini"),
                ],
            ).ask()
            if provider is None:
                raise KeyboardInterrupt("Interactive manual provider selection cancelled.")
            model_id = questionary.text("Enter runnable model id").ask()
            if not model_id:
                raise KeyboardInterrupt("Interactive manual model entry cancelled.")

            manual_project = resolved_google_project
            manual_location = resolved_google_location
            if provider == "gemini":
                if manual_project is None:
                    manual_project, manual_location = _prompt_google_context(
                        current_project=resolved_google_project,
                        current_location=resolved_google_location,
                    )
            return (
                RunnableModel(
                    provider=provider,
                    runnable_id=model_id.strip(),
                    kind="final",
                    title=f"[{provider} manual] {model_id.strip()}",
                    lineage_id=None,
                    base_model=None,
                    source="manual",
                    project=manual_project,
                    location=manual_location,
                ),
                manual_project,
                manual_location,
            )

        if not isinstance(selected, RunnableModel):
            raise RuntimeError(
                f"Unexpected model selection value: {selected!r}"
            )
        return selected, resolved_google_project, resolved_google_location


def run_interactive_wizard(
    *,
    openai_api_key: str | None,
    google_project: str | None,
    google_location: str | None,
    workflow: str | None,
    team_name: str,
    raw_dir: str,
    runs_dir: str,
    final_results_dir: str,
    raw_path: str | None,
    report_path: str | None,
    run_slug: str | None,
    max_workers: int,
    max_retries: int,
    max_output_tokens: int,
    report_every: int,
    save_every: int,
) -> int:
    if __package__ in {None, ""}:
        from fine_tuning import cli as fine_tuning_cli
    else:
        from . import cli as fine_tuning_cli

    try:
        resolved_workflow = workflow or choose_workflow()
        selected_model, resolved_google_project, resolved_google_location = (
            choose_runnable_model(
                openai_api_key=openai_api_key,
                google_project=google_project,
                google_location=google_location,
            )
        )
        if resolved_workflow == "evaluate":
            dataset_ids = choose_dataset_ids()
        elif resolved_workflow == "submit-final":
            dataset_ids = choose_final_dataset_ids()
        else:
            raise ValueError(f"Unsupported interactive workflow: {resolved_workflow}")
    except KeyboardInterrupt as exc:
        print(str(exc))
        return 1

    print(f"selected_workflow={resolved_workflow}")
    print(f"selected_provider={selected_model.provider}")
    print(f"selected_model={selected_model.runnable_id}")
    if selected_model.provider == "gemini":
        print(f"selected_google_project={resolved_google_project or '(unset)'}")
        print(f"selected_google_location={resolved_google_location or '(unset)'}")
    print(
        "selected_dataset_ids=" + ",".join(str(dataset_id) for dataset_id in dataset_ids)
    )

    common_args = argparse.Namespace(
        model=selected_model.runnable_id,
        dataset_ids=",".join(str(dataset_id) for dataset_id in dataset_ids),
        raw_path=raw_path,
        raw_dir=raw_dir,
        run_slug=run_slug,
        max_workers=max_workers,
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        report_every=report_every,
        save_every=save_every,
        api_key=openai_api_key,
        project=resolved_google_project or selected_model.project,
        location=resolved_google_location or selected_model.location,
    )

    if resolved_workflow == "evaluate":
        evaluate_args = argparse.Namespace(
            **vars(common_args),
            collection=None,
            runs_dir=runs_dir,
            report_path=report_path,
        )
        if selected_model.provider == "openai":
            return fine_tuning_cli.cmd_openai_evaluate(evaluate_args)
        if selected_model.provider == "gemini":
            return fine_tuning_cli.cmd_gemini_evaluate(evaluate_args)
    elif resolved_workflow == "submit-final":
        submit_args = argparse.Namespace(
            **vars(common_args),
            team_name=team_name,
            output_dir=final_results_dir,
        )
        if selected_model.provider == "openai":
            return fine_tuning_cli.cmd_openai_submit_final(submit_args)
        if selected_model.provider == "gemini":
            return fine_tuning_cli.cmd_gemini_submit_final(submit_args)

    raise ValueError(f"Unsupported provider: {selected_model.provider}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fine_tuning.interactive",
        description=(
            "Open a single interactive wizard for evaluating fine-tuned OpenAI "
            "and Gemini models or generating final competition detections."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional OPENAI_API_KEY override.",
    )
    parser.add_argument(
        "--google-project",
        default=None,
        help="Optional Google Cloud project id for Vertex model refresh/evaluation.",
    )
    parser.add_argument(
        "--google-location",
        default=None,
        help=(
            "Optional Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or "
            f"{DEFAULT_GEMINI_LOCATION}."
        ),
    )
    parser.add_argument(
        "--workflow",
        default=None,
        choices=("evaluate", "submit-final"),
        help="Optional workflow override. When omitted, the wizard asks.",
    )
    parser.add_argument(
        "--team-name",
        default="maxilillian",
        help="Team name used for final submission file names.",
    )
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Optional explicit raw results path. Reusing a path resumes the run.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Directory for persisted raw inference artifacts.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory for generated human-readable run reports.",
    )
    parser.add_argument(
        "--final-results-dir",
        default=str(FINAL_RESULTS_DIR),
        help="Directory for final competition-formatted detection files.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the generated run report.",
    )
    parser.add_argument(
        "--run-slug",
        default=None,
        help="Optional suffix used when auto-generating raw/report paths.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent inference workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per example before failing the run.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max tokens requested from the model for each classification.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="Print progress every N completed examples.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist the raw artifact every N completed examples.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_interactive_wizard(
        openai_api_key=args.openai_api_key,
        google_project=args.google_project,
        google_location=args.google_location,
        workflow=args.workflow,
        team_name=args.team_name,
        raw_dir=args.raw_dir,
        runs_dir=args.runs_dir,
        final_results_dir=args.final_results_dir,
        raw_path=args.raw_path,
        report_path=args.report_path,
        run_slug=args.run_slug,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        report_every=args.report_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
