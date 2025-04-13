"""
Workflow module for ImageWand.

This module provides functionality to chain multiple image processing operations
together and save them as reusable workflows.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import click
from tqdm import tqdm

from imagewand.align import align_image
from imagewand.autocrop import autocrop as autocrop_func
from imagewand.config import IMAGEWAND_CONFIG_DIR
from imagewand.filters import apply_filters
from imagewand.resize import resize_image
from imagewand.rmbg import remove_background

# Define workflow storage path
WORKFLOWS_PATH = os.path.join(IMAGEWAND_CONFIG_DIR, "workflows.json")


class Workflow:
    """
    Class representing a sequence of image processing operations.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.steps = []

    def add_step(self, operation: str, params: Dict[str, Any]):
        """
        Add an operation to the workflow.

        Args:
            operation: Operation name (align, autocrop, resize, filter, rmbg)
            params: Parameters for the operation
        """
        self.steps.append({"operation": operation, "params": params})
        return self

    def remove_step(self, index: int):
        """
        Remove a step from the workflow.

        Args:
            index: Index of the step to remove
        """
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
        return self

    def execute(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> str:
        """
        Execute the workflow on an image.

        Args:
            input_path: Path to input image
            output_path: Path to save final output (optional)
            show_progress: Whether to show progress information

        Returns:
            Path to the output image
        """
        if not self.steps:
            return input_path

        current_path = input_path
        temp_files = []

        try:
            total_steps = len(self.steps)
            with tqdm(total=total_steps, disable=not show_progress) as pbar:
                for i, step in enumerate(self.steps):
                    operation = step["operation"]
                    params = step["params"]

                    # Set up intermediate output path if needed
                    is_last_step = i == total_steps - 1
                    step_output = output_path if is_last_step else None

                    # Execute the step
                    if show_progress:
                        pbar.set_description(f"Step {i+1}/{total_steps}: {operation}")

                    if operation == "align":
                        result = align_image(
                            current_path,
                            output_path=step_output,
                            method=params.get("method", "auto"),
                            angle_threshold=params.get("angle_threshold", 1.0),
                        )
                    elif operation == "autocrop":
                        result = autocrop_func(
                            current_path,
                            output_path=step_output,
                            mode=params.get("mode", "auto"),
                            border_percent=params.get("border_percent", -1),
                            margin=params.get("margin", -1),
                        )
                    elif operation == "resize":
                        result = resize_image(
                            current_path,
                            output_path=step_output,
                            width=params.get("width"),
                            height=params.get("height"),
                            percent=params.get("percent"),
                        )
                    elif operation == "filter":
                        filters = params.get("filters", [])
                        if isinstance(filters, str):
                            filters = [f.strip() for f in filters.split(",")]

                        result = apply_filters(
                            current_path, filters, output_path=step_output
                        )
                    elif operation == "rmbg":
                        result = remove_background(
                            current_path,
                            output_path=step_output,
                            alpha_matting=params.get("alpha_matting", False),
                            alpha_matting_foreground_threshold=params.get(
                                "alpha_matting_foreground_threshold", 240
                            ),
                            alpha_matting_background_threshold=params.get(
                                "alpha_matting_background_threshold", 10
                            ),
                            alpha_matting_erode_size=params.get(
                                "alpha_matting_erode_size", 10
                            ),
                            model_name=params.get("model", "u2net"),
                        )
                    else:
                        # Unknown operation, skip
                        result = current_path

                    # If not the last step, this is a temporary file
                    if not is_last_step and result != current_path:
                        temp_files.append(result)

                    # Update current path for next step
                    current_path = result
                    pbar.update(1)

        finally:
            # Clean up temporary files
            if output_path and show_progress:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file) and temp_file != output_path:
                            os.remove(temp_file)
                    except Exception:
                        # Ignore errors during cleanup
                        pass

        return current_path

    def to_dict(self) -> Dict:
        """Convert workflow to dictionary for serialization."""
        return {"name": self.name, "steps": self.steps}

    @classmethod
    def from_dict(cls, data: Dict) -> "Workflow":
        """Create workflow from dictionary."""
        workflow = cls(data.get("name", ""))
        workflow.steps = data.get("steps", [])
        return workflow


def load_workflows() -> Dict[str, Workflow]:
    """
    Load all saved workflows.

    Returns:
        Dictionary of workflow name to Workflow object
    """
    # Make sure parent directory exists
    workflows_path = Path(WORKFLOWS_PATH)
    workflows_dir = workflows_path.parent
    workflows_dir.mkdir(exist_ok=True)

    if not workflows_path.exists():
        return {}

    try:
        with open(WORKFLOWS_PATH, "r") as f:
            data = json.load(f)

        workflows = {}
        for name, workflow_data in data.items():
            workflows[name] = Workflow.from_dict(workflow_data)

        return workflows
    except Exception as e:
        click.echo(f"Error loading workflows: {e}")
        return {}


def save_workflow(workflow: Workflow):
    """
    Save a workflow.

    Args:
        workflow: Workflow to save
    """
    # Make sure parent directory exists
    workflows_path = Path(WORKFLOWS_PATH)
    workflows_dir = workflows_path.parent
    workflows_dir.mkdir(exist_ok=True)

    # Load existing workflows
    workflows = load_workflows()
    workflows[workflow.name] = workflow

    # Convert to serializable format
    data = {name: wf.to_dict() for name, wf in workflows.items()}

    # Save workflows
    with open(WORKFLOWS_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Print debug info about where it was saved
    click.echo(f"Workflow saved to: {WORKFLOWS_PATH}")


def delete_workflow(name: str) -> bool:
    """
    Delete a workflow.

    Args:
        name: Name of workflow to delete

    Returns:
        True if workflow was deleted, False otherwise
    """
    workflows = load_workflows()
    if name in workflows:
        del workflows[name]

        # Convert to serializable format
        data = {name: wf.to_dict() for name, wf in workflows.items()}

        # Save workflows
        with open(WORKFLOWS_PATH, "w") as f:
            json.dump(data, f, indent=2)

        return True

    return False


@click.command()
@click.argument("input_path", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", type=click.Path(), help="Output path")
@click.option("--workflow", "-w", help="Workflow to use")
@click.option(
    "--list", "-l", "list_workflows", is_flag=True, help="List available workflows"
)
@click.option("--delete", "-d", help="Delete a workflow")
def workflow_command(
    input_path: Optional[str] = None,
    output: Optional[str] = None,
    workflow: Optional[str] = None,
    list_workflows: bool = False,
    delete: Optional[str] = None,
):
    """
    Execute a saved workflow on an image.

    A workflow is a sequence of operations (align, autocrop, resize, filter, rmbg)
    that are applied to an image in order.

    Examples:
        imagewand workflow input.jpg -w my_workflow
        imagewand workflow --list  # List available workflows
        imagewand workflow --delete my_workflow  # Delete a workflow
    """
    # Handle listing workflows without requiring input path
    if list_workflows:
        workflows = load_workflows()
        if not workflows:
            click.echo("No workflows found.")
            return

        click.echo("Available workflows:")
        for name, wf in workflows.items():
            steps_str = " → ".join(step["operation"] for step in wf.steps)
            click.echo(f"  {name}: {steps_str}")
        return

    # Handle deleting workflow without requiring input path
    if delete:
        if delete_workflow(delete):
            click.echo(f"Workflow '{delete}' deleted.")
        else:
            click.echo(f"Workflow '{delete}' not found.")
        return

    # For executing a workflow, we need input path and workflow name
    if not input_path:
        click.echo("Input path is required when executing a workflow.")
        return

    if not workflow:
        click.echo(
            "No workflow specified. Use --workflow to specify a workflow or --list to see available workflows."
        )
        return

    workflows = load_workflows()
    if workflow not in workflows:
        click.echo(f"Workflow '{workflow}' not found.")
        return

    result = workflows[workflow].execute(input_path, output)
    click.echo(f"Workflow completed. Output saved to: {result}")


if __name__ == "__main__":
    workflow_command()
