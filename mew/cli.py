import typer
import time
import random
from rich.console import Console

from mew.train.toy import ToyModel, toy_dataloader
import pytorch_lightning as pl

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("version")
def version():
    """Print MEW version."""
    from .version import __version__

    console.print(f"MEW {__version__}")


@app.command("train-toy")
def train_toy(steps: int = 200, seed: int = 42):
    """
    Minimal 'training' loop on random tensors so the pipeline and logs work.
    Safe to run CPU-only or on MPS.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # fake data: 1024 x 16 -> 8
    x = torch.randn(1024, 16)
    w = torch.randn(16, 8, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)

    for t in range(1, steps + 1):
        y_pred = x @ w
        target = torch.zeros_like(y_pred)
        loss = ((y_pred - target) ** 2).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        if t % 50 == 0 or t == 1 or t == steps:
            console.log(f"[toy] step={t} loss={loss.item():.4f}")
            time.sleep(0.01)  # keep logs readable

    console.print("[bold green]Toy training finished.[/bold green]")


@app.command("train")
def train(exp: str = "toy"):
    if exp == "toy":
        model = ToyModel()
        trainer = pl.Trainer(max_epochs=1, limit_train_batches=10)
        trainer.fit(model, toy_dataloader())
        typer.echo("Toy training run finished")


@app.command("data.build")
def data_build(data: str = "toy"):
    typer.echo(f"[stub] building dataset: {data}")


@app.command("eval")
def eval(exp: str = "toy"):
    typer.echo(f"[stub] evaluating experiment {exp}")


@app.command("explore")
def explore(run: str = "toy"):
    typer.echo(f"[stub] exploring run {run}")


@app.command("export.artifact")
def export_artifact(run: str = "toy"):
    typer.echo(f"[stub] exporting artifacts for {run}")


if __name__ == "__main__":
    app()
