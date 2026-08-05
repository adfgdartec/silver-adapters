from typing import List

from .models import NotebookCell, NotebookDocument


def notebook_document(cells: List[NotebookCell]) -> NotebookDocument:
    return NotebookDocument(
        cells=[
            NotebookCell(
                cell_type=cell.cell_type,
                metadata=dict(cell.metadata),
                source=list(cell.source),
                outputs=None if cell.outputs is None else list(cell.outputs),
                execution_count=cell.execution_count,
            )
            for cell in cells
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def silver_notebook_cells() -> List[NotebookCell]:
    return [
        NotebookCell(
            cell_type="markdown",
            metadata={"language": "markdown"},
            source=[
                "# Silver training bridge",
                "",
                "Inspect a dataset, launch a backend, and keep the run observable.",
            ],
        ),
        NotebookCell(
            cell_type="code",
            metadata={"language": "python"},
            source=[
                "from silver_pandas_bridge import load",
                "dataset = load('train.csv')",
                "print(dataset.report())",
            ],
        ),
        NotebookCell(
            cell_type="code",
            metadata={"language": "python"},
            source=[
                "# Emit silver-jsonl-v1 events from your training loop",
                'print(\'{"kind": "epoch", "epoch": 1, "metrics": {"loss": 0.5}}\')',
            ],
        ),
    ]
