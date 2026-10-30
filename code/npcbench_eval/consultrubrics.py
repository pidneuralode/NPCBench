"""CLI wrapper for the NPCBench ConsultRubrics evaluator."""
from __future__ import annotations

def main() -> None:
    from o1_eval.simple_evals_main.npc_o1_eval import main as o1_main

    o1_main()


if __name__ == "__main__":
    main()
