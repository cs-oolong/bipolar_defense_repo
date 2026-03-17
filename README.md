# Isolating Refusal and Compliance Heads: Unconditional Bipolar Defenses Against White-Box Attacks

This repository contains the official code and paper source for the study of internal safety circuits in instruction-tuned LLMs (Qwen2.5-7B) and the implementation of "Bipolar" mechanistic defenses.

## Abstract
Recent advances in mechanistic interpretability have demonstrated that safety mechanisms in Large Language Models (LLMs) often reside in sparse, targetable circuits. We present an empirical evaluation of internal safety circuits in Qwen2.5-7B through activation patching. We introduce a purely mechanized "bipolar" defense that simultaneously amplifies refusal heads and suppresses compliance heads. Against white-box adversarial attacks (nanoGCG), we find that unconditional bipolar assertion achieves a lower attack success rate (33% ASR) than conditional variants (36% ASR). This reveals a selectivity--robustness tradeoff: the precision benefit of conditional gating creates an exploitable structural vulnerability under white-box optimization.

## Repository Structure

- `paper/`: Contains the LaTeX source for the research paper.
- `code/`: Core evaluation and circuit discovery scripts.
  - `circuit_discovery_gcg.py`: Mapping safety-relevant attention heads.
  - `adaptive_defense.py`: Implementation of the zero-shot threshold gating.
  - `eval_full.py`: Full evaluation suite against AdvBench and JBB-Behaviors.
- `figures/`: Key experimental results and visualizations (referenced in the paper).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Key Findings

- **Selectivity vs. Robustness**: Conditional defenses can be bypassed by white-box optimization that evades the downstream intervention rather than the detector itself.
- **Structural Vulnerabilities**: GCG attacks precisely target "Compliance Heads" (like L25-H4) to override safety mechanisms.
- **Template Dependency**: Internal safety circuits are rigid and rely on exact chat template formatting.

## License
[Specify License, e.g., MIT]

## Citation
If you use this work, please consider citing:
```bibtex
@article{dubey2026bipolar,
  title={Isolating Refusal and Compliance Heads: Unconditional Bipolar Defenses Against White-Box Attacks},
  author={Dubey, Shivam},
  year={2026}
}
```
