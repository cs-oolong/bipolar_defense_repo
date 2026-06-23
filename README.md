# Isolating Refusal and Compliance Heads: Unconditional Bipolar Defenses Against White-Box and Multi-Turn Attacks

This repository contains the official code and paper source for the study of internal safety circuits in instruction-tuned LLMs (Qwen2.5-7B and Qwen2.5-1.5B) and the implementation of "Bipolar" mechanistic defenses against both single-turn white-box attacks (GCG) and multi-turn conversational attacks (Crescendo).

## Abstract
Recent advances in mechanistic interpretability have demonstrated that safety mechanisms in Large Language Models (LLMs) often reside in sparse, targetable circuits. We present an empirical evaluation of internal safety circuits in Qwen2.5-7B through activation patching. We introduce a purely mechanized "bipolar" defense that simultaneously amplifies refusal heads and suppresses compliance heads. Against white-box adversarial attacks (nanoGCG), we find that unconditional bipolar assertion achieves a lower attack success rate (33% ASR) than conditional variants (36-42% ASR). This reveals a selectivity-robustness tradeoff: the precision benefit of conditional gating creates an exploitable structural vulnerability under white-box optimization.

We extend this analysis to multi-turn Crescendo attacks on Qwen2.5-1.5B, showing that conversational escalation erodes safety via **KV-cache poisoning**: by the final turn, assistant-generated tokens vastly outnumber the system prompt in the attention context, diluting the safety signal even though refusal-head activation norms remain stable. The same unconditional bipolar defense blocks the terminal harmful request without disabling the model's ability to discuss the underlying topic in earlier, benign turns.

On the larger Qwen2.5-7B-Instruct, the same defense only works as **additive steering-vector injection**, not multiplicative amplification: the dominant refusal head's current direction already points toward compliance-opener vocabulary ("Sure", "Certainly") at every Crescendo turn, including the terminal one, so amplifying it in place only strengthens the wrong direction. Switching to additive injection of a precomputed contrastive steering vector — the same mechanism as the GCG-side defense — drops compliance from 100% to 20% (RDX synthesis) and 100% to 0% (vehicle hotwiring) across two of three tested escalation scenarios; on the third (methamphetamine synthesis), the undefended model already refuses 100% of the time, showing that Crescendo's escalation efficacy is itself sharply domain-dependent. Combining the steering vectors of all implicated heads consistently outperforms injecting the single most causally important head alone, even though that head's activation-patching score is ~3x larger than any other's.

## Reproducibility

**GCG / single-turn (Qwen2.5-7B):**
```bash
cd code/
python eval_full.py
```
*Note: `circuit_map.json`, `advbench.csv`, `template_ablation.json`, and `steering_vectors_50.pt` must be present in `code/` (included in this repository).*

**Crescendo / multi-turn (Qwen2.5-1.5B and Qwen2.5-7B):**
```bash
cd crescendo/
python crescendo_baseline.py            # undefended trajectory
python crescendo_bipolar_defense.py      # defended trajectory
python crescendo_circuit_discovery_v3.py # context-override circuit discovery
python crescendo_7b_pipeline.py --defense   # 7B: matched baseline vs. additive-injection defense, multi-scenario
python single_head_vs_all_poc.py --all      # 7B: single dominant head vs. all-11-heads steering comparison
```
*Note: `circuit_map_1_5B.json`, `crescendo_override_circuit_v3.json`, and `patching_results_1_5B.json` must be present in `crescendo/` (included in this repository). The 7B scripts additionally require `steering_vectors_50.pt`. See `FINDINGS_7B_DEFENSE.md` for the full write-up of the 7B results.*

## Repository Structure

- `paper/`: Contains the finalized research paper (PDF).
- `code/`: Core evaluation and circuit discovery scripts for the single-turn (GCG) study.
  - `circuit_discovery_gcg.py`: Mapping safety-relevant attention heads on Qwen2.5-7B.
  - `adaptive_defense.py`: Implementation of the zero-shot threshold gating.
  - `eval_full.py`: Full evaluation suite against AdvBench and JBB-Behaviors.
- `crescendo/`: Multi-turn Crescendo attack analysis and defense.
  - `circuit_discovery_1_5B.py` / `crescendo_circuit_discovery_v3.py`: Context-override circuit discovery on Qwen2.5-1.5B via neutral-filler contrastive patching.
  - `crescendo_baseline.py`: Undefended 5-turn KV-cache trajectory (refusal/compliance norms per turn).
  - `crescendo_bipolar_defense.py`: Bipolar defense applied across the Crescendo turn sequence.
  - `crescendo_7b_pipeline.py`: 10-scenario benchmark across threat domains on Qwen2.5-7B, additive steering-vector defense.
  - `logit_lens_poc.py`: Projects a head's current output through the final norm + unembedding at each turn, to check which vocabulary it's promoting (revealed the dominant refusal head naturally promotes compliance-opener tokens — the bug behind the additive-injection fix).
  - `single_head_vs_all_poc.py`: Compares additive injection of the single dominant refusal head against injecting all 11 implicated heads combined, across multipliers and scenarios.
  - `steering_vectors_50.pt`: Precomputed per-head contrastive steering vectors (chat-template vs. raw-text activation means across 50 harmful prompts), used for additive injection on both the GCG and Crescendo sides.
  - `dla_decomposition_poc.py` / `attention_entropy_poc.py` / `mlp_compensation_poc.py`: Per-turn, per-scenario mechanistic decomposition (direct logit attribution, attention entropy, MLP-output compensation) used to explain why Crescendo's undefended success rate varies by harm domain.
  - `FINDINGS_7B_DEFENSE.md`: Full write-up of the 7B defense experiments and the mechanistic explanation for domain-dependent Crescendo resistance.
  - `crescendo_generations.txt` / `crescendo_generations_defense.txt`: Raw model outputs per turn, undefended vs. defended.
- `figures/`: Key experimental results and visualizations (referenced in the paper), including circuit heatmaps and KV-cache decay plots for both attack surfaces.

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
- **KV-Cache Poisoning**: In multi-turn Crescendo attacks, the model's own prior outputs dominate the attention context by the terminal turn, diluting the system prompt's safety signal even though refusal-head activation norms stay stable.
- **Norm ≠ Behavior**: Elevated refusal-head activation does not guarantee refusal behavior — the defended model's refusal norm is comparable across a benign turn (complies) and the harmful terminal turn (refuses), showing that norm magnitude alone is not a reliable behavioral predictor even under defense.
- **Diffuse vs. Sparse Circuits**: The Crescendo context-override circuit is distributed across many heads (top-10 heads carry only ~44% of causal mass), unlike the highly concentrated GCG safety circuit — making Crescendo harder to defend against via targeted head ablation alone.
- **Additive Injection, Not Amplification (Qwen2.5-7B)**: Naively amplifying a head's existing direction (`hidden *= multiplier`) fails as a Crescendo defense because the dominant refusal head's natural direction already points toward compliance-opener vocabulary ("Sure", "Certainly") at every turn, including the harmful terminal one. Only switching to additive injection of a precomputed contrastive steering vector — the same mechanism the GCG-side defense uses — produces a real, coherent compliance-rate drop.
- **Sparse Discovery ≠ Sparse Intervention**: Although one head (L25-H1) has ~3x the causal-patching score of any other on Qwen2.5-7B, injecting its steering vector alone consistently underperforms injecting all 11 implicated heads' vectors combined, confirmed across two independent Crescendo scenarios. A head's importance for *detecting* refusal does not predict its sufficiency for *injecting* it.
- **Crescendo's Success Is Domain-Dependent**: The identical 5-turn escalation pattern jailbreaks the undefended Qwen2.5-7B model 100% of the time on RDX synthesis and vehicle hotwiring, but 0% of the time on methamphetamine synthesis — escalation efficacy varies sharply by harm category, not only by model or defense.
- **Mechanistic Driver of Domain Resistance**: On the resistant scenario, the dominant refusal head's direct logit attribution sign-flips from compliance-promoting to refusal-promoting specifically at the terminal turn — the only such flip observed across all tested scenarios and turns — coinciding with a sharp collapse in the same head's attention entropy, while compliance-head contributions stay uniform regardless of outcome.

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
