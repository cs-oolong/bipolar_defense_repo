Circuit Discovery: Baby Steps

### What even is a "circuit" in an LLM?

A large language model has billions of parameters. But for any specific behavior — like **refusing a harmful request** — only a tiny fraction of those parameters actually matter.

A **circuit** is that tiny fraction: a small set of attention heads (and sometimes MLP neurons) that work together to produce one specific behavior.

Think of the model like a big orchestra. A "circuit" is the small group of musicians actually playing the "refusal melody." Circuit discovery is the process of figuring out *which* musicians are playing that melody.

---

### What is circuit discovery?

**Circuit discovery** = running experiments on the model to find which internal components cause a specific output.

The basic idea:
1. Pick a behavior you care about (e.g., "the model refuses a harmful prompt").
2. Find a "clean" input that triggers it and a "corrupt" input where it disappears.
3. Intervene inside the model: swap activations from the clean run into the corrupt run, one component at a time.
4. Measure which swap makes the corrupt run behave like the clean run.

If swapping head `L25-H1` makes the model refuse again, that head is probably part of the refusal circuit.
