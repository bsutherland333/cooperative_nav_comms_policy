# Cooperative Navigation Communication Policy

This project studies learned communication strategies for cooperative localization. Each agent maintains its own estimate of the fleet, chooses whether to communicate with one other agent at each timestep, and uses successful communications to exchange range measurements and shared fleet information. The long-term goal is to reduce total group estimation uncertainty while minimizing communication.

Algorithm decisions should reference `algorithms_for_decision_making.pdf`, especially Chapter 13 for stochastic actor-critic methods. The initial scaffold follows the book's value-function critic framing: a shared stochastic local actor and a centralized value critic used for actor-critic training.

## Architecture

- `src/policy/`: actor, value critic, function-provider interface, action encoding, and state encoding abstractions.
- `src/simulation/`: simulator and rollout/result dataclasses. Will contains multiple types of simulations.
- `src/training/`: trainer orchestration for single-episode training and optional evaluation hooks.
- `src/main.py`: sparse CLI entrypoint. It should wire components together but not hold simulation or learning logic.

## Style

- Keep this research Python code readable and direct.
- Avoid unnecessary default arguments. If an object is required by an algorithm or interface, require it explicitly.
- Keep parameter definitions near the classes or files that use them.
- Add short comments for classes/functions and for non-obvious algorithmic steps. Avoid comments that merely restate simple code.
- Prefix internal-only methods and helper functions with a leading underscore.

## Running

Install local dependencies with:

```bash
python -m pip install -e '.[dev]'
```

Run tests with:

```bash
python -m pytest
```

The current CLI is a placeholder:

```bash
python src/main.py
```
