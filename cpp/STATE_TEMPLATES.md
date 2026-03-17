# Direction State Templates

## Scope

This document defines single-direction local states around the newly placed stone.

Notation:

- `X`: same color as the current stone
- `O`: blocked by opponent stone or board edge
- `_`: empty
- All templates must contain the current stone

The full local line is length `11`, but template matching is done on shorter windows that contain the current stone.

## State Families

### 1 stone

- `live1`
  - Core meaning: both sides open
  - Canonical template: `_X_`

- `sleep1`
  - Core meaning: one side blocked, one side open
  - Canonical templates:
    - `OX_`
    - `_XO`

- `dead1`
  - Core meaning: both sides blocked
  - Canonical template:
    - `OXO`

### 2 stones

- `live2`
  - Both sides open
  - Templates:
    - `_XX_`
    - `_X_X_`
    - `_X__X_`
    - `__XX__`
    - `__X_X__`

- `sleep2`
  - One side blocked
  - Typical templates:
    - `OXX_`
    - `_XXO`
    - `OX_X_`
    - `_X_XO`
    - `OX__X_`
    - `_X__XO`

- `dead2`
  - Both sides blocked
  - Typical templates:
    - `OXXO`
    - `OX_XO`
    - `OX__XO`

### 3 stones

- `live3`
  - Both sides open
  - Templates:
    - `_XXX_`
    - `_XX_X_`
    - `_X_XX_`
    - `_XX__X_`
    - `_X__XX_`

- `sleep3`
  - One side blocked
  - Templates:
    - `OXXX__`
    - `__XXXO`
    - `O_XXX_`
    - `_XXX_O`
    - `OX_XX_`
    - `_XX_XO`
    - `OXX_X_`
    - `_X_XXO`
    - `OXX__X`
    - `X__XXO`
    - `OX__XX`
    - `XX__XO`
    - `OXX__X_`
    - `_X__XXO`
    - `OX__XX_`
    - `_XX__XO`

- `dead3`
  - Both sides blocked
  - Typical templates:
    - `OXXXO`
    - `OX_XXO`
    - `OXX_XO`
    - `OXX__XO`
    - `OX__XXO`

### 4 stones

- `live4`
  - Both sides open
  - Templates:
    - `_XXXX_`
    - `_XXX_X_`
    - `_XX_XX_`
    - `_X_XXX_`

- `sleep4`
  - One side blocked
  - Use rush-four logic in code
  - Typical templates:
    - `OXXXX_`
    - `_XXXXO`
    - `OXXX_X_`
    - `_XXX_XO`
    - `OX_XXX_`
    - `_X_XXXO`
    - `OXX_XX_`
    - `_XX_XXO`

- `dead4`
  - Both sides blocked
  - Only one structural meaning: four stones but no open winning end
  - Canonical contiguous template:
    - `OXXXXO`

## State Transition Examples

Only state counts matter now. We do not score "block delta" or "destroy delta" separately.

Examples:

- `live1 + self` -> `live2` or `sleep2`
- `sleep1 + self` -> `sleep2`
- `dead1 + self` -> `dead2`
- `live2 + self` -> `live3` / `sleep3`
- `sleep2 + self` -> `sleep3`
- `dead2 + self` -> `dead3`
- `live3 + self` -> `live4` / `sleep4`
- `sleep3 + self` -> `sleep4`
- `dead3 + self` -> `dead4`
- `live4 + self` -> `five`
- `sleep4 + self` -> `five`

Concrete examples:

- `sleep2 + self -> sleep3`
  - `OXX_` + one self on the open side -> `OXXX_`

- `sleep2 + self -> live3`
  - `_XX_O` + one self on the open side can become `_XXX_`

- `live2 + self -> live3`
  - `_XX_` + one self -> `_XXX_`

- `live2 + self -> sleep3`
  - `_XXO` is already blocked on one side, after extension it becomes sleep-three

- `sleep3 + self -> sleep4`
  - `OXXX_` + one self -> `OXXXX_`

- `live3 + self -> live4`
  - `_XXX_` + one self -> `_XXXX_`

- `live4 + self -> five`
  - `_XXXX_` + one self on either side -> `XXXXX`

## Implementation Guidance

- Template matching is the source of truth.
- Every template must verify that `current_index` points to an `X`.
- `X` always means same color as the current stone.
- `O` always means different color or board edge.
- Scoring should be based on final state counts, not delta-event inference.
