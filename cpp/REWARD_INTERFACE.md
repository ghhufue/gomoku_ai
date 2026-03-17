# Reward Interface Contract

## Goal

Define the minimal data contract between Python and the C++ backend for reward evaluation.

Target direction:

- Python only passes board state and move context.
- C++ owns reward config, event extraction, score calculation, and reward aggregation.
- C++ returns only compact numeric results.

## Python -> C++

Target Python call:

```python
evaluate_reward(board_before, row, col, player)
```

Target backend signature:

```cpp
RewardResult evaluate_reward(
    const Board& board_before,
    int row,
    int col,
    int player
);
```

### Input fields

`board_before`
- Type: `numpy.ndarray[int8]`
- Shape: `(15, 15)`
- Values:
  - `0` = empty
  - `1` = black
  - `-1` = white

`row`
- Type: `int`
- Meaning: target row of the move

`col`
- Type: `int`
- Meaning: target col of the move

`player`
- Type: `int`
- Allowed values:
  - `1`
  - `-1`

### Input constraints

- Python passes the board before the move is placed.
- `board_before[row, col]` must be empty.
- C++ simulates the placed stone internally.
- Python does not pass reward coefficients.

## C++ -> Python

Target return payload:

```python
{
    "reward": float,
    "offense_score": float,
    "defense_score": float,
    "events": [
        (event_id: int, count: int),
        ...
    ],
}
```

That is the whole public contract.

No extra nested payloads should be returned by default.

## Field meaning

`reward`
- Final scalar reward for the move
- Computed fully inside C++

`offense_score`
- Offensive contribution of this move
- Computed fully inside C++

`defense_score`
- Defensive contribution of this move
- Computed fully inside C++

`events`
- Compact event summary list
- Each item is a pair:

```python
(event_id, count)
```

Meaning:

- `event_id`: integer enum defined by C++
- `count`: signed or unsigned integer count for how many times the event happened

Examples:

- `(EVENT_CREATE_LIVE_TWO, 2)`
  - created live-two twice
- `(EVENT_BLOCK_LIVE_THREE, 1)`
  - blocked one live-three
- `(EVENT_UNRESOLVED_FATAL, 1)`
  - left one fatal threat unresolved

## Event model

Everything should be represented as events.

This includes:

- created shapes
- destroyed shapes
- upgraded shapes
- downgraded shapes
- converted shapes
- blocked threats
- weakened threats
- unresolved threats
- terminal outcomes if needed

Python should not receive raw shape state snapshots.

Python should not receive:

- before summary
- after summary
- offense delta dict
- defense delta dict
- affected action lists
- candidate details
- direction windows

All of those are internal C++ implementation details.

## Event enum ownership

The event vocabulary should be defined in C++ as an integer enum.

Suggested examples:

```cpp
enum RewardEventId {
    EVENT_CREATE_LIVE_TWO = 1,
    EVENT_CREATE_LIVE_THREE = 2,
    EVENT_CREATE_SLEEP_THREE = 3,
    EVENT_CREATE_LIVE_FOUR = 4,
    EVENT_CREATE_RUSH_FOUR = 5,
    EVENT_BLOCK_WINNING_ACTION = 6,
    EVENT_BLOCK_LIVE_FOUR = 7,
    EVENT_BLOCK_LIVE_THREE = 8,
    EVENT_WEAKEN_LIVE_THREE = 9,
    EVENT_UNRESOLVED_FATAL = 10,
    EVENT_UNRESOLVED_FOUR = 11,
    EVENT_UNRESOLVED_LIVE_THREE = 12,
};
```

The exact enum can change, but Python should treat event ids as opaque.

## Reward config ownership

Target ownership:

- C++ loads `configs/reward.toml`
- C++ owns `RewardConfig`
- C++ computes:
  - offense score
  - defense score
  - final reward

Python should not rebuild reward from returned data.

Python only consumes:

- `reward`
- `offense_score`
- `defense_score`
- `events`

## Python-side usage target

Python-side reward path should reduce to:

```python
result = cpp_backend.evaluate_reward(board_before, row, col, player)
reward = result["reward"]
```

Optional logging:

```python
info = {
    "offense_score": result["offense_score"],
    "defense_score": result["defense_score"],
    "events": result["events"],
}
```

## Non-goals

The backend should not return rich debug structures through the normal reward interface.

If detailed debugging is needed later, add a separate debug-only interface instead of expanding the main reward payload.
