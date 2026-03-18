#pragma once

#include "../direction_encoding.h"
#include "../StateValueRegistry.h"

#include <array>

namespace gomoku {

StateValueCountVector classify_restored_direction_state_counts(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    bool fill_current_stone
);

StateValueCountVector classify_restored_direction_state_counts_with_center(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    RelativeCellState center_state
);

StateValueDeltaVector classify_direction_state_delta(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index
);

StateValueDeltaVector classify_direction_state_delta_with_center(
    const std::array<RelativeCellState, kDirectionSideCount>& side_cells,
    int current_index,
    RelativeCellState filled_center_state
);

}  // namespace gomoku
