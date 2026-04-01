#pragma once
#include <mlx/mlx.h>
namespace mx = mlx::core;

namespace flashmlx {
mx::array sample_token(const mx::array& logits, float temperature);
} // namespace flashmlx
