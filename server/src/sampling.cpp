#include "flashmlx/sampling.h"

namespace flashmlx {

mx::array sample_token(const mx::array& logits, float temperature) {
    if (temperature <= 0.0f) {
        return mx::argmax(logits, -1);
    }
    auto scaled = mx::multiply(logits, mx::array(1.0f / temperature));
    return mx::random::categorical(scaled);
}

} // namespace flashmlx
