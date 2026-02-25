#if defined(__METAL_VERSION__)
#define MTL_CONSTANT constant
#define MTL_DEVICE device
#else
#include <cstdint>
#define MTL_CONSTANT const
#define MTL_DEVICE
typedef uint32_t uint;
#endif

[[kernel]] void add_kernel(
    uint index [[thread_position_in_grid]],
    float MTL_CONSTANT * buf1 [[buffer(0)]],
    float MTL_CONSTANT * buf2 [[buffer(1)]],
    float MTL_DEVICE * result [[buffer(2)]])
{
    result[index] = buf1[index] + buf2[index];
}
