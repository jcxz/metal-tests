#include "kernel.h"

REFL_DECL_STRUCT_BEGIN(Args)
	REFL_STRUCT_FIELD(uint, dstStride)
	REFL_STRUCT_FIELD(uint, srcStride)
	REFL_STRUCT_FIELD(uint8_t MTL_DEVICE *, dst)
	REFL_STRUCT_FIELD(uint8_t MTL_CONSTANT *, src)
REFL_DECL_STRUCT_END(Args)

DECL_KERNEL_2D(Invert, Args)
{
	uint8_t MTL_DEVICE * dst = args.dst + index.y * args.dstStride + index.x;
	uint8_t MTL_CONSTANT * src = args.src + index.y * args.srcStride + index.x;
	*dst = 255 - *src;
}

