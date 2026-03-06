#pragma once

#if defined(__METAL_VERSION__)

#include <metal_stdlib>

#define MTL_CONSTANT constant
#define MTL_DEVICE device
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	kernel void NAME(const ITYPE index [[thread_position_in_grid]], ARGS)

#else

#include "reflection.h"
#include "gpu.h"
#include <simd/simd.h>

#define MTL_CONSTANT const
#define MTL_DEVICE
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	struct NAME \
	{ \
		typedef ARGS ArgsType; \
		typedef ITYPE IndexType; \
		static inline const std::string kName = #NAME; \
		static inline const uint32_t kID = RegisterKernel(#NAME); \
		static inline void Run(const IndexType index, const ArgsType& args); \
	}; \
	inline void NAME::Run(const IndexType index, const ArgsType& args)

template <typename K>
void ExecuteKernel(const uint32_t n, const typename A::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint32_t>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kName, uint3(n, 1, 1), static_cast<const void*>(&args), A::ArgsType::kMetaInfo))
	{
		for (uint32_t i = 0; i < n; ++i)
		{
			K::Run(i, args);
		}
	}
}

template <typename K>
void ExecuteKernel(const uint2 dims, const typename A::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint2>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kName, uint3(dims.x, dims.y, 1), static_cast<const void*>(&args), A::ArgsType::kMetaInfo))
	{
		for (uint32_t y = 0; y < dims.y; ++y)
		{
			for (uint32_t x = 0; x < dims.x; ++x)
			{
				K::Run(uint2(x, y), args);
			}
		}
	}
}

template <typename K>
void ExecuteKernel(const uint3 dims, const typename A::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint3>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kName, dims, static_cast<const void*>(&args), A::ArgsType::kMetaInfo))
	{
		for (uint32_t z = 0; z < dims.z; ++z)
		{
			for (uint32_t y = 0; y < dims.y; ++y)
			{
				for (uint32_t x = 0; x < dims.x; ++x)
				{
					K::Run(uint3(x, y, z), args);
				}
			}
		}
	}
}

#endif

#define DECL_KERNEL_1D(NAME, ARGS) DECL_KERNEL(uint, NAME, ARGS)
#define DECL_KERNEL_2D(NAME, ARGS) DECL_KERNEL(uint2, NAME, ARGS)
#define DECL_KERNEL_3D(NAME, ARGS) DECL_KERNEL(uint3, NAME, ARGS)