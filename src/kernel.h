#pragma once

#if defined(__METAL_VERSION__)

#include <metal_stdlib>

#define MTL_CONSTANT constant
#define MTL_DEVICE device
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	kernel void NAME(const ITYPE index [[thread_position_in_grid]], constant ARGS& args)

#define DECL_KERNEL_ARGS_BEGIN(NAME) struct NAME {
// TODO make variadic
#define DECL_KERNEL_ARGS_FIELD(TYPE, NAME) TYPE NAME;
#define DECL_KERNEL_ARGS_END(NAME) };

#else

#include "reflection.h"
#include "gpu.h"
#include <simd/simd.h>

using namespace simd; // so that the code between CPU and GPU is the same

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

#define DECL_KERNEL_ARGS_BEGIN(NAME) REFL_DECL_STRUCT_BEGIN(NAME)
#define DECL_KERNEL_ARGS_FIELD(...) REFL_DECL_STRUCT_FIELD(__VA_ARGS__)
#define DECL_KERNEL_ARGS_END(NAME) REFL_DECL_STRUCT_END(NAME)

template <typename K>
void ExecuteKernel(const uint32_t n, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint32_t>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kID, n, static_cast<const void*>(&args), K::ArgsType::kMetaInfo))
	{
		for (uint32_t i = 0; i < n; ++i)
		{
			K::Run(i, args);
		}
	}
}

template <typename K>
void ExecuteKernel(const uint32_t nx, const uint32_t ny, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint2>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kID, nx, ny, static_cast<const void*>(&args), K::ArgsType::kMetaInfo))
	{
		for (uint32_t y = 0; y < ny; ++y)
		{
			for (uint32_t x = 0; x < nx; ++x)
			{
				K::Run({ x, y }, args);
			}
		}
	}
}

template <typename K>
void ExecuteKernel(const uint32_t nx, const uint32_t ny, const uint32_t nz, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint3>, "Mismatching kernel dimensionality");
	if (!ExecuteGPUKernel(K::kID, nx, ny, nz, static_cast<const void*>(&args), K::ArgsType::kMetaInfo))
	{
		for (uint32_t z = 0; z < nz; ++z)
		{
			for (uint32_t y = 0; y < ny; ++y)
			{
				for (uint32_t x = 0; x < nx; ++x)
				{
					K::Run({ x, y, z }, args);
				}
			}
		}
	}
}

#endif

#define DECL_KERNEL_1D(NAME, ARGS) DECL_KERNEL(uint, NAME, ARGS)
#define DECL_KERNEL_2D(NAME, ARGS) DECL_KERNEL(uint2, NAME, ARGS)
#define DECL_KERNEL_3D(NAME, ARGS) DECL_KERNEL(uint3, NAME, ARGS)
