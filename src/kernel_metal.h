#pragma once

#include <metal_stdlib>

#define MTL_CONSTANT constant
#define MTL_DEVICE device
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	kernel void NAME(const ITYPE index [[thread_position_in_grid]], constant ARGS& args [[bufer(0)]])

#define DECL_KERNEL_ARGS_BEGIN(NAME) struct NAME {
// TODO make variadic
#define DECL_KERNEL_ARGS_FIELD(TYPE, NAME) TYPE NAME;
#define DECL_KERNEL_ARGS_END(NAME) };