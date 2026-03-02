#include "mtl.h"
#include "invert.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>



static void GenerateData(uint8_t* data, const uint32_t w, const uint32_t h)
{
	static std::random_device sRngSeed;
	static std::mt19937 sRng(sRngSeed());
	static std::uniform_int_distribution<uint32_t> sDist(0, 255);

	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			data[x + y * w] = static_cast<uint8_t>(sDist(sRng));
		}
	}
}

static void InvertCPU(
	const uint8_t* src,
	const uint32_t srcStride,
	uint8_t* dst,
	const uint32_t dstStride,
	const uint32_t w,
	const uint32_t h)
{
	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			InvertArgs args;
			args.src = src;
			args.dst = dst;
			args.srcStride = srcStride;
			args.dstStride = dstStride;
			invert_kernel({ x, y }, args);
		}
	}
}

static void InvertGPU(
	MTL::CommandQueue* pCommandQueue,
	MTL::ComputePipelineState* pPSO,
	MTL::Buffer* src,
	const uint32_t srcStride,
	MTL::Buffer* dst,
	const uint32_t dstStride,
	const uint32_t w,
	const uint32_t h)
{
	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	pEncoder->setComputePipelineState(pPSO);
	pEncoder->setBuffer(src, 0, 0);
	pEncoder->setBuffer(dst, 0, 1);

	MTL::Size gridSize = MTL::Size(w, h, 1);
	MTL::Size threadsPerThreadgroup = MTL::Size(pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);

	pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();
}


bool test3()
{
	// a global autorelease pool for all temporary objects released with autorelease (otherwise they would leak)
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create default system device
	NS::SharedPtr<MTL::Device> pDevice = TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!pDevice)
	{
		std::cerr << "Failed to create system default device." << std::endl;;
		return false;
	}

	// load shaders
	NS::SharedPtr<MTL::Library> pLibrary = TransferPtr(pDevice->newDefaultLibrary());
	if (!pLibrary)
	{
		std::cerr << "Failed to load default library." << std::endl;;
		return false;
	}

	// create compute pipeline for the add_kernel shader
	NS::Error* pError = nullptr;
	NS::SharedPtr<MTL::Function> pInvertFn = TransferPtr(pLibrary->newFunction(NS::String::string("invert_kernel", NS::UTF8StringEncoding)));
	NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(pDevice->newComputePipelineState(pInvertFn.get(), &pError));
	if (!pPSO)
	{
		std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
		return false;
	}

	// create command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> pCommandQueue = TransferPtr(pDevice->newCommandQueue());
	if (!pCommandQueue)
	{
		std::cerr << "Failed to create command queue" << std::endl;
		return false;
	}

	// Create buffers
	const size_t W = 8192;
	const size_t H = 8192;

	NS::SharedPtr<MTL::Buffer> bufSrc = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resCPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resGPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));

	// Create input data
	GenerateData(static_cast<unsigned char*>(bufSrc->contents()), W, H);

	// compute on CPU (reference)
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	InvertCPU(static_cast<const uint8_t*>(bufSrc->contents()), W * sizeof(uint8_t), static_cast<uint8_t*>(resCPU->contents()), W * sizeof(uint8_t), W, H);
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	//InvertGPU(pCommandQueue.get(), pPSO.get(), bufSrc.get(), W * sizeof(uint8_t), resGPU.get(), W * sizeof(uint8_t), W, H);
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	// validate results
	const uint8_t* cpu = static_cast<const uint8_t*>(resCPU->contents());
	const uint8_t* gpu = static_cast<const uint8_t*>(resGPU->contents());
	for (uint32_t y = 0; y < H; ++y)
	{
		for (uint32_t x = 0; x < W; ++x)
		{
			const uint32_t i = x + y * W;
			if (cpu[i] != gpu[i])
			{
				std::cerr << "Invalid results, cpu and gpu differ at element [" << x << ", " << y << "]: " << (uint32_t)cpu[i] << " vs " << (uint32_t)gpu[i] << std::endl;
				return false;
			}
		}
	}

	std::cout << "CPU and GPU produces equivalent outputs" << std::endl;
	std::cout << "CPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_t1 - cpu_t0).count() << " ms" << std::endl;
	std::cout << "GPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(gpu_t1 - gpu_t0).count() << " ms" << std::endl;

	return true;
}
