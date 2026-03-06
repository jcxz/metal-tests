#include "gpu.h"
#include "reflection.h"
#include "mtl.h"

#include <unordered_map>
#include <string>
#include <cassert>
#include <iostream>



namespace
{

class Gpu
{
private:
	struct KernelInfo
	{
		std::string name;
		NS::SharedPtr<MTL::Function> pFunc;
		NS::SharedPtr<MTL::ComputePipelineState> pPSO;
		KernelInfo() = default;
		explicit KernelInfo(const std::string& name)
			: name(name)
			, pFunc(nullptr)
			, pPSO(nullptr)
		{ }
	};

public:
	uint32_t RegisterKernel(const std::string& name)
	{
		const uint32_t id = static_cast<uint32_t>(mKernels.size());
		mKernels.emplace_back(name);
		return id;
	}

	bool ExecuteKernel(
		const uint32_t id,
		const simd::uint3 dims,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo)
	{
		// an autorelease pool to make sure that any temporary (autorelease) allocations (e.g. the command buffer)
		// get destroyed at the end of this function
		NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

		// Initialize GPU if needed
		if (!Init())
		{
			std::cerr << "Failed to initialize GPU" << std::endl;
			return false;
		}

		// Get PSO
		const KernelInfo* pKernel = RequestKernel(id);
		if (pKernel == nullptr)
		{
			std::cerr << "Failed to initialize GPU data for kernel " << pKernel->name << std::endl;
			return false;
		}

		// Enqueue GPU commands and execute them
		MTL::CommandBuffer* pCommandBuffer = mpCommandQueue->commandBuffer();
		MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

		// bind the compute pipeline
		pEncoder->setComputePipelineState(pKernel->pPSO.get());

		// prepare and bind kernel arguments
		if (!EncodeKernelArguments(pEncoder, pKernel, pArgs, pArgsInfo))
		{
			pEncoder->endEncoding();
			std::cerr << "Failed to update arguments for kernel " << pKernel->name << std::endl;
			return false;
		}

		// equeue kernel dispatch
		const MTL::Size gridSize(dims.x, dims.y, dims.z);
		const MTL::Size groupSize(pKernel->pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);
		pEncoder->dispatchThreads(gridSize, groupSize);

		// finish encoding and execute computation on the GPU and wait for it
		pEncoder->endEncoding();
		pCommandBuffer->commit();
		pCommandBuffer->waitUntilCompleted();

		// how long the GPU took to execute the command buffer (i.e. the computation kernel)
		const double kernelStartTime = pCommandBuffer->kernelStartTime();
		const double kernelEndTime = pCommandBuffer->kernelEndTime();
		const double kerneTimelMs = (kernelEndTime - kernelStartTime) * 1000.0;
		const double gpuStartTime = pCommandBuffer->GPUStartTime();
		const double gpuEndTime = pCommandBuffer->GPUEndTime();
		const double gpuTimeMs = (gpuEndTime - gpuStartTime) * 1000.0;
		std::cout << "GPU scheduling time: " << kerneTimelMs << " ms" << std::endl;
		std::cout << "GPU execution time: " << gpuTimeMs << " ms" << std::endl;
	}

	static Gpu* GetInstance()
	{
		static Gpu gpu;
		return &gpu;
	}

private:
	bool Init()
	{
		// the command queue is initialized as last in the initialization process,
		// so if we have a valid queue, we know the initialization has already succeeded in the past
		if (mpCommandQueue)
			return true;

		// create default system device
		if (!(mpDevice = TransferPtr(MTL::CreateSystemDefaultDevice())))
		{
			std::cerr << "Failed to create system default device." << std::endl;;
			return false;
		}

		// load shaders
		if (!(mpLibrary = TransferPtr(mpDevice->newDefaultLibrary())))
		{
			std::cerr << "Failed to load default shader library." << std::endl;;
			return false;
		}

		// create command queue for submitting commands to the GPU
		if (!(mpCommandQueue = TransferPtr(mpDevice->newCommandQueue())))
		{
			std::cerr << "Failed to create command queue" << std::endl;
			return false;
		}

		return true;
	}

	const KernelInfo* RequestKernel(const uint32_t id)
	{
		if (id >= mKernels.size())
		{
			std::cerr << "Invalid kernel id " << id << std::endl;
			return nullptr;
		}
	
		auto& kernel = mKernels[id];
		if (!kernel.pPSO)
		{
			// load the kernel function from shader library
			NS::SharedPtr<MTL::Function> pFunc = TransferPtr(mpLibrary->newFunction(NS::String::string(kernel.name.c_str(), NS::UTF8StringEncoding)));
			if (!pFunc)
			{
				std::cerr << "Kernel function " << kernel.name << " not found in the shader library" << std::endl;;
				return nullptr;
			}

			// create a compute pipeline for the function, so that it can be executed
			NS::Error* pError = nullptr;
			NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(mpDevice->newComputePipelineState(pFunc.get(), &pError));
			if (!pPSO)
			{
				std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
				return nullptr;
			}

			// only after all creation succeeded add the entry to the cache
			kernel.pFunc = pFunc;
			kernel.pPSO = pPSO;
		}

		return &kernel;
	}

	bool EncodeKernelArguments(
		MTL::ComputeCommandEncoder* pEncoder,
		const KernelInfo* pKernel,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo)
	{
		// we assume that all kernels take only one argument, which is an argument buffer containing the arguments structure
		NS::SharedPtr<MTL::ArgumentEncoder> pArgEncoder = TransferPtr(pKernel->pFunc->newArgumentEncoder(0));

		// allocate a new argument buffer to store the arguments of the kernel (TODO: replace this with a ring buffer allocator)
		NS::SharedPtr<MTL::Buffer> pArgBuffer = TransferPtr(mpDevice->newBuffer(pArgEncoder->encodedLength(), MTL::ResourceStorageModeShared));

		// tell argument encoder about the buffer where it will serialize kernel arguments
		pArgEncoder->setArgumentBuffer(pArgBuffer.get(), 0);

		// iterate over properties of the arguments structure and fill them up
		for (const refl::TypeMetaInfo* info = pArgsInfo; info; info = info->next)
		{
			switch (info->type)
			{
				// TODO somehow decode a buffer from the args
				case refl::TypeTag::Pointer:
					//pArgEncoder->setBuffer(buf, 0, info->location);
					//pEncoder->useResource(buf, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
					break;

				case refl::TypeTag::ConstPointer:
					//pArgEncoder->setBuffer(buf, 0, info->location);
					//pEncoder->useResource(buf, MTL::ResourceUsageRead);
					break;

				default:
					std::memcpy(pArgEncoder->constantData(info->location), reinterpret_cast<const uint8_t*>(pArgs) + info->offset, info->size);
					break;
			}
		}

		// bind the argument buffer
		pEncoder->setBuffer(pArgBuffer.get(), 0, 0);

		return true;
	}

private:
	Gpu() = default;
	Gpu(Gpu&& ) = delete;
	Gpu(const Gpu& ) = delete;
	Gpu& operator=(Gpu&& ) = delete;
	Gpu& operator=(const Gpu& ) = delete;

private:
	//! the GPU device that will be used to execute kernels
	NS::SharedPtr<MTL::Device> mpDevice;
	//! shader library with the compile kernels
	// (allows us on demand compute pipeline creation)
	NS::SharedPtr<MTL::Library> mpLibrary;
	//! a command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> mpCommandQueue;
	//! a cache of precompiled pipelines ready to be used for starting a kernel
	std::vector<KernelInfo> mKernels;
	//! argument buffer allocator (here we will store the arguments that we pass to kernels)
	//ArgumentBufferAllocator mAllocator; // ... TODO
};

} // End of private namespace

uint32_t RegisterKernel(const std::string& name)
{
	return Gpu::GetInstance()->RegisterKernel(name);
}

bool ExecuteGPUKernel(
	const uint32_t id,
	const simd::uint3 dims,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo)
{
	return Gpu::GetInstance()->ExecuteKernel(id, dims, pArgs, pArgsInfo);
}
