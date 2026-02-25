#include "mtl.h"
//#include "add.h"
#include <algorithm>
#include <iostream>



int main()
{
	// a global autorelease pool for all temporary objects released with autorelease (otherwise they would leak)
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create default system device
	NS::SharedPtr<MTL::Device> pDevice = TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!pDevice)
	{
		std::cerr << "Failed to create system default device." << std::endl;;
		return 1;
	}

	// load shaders
	NS::SharedPtr<MTL::Library> pLibrary = TransferPtr(pDevice->newDefaultLibrary());
	if (!pLibrary)
	{
		std::cerr << "Failed to load default library." << std::endl;;
		return 1;
	}

	// create compute pipeline for the add_kernel shader
	NS::Error* pError = nullptr;
	NS::SharedPtr<MTL::Function> pAddFn = TransferPtr(pLibrary->newFunction(NS::String::string("add_kernel", NS::UTF8StringEncoding)));
	NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(pDevice->newComputePipelineState(pAddFn.get(), &pError));
	if (!pPSO)
	{
		std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
		return 1;
	}

	// create command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> pCommandQueue = TransferPtr(pDevice->newCommandQueue());
	if (!pCommandQueue)
	{
		std::cerr << "Failed to create command queue" << std::endl;
		return 1;
	}

	// Create input data
	const size_t elementCount = 10;
	std::vector<float> a(elementCount);
	std::vector<float> b(elementCount);

	for (size_t i = 0; i < elementCount; ++i)
	{
		a[i] = float(i);
		b[i] = float(i * 2);
	}

	// Create buffers
	NS::SharedPtr<MTL::Buffer> buf1 = TransferPtr(pDevice->newBuffer(elementCount * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> buf2 = TransferPtr(pDevice->newBuffer(elementCount * sizeof(float), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> result = TransferPtr(pDevice->newBuffer(elementCount * sizeof(float), MTL::ResourceStorageModeShared));

	std::memcpy(buf1->contents(), a.data(), elementCount * sizeof(float));
	std::memcpy(buf2->contents(), b.data(), elementCount * sizeof(float));

	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	pEncoder->setComputePipelineState(pPSO.get());
	pEncoder->setBuffer(buf1.get(), 0, 0);
	pEncoder->setBuffer(buf2.get(), 0, 1);
	pEncoder->setBuffer(result.get(), 0, 2);

	MTL::Size gridSize = MTL::Size(elementCount, 1, 1);

	NS::UInteger threadGroupSize = std::min(pPSO->maxTotalThreadsPerThreadgroup(), elementCount);

	MTL::Size threadsPerThreadgroup = MTL::Size(threadGroupSize, 1, 1);

	pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();

	// Read results
	const float* resultData = static_cast<const float*>(result->contents());

	for (size_t i = 0; i < elementCount; ++i)
	{
		std::cout << a[i] << " + " << b[i] << " = " << resultData[i] << std::endl;
	}

	return 0;
}
