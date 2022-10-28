/*  
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES.
 * 
 * SPDX-License-Identifier: BSD-3-Clause
 */  

// Sphinx: #1
#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
if( err != CUTENSORNET_STATUS_SUCCESS )                           \
{ printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{  const auto err = x;                                            \
   if( err != cudaSuccess )                                       \
   { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); return err; } \
};

struct GPUTimer
{
   GPUTimer(cudaStream_t stream): stream_(stream)
   {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
   }

   ~GPUTimer()
   {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
   }

   void start()
   {
      cudaEventRecord(start_, stream_);
   }

   float seconds()
   {
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float time;
      cudaEventElapsedTime(&time, start_, stop_);
      return time * 1e-3;
   }

   private:
   cudaEvent_t start_, stop_;
   cudaStream_t stream_;
};


int main()
{
   const size_t cuTensornetVersion = cutensornetGetVersion();
   printf("cuTensorNet-vers:%ld\n",cuTensornetVersion);

   cudaDeviceProp prop;
   int deviceId{-1};
   HANDLE_CUDA_ERROR( cudaGetDevice(&deviceId) );
   HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

   printf("===== device info ======\n");
   printf("GPU-name:%s\n", prop.name);
   printf("GPU-clock:%d\n", prop.clockRate);
   printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
   printf("GPU-nSM:%d\n", prop.multiProcessorCount);
   printf("GPU-major:%d\n", prop.major);
   printf("GPU-minor:%d\n", prop.minor);
   printf("========================\n");

   typedef float floatType;
   cudaDataType_t typeData = CUDA_R_32F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

   printf("Include headers and define data types\n");

   // Sphinx: #2
   /**********************
   * Computing: D_{m,x,n,y} = A_{m,h,k,n} B_{u,k,h} C_{x,u,y}
   **********************/

   constexpr int32_t numInputs = 3;

   // Create vector of modes
   std::vector<int32_t> modesA{'m','h'};
   std::vector<int32_t> modesB{'h','u'};
   std::vector<int32_t> modesC{'u','y'};
   std::vector<int32_t> modesD{'m','y'};

   // Extents
   std::unordered_map<int32_t, int64_t> extent;
   extent['m'] = 2;
   extent['n'] = 2;
   extent['u'] = 2;
   extent['h'] = 2;
   extent['k'] = 2;
   extent['x'] = 2;
   extent['y'] = 2;

   // Create a vector of extents for each tensor
   std::vector<int64_t> extentA;
   for (auto mode : modesA)
      extentA.push_back(extent[mode]);
   std::vector<int64_t> extentB;
   for (auto mode : modesB)
      extentB.push_back(extent[mode]);
   std::vector<int64_t> extentC;
   for (auto mode : modesC)
      extentC.push_back(extent[mode]);
   std::vector<int64_t> extentD;
   for (auto mode : modesD)
      extentD.push_back(extent[mode]);

   printf("Define network, modes, and extents\n");

   // Sphinx: #3
   /**********************
   * Allocating data
   **********************/

   size_t elementsA = 1;
   for (auto mode : modesA)
      elementsA *= extent[mode];
   size_t elementsB = 1;
   for (auto mode : modesB)
      elementsB *= extent[mode];
   size_t elementsC = 1;
   for (auto mode : modesC)
      elementsC *= extent[mode];
   size_t elementsD = 1;
   for (auto mode : modesD)
      elementsD *= extent[mode];

   size_t sizeA = sizeof(floatType) * elementsA;
   size_t sizeB = sizeof(floatType) * elementsB;
   size_t sizeC = sizeof(floatType) * elementsC;
   size_t sizeD = sizeof(floatType) * elementsD;
   printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC + sizeD)/1024./1024./1024);

   void* rawDataIn_d[numInputs];
   void* D_d;
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[0], sizeA) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[1], sizeB) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[2], sizeC) );
   HANDLE_CUDA_ERROR( cudaMalloc((void**) &D_d, sizeD));

   floatType *A = (floatType*) malloc(sizeof(floatType) * elementsA);
   floatType *B = (floatType*) malloc(sizeof(floatType) * elementsB);
   floatType *C = (floatType*) malloc(sizeof(floatType) * elementsC);
   floatType *D = (floatType*) malloc(sizeof(floatType) * elementsD);

   if (A == NULL || B == NULL || C == NULL || D == NULL)
   {
      printf("Error: Host allocation of A or C.\n");
      return -1;
   }

   /*******************
   * Initialize data
   *******************/

   for (uint64_t i = 0; i < elementsA-1; i++)
      A[i] = ((floatType) 1)/sqrt(2);
   for (uint64_t i = 0; i < elementsB-1; i++)
      B[i] = ((floatType) 1)/sqrt(2);
   for (uint64_t i = 0; i < elementsC-1; i++)
      C[i] = ((floatType) 1)/sqrt(2);
   memset(D, 0, sizeof(floatType) * elementsD);

   A[elementsA] = ((floatType) -1)/sqrt(2);
   B[elementsB] = ((floatType) -1)/sqrt(2);
   C[elementsC] = ((floatType) -1)/sqrt(2);

   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[0], A, sizeA, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[1], B, sizeB, cudaMemcpyHostToDevice) );
   HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[2], C, sizeC, cudaMemcpyHostToDevice) );

   printf("Allocate memory for data, and initialize data.\n");

   // Sphinx: #4
   /*************************
   * cuTensorNet
   *************************/

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   cutensornetHandle_t handle;
   HANDLE_ERROR( cutensornetCreate(&handle) );

   const int32_t nmodeA = modesA.size();
   const int32_t nmodeB = modesB.size();
   const int32_t nmodeC = modesC.size();
   const int32_t nmodeD = modesD.size();

   /*******************************
   * Create Network Descriptor
   *******************************/

   const int32_t* modesIn[] = {modesA.data(), modesB.data(), modesC.data()};
   int32_t const numModesIn[] = {nmodeA, nmodeB, nmodeC};
   const int64_t* extentsIn[] = {extentA.data(), extentB.data(), extentC.data()};
   const int64_t* stridesIn[] = {NULL, NULL, NULL}; // strides are optional; if no stride is provided, then cuTensorNet assumes a generalized column-major data layout

   // Notice that pointers are allocated via cudaMalloc are aligned to 256 byte
   // boundaries by default; however here we're checking the pointer alignment explicitly
   // to demonstrate how one would check the alginment for arbitrary pointers.

   auto getMaximalPointerAlignment = [](const void* ptr) {
      const uint64_t ptrAddr  = reinterpret_cast<uint64_t>(ptr);
      uint32_t alignment = 1;
      while(ptrAddr % alignment == 0 &&
            alignment < 256) // at the latest we terminate once the alignment reached 256 bytes (we could be going, but any alignment larger or equal to 256 is equally fine)
      {
         alignment *= 2;
      }
      return alignment;
   };
   const uint32_t alignmentsIn[] = {getMaximalPointerAlignment(rawDataIn_d[0]),
                                    getMaximalPointerAlignment(rawDataIn_d[1]),
                                    getMaximalPointerAlignment(rawDataIn_d[2])};
   const uint32_t alignmentOut = getMaximalPointerAlignment(D_d);

   // setup tensor network
   cutensornetNetworkDescriptor_t descNet;
   HANDLE_ERROR( cutensornetCreateNetworkDescriptor(handle,
                                                numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentsIn,
                                                nmodeD, extentD.data(), /*stridesOut = */NULL, modesD.data(), alignmentOut,
                                                typeData, typeCompute,
                                                &descNet) );

   printf("Initialize the cuTensorNet library and create a network descriptor.\n");

   // Sphinx: #5
   /*******************************
   * Choose workspace limit based on available resources.
   *******************************/

   size_t freeMem, totalMem;
   HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem) );
   uint64_t workspaceLimit = totalMem * 0.9;

   /*******************************
   * Find "optimal" contraction order and slicing
   *******************************/

   cutensornetContractionOptimizerConfig_t optimizerConfig;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig) );

   // Set the value of the partitioner imbalance factor, if desired
   int32_t imbalance_factor = 30;
   HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(
                                                               handle,
                                                               optimizerConfig,
                                                               CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR,
                                                               &imbalance_factor,
                                                               sizeof(imbalance_factor)) );


   cutensornetContractionOptimizerInfo_t optimizerInfo;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo) );

   HANDLE_ERROR( cutensornetContractionOptimize(handle,
                                             descNet,
                                             optimizerConfig,
                                             workspaceLimit,
                                             optimizerInfo) );

   int64_t numSlices = 0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
               handle,
               optimizerInfo,
               CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
               &numSlices,
               sizeof(numSlices)) );

   assert(numSlices > 0);

   printf("Find an optimized contraction path with cuTensorNet optimizer.\n");

   // Sphinx: #6
   /*******************************
   * Create workspace descriptor, allocate workspace, and set it.
   *******************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

   uint64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR( cutensornetWorkspaceComputeSizes(handle,
                                          descNet,
                                          optimizerInfo,
                                          workDesc) );

   HANDLE_ERROR( cutensornetWorkspaceGetSize(handle,
                                         workDesc,
                                         CUTENSORNET_WORKSIZE_PREF_MIN,
                                         CUTENSORNET_MEMSPACE_DEVICE,
                                         &requiredWorkspaceSize) );

   void *work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, requiredWorkspaceSize) );

   HANDLE_ERROR( cutensornetWorkspaceSet(handle,
                                         workDesc,
                                         CUTENSORNET_MEMSPACE_DEVICE,
                                         work,
                                         requiredWorkspaceSize) );

   printf("Allocate workspace.\n");

   // Sphinx: #7
   /*******************************
   * Initialize all pair-wise contraction plans (for cuTENSOR).
   *******************************/

   cutensornetContractionPlan_t plan;

   HANDLE_ERROR( cutensornetCreateContractionPlan(handle,
                                                  descNet,
                                                  optimizerInfo,
                                                  workDesc,
                                                  &plan) );


   /*******************************
   * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
   *           for each pairwise contraction.
   *******************************/
   cutensornetContractionAutotunePreference_t autotunePref;
   HANDLE_ERROR( cutensornetCreateContractionAutotunePreference(handle,
                           &autotunePref) );

   const int numAutotuningIterations = 5; // may be 0
   HANDLE_ERROR( cutensornetContractionAutotunePreferenceSetAttribute(
                           handle,
                           autotunePref,
                           CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                           &numAutotuningIterations,
                           sizeof(numAutotuningIterations)) );

   // modify the plan again to find the best pair-wise contractions
   HANDLE_ERROR( cutensornetContractionAutotune(handle,
                           plan,
                           rawDataIn_d,
                           D_d,
                           workDesc,
                           autotunePref,
                           stream) );

   HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );

   printf("Create a contraction plan for cuTensorNet and optionally auto-tune it.\n");

   // Sphinx: #8
   /**********************
   * Run
   **********************/
   cutensornetSliceGroup_t sliceGroup{};

   // Create a cutensornetSliceGroup_t object from a range of slice IDs.
   HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup) );

   GPUTimer timer{stream};
   double minTimeCUTENSOR = 1e100;
   const int numRuns = 3; // to get stable perf results
   for (int i=0; i < numRuns; ++i)
   {
      cudaMemcpy(D_d, D, sizeD, cudaMemcpyHostToDevice); // restore output
      cudaDeviceSynchronize();

      /*
      * Contract over all slices.
      *
      * A user may choose to parallelize over the slices across multiple devices.
      */
      timer.start();

      int32_t accumulateOutput = 0;
      HANDLE_ERROR( cutensornetContractSlices(handle,
                                 plan,
                                 rawDataIn_d,
                                 D_d,
                                 accumulateOutput,
                                 workDesc,
                                 sliceGroup,    // Alternatively, NULL can also be used to contract over all the slices instead of specifying a sliceGroup object.
                                 stream) );

      // Synchronize and measure timing
      auto time = timer.seconds();
      minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
   }

   printf("Contract the network, each slice uses the same contraction plan.\n");


   /*************************/

   double flops{0.};
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
               handle,
               optimizerInfo,
               CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
               &flops,
               sizeof(flops)) );

   printf("numSlices: %ld\n", numSlices);
   printf("%.2f ms / slice\n", minTimeCUTENSOR * 1000.f / numSlices);
   printf("%.2f GFLOPS/s\n", flops/1e9/minTimeCUTENSOR );

   HANDLE_ERROR( cutensornetDestroySliceGroup(sliceGroup) );
   HANDLE_ERROR( cutensornetDestroy(handle) );
   HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
   HANDLE_ERROR( cutensornetDestroyContractionPlan(plan) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerConfig(optimizerConfig) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );

   if (A) free(A);
   if (B) free(B);
   if (C) free(C);
   if (D) free(D);
   if (rawDataIn_d[0]) cudaFree(rawDataIn_d[0]);
   if (rawDataIn_d[1]) cudaFree(rawDataIn_d[1]);
   if (rawDataIn_d[2]) cudaFree(rawDataIn_d[2]);
   if (D_d) cudaFree(D_d);
   if (work) cudaFree(work);

   printf("Free resource and exit.\n");

   return 0;
}
