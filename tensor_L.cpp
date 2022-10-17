// complie line: nvcc tensor_L.cpp -L${CUTENSOR_ROOT}/lib/10.1/ -I${CUTENSOR_ROOT}/include -std=c++11 -lcutensor -o contraction 

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cutensor.h>


#include <unordered_map>
#include <vector>

#include <sstream>
#include <string>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                 \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}


int main(int argc, char** argv)
{
    
  int argsize = sizeof(argv);
  int qbit = -1; int gate = -1;
  
  if (argsize>1){
      
      qbit = std::stoi(argv[1]);
      gate = std::stoi(argv[2]);
      
  }
      

      
     typedef float floatTypeqb; 
     cudaDataType_t typeA[qbit]; 
     for(auto i=0;i<qbit;i++)
          typeA[i] = CUDA_R_32F;

      
  
    
    // Host element type definition
//   typedef float floatTypeA;
//   typedef float floatTypeB;
//   typedef float floatTypeC;
  typedef float floatTypeCompute;

  // CUDA types
//   cudaDataType_t typeA = CUDA_R_32F;
//   cudaDataType_t typeB = CUDA_R_32F;
//   cudaDataType_t typeC = CUDA_R_32F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

  floatTypeCompute alpha = (floatTypeCompute)1.1f;
  floatTypeCompute beta  = (floatTypeCompute)0.9f;

  printf("Include headers and define data types\n");

  /* ***************************** */

  // Create vector of modes
  //std::vector<int> modeC{'m','u','n','v'};
  
  std::string strA, strB;
  int nmodeA[qbit];
  std::vector<std::string> modeA[qbit];
  std::unordered_map<std::string, int64_t> extent;
  for(int i=0;i<qbit;i++){
     strA=std::to_string(i)+"A";
     strB=std::to_string(i)+"B";
     modeA[i][0] = strA;//,strB};
     modeA[i][1] = strB;
     nmodeA[i]=modeA[i].size();
     extent[strA] = 4;
     extent[strB] = 4;
      
  }
      
      
  //std::vector<int> modeB{'u','k','v','h'};
  //int nmodeA[qbit]; //= modeA.size();
  //int nmodeB = modeB.size();
  //int nmodeC = modeC.size();

  // Extents


  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA[qbit];
  for(auto i=0;i<qbit;i++){
    for(auto mode : modeA[i])
        extentA[i].push_back(extent[mode]);
    std::vector<int64_t> extentA[i];
  }


  printf("Define modes and extents\n");

  /* ***************************** */

  // Number of elements of each tensor
  
  size_t elementsA[qbit];
  for(auto i=0;i<qbit;i++){
    elementsA[qbit]= 1;
    for(auto mode : modeA[i])
      elementsA[i] *= extent[mode];
  }
//   size_t elementsB = 1;
//   for(auto mode : modeB)
//       elementsB *= extent[mode];
//   size_t elementsC = 1;
//   for(auto mode : modeC)
//       elementsC *= extent[mode];

  // Size in bytes
  size_t sizeA[qbit];
  for(auto i=0;i<qbit;i++)
     sizeA[i] = sizeof(floatTypeqb) * elementsA[i];

  // Allocate on device
  void *A_d[qbit];
  for(auto i=0;i<qbit;i++)
     cudaMalloc((void**)&A_d[i], sizeA[i]);

  // Allocate on host
  floatTypeqb *A[qbit];
  for(auto i=0;i<qbit;i++)
     A[i] = (floatTypeqb*) malloc(sizeof(floatTypeqb) * elementsA[i]);

  // Initialize data on host
  for(auto i=0;i<qbit;i++){
     for(int64_t j = 0; j < elementsA[i]; j++)
         A[i][j] = (((float) rand())/RAND_MAX - 0.5)*100;
  }
  // Copy to device
  for(auto i=0;i<qbit;i++)
      cudaMemcpy(A_d[i], A[i], sizeA[i], cudaMemcpyHostToDevice);

  printf("Allocate, initialize and transfer tensors\n");

  /* ***************************** */

  // Initialize cuTENSOR library
  cutensorHandle_t handle;
  cutensorInit(&handle);

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA[qbit];
  for(auto i=0;i<qbit;i++){
    HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
                &descA[i],
                nmodeA[i],
                extentA[i].data(),
                NULL,/*stride*/
                typeA[i], CUTENSOR_OP_IDENTITY ) );
  }

//   cutensorTensorDescriptor_t descB;
//   HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
//               &descB,
//               nmodeB,
//               extentB.data(),
//               NULL,/*stride*/
//               typeB, CUTENSOR_OP_IDENTITY ) );
// 
//   cutensorTensorDescriptor_t descC;
//   HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
//               &descC,
//               nmodeC,
//               extentC.data(),
//               NULL,/*stride*/
//               typeC, CUTENSOR_OP_IDENTITY ) );

  printf("Initialize cuTENSOR and tensor descriptors\n");

  /* ***************************** */

   //Retrieve the memory alignment for each tensor
   uint32_t alignmentRequirementA[qbit];
   for(auto i=0;i<qbit;i++){
      HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
                  A_d[i],
                  &descA[i],
                  &alignmentRequirementA[i]) );


  printf("Query best alignment requirement for our pointers\n");

  /* ***************************** */

  // Create the Contraction Descriptor
  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR( cutensorInitContractionDescriptor( &handle,
              &desc,
              &descA, modeA.data(), alignmentRequirementA,
              typeCompute) );

  printf("Initialize contraction descriptor\n");

  /* ***************************** */

  // Set the algorithm to use
  cutensorContractionFind_t find;
  HANDLE_ERROR( cutensorInitContractionFind(
              &handle, &find,
              CUTENSOR_ALGO_DEFAULT) );

  printf("Initialize settings to find algorithm\n");

  /* ***************************** */

  // Query workspace
  size_t worksize = 0;
  HANDLE_ERROR( cutensorContractionGetWorkspace(&handle,
              &desc,
              &find,
              CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

  // Allocate workspace
  void *work = nullptr;
  if(worksize > 0)
  {
      if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
      {
          work = nullptr;
          worksize = 0;
      }
  }

  printf("Query recommended workspace size and allocate it\n");

  /* ***************************** */

  // Create Contraction Plan
  cutensorContractionPlan_t plan;
  HANDLE_ERROR( cutensorInitContractionPlan(&handle,
                                            &plan,
                                            &desc,
                                            &find,
                                            worksize) );

  printf("Create plan for contraction\n");

  /* ***************************** */

  cutensorStatus_t err;

  // Execute the tensor contraction
  err = cutensorContraction(&handle,
                            &plan,
                     (void*)&alpha, A_d[i],
                            work, worksize, 0 /* stream */);
  cudaDeviceSynchronize();

  // Check for errors
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
      printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  printf("Execute contraction from plan\n");

  /* ***************************** */

  if ( A ) free( A );
  if ( A_d ) cudaFree( A_d );
  if ( work ) cudaFree( work );

  printf("Successful completion\n");

  return 0; 
    
    
    
    
    
    
    
    
}
