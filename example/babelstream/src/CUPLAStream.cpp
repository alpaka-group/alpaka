
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "CUPLAStream.h"

void check_error(void)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
CUPLAStream<T>::CUPLAStream(const int ARRAY_SIZE, const int device_index)
{

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  //CUPLA and Alpaka don't support the same device selection that CUDA does
  //int count;
  //cudaGetDeviceCount(&count);
  //check_error();
  //if (device_index >= count)
  //  throw std::runtime_error("Invalid device index");
  //cudaSetDevice(device_index);
  //check_error();

  // Print out device information
  //std::cout << "Using CUDA device " << getDeviceName(device_index) << std::endl;
  //std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  /*cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");
*/

  //CUPLA NOTE: You need to cast pointers for cudaMalloc to void** because 
  //  the underlying cuplaMalloc is C++ 
  // Create device buffers
#if defined(MANAGED)
  cudaMallocManaged((void**)&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged((void**)&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged((void**)&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMallocManaged((void**)&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_b = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_c = (T*)malloc(sizeof(T)*ARRAY_SIZE);
  d_sum = (T*)malloc(sizeof(T)*DOT_NUM_BLOCKS);
#else
  cudaMalloc((void**)&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc((void**)&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc((void**)&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  cudaMalloc((void**)&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();
#endif
}


template <class T>
CUPLAStream<T>::~CUPLAStream()
{
  free(sums);

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(d_sum);
#else
  cudaFree(d_a);
  check_error();
  cudaFree(d_b);
  check_error();
  cudaFree(d_c);
  check_error();
  cudaFree(d_sum);
  check_error();
#endif
}


template <typename T>
struct init_kernel
{
 	template< typename T_Acc >
    	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, T * a, T * b, T * c, const T initA, const T initB, const T initC) const
	{
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  a[i] = initA;
	  b[i] = initB;
	  c[i] = initC;
	}
};

template <class T>
void CUPLAStream<T>::init_arrays(T initA, T initB, T initC)
{
  CUPLA_KERNEL_OPTI(init_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_a, d_b, d_c, initA, initB, initC);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
void CUPLAStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  cudaDeviceSynchronize();
  for (int i = 0; i < array_size; i++)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  cudaMemcpy(a.data(), d_a, a.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(b.data(), d_b, b.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
  cudaMemcpy(c.data(), d_c, c.size()*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif
}


template <typename T>
struct copy_kernel
{
    	template< typename T_Acc >
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, const T * a, T * c) const
	{
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  c[i] = a[i];
	}
};

template <class T>
void CUPLAStream<T>::copy()
{
  CUPLA_KERNEL_OPTI(copy_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_a, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
struct mul_kernel
{
	template< typename T_Acc >
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, T * b, const T * c) const
	{
	  const T scalar = startScalar;
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  b[i] = scalar * c[i];
	}
};

template <class T>
void CUPLAStream<T>::mul()
{
  CUPLA_KERNEL_OPTI(mul_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
struct add_kernel
{
	template< typename T_Acc >
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, const T * a, const T * b, T * c) const
	{
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  c[i] = a[i] + b[i];
	}
};

template <class T>
void CUPLAStream<T>::add()
{
  CUPLA_KERNEL_OPTI(add_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
struct triad_kernel
{
	template< typename T_Acc >
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, T * a, const T * b, const T * c) const
	{
	  const T scalar = startScalar;
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  a[i] = b[i] + scalar * c[i];
	}
};

template <class T>
void CUPLAStream<T>::triad()
{
  CUPLA_KERNEL_OPTI(triad_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <typename T>
struct nstream_kernel
{
	template< typename T_Acc >
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, T * a, const T * b, const T * c) const
	{
	  const T scalar = startScalar;
	  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	  a[i] += b[i] + scalar * c[i];
	}
};

template <class T>
void CUPLAStream<T>::nstream()
{
  CUPLA_KERNEL_OPTI(nstream_kernel<T>)(array_size/TBSIZE, TBSIZE)(d_a, d_b, d_c);
  check_error();
  cudaDeviceSynchronize();
  check_error();
}

template <class T>
struct dot_kernel
{
	template< typename T_Acc>
	ALPAKA_FN_ACC
	void operator()(T_Acc const & acc, const T * a, const T * b, T * sum, const int array_size) const
	{
	  //TODO - test if sharedMem bug is affecting performance here
          sharedMem(tb_sum,cupla::Array<T, TBSIZE>);
	
	  int i = blockDim.x * blockIdx.x + threadIdx.x;
	  const size_t local_i = threadIdx.x;

	  tb_sum[local_i] = 0.0;
	  for (; i < array_size; i += blockDim.x*gridDim.x)
	    tb_sum[local_i] += a[i] * b[i];

	  for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
	  {
	    __syncthreads();
	    if (local_i < offset)
	    {
	      tb_sum[local_i] += tb_sum[local_i+offset];
	    }
	  }

	  if (local_i == 0)
	    sum[blockIdx.x] = tb_sum[local_i];
	}
};

template <class T>
T CUPLAStream<T>::dot()
{
  CUPLA_KERNEL_OPTI(dot_kernel<T>)(DOT_NUM_BLOCKS, TBSIZE)(d_a, d_b, d_sum, array_size);
  check_error();

#if defined(MANAGED) || defined(PAGEFAULT)
  cudaDeviceSynchronize();
  check_error();
#else
  cudaMemcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
#endif

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
  {
#if defined(MANAGED) || defined(PAGEFAULT)
    sum += d_sum[i];
#else
    sum += sums[i];
#endif
  }

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  cudaGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  //cudaDeviceProp props;
  //cudaGetDeviceProperties(&props, device);
  check_error();
  //return std::string(props.name);
  return std::string("Not supported");
}


std::string getDeviceDriver(const int device)
{
  cudaSetDevice(device);
  check_error();
  int driver;
  //TODO - update with Alpaka version
  //cudaDriverGetVersion(&driver);
  check_error();
  return std::string("Not supported");
  //std::to_string(driver);
}

template class CUPLAStream<float>;
template class CUPLAStream<double>;
