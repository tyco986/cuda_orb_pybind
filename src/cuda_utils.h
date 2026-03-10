#pragma once
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>

#define H_PI		1.5707963267948966f

#define CHECK(err)		__check(err, __FILE__, __LINE__)
#define CheckMsg(msg)	__checkMsg(msg, __FILE__, __LINE__)


/* Check cuda runtime api, and print error. */
inline void __check(cudaError err, const char* file, const int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CHECK() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


/* Check cuda runtime api, and print error with Message. */
inline void __checkMsg(const char* msg, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CheckMsg() CUDA error: %s in file <%s>, line %i : %s.\n", msg, file, line, cudaGetErrorString(err));
		exit(-1);
	}
}


/* Initialize device properties */
inline bool initDevice(int dev)
{
	int device_count = 0;
	CHECK(cudaGetDeviceCount(&device_count));
	if (device_count == 0)
	{
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		return false;
	}
	dev = std::max<int>(0, std::min<int>(dev, device_count - 1));
	cudaDeviceProp device_prop;
	CHECK(cudaGetDeviceProperties(&device_prop, dev));
	if (device_prop.major < 1)
	{
		fprintf(stderr, "error: device does not support CUDA.\n");
		return false;
	}
	CHECK(cudaSetDevice(dev));
	return true;
}


__device__ __inline__ int dealBorder(int i, int sz)
{
	if (i < 0)
		return -i;
	if (i >= sz)
		return sz + sz - 2 - i;
	return i;
}


/* Align up */
inline int iAlignUp(const int a, const int b)
{
	return (a % b != 0) ? (a - a % b + b) : a;
}
