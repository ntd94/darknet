#include "dat_custom.h"
#include "yolo_v2_class.hpp"
//#include "cuda.h"
//#include "cuda_runtime_api.h"
//#include "cuda_runtime.h"
#include "dark_cuda.h"
#include "darknet.h"
#include "network.h"

__global__
void cuda_blob_resize_RGBA(unsigned char* input, int in_h, int in_w, float *output, int out_h, int out_w,
					  float scale_x, float scale_y, int N)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	// threadIx should vary from 0 to out_w*out_h;
	if (threadId >= N) return;

	int out_x = threadId % out_w;
	int out_y = threadId / out_w;

	int in_x = out_x / scale_x;
	int in_y = out_y / scale_y;

	// RGBA
	int in_add = (in_y * in_w + in_x)*4;

	// input is uchar (size = 1)
	// output is float (size = 4)
	output[threadId                ] = input[in_add    ] / 255.0;
	output[threadId +   out_h*out_w] = input[in_add + 1] / 255.0;
	output[threadId + 2*out_h*out_w] = input[in_add + 2] / 255.0;
}

__global__
void cuda_blob_RGBA(unsigned char* input, float *output, int out_h, int out_w, int N)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	// threadIx should vary from 0 to out_w*out_h;
	if (threadId >= N) return;

	// input is uchar (size = 1)
	// output is float (size = 4)


	// RGBA means 4*threadId
	output[threadId                ] = input[4*threadId    ] / 255.0;
	output[threadId +   out_h*out_w] = input[4*threadId + 1] / 255.0;
	output[threadId + 2*out_h*out_w] = input[4*threadId + 2] / 255.0;
}

/**
  * force x in range [0..255]
  */
__device__
int clamp(int x)
{
//	return x < 0 ? 0 : x > 255 ? 255 : x;
	return x & 255;
}

/**
  * brief: convert from I420 to RGB, blob, and change to [0..1]
  * x = threadId % out_w;
  * y = threadId / out_w;
  * Y = I420.data[x + y * width];
  * U = I420.data[(y/2) * (width/2) + (x/2) + size];
  * V = I420.data[(y/2) * (width/2) + (x/2) + size + size/4];
  */
__global__
void cuda_blob_I420(unsigned char* input, float *output, int out_h, int out_w, int N)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	// threadIx should vary from 0 to out_w*out_h;
	if (threadId >= N) return;

	int x = threadId % out_w;
	int y = threadId / out_w;

	int size = out_h*out_w;

//	unsigned char Y = input[threadId];
//	unsigned char U = input[x/4 + threadId/4 + size];
//	unsigned char V = input[x/4 + threadId/4 + size + size/4];

	unsigned char Y = input[x + y*out_w];
	unsigned char U = input[y/2 * out_w/2 + x/2 + size];
	unsigned char V = input[y/2 * out_w/2 + x/2 + size + size/4];

	int C = Y - 16;
	int D = U - 128;
	int E = V - 128;

	uchar R = (298 * C           + 409 * E) >> 8;
	uchar G = (298 * C - 100 * D - 208 * E) >> 8;
	uchar B = (298 * C + 516 * D          ) >> 8;

	R = clamp(R);
	G = clamp(G);
	B = clamp(B);

	// input is uchar (size = 1)
	// output is float (size = 4)
	// RGBA means 4*threadId
	output[threadId                ] = R / 255.0;
	output[threadId +   out_h*out_w] = G / 255.0;
	output[threadId + 2*out_h*out_w] = B / 255.0;
}

LIB_API
void preprocess_RGBA(unsigned char* input, int in_h, int in_w, float*output, int out_h, int out_w)
{
	//    CHECK_CUDA(cudaMalloc( (void**)&output, 3*out_h*out_w*sizeof(float) ));
	// remember to cudaMalloc
//	printf("\nPREPROCESSSSSSSSSSSSSSSSSSSSSSSSSSSSS\n");
	int N = out_h * out_w;
	int blockSize = 1024;
	int numBlock = (N + blockSize - 1) / blockSize;
	if (in_w == out_w && in_h == out_h)
	{
		cuda_blob_RGBA<<<numBlock, blockSize>>>(input, output, out_h, out_w, N);
	}
	else
	{
		float scale_x = float(out_w) / in_w;
		float scale_y = float(out_h) / in_h;
		cuda_blob_resize_RGBA<<<numBlock, blockSize>>>(input, in_h, in_w, output, out_h, out_w,
												  scale_x, scale_y, N);
	}
	cudaDeviceSynchronize();
}

LIB_API
void preprocess_I420(unsigned char* input, int in_h, int in_w, float*output, int out_h, int out_w)
{
	//    CHECK_CUDA(cudaMalloc( (void**)&output, 3*out_h*out_w*sizeof(float) ));
	// remember to cudaMalloc
//	printf("\nPREPROCESSSSSSSSSSSSSSSSSSSSSSSSSSSSS\n");
	int N = out_h * out_w;
	int blockSize = 1024;
	int numBlock = (N + blockSize - 1) / blockSize;
	if (in_w == out_w && in_h * 2 / 3 == out_h)
	{
		printf("\n\n================= HERE ===============\n\n");
		cuda_blob_I420<<<numBlock, blockSize>>>(input, output, out_h, out_w, N);
	}
	else
	{
		float scale_x = float(out_w) / in_w;
		float scale_y = float(out_h) / in_h;
		cuda_blob_resize_RGBA<<<numBlock, blockSize>>>(input, in_h, in_w, output, out_h, out_w,
												  scale_x, scale_y, N);
	}
	cudaDeviceSynchronize();
}


__global__
void getROI_blobed_gpu_kernel(unsigned char *input, int in_h, int in_w,
									 float* output, int out_h, int out_w,
									   int roi_top, int roi_left,
									 int roi_width, int roi_height,
									 float scale_x, float scale_y,
													int N)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	// threadIx should vary from 0 to out_w*out_h;
	if (threadId >= N) return;

	int out_x = threadId % out_w;
	int out_y = threadId / out_w;

	int in_x = roi_left + out_x / scale_x;
	int in_y = roi_top  + out_y / scale_y;
	// change to *4 if RGBA
	int in_add = (in_y * in_w + in_x)*4;

	// input is uchar (size = 1)
	// output is float (size = 4)
	output[threadId                ] = input[in_add    ] / 255.0;
	output[threadId +   out_h*out_w] = input[in_add + 1] / 255.0;
	output[threadId + 2*out_h*out_w] = input[in_add + 2] / 255.0;
}


// input: in, top, left, out.h, out.w
// output: out.data
LIB_API
void getROI_blobed_gpu_RGBA(image_t in, image_t blob_resized, int roi_top,
														 int roi_left,
														 int roi_width,
														 int roi_height)
{
	float scale_x = float(blob_resized.w) / roi_width;
	float scale_y = float(blob_resized.h) / roi_height;
	// check ROI inside image
	// cudamalloc output
	// actual blob
	int N = blob_resized.h * blob_resized.w;
	int blockSize = 1024;
	int numBlock = (N + blockSize - 1) / blockSize;
	getROI_blobed_gpu_kernel<<<numBlock, blockSize>>>((unsigned char*)in.data, in.h,  in.w,
																	 blob_resized.data, blob_resized.h, blob_resized.w,
																	  roi_top, roi_left,
																	roi_width, roi_height,
													  scale_x, scale_y,
																			   N);

	cudaDeviceSynchronize();
}

//////////////////
/// \brief network_predict_gpu_custom
/// \param net
/// \param device_input
/// \return

LIB_API
float *network_predict_gpu_custom(network net, float *device_input)
{
	if (net.gpu_index != cuda_get_device())
		cuda_set_device(net.gpu_index);
	int size = get_network_input_size(net) * net.batch;
	network_state state;
	state.index = 0;
	state.net = net;
	//state.input = cuda_make_array(input, size);   // memory will be allocated in the parse_network_cfg_custom()
	state.input = net.input_state_gpu;
//    memcpy(net.input_pinned_cpu, input, size * sizeof(float));
//    cuda_push_array(state.input, net.input_pinned_cpu, size);
	state.input = device_input;
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
//    printf("network_predict_gpu_custom: 1\n");
	forward_network_gpu(net, state);

//    printf("network_predict_gpu_custom: 2\n");
	float *out = get_network_output_gpu(net);
//    printf("network_predict_gpu_custom: 3\n");
	//cuda_free(state.input);   // will be freed in the free_network()
	return out;
}
