#include "cuda_runtime.h"
#include "device_launch_parameters.h"



/* example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void rectification(unsigned char* image, unsigned char* new_image, unsigned int size_of_image) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < size_of_image) {
		if (image[index] - 127 < 0) {
			new_image[index] = 127;
		}
		else {
			new_image[index] = image[index];
		}
	}
}

//__global__ void pooling(unsigned int* width, unsigned int* size, unsigned char* cuda_original_image, unsigned char* cuda_pooled_image) {
//
//}

int main(int argc, char* argv[])
{
	//char* input_filename = argv[1];
	//char* output_filename = argv[2];
	  //printf("inside cpu \n");

	  //unsigned error;
	  //unsigned char* image;
	  //unsigned width, height;

	  //error = lodepng_decode32_file(&image, &width, &height, input_filename);
	  //if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	  //unsigned int image_size = width * height * 4 * sizeof(unsigned char);

	  //// process image
	  ////unsigned char value;
	  ////for (int i = 0; i < height; i++) {
	  ////	for (int j = 0; j < width; j++) {

	  ////		value = image[4 * width * i + 4 * j];

	  ////		new_image[4 * width * i + 4 * j + 0] = value; // r
	  ////		new_image[4 * width * i + 4 * j + 1] = value; // g
	  ////		new_image[4 * width * i + 4 * j + 2] = value; // b
	  ////		new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; // a
	  ////	}
	  ////}

	  //lodepng_encode32_file(output_filename, new_image, width, height);

	  //free(image);
	  //free(new_image);

	float memsettime;
	cudaEvent_t start, stop;

	char* input_filename = argv[1];
	char* output_filename = argv[2];
	unsigned int num_of_threads = atol(argv[3]);

	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	unsigned int size = width * height * 4 * sizeof(unsigned char); // height x width number of pixels, 4 layers (RGBA) for each pixel, 1 char for each value





	unsigned char* cuda_og_image, * cuda_rectified_image;
	cudaMallocManaged((void**)&cuda_og_image, size);
	cudaMallocManaged((void**)&cuda_rectified_image, size);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_og_image, image, size, cudaMemcpyHostToDevice);

	// genereal rule of thumb for creating number of blocks 
	unsigned int num_of_blocks = (size + num_of_threads - 1) / num_of_threads;

	//start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//CUDA kernel call
	rectification << < num_of_blocks, num_of_threads >> > (cuda_og_image, cuda_rectified_image, size);
	cudaDeviceSynchronize();

	//end timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf(" *** CUDA execution time: %f *** \n", memsettime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// CPU copies input data from GPU back to CPU
	unsigned char* rec_image = (unsigned char*)malloc(size);
	cudaMemcpy(rec_image, cuda_rectified_image, size, cudaMemcpyDeviceToHost);
	cudaFree(cuda_og_image);
	cudaFree(cuda_rectified_image);

	lodepng_encode32_file(output_filename, rec_image, width, height);

	//end of recitfication

	//start of pooling

	//unsigned char* cuda_original_image, * cuda_pooled_image;
	//cudaMallocManaged((void**)&cuda_original_image, size);
	//cudaMallocManaged((void**)&cuda_pooled_image, size);

	//// CPU copies input data from CPU to GPU
	//cudaMemcpy(cuda_original_image, image, size, cudaMemcpyHostToDevice);

	//// genereal rule of thumb for creating number of blocks 
	//unsigned int num_of_blks = (size + num_of_threads - 1) / num_of_threads;

	//pooling << <num_of_blks, num_of_threads >> > (width, size, cuda_original_image, cuda_pooled_image);

	return 0;
}
