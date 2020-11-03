#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>  

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void pooling(unsigned char* test_image, unsigned char* output_image, unsigned int num_of_threads, int size, unsigned width, int row_size) {
	int index = blockIdx.x * num_of_threads + threadIdx.x;
	int new_index = index * 8 + row_size * (index / (width / 2));
	if (index < size) {

		// i is rgba vals 
		for (int i = 0; i < 4; i++) {
			
			int u_left = test_image[new_index + i];
			int u_right = test_image[new_index + 4 + i];
			int b_left = test_image[row_size + new_index + i];
			int b_right = test_image[row_size + 4 + new_index + i];

			//gets largest number from four
			int largest = 0;

			if (u_left >= u_right && u_left >= b_left && u_left >= b_right) {
				largest = u_left; 
			}
			else if (u_right >= u_left && u_right >= b_left && u_right >= b_right) {
				largest = u_right;
			}
			else if (b_left >= u_left && b_left >= u_right && b_left >= b_right) {
				largest = b_left;
			}
			else {
				largest = b_right;
			}

			output_image[index * 4 + i] = largest;

		}
	}
}

int main(int argc, char* argv[])
{

	char* input_filename = argv[1];
	char* output_filename = argv[2];
	unsigned int num_of_threads = atoi(argv[3]);

//	char* input_filename = "Test_1.png";
//	char* output_filename = "pool.png";
//	unsigned int num_of_threads = 2;

	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	unsigned int size = width * height * 4 * sizeof(unsigned char);
	unsigned char* og_image, * output_image;
	cudaMallocManaged((void**)&og_image, size);
	cudaMallocManaged((void**)&output_image, size / 4);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(og_image, image, size, cudaMemcpyHostToDevice);

	// genereal rule of thumb for creating number of blocks, size/16 cause of rgba*4
	unsigned int num_of_blocks = ((size / 16) + num_of_threads - 1) / num_of_threads;

	int row_size = width * 4; 

	//CUDA kernel call
	pooling << <num_of_blocks, num_of_threads >> > (og_image, output_image, num_of_threads, size / 16, width, row_size);
	cudaDeviceSynchronize();


	// CPU copies input data from GPU back to CPU
	unsigned char* pooling_image = (unsigned char*)malloc(size / 4);
	cudaMemcpy(pooling_image, output_image, size / 4, cudaMemcpyDeviceToHost);
	cudaFree(og_image);
	cudaFree(output_image);

	lodepng_encode32_file(output_filename, pooling_image, width / 2, height / 2);

	return 0;
}