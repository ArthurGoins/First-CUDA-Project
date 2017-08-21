/*==============================================================================================*
* Project: CUDA Bemchmark based op calculating prime numbers									*
* Developed with Visual Studio 2015 and CUDA Toolkit 8.0										*
* Written by:	Arthur Goins																	*
*				Computer Engineering student at NCSU											*
*				ajgoins@ncsu.edu																*
*																								*
* Function:		To test and quantify the compute performance of a multi-threaded CUDA device	*
*===============================================================================================*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>

#define MAXARRAY 20000000											//calculate primes until this number
#define THREADS 100													//each thread will check 200,000 numbers (each thread will check every 100 integers)

__global__ 
void checkPrime()													//CUDA function
{
	int flag = 0;													//flag to identify how the following for loop exited
	int loop = 1;													//variable to allow the function to exit the while loop
	int x = threadIdx.x;											//save the thread ID as an integer (henceforth referred to as the 'thread integer' and it is the number that will be tested)
	while (loop)
	{
		for (int i = 2; i <= (x / 2); ++i)							//set the integer i to every integer value between 2 and half of the previously saved thread integer
		{		
			if ((x%i) == 0)											//check if the thread integer is evenly divisible by i
			{
				flag = 1;											//the thread integer is not prime, so 'flag' is set to indicate this
				break;												//exit the for loop
			}

		}
		if (flag == 0) {											//if 'flag' is still 0, the thread integer is prime
			x += THREADS;											//increment the thread integer by the macro defined earlier
		}
		else
		{
			x += THREADS;											//increment the thread integer by the macro defined earlier
		}
		if (x >= MAXARRAY)											//once a thread reaches the target number, stop its loop
			loop = 0;
	}
}

int main()
{
	clock_t start, end;												//timing variables
	double cpu_time;											
	double time_per_cycle;

	const int cycles = 150;											//variables for setting the number of cycles to run
	int iteration = 0;

	dim3 dimBlock(1000, 1);											//block size and thread configuration
	dim3 dimGrid(THREADS, 1);

	printf("starting test...\n");

	checkPrime << <dimGrid, dimBlock >> > ();						//run the function a few times to allow the GPU to warm up
	cudaDeviceSynchronize();
	checkPrime << <dimGrid, dimBlock >> > ();
	cudaDeviceSynchronize();
	checkPrime << <dimGrid, dimBlock >> > ();
	cudaDeviceSynchronize();

	start = clock();												//get the current time
	checkPrime << <dimGrid, dimBlock >> > ();						//run the CUDA function
	cudaDeviceSynchronize();										//wait for the function to finish
	end = clock();													//get the current time
	cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;			//determine the time per cycle
	time_per_cycle = cpu_time;										//store the fist time in the accumulation variable
	
	while (iteration < cycles)										//loop the same cycle as above 150 times
	{
		start = clock();
		checkPrime << <dimGrid, dimBlock >> > ();
		cudaDeviceSynchronize();
		end = clock();
		cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
		time_per_cycle = (time_per_cycle + cpu_time) / 2;
		iteration++;
		printf("Time: %f \t %d out of 150\n", cpu_time, iteration);	//print each time
	}

	printf("Average time per cycle: %f\n", time_per_cycle);			//print average time
	printf("Score: %f\n", 1000 / time_per_cycle);					//print score
	printf("Comparisons:\n");
	printf("GTX 1070:\t 17418\n");
	printf("GTX 960:\t 7016\n");
	printf("GTX 680:\t 6192\n");
	printf("GTX 650Ti:\t 2943\n");
	printf("GTX 960M:\t 2628\n");
	system("pause");
	return 0;
}