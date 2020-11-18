
#include <curand_kernel.h>

#define NPART 200

__global__ void genRands()
{
    // iTh is the thread number we use this throughout 
    int iTh=threadIdx.x +  blockIdx.x * blockDim.x;  

    curandState local_rand_state;
    curand_init(1984, iTh, 0, &local_rand_state);
    curand_uniform(&local_rand_state);
}

int main() {

  int num_threads = NPART;
  int num_blocks = 1;

  genRands<<<num_blocks, num_threads>>>();
  cudaDeviceSynchronize();


}



