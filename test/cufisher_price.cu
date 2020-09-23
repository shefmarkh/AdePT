// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// kernel function that generates the 'shower'
__global__
void shower(int n, float *particle_energy)
{
 float r;
 curandState state;

 while (n>=0)
  {
   while (particle_energy[n]>0)
    {
     r = curand_uniform(&state);

     if (r < 0.5f)
      {
       particle_energy[n] = 0;
      }
    }
    n--;
  }
 } 

//////// 

int main()
{
 int N = 1<<20;
 float *particle_energy;


 // Allocate Unified Memory accessible from CPU or GPU
 cudaMallocManaged(&particle_energy, N*sizeof(float));

 // initialize particle_energy array on the host
  for (int i = 0; i < N; i++) 
   {
    particle_energy[i] = 1.0f;
   }

  int n_particles = 100;

  // Run kernel on 100 particles on the GPU
  shower<<<1, 1>>>(n_particles, particle_energy);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // check that the final energy of all particles is 0
  for (int i = 0; i < n_particles; i++)
  {
    std::cout << "energy of particle " << i << " is " << particle_energy[i] << std::endl;
  }

  // Free memory
  cudaFree(particle_energy);

  return 0;
}
