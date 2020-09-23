// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// kernel function that generates the 'shower'
__global__
void shower(int n, float *particle_energy, float *totalEnergyLoss, int *numberOfSecondaries)
 {
 float r;
 curandState state;

 curand_init(0, /* the seed controls the sequence of random values that are produced */
             0, /* the sequence number is only important with multiple cores */
             0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
             &state);

 while (n>=0)
  {
   while (particle_energy[n]>0)
    {
     r = curand_uniform(&state);

    if (r < 0.5f)
      {
       float eloss = 0.2f * particle_energy[n];
       *totalEnergyLoss += (eloss < 0.001f ? particle_energy[n] : eloss);
       particle_energy[n] = (eloss < 0.001f ? 0.0f : (particle_energy[n] - eloss));
      }
    else 
     {
      float eloss = 0.5f * particle_energy[n];

      particle_energy[n] -= eloss;
      //
      // here I need to create a new particle
      //
      
      n++;
      particle_energy[n] = eloss;
      (*numberOfSecondaries)++;

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

 float *totalEnergyLoss;
 int *numberOfSecondaries;

 // Allocate Unified Memory accessible from CPU or GPU
 cudaMallocManaged(&particle_energy, N*sizeof(float));
 cudaMallocManaged(&totalEnergyLoss, sizeof(float));
 cudaMallocManaged(&numberOfSecondaries, sizeof(int));

 *totalEnergyLoss = 0;
 *numberOfSecondaries = 0;

 // initialize particle_energy array on the host
  for (int i = 0; i < N; i++) 
   {
    particle_energy[i] = 10.0f;
   }

  int n_particles = 100;

  // Run kernel on n_particles on the GPU
  shower<<<1, 1>>>(n_particles, particle_energy, totalEnergyLoss, numberOfSecondaries);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  std::cout << "Total energy loss " << *totalEnergyLoss << " number of secondaries " << *numberOfSecondaries << std::endl;

  // Free memory
  cudaFree(particle_energy);
  cudaFree(totalEnergyLoss);
  cudaFree(numberOfSecondaries);
}
