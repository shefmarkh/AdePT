// This is the main include file to give us all of the alpaka classes, functions etc.
#include <alpaka/alpaka.hpp>


struct genRands {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc) const
  {

    using namespace alpaka;
    // iTh is the thread number we use this throughout
    uint32_t iTh = idx::getIdx<Grid, Threads>(acc)[0];
    

    // Create an alpaka random generator using a seed of 1984
    auto generator = rand::generator::createDefault(acc, 1984, iTh);
    auto func(alpaka::rand::distribution::createUniformReal<float>(acc));  
    func(generator);

  }
};

int main()
{

  using namespace alpaka; // alpaka functionality all lives in this namespace

  using Dim = dim::DimInt<1>;
  using Idx = uint32_t;


  // Define the alpaka accelerator to be Nvidia GPU
  using Acc = acc::AccGpuCudaRt<Dim, Idx>;
  //using Acc = acc::AccCpuThreads<Dim, Idx>;

  // Get the first device available of type GPU (i.e should be our sole GPU)/device
  auto const device = pltf::getDevByIdx<Acc>(0u);
  // Create a device for host for memory allocation, using the first CPU available
  auto devHost = pltf::getDevByIdx<dev::DevCpu>(0u);

  uint32_t NPART = 200;
  uint32_t blocksPerGrid     = NPART;
  uint32_t threadsPerBlock   = 1;
  uint32_t elementsPerThread = 1;

  auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, elementsPerThread};

  genRands genRands;

  auto taskGenRands = kernel::createTaskKernel<Acc>(workDiv, genRands);

  auto queue = queue::Queue<Acc, queue::Blocking>{device};
  queue::enqueue(queue, taskGenRands);

  return 0;

}
