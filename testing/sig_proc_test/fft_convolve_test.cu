#include "cufft.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#define SIG_LEN 81920
#define N_SIGS (20*6*3)

/*int main() {
    cufftHandle plan;
    cufftComplex *data;
    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
    if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return 0;
    }

    if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return 0;
    }

    for (int i=0; i<20; i++) {
        cudaEventRecord(start, 0);
        if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: Plan creation failed");
            return 0;
        }
        // Note: * Identical pointers to input and output arrays implies in-place transformation
        if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
            return 0;
        }

        if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
            return 0;  }
        //Results may not be immediately available so block device until all * tasks have completed
        if (cudaDeviceSynchronize() != cudaSuccess){
            fprintf(stderr, "Cuda error: Failed to synchronize\n");
            return 0;
        }
        cufftDestroy(plan);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cumulative_time = cumulative_time + time;
    }


    printf("FFT + IFFT time:  %3.5f ms \n", cumulative_time);


      // Divide by number of elements in data set to get back original data
    cufftDestroy(plan);
    cudaFree(data);
    return 0;
}*/
__global__ void multiply(cuFloatComplex* samples,cuFloatComplex* filter,
                            cuFloatComplex* result, int num_samps) {
    auto channel_off = num_samps * gridDim.y;
    auto samp_off = blockIdx.x * blockDim.x + threadIdx.x;

    cuFloatComplex res;
    res.x = samples[channel_off + samp_off].x * filter[samp_off].x - samples[channel_off + samp_off].y * filter[samp_off].y;
    res.y = samples[channel_off + samp_off].x * filter[samp_off].y + samples[channel_off + samp_off].y * filter[samp_off].x;

    result[channel_off + samp_off] = res;

}




struct ElementWiseProductBasic : public thrust::binary_function<cuFloatComplex,cuFloatComplex,cuFloatComplex>
{
    __host__ __device__
    cuFloatComplex operator()(const cuFloatComplex& v1, const cuFloatComplex& v2) const
    {
        cuFloatComplex res;
        res.x = v1.x * v2.x - v1.y * v2.y;
        res.y = v1.x * v2.y + v1.y * v2.x;
        return res;
    }
};
void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}
int main(){
  int rank = 1;                           // --- 1D FFTs
  int n[] = { SIG_LEN };                 // --- Size of the Fourier transform
  int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
  int idist = SIG_LEN, odist = (SIG_LEN); // --- Distance between batches
  int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
  int batch = N_SIGS;                      // --- Number of batched executions
  cufftHandle plan;
  auto res = cufftPlanMany(&plan, rank, n,
                inembed, istride, idist,
                onembed, ostride, odist, CUFFT_C2C, batch);
  if (res != CUFFT_SUCCESS) {printf("plan create fail\n"); return 1;}

  //cuFloatComplex *h_signal, *d_signal, *h_result, *d_result;
  cudaEvent_t start, stop;
  float time = 0.0;


  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto h_signal = thrust::host_vector<cuFloatComplex>(N_SIGS * SIG_LEN);
  for (int i = 0; i < N_SIGS; i ++)
    for (int j = 0; j < SIG_LEN; j++)
      h_signal[(i*SIG_LEN) + j] = make_cuFloatComplex(sin((i+1)*6.283*j/SIG_LEN), 0);


  //cudaMalloc(&d_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
  //cudaMalloc(&d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex));
  cudaEventRecord(start, 0);
  thrust::device_vector<cuFloatComplex> d_signal = h_signal;
  thrust::device_vector<cuFloatComplex> d_result(N_SIGS * SIG_LEN);
  thrust::device_vector<cuFloatComplex> d_result_modified(N_SIGS * SIG_LEN);
  thrust::device_vector<cuFloatComplex> d_filter(N_SIGS * SIG_LEN);
  //cudaMemcpy(d_signal, h_signal, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

  res = cufftPlanMany(&plan, rank, n,
                inembed, istride, idist,
                onembed, ostride, odist, CUFFT_C2C, batch);
  if (res != CUFFT_SUCCESS) {printf("plan create fail\n"); return 1;}

  //FFT
  auto d_signal_p = thrust::raw_pointer_cast(d_signal.data());
  auto d_result_p = thrust::raw_pointer_cast(d_result.data());
  auto d_result_modified_p = thrust::raw_pointer_cast(d_result_modified.data());
  res = cufftExecC2C(plan, d_signal_p, d_result_p, CUFFT_FORWARD);
  if (res != CUFFT_SUCCESS) {printf("forward transform fail\n"); return 1;}
  //cudaMemcpy(h_result, d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
/*  thrust::transform(d_result.begin(), d_result.end(),
                          d_filter.begin(), d_result_modified.begin(),
                          ElementWiseProductBasic());*/
  dim3 dimGrid(SIG_LEN/1024,N_SIGS,1);
  dim3 dimBlock(1024);
  //multiply<<<dimGrid,dimBlock>>>(d_signal_p,d_result_p,d_result_modified_p,SIG_LEN);
  //throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);

  //IFFT
  res = cufftExecC2C(plan, d_signal_p, d_result_p, CUFFT_INVERSE);
  if (res != CUFFT_SUCCESS) {printf("forward transform fail\n"); return 1;}
  //cudaMemcpy(h_result, d_result, N_SIGS*SIG_LEN*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  auto h_result = d_result;
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("FFT/IFFT time:  %3.5f ms \n", time);

  for (int i = 0; i < N_SIGS; i++){
    for (int j = 0; j < 10; j++)
      printf("%.3f ", cuCrealf(h_result[(i*SIG_LEN)+j]));
    printf("\n"); }

  return 0;
}