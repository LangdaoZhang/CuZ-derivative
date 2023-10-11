#include<cmath>
#include"reduction.h"
#include"macro.h"

__global__ void squareDifferenceKernelUsingByPSNR(float* f0,float* f1,double* tem,int nx,int ny,int nz){

    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int z=blockIdx.z*blockDim.z+threadIdx.z;

    if(x<nx&&y<ny&&z<nz){

        tem[arr3(x,y,z)]=f0[arr3(x,y,z)]-f1[arr3(x,y,z)];
        tem[arr3(x,y,z)]*=tem[arr3(x,y,z)];
    }
}

// __global__ void differenceKernelUsingByPSNR(float* f0,float* f1,float* tem,int nx,int ny,int nz){

//     int x=blockIdx.x*blockDim.x+threadIdx.x;
//     int y=blockIdx.y*blockDim.y+threadIdx.y;
//     int z=blockIdx.z*blockDim.z+threadIdx.z;

//     if(x<nx&&y<ny&&z<nz){

//         tem[arr3(x,y,z)]=abs(f0[arr3(x,y,z)]-f1[arr3(x,y,z)]);
//     }
// }

double findMSEn(float* f0,float* f1,int nx,int ny,int nz){
    
    int n=nx*ny*nz;
    
    dim3 blocksz=dim3(BLOCKSZX,BLOCKSZY,BLOCKSZZ);
    dim3 blocknum=dim3(nx/BLOCKSZX+(nx%BLOCKSZX>0),ny/BLOCKSZY+(ny%BLOCKSZY>0),nz/BLOCKSZZ+(nz%BLOCKSZZ>0));

    double* tem;

    cudaMalloc(&tem,n*sizeof(double));

    squareDifferenceKernelUsingByPSNR<<<blocknum,blocksz>>>(f0,f1,tem,nx,ny,nz);

    cudaDeviceSynchronize();

    double ans=sumupUsingReductionForceDouble(tem,n);

    cudaFree(tem);

    return ans;
}

double findPSNR(float* f0,float* f1,int nx,int ny,int nz){
//double findPSNR(float* f0,float* f1,int nx,int ny,int nz,int output_ae=0){

    int n=nx*ny*nz;

    double Rf0=findMaximumUsingReduction(f0,n)-findMinimumUsingReduction(f0,n);

    if(Rf0<=0) return -inf;

    double MSEn=findMSEn(f0,f1,nx,ny,nz);

    if(MSEn<=0) return inf;

    double PSNR=20.*log10(Rf0*sqrt(n)/sqrt(MSEn));

    // if(output_ae){

    //     float* tem;

    //     cudaMalloc(&tem,n*sizeof(float));

    //     dim3 blocksz=dim3(BLOCKSZX,BLOCKSZY,BLOCKSZZ);
    //     dim3 blocknum=dim3(nx/BLOCKSZX+(nx%BLOCKSZX>0),ny/BLOCKSZY+(ny%BLOCKSZY>0),nz/BLOCKSZZ+(nz%BLOCKSZZ>0));
    //     differenceKernelUsingByPSNR<<<blocknum,blocksz>>>(f0,f1,tem,nx,ny,nz);

    //     float ae=findMaximumUsingReduction(tem,n);

    //     std::cout<<std::scientific<<"Absolute Error="<<ae<<std::endl;
    //     std::cout<<std::fixed;

    //     cudaFree(tem);
    // }

    return PSNR;
}