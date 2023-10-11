#include"derivatives.h"
#include"sobolev.h"
#include"PSNR.h"
#include"macro.h"

// __global__ void testKernel(float* data,int nx,int ny,int nz){
//     int n=nx*ny*nz;
//     float maxval=-inf;
//     for(int i=0;i<n;i++) maxval=max(maxval,data[i]);
//     printf("+++%.12f\n",maxval);
// }

// device -> device 
void derivatives(float* data,int nx,int ny,int nz,
                float *&dx,float *&dy,float *&dz,
                float *&dx2,float *&dy2,float *&dz2,
                float *&dxy,float *&dyz,float *&dzx,
                float *&gl,float *&lap
                ){
    
    int n=nx*ny*nz;

    cudaMalloc(&dx,n*sizeof(float));
    cudaMalloc(&dy,n*sizeof(float));
    cudaMalloc(&dz,n*sizeof(float));
    cudaMalloc(&dx2,n*sizeof(float));
    cudaMalloc(&dy2,n*sizeof(float));
    cudaMalloc(&dz2,n*sizeof(float));
    cudaMalloc(&dxy,n*sizeof(float));
    cudaMalloc(&dyz,n*sizeof(float));
    cudaMalloc(&dzx,n*sizeof(float));
    cudaMalloc(&gl,n*sizeof(float));
    cudaMalloc(&lap,n*sizeof(float));

    dim3 blocksz=dim3(BLOCKSZX,BLOCKSZY,BLOCKSZZ);
    dim3 blocknum=dim3(nx/BLOCKSZX+(nx%BLOCKSZX>0),ny/BLOCKSZY+(ny%BLOCKSZY>0),nz/BLOCKSZZ+(nz%BLOCKSZZ>0));

    derivativesKernel<<<blocknum,blocksz>>>(data,dx,dy,dz,dx2,dy2,dz2,dxy,dyz,dzx,gl,lap,nx,ny,nz);

    cudaDeviceSynchronize();
}

// host -> device -> host

/*
0: PSNR of data
1,2,3: PSNR of dx,dy,dz
4,5,6: PSNR of dx2,dy2,dz2
7,8: PSNR of gradient length, laplacian
9,10,11: relative error of s0,s1,s2
*/

std::vector<float> derivativesPSNR(float* host_f0,float* host_f1,int nx,int ny,int nz){

    int n=nx*ny*nz;

    float *f0;
    cudaMalloc(&f0,n*sizeof(float));
    cudaMemcpy(f0,host_f0,n*sizeof(float),cudaMemcpyHostToDevice);

    float *f0_dx,*f0_dy,*f0_dz;
    float *f0_dx2,*f0_dy2,*f0_dz2;
    float *f0_dxy,*f0_dyz,*f0_dzx;
    float *f0_gl,*f0_lap;

    derivatives(f0,nx,ny,nz,f0_dx,f0_dy,f0_dz,f0_dx2,f0_dy2,f0_dz2,f0_dxy,f0_dyz,f0_dzx,f0_gl,f0_lap);
    std::vector<float> f0_sobolev=sobolev(f0,f0_dx,f0_dy,f0_dz,f0_dx2,f0_dy2,f0_dz2,f0_dxy,f0_dyz,f0_dzx,nx,ny,nz);

    float *f1;
    cudaMalloc(&f1,n*sizeof(float));
    cudaMemcpy(f1,host_f1,n*sizeof(float),cudaMemcpyHostToDevice);

    float *f1_dx,*f1_dy,*f1_dz;
    float *f1_dx2,*f1_dy2,*f1_dz2;
    float *f1_dxy,*f1_dyz,*f1_dzx;
    float *f1_gl,*f1_lap;

    derivatives(f1,nx,ny,nz,f1_dx,f1_dy,f1_dz,f1_dx2,f1_dy2,f1_dz2,f1_dxy,f1_dyz,f1_dzx,f1_gl,f1_lap);
    std::vector<float> f1_sobolev=sobolev(f1,f1_dx,f1_dy,f1_dz,f1_dx2,f1_dy2,f1_dz2,f1_dxy,f1_dyz,f1_dzx,nx,ny,nz);

    std::vector<float> vec(12);

    vec[0]=findPSNR(f0,f1,nx,ny,nz);

    vec[1]=findPSNR(f0_dx,f1_dx,nx,ny,nz);
    vec[2]=findPSNR(f0_dy,f1_dy,nx,ny,nz);
    vec[3]=findPSNR(f0_dz,f1_dz,nx,ny,nz);

    vec[4]=findPSNR(f0_dx2,f1_dx2,nx,ny,nz);
    vec[5]=findPSNR(f0_dy2,f1_dy2,nx,ny,nz);
    vec[6]=findPSNR(f0_dz2,f1_dz2,nx,ny,nz);

    vec[7]=findPSNR(f0_gl,f1_gl,nx,ny,nz);
    vec[8]=findPSNR(f0_lap,f1_lap,nx,ny,nz);

    vec[9]=std::abs(f0_sobolev[0]-f1_sobolev[0])/f0_sobolev[0];
    vec[10]=std::abs(f0_sobolev[1]-f1_sobolev[1])/f0_sobolev[1];
    vec[11]=std::abs(f0_sobolev[2]-f1_sobolev[2])/f0_sobolev[2];

    cudaFree(f0);
    cudaFree(f0_dx),cudaFree(f0_dy),cudaFree(f0_dz);
    cudaFree(f0_dx2),cudaFree(f0_dy2),cudaFree(f0_dz2);
    cudaFree(f0_dxy),cudaFree(f0_dyz),cudaFree(f0_dzx);
    cudaFree(f0_gl),cudaFree(f0_lap);

    cudaFree(f1);
    cudaFree(f1_dx),cudaFree(f1_dy),cudaFree(f1_dz);
    cudaFree(f1_dx2),cudaFree(f1_dy2),cudaFree(f1_dz2);
    cudaFree(f1_dxy),cudaFree(f1_dyz),cudaFree(f1_dzx);
    cudaFree(f1_gl),cudaFree(f1_lap);
    
    return vec;
}