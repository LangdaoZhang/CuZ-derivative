#include<vector>
#include<cmath>
#include"reduction.h"
#include"macro.h"

__global__ void squareMatrixKernelUsingBySobolev(float* data,float* dx,float* dy,float* dz,float* dx2,float* dy2,float* dz2,float* dxy,float* dyz,float* dzx,int nx,int ny,int nz){

    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int z=blockIdx.z*blockDim.z+threadIdx.z;

    if(x<nx&&y<ny&&z<nz){

        data[arr3(x,y,z)]*=data[arr3(x,y,z)];
        dx[arr3(x,y,z)]*=dx[arr3(x,y,z)];
        dy[arr3(x,y,z)]*=dy[arr3(x,y,z)];
        dz[arr3(x,y,z)]*=dz[arr3(x,y,z)];
        dx2[arr3(x,y,z)]*=dx2[arr3(x,y,z)];
        dy2[arr3(x,y,z)]*=dy2[arr3(x,y,z)];
        dz2[arr3(x,y,z)]*=dz2[arr3(x,y,z)];
        dxy[arr3(x,y,z)]*=dxy[arr3(x,y,z)];
        dyz[arr3(x,y,z)]*=dyz[arr3(x,y,z)];
        dzx[arr3(x,y,z)]*=dzx[arr3(x,y,z)];
        
    }
}

std::vector<float> sobolev(float* data,float* dx,float* dy,float* dz,float* dx2,float* dy2,float* dz2,float* dxy,float* dyz,float* dzx,int nx,int ny,int nz){

    dim3 blocksz=dim3(BLOCKSZX,BLOCKSZY,BLOCKSZZ);
    dim3 blocknum=dim3(nx/BLOCKSZX+(nx%BLOCKSZX>0),ny/BLOCKSZY+(ny%BLOCKSZY>0),nz/BLOCKSZZ+(nz%BLOCKSZZ>0));

    squareMatrixKernelUsingBySobolev<<<blocknum,blocksz>>>(data,dx,dy,dz,dx2,dy2,dz2,dxy,dyz,dzx,nx,ny,nz);

    cudaDeviceSynchronize();

    int n=nx*ny*nz;

    std::vector<float> s(3);

    s[0]=0;
    s[0]+=sumupUsingReduction(data,n);
    s[1]=s[0];
    s[1]+=sumupUsingReduction(dx,n);
    s[1]+=sumupUsingReduction(dy,n);
    s[1]+=sumupUsingReduction(dz,n);
    s[2]=s[1];
    s[2]+=sumupUsingReduction(dx2,n);
    s[2]+=sumupUsingReduction(dy2,n);
    s[2]+=sumupUsingReduction(dz2,n);
    s[2]+=sumupUsingReduction(dxy,n);
    s[2]+=sumupUsingReduction(dyz,n);
    s[2]+=sumupUsingReduction(dzx,n);

    s[0]=sqrt(s[0]/n);
    s[1]=sqrt(s[1]/n);
    s[2]=sqrt(s[2]/n);

    return s;
}