#ifndef _REDUCTION_H
#define _REDUCTION_H

#include"macro.h"

#define blksz 1024
// must be 2^k, and 0<= k <= 10

__global__ void maximumReduction(float* data,int p,int n){

    int id=(blockIdx.x*blockDim.x+threadIdx.x)*p;

    if(id>=n) return;

    for(int stride=p;stride<p*blksz;stride<<=1){

        if(id%(stride<<1)==0){
            
            if(id+stride<n){

                if(data[id+stride]>data[id]){

                    data[id]=data[id+stride];
                }
            }
        }

        __syncthreads();
    }
}

float findMaximumUsingReduction(float* data,int n){

    float* tem;

    cudaMalloc(&tem,n*sizeof(float));

    cudaMemcpy(tem,data,n*sizeof(float),cudaMemcpyDeviceToDevice);

    int p=1;

    while(true){

        int blknum=n/(p*blksz)+(n%(p*blksz)>0);

        maximumReduction<<<blknum,blksz>>>(tem,p,n);

        cudaDeviceSynchronize();

        if(blknum==1) break;

        p*=blksz;
    }

    float ans;

    cudaMemcpy(&ans,tem,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(tem);

    return ans;
}

__global__ void minimumReduction(float* data,int p,int n){

    int id=(blockIdx.x*blockDim.x+threadIdx.x)*p;

    if(id>=n) return;

    for(int stride=p;stride<p*blksz;stride<<=1){

        if(id%(stride<<1)==0){
            
            if(id+stride<n){

                if(data[id+stride]<data[id]){

                    data[id]=data[id+stride];
                }
            }
        }

        __syncthreads();
    }
}

float findMinimumUsingReduction(float* data,int n){

    float* tem;

    cudaMalloc(&tem,n*sizeof(float));

    cudaMemcpy(tem,data,n*sizeof(float),cudaMemcpyDeviceToDevice);

    int p=1;

    while(true){

        int blknum=n/(p*blksz)+(n%(p*blksz)>0);

        minimumReduction<<<blknum,blksz>>>(tem,p,n);

        cudaDeviceSynchronize();

        if(blknum==1) break;

        p*=blksz;
    }

    float ans;

    cudaMemcpy(&ans,tem,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(tem);

    return ans;
}

__global__ void sumReduction(float* data,int p,int n){

    int id=(blockIdx.x*blockDim.x+threadIdx.x)*p;

    if(id>=n) return;

    for(int stride=p;stride<p*blksz;stride<<=1){

        if(id%(stride<<1)==0){
            
            if(id+stride<n){
                
                data[id]+=data[id+stride];
            }
        }

        __syncthreads();
    }
}

float sumupUsingReduction(float* data,int n){

    float* tem;

    cudaMalloc(&tem,n*sizeof(float));

    cudaMemcpy(tem,data,n*sizeof(float),cudaMemcpyDeviceToDevice);

    int p=1;

    while(true){

        int blknum=n/(p*blksz)+(n%(p*blksz)>0);

        sumReduction<<<blknum,blksz>>>(tem,p,n);

        cudaDeviceSynchronize();

        if(blknum==1) break;

        p*=blksz;
    }

    float ans;

    cudaMemcpy(&ans,tem,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(tem);

    return ans;
}

__global__ void sumReductionForceDouble(double* data,int p,int n){

    int id=(blockIdx.x*blockDim.x+threadIdx.x)*p;

    if(id>=n) return;

    for(int stride=p;stride<p*blksz;stride<<=1){

        if(id%(stride<<1)==0){
            
            if(id+stride<n){
                
                data[id]+=data[id+stride];
            }
        }

        __syncthreads();
    }
}

double sumupUsingReductionForceDouble(double* data,int n){

    double* tem;

    cudaMalloc(&tem,n*sizeof(double));

    cudaMemcpy(tem,data,n*sizeof(double),cudaMemcpyDeviceToDevice);

    int p=1;

    while(true){

        int blknum=n/(p*blksz)+(n%(p*blksz)>0);

        sumReductionForceDouble<<<blknum,blksz>>>(tem,p,n);

        cudaDeviceSynchronize();

        if(blknum==1) break;

        p*=blksz;
    }

    double ans;

    cudaMemcpy(&ans,tem,sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(tem);

    return ans;
}

#undef blksz

#endif