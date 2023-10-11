#include<cmath>
#include"macro.h"

__global__ void derivatives(float* data,float* dx,float* dy,float* dz,float* dx2,float* dy2,float* dz2,float* dxy,float* dyz,float* dzx,float* gl,float* lap,int nx,int ny,int nz){

    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int z=blockIdx.z*blockDim.z+threadIdx.z;

    if(x<nx&&y<ny&&z<nz){

        // 1st order derivatives

        if(x<nx-1){
            dx[arr3(x,y,z)]=data[arr3(x+1,y,z)]-data[arr3(x,y,z)];
        }
        else{
            dx[arr3(x,y,z)]=0;
        }

        if(y<ny-1){
            dy[arr3(x,y,z)]=data[arr3(x,y+1,z)]-data[arr3(x,y,z)];
        }
        else{
            dy[arr3(x,y,z)]=0;
        }

        if(z<nz-1){
            dz[arr3(x,y,z)]=data[arr3(x,y,z+1)]-data[arr3(x,y,z)];
        }
        else{
            dz[arr3(x,y,z)]=0;
        }

        // 2rd order derivatives

        if(x<nx-2){
            dx2[arr3(x,y,z)]=data[arr3(x+2,y,z)]-2.*data[arr3(x+1,y,z)]+data[arr3(x,y,z)];
        }
        else{
            dx2[arr3(x,y,z)]=0;
        }

        if(y<ny-2){
            dy2[arr3(x,y,z)]=data[arr3(x,y+2,z)]-2.*data[arr3(x,y+1,z)]+data[arr3(x,y,z)];
        }
        else{
            dy2[arr3(x,y,z)]=0;
        }

        if(z<nz-2){
            dz2[arr3(x,y,z)]=data[arr3(x,y,z+2)]-2.*data[arr3(x,y,z+1)]+data[arr3(x,y,z)];
        }
        else{
            dz2[arr3(x,y,z)]=0;
        }

        if(x<nx-1&&y<ny-1){

            dxy[arr3(x,y,z)]=data[arr3(x+1,y+1,z)]-data[arr3(x+1,y,z)]-data[arr3(x,y+1,z)]+data[arr3(x,y,z)];
        }
        else{

            dxy[arr3(x,y,z)]=0;
        }

        if(y<ny-1&&z<nz-1){

            dyz[arr3(x,y,z)]=data[arr3(x,y+1,z+1)]-data[arr3(x,y+1,z)]-data[arr3(x,y,z+1)]+data[arr3(x,y,z)];
        }
        else{

            dyz[arr3(x,y,z)]=0;
        }

        if(z<nz-1&&x<nx-1){

            dzx[arr3(x,y,z)]=data[arr3(x+1,y,z+1)]-data[arr3(x,y,z+1)]-data[arr3(x+1,y,z)]+data[arr3(x,y,z)];
        }
        else{

            dzx[arr3(x,y,z)]=0;
        }

        //gradient length

        gl[arr3(x,y,z)]=sqrt(square(dx[arr3(x,y,z)])+square(dy[arr3(x,y,z)])+square(dz[arr3(x,y,z)]));

        //laplacian

        lap[arr3(x,y,z)]=square(dx2[arr3(x,y,z)])+square(dy2[arr3(x,y,z)])+square(dz2[arr3(x,y,z)]);
    }
}