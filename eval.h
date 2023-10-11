#include<iostream>
#include<iomanip>
#include<fstream>
#include<assert.h>
#include"derivatives.h"
#include"sobolev.h"
#include"PSNR.h"
#include"macro.h"

#define __USE_READFILE 1

using namespace std;

template<typename Type>
void readFile(const char *file, const size_t num, Type *data){
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cout << " Error, Couldn't find the file: " << file << "\n";
        exit(0);
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    assert(num_elements == num && "File size is not equals to the input setting");
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(data), num_elements * sizeof(Type));
    fin.close();
}

template<typename Type>
void writeFile(const char *file, const size_t num_elements, Type *data){
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char *>(&data[0]), num_elements * sizeof(Type));
    fout.close();
}

char f0_file_name[64],f1_file_name[64],buf[64];
int nx,ny,nz,n;
float *f0_host_data,*f0_device_data;
float *f1_host_data,*f1_device_data;

float *f0_device_dx,*f0_device_dy,*f0_device_dz;
float *f0_device_dx2,*f0_device_dy2,*f0_device_dz2;
float *f0_device_dxy,*f0_device_dyz,*f0_device_dzx;
float *f0_device_gl,*f0_device_lap;

float *f1_device_dx,*f1_device_dy,*f1_device_dz;
float *f1_device_dx2,*f1_device_dy2,*f1_device_dz2;
float *f1_device_dxy,*f1_device_dyz,*f1_device_dzx;
float *f1_device_gl,*f1_device_lap;

signed main(){

    printf("please input the original file name:\n");

    scanf("%s",f0_file_name);

    printf("please input the unzipped file name:\n");

    scanf("%s",f1_file_name);

    printf("please input the size like [nx] [ny] [nz]:\n");

    scanf("%d %d %d",&nx,&ny,&nz);

    n=nx*ny*nz;

    f0_host_data=(float*)malloc(n*sizeof(float));
    f1_host_data=(float*)malloc(n*sizeof(float));

    cudaMalloc(&f0_device_data,n*sizeof(float));
    cudaMalloc(&f0_device_dx,n*sizeof(float));
    cudaMalloc(&f0_device_dy,n*sizeof(float));
    cudaMalloc(&f0_device_dz,n*sizeof(float));
    cudaMalloc(&f0_device_dx2,n*sizeof(float));
    cudaMalloc(&f0_device_dy2,n*sizeof(float));
    cudaMalloc(&f0_device_dz2,n*sizeof(float));
    cudaMalloc(&f0_device_dxy,n*sizeof(float));
    cudaMalloc(&f0_device_dyz,n*sizeof(float));
    cudaMalloc(&f0_device_dzx,n*sizeof(float));
    cudaMalloc(&f0_device_gl,n*sizeof(float));
    cudaMalloc(&f0_device_lap,n*sizeof(float));

    cudaMalloc(&f1_device_data,n*sizeof(float));
    cudaMalloc(&f1_device_dx,n*sizeof(float));
    cudaMalloc(&f1_device_dy,n*sizeof(float));
    cudaMalloc(&f1_device_dz,n*sizeof(float));
    cudaMalloc(&f1_device_dx2,n*sizeof(float));
    cudaMalloc(&f1_device_dy2,n*sizeof(float));
    cudaMalloc(&f1_device_dz2,n*sizeof(float));
    cudaMalloc(&f1_device_dxy,n*sizeof(float));
    cudaMalloc(&f1_device_dyz,n*sizeof(float));
    cudaMalloc(&f1_device_dzx,n*sizeof(float));
    cudaMalloc(&f1_device_gl,n*sizeof(float));
    cudaMalloc(&f1_device_lap,n*sizeof(float));

    if(__USE_READFILE){

        readFile<float>(f0_file_name,n,f0_host_data);
        readFile<float>(f1_file_name,n,f1_host_data);
    }
    else{

    }
    
    cudaMemcpy(f0_device_data,f0_host_data,n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(f1_device_data,f1_host_data,n*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blocksz=dim3(BLOCKSZX,BLOCKSZY,BLOCKSZZ);
    dim3 blocknum=dim3(nx/BLOCKSZX+(nx%BLOCKSZX>0),ny/BLOCKSZY+(ny%BLOCKSZY>0),nz/BLOCKSZZ+(nz%BLOCKSZZ>0));

    derivatives<<<blocknum,blocksz>>>(f0_device_data,f0_device_dx,f0_device_dy,f0_device_dz,f0_device_dx2,f0_device_dy2,f0_device_dz2,f0_device_dxy,f0_device_dyz,f0_device_dzx,f0_device_gl,f0_device_lap,nx,ny,nz);
    derivatives<<<blocknum,blocksz>>>(f1_device_data,f1_device_dx,f1_device_dy,f1_device_dz,f1_device_dx2,f1_device_dy2,f1_device_dz2,f1_device_dxy,f1_device_dyz,f1_device_dzx,f1_device_gl,f1_device_lap,nx,ny,nz);

    cudaDeviceSynchronize();

    //then calculate each PSNR, then output

    cout<<fixed<<setprecision(2);

    cout<<"PSNR of data = "<<findPSNR(f0_device_data,f1_device_data,nx,ny,nz)<<endl;
    cout<<"PSNR of dx = "<<findPSNR(f0_device_dx,f1_device_dx,nx,ny,nz)<<endl;
    cout<<"PSNR of dy = "<<findPSNR(f0_device_dy,f1_device_dy,nx,ny,nz)<<endl;
    cout<<"PSNR of dz = "<<findPSNR(f0_device_dz,f1_device_dz,nx,ny,nz)<<endl;
    cout<<"PSNR of dx2 = "<<findPSNR(f0_device_dx2,f1_device_dx2,nx,ny,nz)<<endl;
    cout<<"PSNR of dy2 = "<<findPSNR(f0_device_dy2,f1_device_dy2,nx,ny,nz)<<endl;
    cout<<"PSNR of dz2 = "<<findPSNR(f0_device_dz2,f1_device_dz2,nx,ny,nz)<<endl;
    cout<<"PSNR of gradient length = "<<findPSNR(f0_device_gl,f1_device_gl,nx,ny,nz)<<endl;
    cout<<"PSNR of laplacian = "<<findPSNR(f0_device_lap,f1_device_lap,nx,ny,nz)<<endl;
    
    vector<float> f0_s=sobolev(f0_device_data,f0_device_dx,f0_device_dy,f0_device_dz,f0_device_dx2,f0_device_dy2,f0_device_dz2,f0_device_dxy,f0_device_dyz,f0_device_dzx,nx,ny,nz);
    vector<float> f1_s=sobolev(f1_device_data,f1_device_dx,f1_device_dy,f1_device_dz,f1_device_dx2,f1_device_dy2,f1_device_dz2,f1_device_dxy,f1_device_dyz,f1_device_dzx,nx,ny,nz);

    cout<<setprecision(12);
    cout<<"s0="<<"{"<<f0_s[0]<<","<<f1_s[0]<<"}"<<endl;
    cout<<"s1="<<"{"<<f0_s[1]<<","<<f1_s[1]<<"}"<<endl;
    cout<<"s2="<<"{"<<f0_s[2]<<","<<f1_s[2]<<"}"<<endl;

    cout<<scientific<<setprecision(2);
    cout<<"Relative Error of s0 = "<<abs(f0_s[0]-f1_s[0])/f0_s[0]<<endl;
    cout<<"Relative Error of s1 = "<<abs(f0_s[1]-f1_s[1])/f0_s[1]<<endl;
    cout<<"Relative Error of s2 = "<<abs(f0_s[2]-f1_s[2])/f0_s[2]<<endl;

    free(f0_host_data);
    free(f1_host_data);

    cudaFree(f0_device_data);
    cudaFree(f0_device_dx);
    cudaFree(f0_device_dy);
    cudaFree(f0_device_dz);
    cudaFree(f0_device_dx2);
    cudaFree(f0_device_dy2);
    cudaFree(f0_device_dz2);
    cudaFree(f0_device_dxy);
    cudaFree(f0_device_dyz);
    cudaFree(f0_device_dzx);
    cudaFree(f0_device_gl);
    cudaFree(f0_device_lap);

    cudaFree(f1_device_data);
    cudaFree(f1_device_dx);
    cudaFree(f1_device_dy);
    cudaFree(f1_device_dz);
    cudaFree(f1_device_dx2);
    cudaFree(f1_device_dy2);
    cudaFree(f1_device_dz2);
    cudaFree(f1_device_dxy);
    cudaFree(f1_device_dyz);
    cudaFree(f1_device_dzx);
    cudaFree(f1_device_gl);
    cudaFree(f1_device_lap);

end:
    return 0;
}