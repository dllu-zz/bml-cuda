/*
This file is part of bml-cuda by Daniel Lu
Copyright (C) 2013  Daniel Lu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "bml.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define CATCH_ERROR s = cudaGetLastError(); if(s)printf("\n          Cuda Error: %s\n",cudaGetErrorString(s));
cudaError_t s;

// performs update that moves cars of type 1.
__global__ void step1(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	){
	int x = threadIdx.x;
	int y = blockIdx.x;

	char a, b, c;
	a=state1[y*xsize+(x+xsize-1)%xsize];
	b=state1[y*xsize+x];
	c=state1[y*xsize+(x+1)%xsize];

	state2[y*xsize+x] = omnistate[c+3*b+9*a];
}

// performs update that moves cars of type 2.
__global__ void step2(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	){
	int x = threadIdx.x;
	int y = blockIdx.x;

	char a, b, c;
	b=state2[y*xsize+x];
	a=state2[x+((y+ysize-1)%ysize)*xsize];
	c=state2[x+((y+1)%ysize)*xsize];

	state1[y*xsize+x]  = omnistate[c+3*b+9*a];
}

// host wrapper that calls kernels step1 and step2
__host__ void h_step(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	){
	step1<<<ysize, xsize>>>(state1, state2, xsize, ysize);
	step2<<<ysize, xsize>>>(state1, state2, xsize, ysize);
}

/////////////////////////////////////////////////////////////

// performs update that moves cars of type 1, while computing the mobility (number of cars that move)
__global__ void speed1(
	char *state1,
	char *state2,
    char *mobility,
	int xsize,
	int ysize
	){
	int x = threadIdx.x;
	int y = blockIdx.x;

	char a, b, c;
	a=state1[y*xsize+(x+xsize-1)%xsize];
	b=state1[y*xsize+x];
	c=state1[y*xsize+(x+1)%xsize];

	state2[y*xsize+x] = omnistate[c+3*b+9*a];
    mobility[y*xsize+x] += (omnistate[c+3*b+9*a]!=0 && b==0)?1:0;
}

// performs update that moves cars of type 2, while computing the mobility (number of cars that move)
__global__ void speed2(
	char *state1,
	char *state2,
    char *mobility,
	int xsize,
	int ysize
	){
	int x = threadIdx.x;
	int y = blockIdx.x;

	char a, b, c;
	b=state2[y*xsize+x];
	a=state2[x+((y+ysize-1)%ysize)*xsize];
	c=state2[x+((y+1)%ysize)*xsize];

	state1[y*xsize+x]  = omnistate[c+3*b+9*a];
    mobility[y*xsize+x] += (omnistate[c+3*b+9*a]!=0 && b==0)?1:0;
}

// host wrapper that calls speed1 and speed2
__host__ double h_speed(
	char *state1,
	char *state2,
	int xsize,
	int ysize,
    double density
	){
    char 
        *mobility,
        *state1copy,
        *state2copy;
    double v;
    cudaMalloc((void**)&mobility,  sizeof(char)*xsize*ysize);
    cudaMalloc((void**)&state1copy,sizeof(char)*xsize*ysize);
    cudaMalloc((void**)&state2copy,sizeof(char)*xsize*ysize);

    cudaMemcpy(state1copy,state1, sizeof(char)*xsize*ysize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(state2copy,state2, sizeof(char)*xsize*ysize, cudaMemcpyDeviceToDevice);

    cudaMemset(mobility, 0, sizeof(char)*xsize*ysize);
	speed1<<<ysize, xsize>>>(state1copy, state2copy, mobility, xsize, ysize);
	speed2<<<ysize, xsize>>>(state1copy, state2copy, mobility, xsize, ysize);
    thrust::device_ptr<char>dev_ptr(mobility);
    v = (double)(thrust::reduce(dev_ptr,dev_ptr+xsize*ysize,0,thrust::plus<long>())) / (2*((int)(density*xsize*ysize*0.5)));

    cudaFree(mobility);
    cudaFree(state1copy);
    cudaFree(state2copy);


    return v;
}