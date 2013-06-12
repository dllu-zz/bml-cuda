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
#include "bml.h"
#include "bml.cuh"

#define CATCH_ERROR s = cudaGetLastError(); if(s)printf("\n          Cuda Error: %s\n",cudaGetErrorString(s));

bml::bml(int x, int y, double d){
	xsize = x;
	ysize = y;
	density = d;
	reset();
}

bml::~bml(){
	cudaFree(d_state1);
	cudaFree(d_state2);
}

// initialise the state arrays with randomly placed cars. 
void bml::reset(){
	char * h_state = new char[xsize*ysize];
	int xn, yn;
	for(int i=0; i<xsize*ysize; i++){
		h_state[i] = 0;
	}
	for(int i=0; i<(int)(density*xsize*ysize*0.5); i++){
		do{
			xn=rand()%(xsize);
			yn=rand()%(ysize);
		}while(h_state[yn*xsize+xn]!=0);//find a random place that's empty
		h_state[yn*xsize+xn]=1;         //set that as a one
		do{
			xn=rand()%(xsize);
			yn=rand()%(ysize);
		}while(h_state[yn*xsize+xn]!=0);//find a random place that's empty
		h_state[yn*xsize+xn]=2;         //set that as a two
	}
	cudaMalloc((void**)&d_state1,sizeof(char)*xsize*ysize);
	cudaMalloc((void**)&d_state2,sizeof(char)*xsize*ysize);
	cudaMemcpy(d_state1,h_state, sizeof(char)*xsize*ysize, cudaMemcpyHostToDevice);

	delete[]h_state;
}

void bml::run(){
	h_step(d_state1,d_state2,xsize,ysize);
}

double bml::speed(){
    return h_speed(d_state1,d_state2,xsize,ysize,density);
}

// save a bitmap.
// The bmp function was derived from a program on Wikipedia at 
// https://en.wikipedia.org/wiki/User:Evercat/Buddhabrot.c
// the next 4 lines of comment are the original licensing information for that snippet
	// Written by User:Evercat
	// Released under the GNU Free Documentation License
	// or the GNU Public License, whichever you prefer:
	// November 23, 2004

void bml::bmp(const char*name){
	char * h_state = new char[xsize*ysize];
	cudaMemcpy(h_state,d_state1, sizeof(char)*xsize*ysize, cudaMemcpyDeviceToHost);
	unsigned int headers[13];
	FILE * outfile;
	int extrabytes;
	int paddedsize;
	int x, y, n;

	extrabytes = 4 - ((xsize * 3) % 4); 

	char filename[200];
	
	sprintf(filename, "%s x %d y %d.bmp", name, xsize, ysize);

	if (extrabytes == 4) extrabytes = 0;

	paddedsize = ((xsize * 3) + extrabytes) * ysize;

	// Headers...
	// Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".
                     
	headers[0]  = paddedsize + 54;      // bfSize (whole file size)
	headers[1]  = 0;                    // bfReserved (both)
	headers[2]  = 54;                   // bfOffbits
	headers[3]  = 40;                   // biSize
	headers[4]  = xsize;  // biWidth
	headers[5]  = ysize; // biHeight

	// Would have biPlanes and biBitCount in position 6, but they're shorts.
	// It's easier to write them out separately (see below) than pretend
	// they're a single int, especially with endian issues...

	headers[7]  = 0;                    // biCompression
	headers[8]  = paddedsize;           // biSizeImage
	headers[9]  = 0;                    // biXPelsPerMeter
	headers[10] = 0;                    // biYPelsPerMeter
	headers[11] = 0;                    // biClrUsed
	headers[12] = 0;                    // biClrImportant

	outfile = fopen(filename, "wb");

	//
	// Headers begin...
	// When printing ints and shorts, we write out 1 character at a time to avoid endian issues.
	//

	fprintf(outfile, "BM");

	for (n = 0; n <= 5; n++){
	   fprintf(outfile, "%c", headers[n] & 0x000000FF);
	   fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
	   fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
	   fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
	}

	// These next 4 characters are for the biPlanes and biBitCount fields.

	fprintf(outfile, "%c", 1);
	fprintf(outfile, "%c", 0);
	fprintf(outfile, "%c", 24);
	fprintf(outfile, "%c", 0);

	for (n = 7; n <= 12; n++){
	   fprintf(outfile, "%c", headers[n] & 0x000000FF);
	   fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
	   fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
	   fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
	}

	//
	// Headers done, now write the data...
	//
	for(y = ysize - 1; y >= 0; y--){
		for(x = 0; x <= xsize - 1; x++){
			if(h_state[y*xsize+x]==0){
				fprintf(outfile, "%c", 255); fprintf(outfile, "%c", 255); fprintf(outfile, "%c", 255);
			}else if(h_state[y*xsize+x]==1){
				fprintf(outfile, "%c", 0); fprintf(outfile, "%c", 0);   fprintf(outfile, "%c", 255);
			}else if(h_state[y*xsize+x]==2){
				fprintf(outfile, "%c", 255);   fprintf(outfile, "%c", 0);   fprintf(outfile, "%c", 0);
			}
		}
		if (extrabytes){     // See above - BMP lines must be of lengths divisible by 4.
			for (n = 1; n <= extrabytes; n++){
				fprintf(outfile, "%c", 0);
			}
		}
	}
	printf("file printed: %s\n", filename); 
	fclose(outfile);
	delete[]h_state;
}
