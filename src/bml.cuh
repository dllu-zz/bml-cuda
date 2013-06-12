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
#ifndef __BIHAM_MIDDLETON_LEVINE_CUH__
#define __BIHAM_MIDDLETON_LEVINE_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// omnistate is the cellular automaton rule. It is in base 3 and is analogous to rulestrings like Rule 110.
// see: http://dllu.net/bml/ for more information on the rulestring. 
__constant__ char  omnistate[27] = {0,0,0,0,2,2,1,1,1,2,2,2,0,2,2,1,1,1,0,0,0,0,2,2,1,1,1};

__host__ void h_step(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	);
__host__ double h_speed(
	char *state1,
	char *state2,
	int xsize,
	int ysize,
    double density
	);

__global__ void step1(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	);
__global__ void step2(
	char *state1,
	char *state2,
	int xsize,
	int ysize
	);
__global__ void speed1(
	char *state1,
	char *state2,
    char *mobility,
	int xsize,
	int ysize
	);
__global__ void speed2(
	char *state1,
	char *state2,
    char *mobility,
	int xsize,
	int ysize
	);

#endif