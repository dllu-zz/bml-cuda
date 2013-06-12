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
#ifndef __BIHAM_MIDDLETON_LEVINE_H__
#define __BIHAM_MIDDLETON_LEVINE_H__
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class bml{
public:
	bml(int x = 100, int y = 100, double d = 0.3);
	~bml();
	void reset();
	void run();
    double speed();
	void bmp(const char*name);
private:
	char
		*d_state1,
		*d_state2;
	int
		xsize,
		ysize;
    cudaError_t s;
	double density;
};

#endif