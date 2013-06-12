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
#include <cstdio>
#include <ctime>
#include "bml.h"
int main(){
    int xsize = 128, ysize = 128;
    int limit = 1000000;
    double thresh = 0.9/(xsize*ysize);
    long long start = clock();
    for(double percent = 0.4; percent < 0.45; percent +=0.01){
        char mobilityfilename[200];
        sprintf(mobilityfilename, "mobility %f.txt", percent);
        FILE * mobilities = fopen(mobilityfilename,"w");
        for(int j=0; j<1000; j++){
	        srand ( time(NULL) );
	        bml test(xsize,ysize,percent);
            char filename[200];
            sprintf(filename,"%f %d start",percent,j);
            test.bmp(filename);
            bool runornot = 1;
	        for(int i=0; i<limit; i++){
		        if(runornot) test.run();
                if(i%1000==0 || i>=limit-8192){
                    double v = test.speed();
                    /*printf("%f\n",v);
                    char filename2[200];
                    sprintf(filename2,"run %d %f",i,v);
                    test.bmp(filename2);*/
                    if(i>=limit-8192) fprintf(mobilities, "%f ", v);
                    if(v<thresh||v>=1.0 - thresh) runornot = 0;
                }
	        }
            fprintf(mobilities, "\n");
        
            char filename2[200];
            sprintf(filename2,"%f %d stop %f",percent,j,test.speed());
            test.bmp(filename2);
        }
        fclose(mobilities);
    }
    long long stop = clock();
    printf("Calculation done, printing now... time elapsed: %f\n", (stop-start)/(double)CLOCKS_PER_SEC);
	scanf("%*x");
	return 0;
}