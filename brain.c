#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

matrix* transpose_matrix(matrix* mat);


int main()
{
	struct timespec start, end;
	matrix* mat = allocate_matrix(5,5);
	fill_matrix(mat,0,100);
	clock_gettime(CLOCK_MONOTONIC, &start);
	print_matrix(mat);
	clock_gettime(CLOCK_MONOTONIC,&end);
	i64 t1 = time_diff(start,end);
	printf("\n-------------\n");
	clock_gettime(CLOCK_MONOTONIC,&start);
	print_matrix2(mat);
	clock_gettime(CLOCK_MONOTONIC,&end);
	i64 t2 = time_diff(start,end);
	printf("Elapsed 1: %ld ns\n",t1); 
	printf("Elapsed 2: %ld ns\n",t2); 
	printf("Elapsed diff: %ld ns\n",t1 - t2); 
	matrix* mat_t = transpose_matrix(mat);
	//print_matrix(mat_t);
	free_matrix(mat);
	free_matrix(mat_t);
	return 1;
}


i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
matrix* transpose_matrix(matrix* mat)
{
	u32 columns = mat->columns;
	u32 rows = mat->rows;
	matrix* mat_t = allocate_matrix(mat->columns,mat->rows);
	for (u32 i = 0; i < columns; i++)
	{
		for (u32 j = 0; j <rows; j++)
		{ 
			mat_t->data[(rows*i) + j] = mat->data[(columns*i) + j];
		}
	}
	return mat_t;
}
