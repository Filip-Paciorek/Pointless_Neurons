#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

matrix* transpose_matrix(matrix* mat);


int main()
{
	matrix* mat = allocate_matrix(6,5);
	fill_matrix(mat,0,100);
	print_matrix(mat);
	printf("\n-------------\n"); 
	matrix* mat_t = transpose_matrix(mat);
	print_matrix(mat_t);
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
			mat_t->data[(rows*i) + j] = mat->data[(rows*j) + i];
		}
	}
	return mat_t;
}
