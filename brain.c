#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

matrix* transpose_matrix(matrix* mat);


int main()
{
	matrix* mat = allocate_matrix(4,3);
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
	u32 c1 = 0;
	u32 c2 = 0;
	matrix* mat_t = allocate_matrix(mat->columns,mat->rows);
	for (u32 i = 0; i < rows; i++)
	{
		for (u32 j = 0; j <columns; j++)
		{ 
			u32 original_index = (i*columns) +j;
			u32 transposed_index = (j*rows)+i;
			mat_t->data[transposed_index] = mat->data[original_index];
		}
	}
	return mat_t;
}
matrix* multiply_matrices(matrix* mat1, matrix* mat2)
{
	if (mat1->rows != mat2->columns && mat1->columns != mat2->rows)
	{
		printf("Wrong matrice dimensions!");
		return NULL;
	}
	u32 rows = mat1->rows;
	u32 columns = mat2->columns;
	matrix* mult_mat = allocate_matrix(rows,rows);
	
	for (u32 i = 0; i < rows; i++)
	{

		for (u32 j =0; j < columns; j++)
		{

		}

	}
}
