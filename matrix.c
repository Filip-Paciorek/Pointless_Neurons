#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
matrix* allocate_matrix(u32 rows, u32 columns)
{
/* A function which takes as arguments num of rows and columns and returns
 * a flattened 2D matrix space */
	matrix* mat = malloc(sizeof(*mat));

	if (mat == NULL)
		return NULL;
	mat->rows = rows;
	mat->columns = columns;
	mat->data = malloc(rows*columns*sizeof(*mat->data));
	if (mat->data == NULL)
	{
		free(mat);
		return NULL;
	}
	//Check if the memory was properly allocated
	return mat;
}

void free_matrix(matrix* mat)
{
/* A function freeing the matrix, important for the sake of symmetry and future-proofing*/ 
	free(mat->data);
	free(mat);
};


void print_matrix(matrix* mat)
{
	for (int i = 0; i < mat->rows; i++)
	{
		printf("\n");
		for (int j = 0; j < mat->columns; j++)
		{
			printf("%f, ", (mat->data)[(i*(mat->columns))+j]);
		}
	}
}
void fill_matrix(matrix* mat,int start, int end)
{	
/* A function that fills the flattened matrix with numbers in range start end*/ 
	int sum_of_data = mat->rows*mat->columns;
	for (int i = 0; i < sum_of_data;i++)
	{
		(mat->data)[i] = rand() % (end-start) + start;
	}
}

matrix* transpose_matrix(matrix* mat)
{
	u32 columns = mat->columns;
	u32 rows = mat->rows;
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

	if (mat1->columns != mat2->rows)
	{
		printf("Wrong matrice dimensions!");
		return NULL;
	}
	float mult_sum = 0;
	matrix* mult_mat = allocate_matrix(mat1->rows,mat2->columns);
	for (u32 i =0; i < mat1->rows; i++)
	{
		for (u32 j =0; j < mat2->columns; j++)
		{
			mult_sum = 0;
			for (u32 k = 0; k < mat1->columns; k++)
			{
				float a = mat1->data[i*mat1->columns +k];
				float b = mat2->data[k*mat2->columns + j];
				mult_sum += (float)a * (float)b;
			}
		mult_mat->data[(i*mat2->columns) + j] = mult_sum;
		}
	
	}

	return mult_mat;
}
