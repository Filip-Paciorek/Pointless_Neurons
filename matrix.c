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
