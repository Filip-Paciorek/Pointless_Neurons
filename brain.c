#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

matrix* transpose_matrix(matrix* mat);

int main()
{
	matrix* mat = allocate_matrix(4,3);
	fill_matrix(mat,0,100);
	print_matrix(mat);
	printf("-------------");
	matrix* mat_t = transpose_matrix(mat);
	free_matrix(mat);
	free_matrix(mat_t);
	return 1;
}



matrix* transpose_matrix(matrix* mat)
{
	u32 columns = mat->columns;
	matrix* mat_t = allocate_matrix(mat->rows,mat->columns);
	u32 num_of_data = (mat->rows)*(mat->columns);
	u32 k = 0;
	u32 z = 0;
	i32 item = 0;
	for (u32 i = 0; i < num_of_data;i++)
	{
		if (z == columns)
			z = 0;
		if (i != 0 && i % mat->columns == 0)
		{	k++;
			printf("\n");
		}
		item = mat->data[(k*mat->rows) + z];
		mat_t->data[i] = item;
		z++;
		printf("%f ,", mat_t->data[i]);
	}
	return mat_t;
}
