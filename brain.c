#include <stdio.h>
#include <stdlib.h>
#include "nn.h"




int main()
{
	matrix* mat = allocate_matrix(10,4);
	fill_matrix(mat,0,100);
	print_matrix(mat);
	free_matrix(mat);
	return 1;
}
