#include <stdio.h>
#include <mpi.h>
#include "firstTask.h"
#include "fourthTask.h"

int main() {

	int my_rank;
	int comm_sz;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Первое задание
	first_task_execute(my_rank, comm_sz);
	
	// Четвертое задание
	fourth_task_execute(my_rank, comm_sz); // для демонстрации работы
	fourth_task_report(my_rank, comm_sz); // для анализа работы алгоритма
	
	MPI_Finalize();

	return 0;
}
