#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

// Функция для генерации случайных точек и подсчета попаданий в окружность
// Условия, что точка в квадрате: 0 <= x <= 2 && 0 <= y <= 2
// Условия, что точка в круге: (x - 1)^2 + (y - 1)^2 <= 1
long long monte_carlo_pi(long long num_points, int rank) {
	long long local_hits = 0;

	srand(time(NULL) + rank);

	for (long long i = 0; i < num_points; i++) {
		double x = (double)rand() / RAND_MAX * 2.0;
		double y = (double)rand() / RAND_MAX * 2.0;

		if ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) <= 1.0) {
			local_hits++;
		}
	}

	return local_hits;
}

void report_first_task(long long total_points, long long total_hits, double execution_time, int comm_sz, int rank) {
	if (rank == 0) {
		double pi_estimate = 4.0 * (double)total_hits / (double)total_points;

		printf("===============================================\n");
		printf("The Monte Carlo method for calculating PI\n");
		printf("===============================================\n");
		printf("Total number of points: %lld\n", total_points);
		printf("Number of processes: %d\n", comm_sz);
		printf("Number of hits in the circle: %lld\n", total_hits);
		printf("Calculated value of PI: %.10f\n", pi_estimate);
		printf("Exact value of PI: %.10f\n", M_PI);
		printf("Error rate: %.10f\n", fabs(pi_estimate - M_PI));
		printf("Execution time: %.6f seconds\n", execution_time);
		printf("===============================================\n");
	}

}

void first_task_execute(int my_rank, int comm_sz) {
	long long total_points = 100000000;
	long long points_per_process = total_points / comm_sz;

	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	long long local_hits = monte_carlo_pi(points_per_process, my_rank);
	long long total_hits = 0;
	MPI_Reduce(&local_hits, &total_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	double end_time = MPI_Wtime();
	double execution_time = end_time - start_time;

	report_first_task(total_points, total_hits, execution_time, comm_sz, my_rank);
}