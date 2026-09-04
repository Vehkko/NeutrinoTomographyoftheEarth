#include <mpi.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1;
    int size = -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor[MPI_MAX_PROCESSOR_NAME]{};
    int  name_length = 0;

    MPI_Get_processor_name(processor, &name_length);

    std::cout << "[MPI] rank " << rank << " / " << size << " on " << std::string(processor, name_length) << '\n';

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[OK] Intel MPI communication works\n";
    }

    MPI_Finalize();

    return 0;
}
