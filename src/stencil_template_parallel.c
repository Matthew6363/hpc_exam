
/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */

#include "../libraries/stencil_template_parallel.h"

// ------------------------------------------------------------------
// ------------------------------------------------------------------

// every rank executes this

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD; // communicator
  int Rank, Ntasks;      // id and number of MPI tasks
  int neighbours[4];     // N/S/E/W neighbours

  int Niterations; // number of iterations to be performed
  int periodic;    // periodic boundary condition tag
  vec2_t S, N;     // S array with size of the global grid (x,y) and N (x,y) for the process grid

  int Nsources;
  int Nsources_local;    // nr of heat sources within each process
  vec2_t *sources_local; // array with local sources coordinates
  double energy_per_source;

  plane_t planes[2];    // two planes to be used in this simulation plane[OLD] and plane[NEW]
  buffers_t buffers[2]; // two sets of buffers for sending and receiving

  int output_energy_stat_perstep;

  /* initialize MPI envrionment */
  {
    int level_obtained; // thread support level obtained

    // NOTE: change MPI_FUNNELED if appropriate
    // allows to use mpi calls, and multithreading --> funneled, just the master can perform mpi calls
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    if (level_obtained < MPI_THREAD_FUNNELED)
    {
      printf("MPI_thread level obtained is %d instead of %d\n",
             level_obtained, MPI_THREAD_FUNNELED);
      MPI_Finalize();
      exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);        // get rank's id
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);      // get number of ranks
    MPI_Comm_dup(MPI_COMM_WORLD, &myCOMM_WORLD); // obtain a private communicator
  }

  /* argument checking and setting */
  int ret = initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
                       neighbours, &Niterations,
                       &Nsources, &Nsources_local, &sources_local, &energy_per_source,
                       &planes[0], &buffers[0]);

  // check if initialisation was succesful otherwise exit with error code 0
  if (ret)
  {
    printf("task %d is opting out with termination code %d\n",
           Rank, ret);

    MPI_Finalize();
    return 0;
  }

  int current = OLD;       // set OLD as reading plane
  double t1 = MPI_Wtime(); /* take wall-clock time */

  for (int iter = 0; iter < Niterations; ++iter)

  {
    MPI_Request reqs[8]; // array of requests for handling mpi calls
    for (int k = 0; k < 8; k++)
      reqs[k] = MPI_REQUEST_NULL; // initialize all requests to NULL

    /* new energy from sources */
    inject_energy(periodic, Nsources_local, sources_local, energy_per_source, &planes[current], N);

    /* -------------------------------------- */

    // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position
    fill_buffers(buffers, &planes[current], neighbours, periodic, N);

    // [B] perform the halo communications
    MPI_calls(buffers, planes[current].size, neighbours, myCOMM_WORLD, reqs); // perform MPI_calls for send and receive between neighbouring processes
    
    //update_inner_plane(&planes[current], &planes[!current]);
    MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);                                  // wait for all communications to be completed

    // [C] copy the haloes data
    copy_received_halos(buffers, &planes[current], neighbours, periodic, N); // fill halo regions with data received from neighbours

    /* --------------------------------------  */
    /* update grid points */
    //update_border_plane(periodic, N, &planes[current], &planes[!current]);

    update_plane(periodic, N, &planes[current], &planes[!current]);

    /* output if needed */
    if (output_energy_stat_perstep)
    {
      output_energy_stat(iter, &planes[!current], (iter + 1) * Nsources * energy_per_source, Rank, &myCOMM_WORLD);
    }

    // optional: dump data for visualization
    //commented out to avoid creation of too many files
    // char filename[100];
    // sprintf(filename, "src/data_parallel/%d_plane_%05d.bin", Rank, iter);
    // int dump_status = dump(planes[!current].data, planes[!current].size, filename);
    // if (dump_status != 0)
    // {
    //   fprintf(stderr, "Error in dump_status. Exit with %d\n", dump_status);
    // }

    /* swap plane indexes for the new iteration */
    current = !current;
  }

  t1 = MPI_Wtime() - t1;

  printf("---------Rank: %d \t Elapsed time:%.6f---------\n", Rank, t1);

  output_energy_stat(-1, &planes[!current], Niterations * Nsources * energy_per_source, Rank, &myCOMM_WORLD);

  memory_release(planes, buffers); // free allocated memory

  MPI_Finalize(); // finalize MPI environment
  // if (Rank == 0)
  //   printf("Total time: %f seconds\n", t1);
  return 0;
}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */

void fill_buffers(buffers_t *buffers, plane_t *plane, int *neighbours, int periodic, vec2_t N)
{
  const uint sizey = plane->size[_y_];
  const uint sizex = plane->size[_x_];
  #define IDX(i, j) ((j) * (sizex + 2) + (i))

  // direct point buffers to correct position within the plane (no memory allocation needed due contiguity)
  if (neighbours[NORTH] != MPI_PROC_NULL ){
    buffers[SEND][NORTH] = &plane->data[IDX(1, 1)];         // point to first element of first internal row
    buffers[RECV][NORTH] = &plane->data[IDX(1, 0)];         // point to firs element of first halo row
  }

  if (neighbours[SOUTH] != MPI_PROC_NULL ){
    buffers[SEND][SOUTH] = &plane->data[IDX(1, sizey)];     // point to first element of last internal row
    buffers[RECV][SOUTH] = &plane->data[IDX(1, sizey + 1)]; // point to first element of last halo row
  }

  // handle periodic boundary conditions for NORTH and SOUTH
  if (periodic && N[_y_] == 2)
  {
    buffers[RECV][NORTH] = &plane->data[IDX(1, sizey + 1)];
    buffers[RECV][SOUTH] = &plane->data[IDX(1, 0)];
  }

  // pack WEST and EAST send buffers (due to non-contiguity)
  if (neighbours[WEST] != MPI_PROC_NULL)
  {
    for (int j = 0; j < sizey; j++)
      buffers[SEND][WEST][j] = plane->data[IDX(1, j + 1)];
  }
  if (neighbours[EAST] != MPI_PROC_NULL)
  {
    for (int j = 0; j < sizey; j++)
      buffers[SEND][EAST][j] = plane->data[IDX(sizex, j + 1)];
  }
#undef IDX
}

// perform MPI_calls for send and receive between neighbouring processes
int MPI_calls(buffers_t *buffers, vec2_t size, int *neighbours, MPI_Comm comm, MPI_Request reqs[8])
{

  const uint sizex = size[_x_];
  const uint sizey = size[_y_];
  // post non-blocking receive and send requests
  if (neighbours[NORTH] != MPI_PROC_NULL)
  {
    MPI_Irecv(buffers[RECV][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 0, comm, &reqs[0]);
    MPI_Isend(buffers[SEND][NORTH], sizex, MPI_DOUBLE, neighbours[NORTH], 0, comm, &reqs[1]);
  }

  if (neighbours[SOUTH] != MPI_PROC_NULL)
  {
    MPI_Irecv(buffers[RECV][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 0, comm, &reqs[2]);
    MPI_Isend(buffers[SEND][SOUTH], sizex, MPI_DOUBLE, neighbours[SOUTH], 0, comm, &reqs[3]);
  }

  if (neighbours[EAST] != MPI_PROC_NULL)
  {
    MPI_Irecv(buffers[RECV][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 0, comm, &reqs[4]);
    MPI_Isend(buffers[SEND][EAST], sizey, MPI_DOUBLE, neighbours[EAST], 0, comm, &reqs[5]);
  }

  if (neighbours[WEST] != MPI_PROC_NULL)
  {
    MPI_Irecv(buffers[RECV][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 0, comm, &reqs[6]);
    MPI_Isend(buffers[SEND][WEST], sizey, MPI_DOUBLE, neighbours[WEST], 0, comm, &reqs[7]);
  }

  return 0;
}

inline double stencil_computation(const double *restrict old,
                                  const uint fxsize,
                                  const uint i,
                                  const uint j)
{
  const uint idx = j * fxsize + i;
  return old[idx] * 0.5 + (old[idx - 1] + old[idx + 1] +
                           old[idx - fxsize] + old[idx + fxsize]) * 0.125;
  }



void copy_received_halos(buffers_t *buffers, plane_t *plane,
                         int *neighbours, int periodic, vec2_t N)
{

  const uint sizey = plane->size[_y_];
  const uint sizex = plane->size[_x_];

#define IDX(i, j) ((j) * (sizex + 2) + i)

  // Copy WEST halo (from EAST neighbor)
  if (neighbours[WEST] != MPI_PROC_NULL)
  {
    for (int j = 0; j < sizey; j++)
    {
      plane->data[IDX(0, j + 1)] = buffers[RECV][WEST][j];
    }
  }

  // Copy EAST halo (from WEST neighbor)
  if (neighbours[EAST] != MPI_PROC_NULL)
  {
    for (int j = 0; j < sizey; j++)
    {
      plane->data[IDX(sizex + 1, j + 1)] = buffers[RECV][EAST][j];
    }
  }
  // handle periodic boundary conditions for west and east
  if (periodic && N[_x_] == 2)
  {
    for (int j = 0; j < sizey; j++)
    {
      plane->data[IDX(0, j + 1)] = buffers[RECV][EAST][j];
      plane->data[IDX(sizex + 1, j + 1)] = buffers[RECV][WEST][j];
    }
  }

#undef IDX
}

inline int update_inner_plane(const plane_t *oldplane,
                              plane_t *newplane)
{
  uint fxsize = oldplane->size[_x_] + 2;
  uint fysize = oldplane->size[_y_] + 2;

  uint xsize = oldplane->size[_x_];
  uint ysize = oldplane->size[_y_];

#define IDX(i, j) ((j) * fxsize + (i))

  double *restrict old = oldplane->data;
  double *restrict new = newplane->data;

#pragma omp parallel for collapse(2) schedule(static)
  for (uint j = 2; j <= ysize - 1; j++)
    for (uint i = 2; i <= xsize - 1; i++)
      new[IDX(i, j)] = stencil_computation(old, fxsize, i, j);

#undef IDX
  return 0;
}

inline int update_border_plane(const int periodic,
                               const vec2_t N,
                               const plane_t *oldplane,
                               plane_t *newplane)
{
  uint register fxsize = oldplane->size[_x_] + 2;
  uint register fysize = oldplane->size[_y_] + 2;

  uint register xsize = oldplane->size[_x_];
  uint register ysize = oldplane->size[_y_];

#define IDX(i, j) ((j) * fxsize + (i))

  double *restrict old = oldplane->data;
  double *restrict new = newplane->data;

#pragma omp parallel for schedule(static)
  for (uint j = 1; j <= ysize; j++)
  {
    new[IDX(1, j)] = stencil_computation(old, fxsize, 1, j);         // left border
    new[IDX(xsize, j)] = stencil_computation(old, fxsize, xsize, j); // right border
  }

#pragma omp parallel for schedule(static)
  for (uint i = 1; i <= xsize; i++)
  {
    new[IDX(i, 1)] = stencil_computation(old, fxsize, i, 1);         // top border
    new[IDX(i, ysize)] = stencil_computation(old, fxsize, i, ysize); // bottom border
  }

  // If periodic, wrap
  if (periodic)
  {
    if (N[_x_] == 1)
    {
      for (uint j = 1; j <= ysize; j++)
      {
        new[IDX(0, j)] = new[IDX(xsize, j)];     // left ghost <-- right inner boundary
        new[IDX(xsize + 1, j)] = new[IDX(1, j)]; // right ghost <-- left inner boundary
      }
    }

    if (N[_y_] == 1)
    {
      for (uint i = 0; i <= xsize + 1; i++)
      {
        new[IDX(i, 0)] = new[IDX(i, ysize)];     // bottom ghost <-- top inner boundary
        new[IDX(i, ysize + 1)] = new[IDX(i, 1)]; // top ghost <-- bottom inner boundary
      }
    }
  }

#undef IDX
  return 0;
}
/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */

uint simple_factorization(uint, int *, uint **);

int initialize_sources(int,
                       int,
                       MPI_Comm *,
                       uint[2],
                       int,
                       int *,
                       vec2_t **);

int memory_allocate(const int *,
                    const vec2_t,
                    buffers_t *,
                    plane_t *);

int initialize(MPI_Comm *Comm,
               int Me,        // the rank of the calling process
               int Ntasks,    // the total number of MPI ranks
               int argc,      // the argc from command line
               char **argv,   // the argv from command line
               vec2_t *S,     // the size of the plane
               vec2_t *N,     // two-uint array defining the MPI tasks' grid
               int *periodic, // periodic-boundary tag
               int *output_energy_stat,
               int *neighbours,  // four-int array that gives back the neighbours of the calling task
               int *Niterations, // how many iterations
               int *Nsources,    // how many heat sources
               int *Nsources_local,
               vec2_t **sources_local,
               double *energy_per_source, // how much heat per souresulte
               plane_t *planes,
               buffers_t *buffers)
{
  int halt = 0;
  int ret; // value to show if init was succesful
  int verbose = 0;

  // ··································································
  // set deffault values

  (*S)[_x_] = 10000;
  (*S)[_y_] = 10000;
  *periodic = 0;            // non-periodic boundary for default
  *Nsources = 4;            // defaul number of heat sources
  *Nsources_local = 0;      // number of local heat sources
  *sources_local = NULL;    // array with local sources coordinates
  *Niterations = 1000;      // default number of iterations
  *energy_per_source = 1.0; // default energy per source

  if (planes == NULL)
  {
    // manage the situation
    fprintf(stderr, "planes not allocated\n");
    return 1;
  }

  // initializes planes sizes to 0
  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;

  // assume no neighbours at start
  for (int i = 0; i < 4; i++)
    neighbours[i] = MPI_PROC_NULL;

  // initializes buffers to NULL, they are not allocated yet
  for (int b = 0; b < 2; b++)
    for (int d = 0; d < 4; d++)
      buffers[b][d] = NULL;

  // ··································································
  // process the commadn line
  //
  while (1)
  {
    int opt;
    while ((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:")) != -1)
    {
      switch (opt)
      {
      case 'x':
        (*S)[_x_] = (uint)atoi(optarg);
        break;

      case 'y':
        (*S)[_y_] = (uint)atoi(optarg);
        break;

      case 'e':
        *Nsources = atoi(optarg);
        break;

      case 'E':
        *energy_per_source = atof(optarg);
        break;

      case 'n':
        *Niterations = atoi(optarg);
        break;

      case 'o':
        *output_energy_stat = (atoi(optarg) > 0);
        break;

      case 'p':
        *periodic = (atoi(optarg) > 0);
        break;

      case 'v':
        verbose = atoi(optarg);
        break;

      case 'h':
      {
        if (Me == 0)
          printf("\nvalid options are ( values btw [] are the default values ):\n"
                 "-x    x size of the plate [10000]\n"
                 "-y    y size of the plate [10000]\n"
                 "-e    how many energy sources on the plate [4]\n"
                 "-E    how many energy sources on the plate [1.0]\n"
                 "-n    how many iterations [1000]\n"
                 "-p    whether periodic boundaries applies  [0 = false]\n\n");
        halt = 1;
      }
      break;

      case ':':
        printf("option -%c requires an argument\n", optopt);
        break;

      case '?':
        printf(" -------- help unavailable ----------\n");
        break;
      }
    }

    if (opt == -1)
      break;
  }

  if (halt)
    return 1;

  // ··································································
  /*
   * here we should check for all the parms being meaningful
   *
   */

  // ...

  if ((*S)[_x_] <= 0 || (*S)[_y_] <= 0)
  {
    if (Me == 0)
      fprintf(stderr, "Error: grid size must be a positive integer, using default values x = y = 10000\n");
    (*S)[_x_] = 10000;
    (*S)[_y_] = 10000;
  }

  if (*Nsources < 0)
  {
    if (Me == 0)
      fprintf(stderr, "Error: Nsources must be a positive integer, using default value Nsources = 4\n");
    *Nsources = 4;
  }

  if (Niterations <= 0)
  {
    if (Me == 0)
      fprintf(stderr, "Error: Niteratiions must be a positive integer, usinge default value = 1000\n");
    *Niterations = 1000;
  }

  if (verbose < 0)
    verbose = 0;

  // ··································································
  /*
   * find a suitable domain decomposition
   * very simple algorithm, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   */

  vec2_t Grid;
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_] / (*S)[_y_] : (double)(*S)[_y_] / (*S)[_x_]);
  int dimensions = 2 - (Ntasks <= ((int)formfactor + 1));

  if (dimensions == 1)
  {
    if ((*S)[_x_] >= (*S)[_y_])
      Grid[_x_] = Ntasks, Grid[_y_] = 1;
    else
      Grid[_x_] = 1, Grid[_y_] = Ntasks;
  }
  else
  {
    int Nf;
    uint *factors;
    uint first = 1;
    ret = simple_factorization(Ntasks, &Nf, &factors);

    for (int i = 0; (i < Nf) && ((Ntasks / first) / first > formfactor); i++)
      first *= factors[i];

    if ((*S)[_x_] > (*S)[_y_])
      Grid[_x_] = Ntasks / first, Grid[_y_] = first;
    else
      Grid[_x_] = first, Grid[_y_] = Ntasks / first;
  }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];

  // ··································································
  // my cooridnates in the grid of processors
  //
  int X = Me % Grid[_x_];
  int Y = Me / Grid[_x_];

  // ··································································
  // find my neighbours
  //

  if (Grid[_x_] > 1)
  {
    if (*periodic)
    {
      neighbours[EAST] = Y * Grid[_x_] + (Me + 1) % Grid[_x_];
      neighbours[WEST] = (X % Grid[_x_] > 0 ? Me - 1 : (Y + 1) * Grid[_x_] - 1);
    }

    else
    {
      neighbours[EAST] = (X < Grid[_x_] - 1 ? Me + 1 : MPI_PROC_NULL);
      neighbours[WEST] = (X > 0 ? (Me - 1) % Ntasks : MPI_PROC_NULL);
    }
  }

  if (Grid[_y_] > 1)
  {
    if (*periodic)
    {
      neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
      neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks;
    }

    else
    {
      neighbours[NORTH] = (Y > 0 ? Me - Grid[_x_] : MPI_PROC_NULL);
      neighbours[SOUTH] = (Y < Grid[_y_] - 1 ? Me + Grid[_x_] : MPI_PROC_NULL);
    }
  }

  // ··································································
  // the size of my patch
  //

  /*
   * every MPI task determines the size sx x sy of its own domain
   * REMIND: the computational domain will be embedded into a frame
   *         that is (sx+2) x (sy+2)
   *         the outern frame will be used for halo communication or
   */

  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  // old and new planes have the same size

  planes[OLD].size[0] = planes[NEW].size[0] = mysize[0];
  planes[OLD].size[1] = planes[NEW].size[1] = mysize[1];

  if (verbose > 0)
  {
    if (Me == 0)
    {
      printf("Tasks are decomposed in a grid %d x %d\n\n",
             Grid[_x_], Grid[_y_]);
      fflush(stdout);
    }

    MPI_Barrier(*Comm); // one rank at a time prints its info

    /*each rank shows:
     - its grid coordinates (X,Y) inside px x py
     - the neighbor rank IDs in the N/E/S/W directions */
    for (int t = 0; t < Ntasks; t++)
    {
      if (t == Me)
      {
        printf("Task %4d :: "
               "\tgrid coordinates : %3d, %3d\n"
               "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
               Me, X, Y,
               neighbours[NORTH], neighbours[EAST],
               neighbours[SOUTH], neighbours[WEST]);
        fflush(stdout);
      }

      MPI_Barrier(*Comm);
    }
  }

  // ··································································
  //
  ret = memory_allocate(neighbours, *N, buffers, planes); // allocate planes and communication buffers

  // ··································································
  //
  ret = initialize_sources(Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, sources_local); // allocate heat sources

  return 0;
}

uint simple_factorization(uint A, int *Nfactors, uint **factors)
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while (f < A)
  {
    while (_A_ % f == 0)
    {
      N++;
      _A_ /= f;
    }

    f++;
  }

  *Nfactors = N;
  uint *_factors_ = (uint *)malloc(N * sizeof(uint));

  N = 0;
  f = 2;
  _A_ = A;

  while (f < A)
  {
    while (_A_ % f == 0)
    {
      _factors_[N++] = f;
      _A_ /= f;
    }
    f++;
  }

  *factors = _factors_;
  return 0;
}

int initialize_sources(int Me,
                       int Ntasks,
                       MPI_Comm *Comm,
                       vec2_t mysize,
                       int Nsources,
                       int *Nsources_local,
                       vec2_t **sources)

{

  srand48(time(NULL) ^ Me);
  int *tasks_with_sources = (int *)malloc(Nsources * sizeof(int));

  if (Me == 0)
  {
    for (int i = 0; i < Nsources; i++)
      tasks_with_sources[i] = (int)lrand48() % Ntasks;
  }

  MPI_Bcast(tasks_with_sources, Nsources, MPI_INT, 0, *Comm);

  int nlocal = 0;
  for (int i = 0; i < Nsources; i++)
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;

  if (nlocal > 0)
  {
    vec2_t *restrict helper = (vec2_t *)malloc(nlocal * sizeof(vec2_t));
    for (int s = 0; s < nlocal; s++)
    {
      helper[s][_x_] = 1 + lrand48() % mysize[_x_];
      helper[s][_y_] = 1 + lrand48() % mysize[_y_];
    }

    *sources = helper;
  }

  free(tasks_with_sources);

  return 0;
}

int memory_allocate(const int *neighbours,
                    const vec2_t N,
                    buffers_t *buffers_ptr,
                    plane_t *planes_ptr)

{
  /*
    here you allocate the memory buffers that you need to
    (i)  hold the results of your computation
    (ii) communicate with your neighbours

    The memory layout that I propose to you is as follows:

    (i) --- calculations
    you need 2 memory regions: the "OLD" one that contains the
    results for the step (i-1)th, and the "NEW" one that will contain
    the updated results from the step ith.

    Then, the "NEW" will be treated as "OLD" and viceversa.

    These two memory regions are indexed by *plate_ptr:

    planew_ptr[0] ==> the "OLD" region
    plamew_ptr[1] ==> the "NEW" region


    (ii) --- communications

    you may need two buffers (one for sending and one for receiving)
    for each one of your neighnours, that are at most 4:
    north, south, east amd west.

    To them you need to communicate at most mysizex or mysizey
    daouble data.

    These buffers are indexed by the buffer_ptr pointer so
    that

    (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
    (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions

    --->> Of course you can change this layout as you prefer

   */

  if (planes_ptr == NULL)
  // an invalid pointer has been passed
  // manage the situation
  {
    fprintf(stderr, "Error: invalid pointer to planes structure\n");
    return 4;
  }

  if (buffers_ptr == NULL)
  // an invalid pointer has been passed
  // manage the situation
  {
    fprintf(stderr, "Error: invalid pointer to buffers\n");
    return 5;
  }

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame
  // that will contains data form neighbouring MPI tasks
  unsigned int frame_size = (planes_ptr[OLD].size[_x_] + 2) * (planes_ptr[OLD].size[_y_] + 2);

  planes_ptr[OLD].data = (double *)malloc(frame_size * sizeof(double));
  if (planes_ptr[OLD].data == NULL)
  {
    // manage malloc fail
    fprintf(stderr, "Error: malloc failed for plane data\n");
    return 6;
  }
  memset(planes_ptr[OLD].data, 0, frame_size * sizeof(double));

  planes_ptr[NEW].data = (double *)malloc(frame_size * sizeof(double));
  if (planes_ptr[NEW].data == NULL)
  {
    fprintf(stderr, "Error: malloc failed for plane data\n");
    free(planes_ptr[OLD].data);
    return 7;
  }

  memset(planes_ptr[NEW].data, 0, frame_size * sizeof(double));

  // ··················································
  // buffers for north and south communication
  // are not really needed
  //
  // in fact, they are already contiguous, just the
  // first and last line of every rank's plane
  //
  // you may just make some pointers pointing to the
  // correct positions
  //

  // or, if you prefer, just go on and allocate buffers
  // also for north and south communications

  // ··················································
  // allocate buffers
  //

  uint sizex = planes_ptr[OLD].size[_x_];
  uint sizey = planes_ptr[OLD].size[_y_];

  #define IDX(i, j) ((j) * (sizex + 2) + (i))

  // pointers are already initialized to NULL
  // allocate only if there is a neighbour
  // and free allocated memory if something goes wrong

  if (neighbours[WEST] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][WEST] = (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
    if (buffers_ptr[SEND][WEST] == NULL)
    {
      fprintf(stderr, "Error: malloc failed for buffer send west\n");
      return 7;
    }
  }

  buffers_ptr[RECV][WEST] = (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  if (buffers_ptr[RECV][WEST] == NULL)
  {
    fprintf(stderr, "Error: malloc failed for buffer recv west\n");
    free(buffers_ptr[SEND][WEST]);
    return 8;
  }

  if (neighbours[EAST] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][EAST] = (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));
    if (buffers_ptr[SEND][EAST] == NULL)
    {
      fprintf(stderr, "Error: malloc failed for buffer send east\n");
      return 9;
    }
  }

  buffers_ptr[RECV][EAST] = (double *)malloc(planes_ptr[OLD].size[_y_] * sizeof(double));

  if (buffers_ptr[RECV][EAST] == NULL)
  {
    fprintf(stderr, "Error: malloc failed for buffer recv east\n");
    free(buffers_ptr[SEND][EAST]);
    return 10;
  }

#undef IDX
  return 0;
}

// ··················································

// free all allocated memory structures

int memory_release(plane_t *planes, buffers_t *buffers)


{
  if (planes != NULL)
  {
    if (planes[OLD].data != NULL)
      free(planes[OLD].data);

    if (planes[NEW].data != NULL)
      free(planes[NEW].data);
  }

  if (buffers != NULL)
  {
    if (buffers[SEND][WEST] != NULL)
      free(buffers[SEND][WEST]);

    if (buffers[RECV][WEST] != NULL)
      free(buffers[RECV][WEST]);

    if (buffers[SEND][EAST] != NULL)
      free(buffers[SEND][EAST]);

    if (buffers[RECV][EAST] != NULL)
      free(buffers[RECV][EAST]);
  }

  return 0;
}



int output_energy_stat(int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm)
{

  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy(plane, &system_energy);

  // each rank contributes to the global enery separately
  MPI_Reduce(&system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm);

  if (Me == 0)
  {
    if (step >= 0)
      printf(" [ step %4d ] ", step);
    fflush(stdout);

    printf("total injected energy is %g, "
           "system energy is %g "
           "( in avg %g per grid point)\n",
           budget,
           tot_system_energy,
           tot_system_energy / (plane->size[_x_] * plane->size[_y_]));
  }

  return 0;
}

//dump function for data visualization:
//creidts: Davide Zorzetto
// int dump(const double *data, const uint size[2], const char *filename)
// {
//   if ((filename != NULL) && (filename[0] != '\0'))
//   {
//     FILE *outfile = fopen(filename, "w");
//     if (outfile == NULL)
//       return 2;

//     float *array = (float *)malloc(size[0] * sizeof(float));

//     for (int j = 1; j <= size[1]; j++)
//     {
//       const double *restrict line = data + j * (size[0] + 2);
//       for (int i = 1; i <= size[0]; i++)
//       {
//         // int cut = line[i] < 100;
//         array[i - 1] = (float)line[i];
//       }
//       // printf("\n");
//       fwrite(array, sizeof(float), size[0], outfile);
//     }

//     free(array);

//     fclose(outfile);
//     return 0;
//   }

//   return 1;
// }