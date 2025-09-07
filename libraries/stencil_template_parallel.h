/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>


#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

typedef uint    vec2_t[2];
typedef double *restrict buffers_t[4];

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;



extern int inject_energy ( const int      ,
                           const int      ,
			   const vec2_t  *,
			   const double   ,
                                 plane_t *,
                           const vec2_t   );


extern int update_plane ( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                                plane_t * );


extern int get_total_energy( plane_t *,
                             double  * );

int initialize ( MPI_Comm *,
                 int       ,
		 int       ,
		 int       ,
		 char    **,
                 vec2_t   *,
                 vec2_t   *,                 
		 int      *,
                 int      *,
		 int      *,
		 int      *,
		 int      *,
		 int      *,
                 vec2_t  **,
                 double   *,
                 plane_t  *,
                 buffers_t * );


int memory_release (plane_t   * , buffers_t *);


int output_energy_stat ( int      ,
                         plane_t *,
                         double   ,
                         int      ,
                         MPI_Comm *);



inline int inject_energy ( const int      periodic,
                           const int      Nsources,
			   const vec2_t  *Sources,
			   const double   energy,
                                 plane_t *plane,
                           const vec2_t   N
                           )
{
    const uint register sizex = plane->size[_x_]+2;
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) )
    for (int s = 0; s < Nsources; s++)
        {
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];
            
            data[ IDX(x,y) ] += energy;
            
            if ( periodic )
                {
                    if ( (N[_x_] == 1)  )
                        {
                            if (x==1){
                            data[IDX(plane->size[_x_]+1,y)] += energy;
                        }

                            if (x==plane->size[_x_]){
                                data[IDX(0, y)] += energy;
                            }
                        }
                    
                    if ( (N[_y_] == 1) )
                        {
                            if (y==1){
                            data[IDX(x,plane->size[_y_]+1)] += energy;
                        }

                            if (y==plane->size[_y_]){
                                data[IDX(x, 0)] += energy;
                            
                        }
                }                
        }
    }
 #undef IDX
    
  return 0;
}

extern inline double stencil_computation(const double *restrict,
                                  const uint,
                                  const uint,
                                  const uint);

extern inline int update_inner_plane(const plane_t *,
                              plane_t *);

extern inline int update_border_plane(const int ,
                               const vec2_t ,
                               const plane_t *,
                               plane_t *);

extern void fill_buffers(buffers_t *, plane_t *, int *, int, vec2_t );

extern int MPI_calls(buffers_t *, vec2_t , int *, MPI_Comm, MPI_Request * );

extern void copy_received_halos(buffers_t *, plane_t *,
                         int *, int, vec2_t);

inline int update_plane ( const int      periodic, 
                          const vec2_t   N,         // the grid of MPI tasks
                          const plane_t *oldplane,
                                plane_t *newplane
                          )
    
{
    uint register fxsize = oldplane->size[_x_]+2;
    uint register fysize = oldplane->size[_y_]+2;
    
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) )
    
    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    //
    // HINT: in any case, this loop is a good candidate
    //       for openmp parallelization

    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;
    
    # pragma gcc unroll 8
    for (uint j = 1; j <= ysize; j++)
        for ( uint i = 1; i <= xsize; i++)
            {
                
                // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
                //       if this patch is at some border without periodic conditions;
                //       in that case it is assumed that the +-1 points are outside the
                //       plate and always have a value of 0, i.e. they are an
                //       "infinite sink" of heat
                
                // five-points stencil formula
                //
                // HINT : check the serial version for some optimization
                //
                new[ IDX(i,j) ] =
                    old[ IDX(i,j) ] * 0.5 + ( old[IDX(i-1, j)] + old[IDX(i+1, j)] +
                                              old[IDX(i, j-1)] + old[IDX(i, j+1)] ) * 0.125;
                
            }

    if ( periodic )
        {
            if (N[_x_] == 1) // all proc in a single column, called by every proc
        {
            // propagate the boundaries as needed

            for(int j = 0; j < ysize; j++){
                new[IDX(0, j+1)] = new[IDX(xsize, j+1)];
                new[IDX(xsize+1, j+1)] = new[IDX(1, j+1)];
            }
            // check the serial version
        }

        if (N[_y_] == 1)
        {

            for(int j = 0; j < xsize; j++){
                new[IDX(j+1, 0)] = new[IDX(j+1, ysize)];
                new[IDX(j+1, ysize+1)] = new[IDX(j+1, 1)];
            }
        }
        }

    
 #undef IDX
  return 0;
}



inline int get_total_energy( plane_t *plane,
                             double  *energy )
/*
 * NOTE: this routine a good candiadate for openmp
 *       parallelization
 */
{

    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    #pragma omp parallel for reduction(+:totenergy) schedule (static)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}

int dump ( const double *data, const uint size[2], const char *filename);



