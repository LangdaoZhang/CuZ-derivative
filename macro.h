// #ifndef _MACRO_H
// #define _MACRO_H

#ifndef float
#define float float
#endif

#ifndef arr3
#define arr3(x,y,z) ((x)*ny*nz+(y)*nz+(z))
#endif

#ifndef square
#define square(x) ((x)*(x))
#endif

#ifndef _BLOCK_SZ
#define _BLOCK_SZ
#define BLOCKSZX 8
#define BLOCKSZY 8
#define BLOCKSZZ 8
#endif

#ifndef _INF
#define _INF
const float inf=1./0.;
#endif

// #ifndef _EPS
// #define _EPS
// const float eps=1e-20;
// #endif

// #endif