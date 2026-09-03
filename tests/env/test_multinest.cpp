#include <multinest.h>

#include <iostream>

struct Context {
    std::size_t evaluations = 0;
};

void loglike(
    double* cube,
    int&    ndim,
    int&    npars,
    double& loglike,
    void*   context) {
    (void)ndim;
    (void)npars;

    auto& ctx =
        *static_cast<Context*>(context);

    ++ctx.evaluations;

    const double x = cube[0];

    // Simple Gaussian likelihood around x = 0.5.
    const double z =
        (x - 0.5) / 0.1;

    loglike = -0.5 * z * z;
}

void dumper(
    int&     nSamples,
    int&     nlive,
    int&     nPar,
    double** physLive,
    double** posterior,
    double** paramConstr,
    double&  maxLogLike,
    double&  logZ,
    double&  INSlogZ,
    double&  logZerr,
    void*    context) {
    (void)nSamples;
    (void)nlive;
    (void)nPar;
    (void)physLive;
    (void)posterior;
    (void)paramConstr;
    (void)maxLogLike;
    (void)logZ;
    (void)INSlogZ;
    (void)logZerr;
    (void)context;
}

int main() {
    Context ctx;

    int ndims   = 1;
    int nPar    = 1;
    int nClsPar = 1;

    int pWrap[1] = {0};

#ifdef NT_TEST_MULTINEST_MPI
    constexpr int initMPI = 1;
    std::cout << "[TEST] MultiNest MPI\n";
#else
    constexpr int initMPI = 0;
    std::cout << "[TEST] MultiNest serial\n";
#endif

    nested::run(
        0,   // IS
        0,   // mmodal
        0,   // ceff
        20,  // nlive
        0.5, // tol
        0.8, // efr

        ndims,
        nPar,
        nClsPar,

        10,    // maxModes
        10,    // updInt
        -1e90, // Ztol

        "build/test_tmp/mnest-",

        -1, // seed
        pWrap,

        0, // fb
        0, // resume
        0, // outfile

        initMPI,

        -1e90, // logZero
        50,    // maxiter

        loglike,
        dumper,
        &ctx);

    std::cout
        << "[OK] MultiNest returned successfully"
        << " | evaluations="
        << ctx.evaluations
        << '\n';

    return 0;
}
