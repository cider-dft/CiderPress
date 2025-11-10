// CiderPress: Machine-learning based density functional theory calculations
// Copyright (C) 2024 The President and Fellows of Harvard College
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
//
// Author: Kyle Bystrom <kylebystrom@gmail.com>
//

#include "fblas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const double SQRT_PI = 1.77245385090551602729;
const double FOUR_PI = 4.0 * M_PI;

// Compute erfcx(x) = exp(x^2) * erfc(x) in a numerically stable way
// For large x, use asymptotic expansion to avoid overflow
static inline double erfcx(double x) {
    if (isnan(x)) return x;
    if (x == INFINITY) return 0.0;
    if (x == -INFINITY) return INFINITY;

    if (x >= 0.0) {
        if (x <= 5.0) {
            // Safe & accurate region for direct formula
            return exp(x*x) * erfc(x);
        } else {
            // Asymptotic: erfcx(x) ~ (1/(sqrt(pi)*x)) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6) + 105/(16x^8) - ...)
            double t = 1.0/(x*x);
            // 5-term Horner: 1 - 1/2 t + 3/4 t^2 - 15/8 t^3 + 105/16 t^4
            double s = ((((105.0/16.0)*t - 15.0/8.0)*t + 3.0/4.0)*t - 0.5)*t + 1.0;
            return (s / (SQRT_PI * x));
        }
    } else {
        // Identity: erfcx(-x) = 2*exp(x^2) - erfcx(x)
        double ax = -x;
        if (ax <= 5.0) {
            return exp(x*x) * erfc(x); // definition is fine here
        } else {
            // Use positive-branch asymptotic via the identity
            // Note: for very large negative x, the true value overflows (it grows ~ 2*e^{x^2})
            double pos = erfcx(ax);          // recurse to +x branch
            double ex2 = exp(x*x);
            if (!isfinite(ex2)) return INFINITY;
            return fma(2.0, ex2, -pos);      // 2*e^{x^2} - erfcx(|x|)
        }
    }
}

#define CIDER_FEAT_R0_GAUSSIAN 0
#define FILL_CIDER_R0_GAUSSIAN(ind)                                            \
    /* if (g == 0 && a == 0) printf("  -> Executing SE (R0_GAUSSIAN) kernel\n"); */ \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    p[ind] = pi32 * tmp * tmp1;                                                \
    dp[ind] = -1.5 * p[ind] * tmp;                                             \
    /* if (g == 0 && a == 0) printf("     SE result: p[0]=%f, exp_g[0]=%f, alphas[0]=%f\n", p[ind], exp_g[g], alphas[a]); */
#define CIDER_FEAT_R2_GAUSSIAN 1
#define FILL_CIDER_R2_GAUSSIAN(ind)                                            \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    tmp2 = 1.5 * pi32 * tmp * tmp * tmp1;                                      \
    p[ind] = tmp2 * exp_g[g];                                                  \
    dp[ind] = -2.5 * p[ind] * tmp + tmp2;
#define CIDER_FEAT_R4_GAUSSIAN 2
#define FILL_CIDER_R4_GAUSSIAN(ind)                                            \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp1 = sqrt(tmp);                                                          \
    tmp2 = 3.75 * pi32 * tmp * tmp * tmp * tmp1 * exp_g[g];                    \
    p[ind] = tmp2 * exp_g[g];                                                  \
    dp[ind] = -3.5 * p[ind] * tmp + 2 * tmp2;
#define CIDER_FEAT_ERF_GAUSSIAN 3
#define FILL_CIDER_ERF_GAUSSIAN(ind)                                           \
    tmp = 1.0 / (exp_g[g] + alphas[a]);                                        \
    tmp2 = 1.0 / sqrt(1 + extra_args[0] * exp_g[g] * tmp);                     \
    tmp1 = sqrt(tmp);                                                          \
    p[ind] = pi32 * tmp * tmp1 * tmp2;                                         \
    dp[ind] = -1.5 * p[ind] * tmp;                                             \
    tmp2 = tmp2 * tmp;                                                         \
    dp[ind] -= 0.5 * p[ind] * tmp2 * tmp2 * extra_args[0] * alphas[a];
// Kernel: g(a, r) = 1 / (a r^2 + 1)
#define CIDER_FEAT_RINV2_GAUSSIAN 4
#define FILL_CIDER_RINV2_GAUSSIAN(ind)                                     \
    double aval = exp_g[g];                                                \
    double alpha = alphas[a];                                              \
    double aval_safe = fmax(aval, 1e-20);                                  \
    double alpha_safe = fmax(alpha, 1e-20);                                \
    double sqrt_alpha = sqrt(alpha_safe);                                  \
    double sqrt_aval = sqrt(aval_safe);                                    \
    double a_pow_1_5 = aval_safe * sqrt_aval;                              \
    double a_pow_2 = aval_safe * aval_safe;                                \
    double a_pow_2_5 = a_pow_2 * sqrt_aval;                                \
    double a_pow_3 = a_pow_2 * aval_safe;                                  \
    double a_pow_3_5 = a_pow_3 * sqrt_aval;                                \
    double ratio = alpha_safe / aval_safe;                                 \
    double sqrt_ratio = sqrt(ratio);                                       \
    /* Print basic info and powers - COMMENTED TO AVOID SEGFAULT */        \
    /* printf("RINV2[g=%d,a=%d]: aval=%.3e, alpha=%.3e, ratio=%.3e, sqrt_ratio=%.3e\n", \
           g, a, aval, alpha, ratio, sqrt_ratio);                          \
    printf("  Powers: a^1.5=%.3e, a^2=%.3e, a^2.5=%.3e, a^3=%.3e, a^3.5=%.3e\n", \
           a_pow_1_5, a_pow_2, a_pow_2_5, a_pow_3, a_pow_3_5); */                                                                       \
    /* Calculate all three methods for exp*erfc term */                    \
    double exp_ratio = exp(ratio);                                         \
    double erfc_term = erfc(sqrt_ratio);                                   \
    /* Method 1: No fix (original - causes NaN) */                         \
    double exp_erfc_nofix = exp_ratio * erfc_term;                         \
    /* Method 2: Zero fix (sets to 0 when overflow) */                     \
    double exp_erfc_zerofix = isinf(exp_ratio) ? 0.0 : exp_ratio * erfc_term; \
    /* Method 3: erfcx fix (numerically stable) */                         \
    /* Note: erfcx(x) = exp(x^2)*erfc(x), so erfcx(sqrt_ratio) = exp(ratio)*erfc(sqrt_ratio) */ \
    double erfcx_val = erfcx(sqrt_ratio);                                  \
    double exp_erfc_erfcx = erfcx_val;  /* erfcx already gives us exp(ratio)*erfc(sqrt_ratio) */ \
    /* Calculate Term 1 (non-erfcx term) - same for all methods */         \
    double term1_p = SQRT_PI / (2.0 * aval_safe * sqrt_alpha);             \
    /* Calculate Term 2 for each method (erfcx-dependent term) */          \
    double term2_p_nofix = (M_PI * exp_erfc_nofix) / (2.0 * a_pow_1_5);    \
    double term2_p_zerofix = (M_PI * exp_erfc_zerofix) / (2.0 * a_pow_1_5); \
    double term2_p_erfcx = (M_PI * exp_erfc_erfcx) / (2.0 * a_pow_1_5);    \
    /* Final p = 4π * (term1 - term2) */                                   \
    double p_nofix = FOUR_PI * (term1_p - term2_p_nofix);                  \
    double p_zerofix = FOUR_PI * (term1_p - term2_p_zerofix);              \
    double p_erfcx = FOUR_PI * (term1_p - term2_p_erfcx);                  \
    p[ind] = p_erfcx;  /* Use erfcx method */                              \
    /* Print detailed comparison - COMMENTED TO AVOID SEGFAULT */          \
    /* printf("RINV2[g=%d,a=%d]: exp(ratio)=%.3e, erfc=%.3e, erfcx=%.3e\n",   \
           g, a, exp_ratio, erfc_term, erfcx_val);                         \
    printf("  Term1=%.3e (same for all)\n", term1_p);                      \
    printf("  Term2: NOFIX=%.3e, ZEROFIX=%.3e, ERFCX=%.3e\n",              \
           term2_p_nofix, term2_p_zerofix, term2_p_erfcx);                 \
    printf("  p[%d]: NOFIX=%.3e (nan=%d), ZEROFIX=%.3e, ERFCX=%.3e\n",     \
           ind, p_nofix, isnan(p_nofix), p_zerofix, p_erfcx); */              \
    /* Derivative terms */                                                 \
    double exp_erfc_term = exp_erfc_erfcx;                                 \
    double term1_dp = (M_PI * alpha_safe * exp_erfc_term) / (2.0 * a_pow_3_5); \
    double term2_dp = (3.0 * M_PI * exp_erfc_term) / (4.0 * a_pow_2_5);      \
    double term3_dp = (SQRT_PI * sqrt_alpha) / (2.0 * a_pow_3);            \
    double term4_dp = SQRT_PI / (2.0 * a_pow_2 * sqrt_alpha);              \
    dp[ind] = FOUR_PI * (term1_dp + term2_dp - term3_dp - term4_dp);
// Kernel: g(a, r) = (1 / (a r^2 + 1))^2
#define CIDER_FEAT_RINV4_GAUSSIAN 5
#define FILL_CIDER_RINV4_GAUSSIAN(ind)                                     \
    double aval = exp_g[g];                                                \
    double alpha = alphas[a];                                              \
    double aval_safe = fmax(aval, 1e-20);                                  \
    double alpha_safe = fmax(alpha, 1e-20);                                \
    double sqrt_alpha = sqrt(alpha_safe);                                  \
    double sqrt_aval = sqrt(aval_safe);                                    \
    double ratio = alpha_safe / aval_safe;                                 \
    double sqrt_ratio = sqrt(ratio);                                       \
    double a_plus_2alpha = aval_safe + 2.0 * alpha_safe;                   \
    double a_pow_0_5 = sqrt_aval;                                          \
    double a_pow_1_5 = aval_safe * a_pow_0_5;                              \
    double a_pow_2 = aval_safe * aval_safe;                                \
    double a_pow_2_5 = a_pow_2 * a_pow_0_5;                                \
    double a_pow_3_5 = a_pow_2_5 * aval_safe;                              \
    double a_pow_3 = a_pow_2 * aval_safe;                                  \
    /* Print basic info and powers - COMMENTED TO AVOID SEGFAULT */        \
    /* printf("RINV4[g=%d,a=%d]: aval=%.3e, alpha=%.3e, ratio=%.3e, sqrt_ratio=%.3e\n", \
           g, a, aval, alpha, ratio, sqrt_ratio);                          \
    printf("  Powers: a^0.5=%.3e, a^1.5=%.3e, a^2=%.3e, a^2.5=%.3e, a^3=%.3e, a^3.5=%.3e\n", \
           a_pow_0_5, a_pow_1_5, a_pow_2, a_pow_2_5, a_pow_3, a_pow_3_5);   \
    printf("  a_plus_2alpha=%.3e\n", a_plus_2alpha); */                                                                       \
    /* Calculate all three methods for exp*erfc term */                    \
    double exp_ratio = exp(ratio);                                         \
    double erfc_term = erfc(sqrt_ratio);                                   \
    /* Method 1: No fix (original - causes NaN) */                         \
    double exp_erfc_nofix = exp_ratio * erfc_term;                         \
    /* Method 2: Zero fix (sets to 0 when overflow) */                     \
    double exp_erfc_zerofix = isinf(exp_ratio) ? 0.0 : exp_ratio * erfc_term; \
    /* Method 3: erfcx fix (numerically stable) */                         \
    /* Note: erfcx(x) = exp(x^2)*erfc(x), so erfcx(sqrt_ratio) = exp(ratio)*erfc(sqrt_ratio) */ \
    double erfcx_val = erfcx(sqrt_ratio);                                  \
    double exp_erfc_erfcx = erfcx_val;  /* erfcx already gives us exp(ratio)*erfc(sqrt_ratio) */ \
    /* Calculate numerators for each method */                             \
    /* Term 1 numerator (erfcx-dependent) - different for each method */   \
    double term1_num_nofix = M_PI * exp_erfc_nofix * a_plus_2alpha;        \
    double term1_num_zerofix = M_PI * exp_erfc_zerofix * a_plus_2alpha;    \
    double term1_num_erfcx = M_PI * exp_erfc_erfcx * a_plus_2alpha;        \
    /* Term 2 numerator (non-erfcx term) - same for all methods */         \
    double term2_num = 2.0 * SQRT_PI * sqrt_aval * sqrt_alpha;             \
    /* Common denominator */                                               \
    double common_den = 4.0 * a_pow_2_5;                                   \
    /* Final p = 4π * (term1_num - term2_num) / common_den */              \
    /* double p_nofix = FOUR_PI * (term1_num_nofix - term2_num) / common_den; \ */
    /* double p_zerofix = FOUR_PI * (term1_num_zerofix - term2_num) / common_den; \ */
    double p_erfcx = FOUR_PI * (term1_num_erfcx - term2_num) / common_den; \
    p[ind] = p_erfcx;  /* Use erfcx method */                              \
    /* Print detailed comparison - COMMENTED TO AVOID SEGFAULT */          \
    /* printf("RINV4[g=%d,a=%d]: exp(ratio)=%.3e, erfc=%.3e, erfcx=%.3e\n",   \
           g, a, exp_ratio, erfc_term, erfcx_val);                         \
    printf("  Term1_num: NOFIX=%.3e, ZEROFIX=%.3e, ERFCX=%.3e\n",          \
           term1_num_nofix, term1_num_zerofix, term1_num_erfcx);           \
    printf("  Term2_num=%.3e (same for all), Common_den=%.3e\n",           \
           term2_num, common_den);                                         \
    printf("  p[%d]: NOFIX=%.3e (nan=%d), ZEROFIX=%.3e, ERFCX=%.3e\n",     \
           ind, p_nofix, isnan(p_nofix), p_zerofix, p_erfcx); */              \
    /* Derivative terms */                                                 \
    double exp_erfc_term = exp_erfc_erfcx;                                 \
    double term1_d1 = SQRT_PI * sqrt_alpha * a_plus_2alpha / a_pow_1_5;    \
    double term2_d1 = M_PI * alpha_safe * exp_erfc_term * a_plus_2alpha / a_pow_2; \
    double term3_d1 = SQRT_PI * sqrt_alpha / a_pow_0_5;                    \
    double term4_d1 = M_PI * exp_erfc_term;                                \
    double num_d1 = term1_d1 - term2_d1 - term3_d1 + term4_d1;             \
    double den_d1 = common_den;                                            \
    double num_d2 = term1_num_erfcx - term2_num;                           \
    double den_d2 = 8.0 * a_pow_3_5;                                       \
    dp[ind] = FOUR_PI * ( (num_d1 / den_d1) - (5.0 * num_d2 / den_d2) );

#define CIDER_GQ_LOOP(FEATNAME)                                                \
    for (g = 0; g < ngrids; g++) {                                             \
        p = p_ga + g * nalpha;                                                 \
        dp = dp_ga + g * nalpha;                                               \
        for (a = 0; a < nalpha; a++) {                                         \
            FILL_CIDER_##FEATNAME(a);                                          \
        }                                                                      \
    }                                                                          \
    break;
#define CIDER_QG_LOOP(FEATNAME)                                                \
    for (g = 0; g < ngrids; g++) {                                             \
        FILL_CIDER_##FEATNAME(g);                                              \
    }                                                                          \
    break;
#define MC_EXPNT 12

/**
 * Fills the coefficients p_ga for auxiliary basis dcomposition.
 *
 * p_ga : coefficient array nalpha x ngrids
 * dp_ga : derivatives of coefficients wrt exponent alpha
 * exp_g : exponent array
 * alphas : control points
 * ngrids : number of realspace grids
 * nalpha : number of control points
 */
void cider_coefs_gto_gq(double *p_ga, double *dp_ga, double *exp_g,
                        double *alphas, int ngrids, int nalpha, int featid,
                        double *extra_args) {
    double pi32 = pow(4 * atan(1.0), 1.5);
    /* printf("DEBUG cider_coefs_gto_gq: featid=%d, ngrids=%d, nalpha=%d\n", featid, ngrids, nalpha); */
#pragma omp parallel
    {
        double tmp, tmp1, tmp2;
        int g, a;
        double *p, *dp;
        switch (featid) {
        case CIDER_FEAT_R0_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R0_GAUSSIAN);
        case CIDER_FEAT_R2_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R2_GAUSSIAN);
        case CIDER_FEAT_R4_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(R4_GAUSSIAN);
        case CIDER_FEAT_ERF_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(ERF_GAUSSIAN);
        case CIDER_FEAT_RINV2_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(RINV2_GAUSSIAN);
        case CIDER_FEAT_RINV4_GAUSSIAN:
#pragma omp for
            CIDER_GQ_LOOP(RINV4_GAUSSIAN);
        default:
            printf("INTERNAL CIDER ERROR\n");
        }
    }
}

void cider_coefs_gto_qg(double *p_ag, double *dp_ag, double *exp_g,
                        double *alphas, int ngrids, int nalpha, int featid,
                        double *extra_args) {
    double pi32 = pow(4 * atan(1.0), 1.5);
    /* printf("DEBUG cider_coefs_gto_qg: featid=%d\n", featid); */
#pragma omp parallel
    {
        double tmp, tmp1, tmp2;
        int g, a;
        double *p, *dp;
        for (a = 0; a < nalpha; a++) {
            p = p_ag + a * ngrids;
            dp = dp_ag + a * ngrids;
            switch (featid) {
            case CIDER_FEAT_R0_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R0_GAUSSIAN);
            case CIDER_FEAT_R2_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R2_GAUSSIAN);
            case CIDER_FEAT_R4_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(R4_GAUSSIAN);
            case CIDER_FEAT_ERF_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(ERF_GAUSSIAN);
            case CIDER_FEAT_RINV2_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(RINV2_GAUSSIAN);
            case CIDER_FEAT_RINV4_GAUSSIAN:
#pragma omp for
                CIDER_QG_LOOP(RINV4_GAUSSIAN);
            default:
                printf("INTERNAL CIDER ERROR\n");
            }
        }
    }
}

/**
 * Forward pass coefs for version k descriptors
 */
void cider_coefs_vk1_gq(double *p_ga, double *dp_ga, double *exp_g,
                        double *alphas, int ngrids, int nalpha) {
#pragma omp parallel
    {
        double *tmps = (double *)malloc(nalpha * sizeof(double));
        int g, a;
        for (a = 0; a < nalpha; a++) {
            tmps[a] = -1.5 / alphas[a];
        }
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            for (a = 0; a < nalpha; a++) {
                p_ga[g * nalpha + a] = exp(tmps[a] * exp_g[g]);
                dp_ga[g * nalpha + a] = tmps[a] * p_ga[g * nalpha + a];
            }
        }
    }
}

/**
 * Forward pass coefs for version k descriptors
 */
void cider_coefs_vk1_qg(double *p_ag, double *dp_ag, double *exp_g,
                        double *alphas, int ngrids, int nalpha) {
#pragma omp parallel
    {
        double tmp;
        int g, a;
        for (a = 0; a < nalpha; a++) {
            tmp = -1.5 / alphas[a];
#pragma omp for
            for (g = 0; g < ngrids; g++) {
                p_ag[a * ngrids + g] = exp(tmp * exp_g[g]);
                dp_ag[a * ngrids + g] = tmp * p_ag[a * ngrids + g];
            }
        }
    }
}

void cider_ind_etb(double *di_g, double *derivi_g, double *exp_g, int ngrids,
                   double alpha0, double lambd) {
#pragma omp parallel
    {
        int g;
        double ratio = 1.0 / log(lambd);
        double inva = 1.0 / alpha0;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            derivi_g[g] = ratio / exp_g[g];
            di_g[g] = log(exp_g[g] * inva) * ratio;
        }
    }
}

void cider_ind_zexp(double *di_g, double *derivi_g, double *exp_g, int ngrids,
                    double alpha0, double lambd) {
#pragma omp parallel
    {
        int g;
        double ratio = 1.0 / log(lambd);
        double inva = 1.0 / alpha0;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            derivi_g[g] = ratio * inva / (exp_g[g] * inva + 1);
            di_g[g] = log(exp_g[g] * inva + 1) * ratio;
        }
    }
}

void cider_ind_clip(double *di_g, double *derivi_g, int sizem1, int ngrids) {
#pragma omp parallel
    {
        int g;
        double di;
        int cond;
        double dsize = sizem1 - 1e-10;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            di = di_g[g];
            cond = di > 0;
            derivi_g[g] = (cond ? derivi_g[g] : 0);
            di_g[g] = (cond ? di : 0);
            di = di_g[g];
            cond = di < sizem1;
            derivi_g[g] = (cond ? derivi_g[g] : 0);
            di_g[g] = (cond ? di : dsize);
        }
    }
}

void cider_coefs_spline_gq(double *p_ga, double *dp_ga, double *di_g,
                           double *w_iap, int ngrids, int nalpha,
                           double lambd) {
    /* Debug: Print first few spline coefficients and input values
    if (ngrids > 0) {
        fprintf(stderr, "DEBUG cider_coefs_spline_gq: ngrids=%d, nalpha=%d, lambd=%f\n", 
                ngrids, nalpha, lambd);
        fprintf(stderr, "  -> di_g[0]=%f, di_g[1]=%f (input exponents)\n", 
                di_g[0], ngrids > 1 ? di_g[1] : 0.0);
        fprintf(stderr, "  -> w_iap[0]=%f, w_iap[1]=%f (spline coefs)\n", 
                w_iap[0], w_iap[1]);
    } */
#pragma omp parallel
    {
        double di;
        int g, a, i;
        double *w_ap, *w_p, *p_a, *dp_a;
#pragma omp for
        for (g = 0; g < ngrids; g++) {
            p_a = p_ga + g * nalpha;
            dp_a = dp_ga + g * nalpha;
            i = (int)di_g[g];
            di = di_g[g] - i;
            w_ap = w_iap + i * 4 * nalpha;
            for (a = 0; a < nalpha; a++) {
                w_p = w_ap + 4 * a;
                p_a[a] = w_p[0] + di * (w_p[1] + di * (w_p[2] + di * w_p[3]));
                dp_a[a] = w_p[1] + di * (2 * w_p[2] + di * 3 * w_p[3]);
            }
            /* Debug: Print result for first grid point
            if (g == 0) {
                fprintf(stderr, "  -> Result p_a[0]=%f (first grid point, first alpha)\n", p_a[0]);
            } */
        }
    }
}

void cider_coefs_spline_qg(double *p_ag, double *dp_ag, double *di_g,
                           double *w_iap, int ngrids, int nalpha,
                           double lambd) {
#pragma omp parallel
    {
        double di;
        int g, a, i;
        double *w_p, *p_g, *dp_g;
        for (a = 0; a < nalpha; a++) {
            p_g = p_ag + a * ngrids;
            dp_g = dp_ag + a * ngrids;
#pragma omp for
            for (g = 0; g < ngrids; g++) {
                i = (int)di_g[g];
                di = di_g[g] - i;
                w_p = w_iap + (i * nalpha + a) * 4;
                p_g[g] = w_p[0] + di * (w_p[1] + di * (w_p[2] + di * w_p[3]));
                dp_g[g] = w_p[1] + di * (2 * w_p[2] + di * 3 * w_p[3]);
            }
        }
    }
}

inline double _expnt_sat_func(double x, double cut) {
    int m;
    double tot = 0;
    x = x / cut;
    double xm = 1;
    for (m = 1; m <= MC_EXPNT; m++) {
        xm *= x;
        tot += xm / m;
    }
    return cut * (1 - exp(-tot));
}

inline double _expnt_sat_deriv(double x, double cut) {
    int m;
    double tot = 0;
    double dtot = 0;
    x = x / cut;
    double xm = 1;
    for (m = 1; m <= MC_EXPNT; m++) {
        dtot += xm;
        xm *= x;
        tot += xm / m;
    }
    return exp(-tot) * dtot;
}

void smooth_cider_exponents(double *a, double **da, double amax, int ng,
                            int nd) {
#pragma omp parallel
    {
        double c, dc;
        int g, d;
#pragma omp for
        for (g = 0; g < ng; g++) {
            c = _expnt_sat_func(a[g], amax);
            dc = _expnt_sat_deriv(a[g], amax);
            a[g] = c;
            for (d = 0; d < nd; d++) {
                da[d][g] *= dc;
            }
        }
    }
}
