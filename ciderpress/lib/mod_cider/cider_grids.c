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
#include <stdlib.h>

void reduce_angc_to_ylm(double *theta_rlmq, double *y_glm, double *theta_gq,
                        int *rad_loc, int *ylm_loc, int nalpha, int nrad,
                        int ngrids, int nlm, int stride, int offset) {
    theta_gq = theta_gq + offset;
#pragma omp parallel
    {
        double ALPHA = 1.0;
        double BETA = 0.0;
        char NTRANS = 'N';
        char TRANS = 'T';
        int r, nw;
        double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
        for (r = 0; r < nrad; r++) {
            y_wlm = y_glm + ylm_loc[r] * nlm;
            nw = rad_loc[r + 1] - rad_loc[r];
            theta_lmq = theta_rlmq + r * nlm * nalpha;
            theta_wq = theta_gq + rad_loc[r] * stride;
            dgemm_(&NTRANS, &TRANS, &nalpha, &nlm, &nw, &ALPHA, theta_wq,
                   &stride, y_wlm, &nlm, &BETA, theta_lmq, &nalpha);
        }
    }
}

void reduce_ylm_to_angc(double *theta_rlmq, double *y_glm, double *theta_gq,
                        int *rad_loc, int *ylm_loc, int nalpha, int nrad,
                        int ngrids, int nlm, int stride, int offset) {
    theta_gq = theta_gq + offset;
#pragma omp parallel
    {
        double ALPHA = 1.0;
        double BETA = 0.0;
        char NTRANS = 'N';
        char TRANS = 'T';
        int r, nw;
        double *y_wlm, *theta_lmq, *theta_wq;
#pragma omp for schedule(dynamic, 4)
        for (r = 0; r < nrad; r++) {
            y_wlm = y_glm + ylm_loc[r] * nlm;
            nw = rad_loc[r + 1] - rad_loc[r];
            theta_lmq = theta_rlmq + r * nlm * nalpha;
            theta_wq = theta_gq + rad_loc[r] * stride;
            dgemm_(&NTRANS, &NTRANS, &nalpha, &nw, &nlm, &ALPHA, theta_lmq,
                   &nalpha, y_wlm, &nlm, &BETA, theta_wq, &stride);
        }
    }
}

// TODO move this
void VXC_r2vv10a(double *Fvec, double *Uvec, double *Wvec, double *vvcoords,
                 double *coords, double *Qp, double *Q, double *PRpW,
                 int vvngrids, int ngrids) {
#pragma omp parallel
    {
        double DX, DY, DZ, R2;
        double gp, g, gt, T, F, U, W;
        int i, j;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
            F = 0;
            U = 0;
            W = 0;
            for (j = 0; j < vvngrids; j++) {
                DX = vvcoords[j * 3 + 0] - coords[i * 3 + 0];
                DY = vvcoords[j * 3 + 1] - coords[i * 3 + 1];
                DZ = vvcoords[j * 3 + 2] - coords[i * 3 + 2];
                R2 = DX * DX + DY * DY + DZ * DZ;
                gp = R2 * Qp[j] + 1;
                g = R2 * Q[i] + 1;
                gt = g + gp;
                T = PRpW[j] / (g * gp * gt);
                F += T;
                T *= 1. / g + 1. / gt;
                U += T;
                W += T * R2;
            }
            Fvec[i] = F * -1.5;
            Uvec[i] = U;
            Wvec[i] = W;
        }
    }
}
