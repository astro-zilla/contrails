"""
Numerical flux schemes for the 1D compressible flow solver.

Optimized with vectorized HLLC flux computation.
"""

import numpy as np
from abc import ABC, abstractmethod

from .gas import GasProperties


class FluxScheme(ABC):
    """Abstract base class for numerical flux schemes."""

    @abstractmethod
    def compute_flux(self, UL: np.ndarray, UR: np.ndarray,
                     gas: GasProperties) -> np.ndarray:
        """
        Compute numerical flux at a face.

        Args:
            UL: Left state conservative variables (n_vars,)
            UR: Right state conservative variables (n_vars,)
            gas: Gas properties

        Returns:
            Numerical flux (n_vars,)
        """
        pass

    def compute_flux_vectorized(self, UL: np.ndarray, UR: np.ndarray,
                                gas: GasProperties) -> np.ndarray:
        """
        Compute numerical fluxes at all faces (vectorized).

        Args:
            UL: Left states (n_vars, n_faces)
            UR: Right states (n_vars, n_faces)
            gas: Gas properties

        Returns:
            Fluxes at all faces (n_vars, n_faces)
        """
        # Default implementation: loop over faces
        n_vars, n_faces = UL.shape
        F = np.zeros((n_vars, n_faces))
        for i in range(n_faces):
            F[:, i] = self.compute_flux(UL[:, i], UR[:, i], gas)
        return F


class HLLCFlux(FluxScheme):
    """
    HLLC approximate Riemann solver with vectorized implementation.

    A robust and accurate flux scheme that resolves contact discontinuities.
    """

    def compute_flux_vectorized(self, UL: np.ndarray, UR: np.ndarray,
                                gas: GasProperties) -> np.ndarray:
        """
        Vectorized HLLC flux computation for all faces at once.

        This is the performance-critical function - fully vectorized.
        """
        gamma = gas.gamma
        gm1 = gamma - 1
        n_vars, n_faces = UL.shape
        n_scalars = n_vars - 3

        # Extract primitives from left state (vectorized)
        rhoL = UL[0]
        uL = UL[1] / rhoL
        EL = UL[2] / rhoL
        pL = gm1 * (UL[2] - 0.5 * rhoL * uL**2)
        aL = np.sqrt(gamma * pL / rhoL)
        HL = EL + pL / rhoL

        # Extract primitives from right state (vectorized)
        rhoR = UR[0]
        uR = UR[1] / rhoR
        ER = UR[2] / rhoR
        pR = gm1 * (UR[2] - 0.5 * rhoR * uR**2)
        aR = np.sqrt(gamma * pR / rhoR)
        HR = ER + pR / rhoR

        # Roe averages for wave speed estimates (vectorized)
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        denom_inv = 1.0 / (sqrt_rhoL + sqrt_rhoR)

        u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) * denom_inv
        H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) * denom_inv
        a_roe = np.sqrt(gm1 * (H_roe - 0.5 * u_roe**2))

        # Wave speed estimates (vectorized)
        SL = np.minimum(uL - aL, u_roe - a_roe)
        SR = np.maximum(uR + aR, u_roe + a_roe)

        # Contact wave speed (vectorized)
        SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
             (rhoL * (SL - uL) - rhoR * (SR - uR))

        # Initialize final flux array
        F = np.zeros((n_vars, n_faces))

        # Compute wave structure masks
        mask_left = SL >= 0  # Use left flux
        mask_right = SR <= 0  # Use right flux
        mask_star_left = (SM >= 0) & (SL < 0) & (SR > 0)  # Left star state
        mask_star_right = (SM < 0) & (SL < 0) & (SR > 0)  # Right star state

        # Left flux (for simple left and left star regions)
        rhoL_uL = rhoL * uL
        F[0] = rhoL_uL
        F[1] = rhoL_uL * uL + pL
        F[2] = rhoL_uL * HL

        # Right flux regions
        if np.any(mask_right):
            rhoR_uR = rhoR[mask_right] * uR[mask_right]
            F[0, mask_right] = rhoR_uR
            F[1, mask_right] = rhoR_uR * uR[mask_right] + pR[mask_right]
            F[2, mask_right] = rhoR_uR * HR[mask_right]

        # Handle scalars for left and right fluxes
        if n_scalars > 0:
            YL = UL[3:] / rhoL
            F[3:] = rhoL_uL * YL
            if np.any(mask_right):
                rhoR_uR_full = rhoR[mask_right] * uR[mask_right]
                YR_mask = UR[3:, mask_right] / rhoR[mask_right]
                F[3:, mask_right] = rhoR_uR_full * YR_mask

        # Left star state correction
        if np.any(mask_star_left):
            SL_m = SL[mask_star_left]
            uL_m = uL[mask_star_left]
            rhoL_m = rhoL[mask_star_left]
            SM_m = SM[mask_star_left]

            coeffL = rhoL_m * (SL_m - uL_m) / (SL_m - SM_m)

            # Compute U* - U
            dU0 = coeffL - rhoL_m
            dU1 = coeffL * SM_m - rhoL_m * uL_m
            dU2 = coeffL * (EL[mask_star_left] + (SM_m - uL_m) *
                           (SM_m + pL[mask_star_left] / (rhoL_m * (SL_m - uL_m)))) - UL[2, mask_star_left]

            # F* = F + SL * (U* - U)
            F[0, mask_star_left] += SL_m * dU0
            F[1, mask_star_left] += SL_m * dU1
            F[2, mask_star_left] += SL_m * dU2

            if n_scalars > 0:
                YL_m = UL[3:, mask_star_left] / rhoL_m
                dU_scalars = coeffL * YL_m - UL[3:, mask_star_left]
                F[3:, mask_star_left] += SL_m * dU_scalars

        # Right star state correction
        if np.any(mask_star_right):
            SR_m = SR[mask_star_right]
            uR_m = uR[mask_star_right]
            rhoR_m = rhoR[mask_star_right]
            SM_m = SM[mask_star_right]

            coeffR = rhoR_m * (SR_m - uR_m) / (SR_m - SM_m)

            # Recompute right flux for this region
            rhoR_uR_m = rhoR_m * uR_m
            FR0 = rhoR_uR_m
            FR1 = rhoR_uR_m * uR_m + pR[mask_star_right]
            FR2 = rhoR_uR_m * HR[mask_star_right]

            # Compute U* - U
            dU0 = coeffR - rhoR_m
            dU1 = coeffR * SM_m - rhoR_m * uR_m
            dU2 = coeffR * (ER[mask_star_right] + (SM_m - uR_m) *
                           (SM_m + pR[mask_star_right] / (rhoR_m * (SR_m - uR_m)))) - UR[2, mask_star_right]

            # F* = F_R + SR * (U* - U)
            F[0, mask_star_right] = FR0 + SR_m * dU0
            F[1, mask_star_right] = FR1 + SR_m * dU1
            F[2, mask_star_right] = FR2 + SR_m * dU2

            if n_scalars > 0:
                YR_m = UR[3:, mask_star_right] / rhoR_m
                FR_scalars = rhoR_uR_m * YR_m
                dU_scalars = coeffR * YR_m - UR[3:, mask_star_right]
                F[3:, mask_star_right] = FR_scalars + SR_m * dU_scalars

        return F

    def compute_flux(self, UL: np.ndarray, UR: np.ndarray,
                     gas: GasProperties) -> np.ndarray:
        """Single-face flux computation (for backward compatibility)."""
        # Reshape to (n_vars, 1), compute, then squeeze
        UL_2d = UL.reshape(-1, 1)
        UR_2d = UR.reshape(-1, 1)
        F_2d = self.compute_flux_vectorized(UL_2d, UR_2d, gas)
        return F_2d.squeeze()


class RusanovFlux(FluxScheme):
    """
    Rusanov (Local Lax-Friedrichs) flux - much simpler and faster than HLLC.

    Less accurate for contact discontinuities but very robust and fast.
    Good for smooth flows.
    """

    def compute_flux_vectorized(self, UL: np.ndarray, UR: np.ndarray,
                                gas: GasProperties) -> np.ndarray:
        """
        Vectorized Rusanov flux computation - very fast.
        """
        gamma = gas.gamma
        n_vars, n_faces = UL.shape
        n_scalars = n_vars - 3

        # Left state
        rhoL = UL[0]
        uL = UL[1] / rhoL
        pL = (gamma - 1) * (UL[2] - 0.5 * rhoL * uL**2)
        aL = np.sqrt(gamma * pL / rhoL)
        HL = UL[2] / rhoL + pL / rhoL

        # Right state
        rhoR = UR[0]
        uR = UR[1] / rhoR
        pR = (gamma - 1) * (UR[2] - 0.5 * rhoR * uR**2)
        aR = np.sqrt(gamma * pR / rhoR)
        HR = UR[2] / rhoR + pR / rhoR

        # Maximum wave speed
        smax = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)

        # Physical fluxes
        FL = np.zeros((n_vars, n_faces))
        FL[0] = rhoL * uL
        FL[1] = rhoL * uL**2 + pL
        FL[2] = rhoL * uL * HL
        if n_scalars > 0:
            FL[3:] = (rhoL * uL) * (UL[3:] / rhoL)

        FR = np.zeros((n_vars, n_faces))
        FR[0] = rhoR * uR
        FR[1] = rhoR * uR**2 + pR
        FR[2] = rhoR * uR * HR
        if n_scalars > 0:
            FR[3:] = (rhoR * uR) * (UR[3:] / rhoR)

        # Rusanov flux: F = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)
        F = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)

        return F

    def compute_flux(self, UL: np.ndarray, UR: np.ndarray,
                     gas: GasProperties) -> np.ndarray:
        """Single-face flux computation."""
        UL_2d = UL.reshape(-1, 1)
        UR_2d = UR.reshape(-1, 1)
        F_2d = self.compute_flux_vectorized(UL_2d, UR_2d, gas)
        return F_2d.squeeze()
