function [g,dgdx] = spm_gx_fmri_modified_free_h_para(x,u,P,M)
% Simulated BOLD response to input
% FORMAT [g,dgdx] = spm_gx_fmri(x,u,P,M)
% g          - BOLD response (%)
% x          - state vector     (see spm_fx_fmri)
% P          - Parameter vector (see spm_fx_fmri)
% M          - model specification structure (see spm_nlsi)
%__________________________________________________________________________
%
% This function implements the BOLD signal model described in: 
%
% Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
% Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston & Klaas Enno Stephan
% $Id: spm_gx_fmri.m 6262 2014-11-17 13:47:56Z karl $
 
 
% Biophysical constants for 1.5T
%==========================================================================
 
% time to echo (TE) (default 0.04 sec)
%--------------------------------------------------------------------------
% MODIFIED
% TE  = 0.04;
TE = P.TE;

% resting venous volume (%)
%--------------------------------------------------------------------------
% MODIFIED
% V0  = 4;
V0 = P.V0;

% estimated region-specific ratios of intra- to extra-vascular signal 
%--------------------------------------------------------------------------
ep  = exp(P.epsilon);
 
% slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
% saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
%--------------------------------------------------------------------------
% MODIFIED 
% r0  = 25;
r0 = P.r0;
 
% frequency offset at the outer surface of magnetized vessels (Hz)
%--------------------------------------------------------------------------
% MODIFIED
% nu0 = 40.3;
theta0 = P.theta0;
nu0 = theta0;
 
% resting oxygen extraction fraction
%--------------------------------------------------------------------------
% MODIFIED
% E0  = 0.4;
E0 = P.E0;
 
%-Coefficients in BOLD signal model
%==========================================================================
k1  = 4.3.*nu0.*E0.*TE;
k2  = ep.*r0.*E0.*TE;
k3  = 1 - ep;
 
%-Output equation of BOLD signal model
%==========================================================================
v   = exp(x(:,4));
q   = exp(x(:,5));
g   = V0.*(k1 - k1.*q + k2 - k2.*q./v + k3 - k3.*v);

if nargout == 1, return, end


%-derivative dgdx
%==========================================================================
[n m]      = size(x);
dgdx       = cell(1,m);
[dgdx{:}]  = deal(sparse(n,n));
dgdx{1,4}  = diag(-V0*(k3.*v - k2.*q./v));
dgdx{1,5}  = diag(-V0*(k1.*q + k2.*q./v));
dgdx       = spm_cat(dgdx);