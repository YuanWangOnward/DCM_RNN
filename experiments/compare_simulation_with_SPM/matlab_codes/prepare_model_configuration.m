function [M] = prepare_model_configuration(P)
% beginning part of spm_dcm_estimate
 
SVNid = '$Rev: 6755 $';
 
%-Load DCM structure
%--------------------------------------------------------------------------
if ~nargin
    
    %-Display model details
    %----------------------------------------------------------------------
    Finter = spm_figure('GetWin','Interactive');
    set(Finter,'name','Dynamic Causal Modelling')
    
    %-Get DCM
    %----------------------------------------------------------------------
    [P, sts] = spm_select(1,'^DCM.*\.mat$','select DCM_???.mat');
    if ~sts, DCM = []; return; end
    spm('Pointer','Watch')
    spm('FigName','Estimation in progress');
    
end
 
if isstruct(P)
    DCM = P;
    P   = ['DCM-' date '.mat'];
else
    load(P)
end
 
% check options
%==========================================================================
try, DCM.options.two_state;  catch, DCM.options.two_state  = 0;     end
try, DCM.options.stochastic; catch, DCM.options.stochastic = 0;     end
try, DCM.options.nonlinear;  catch, DCM.options.nonlinear  = 0;     end
try, DCM.options.centre;     catch, DCM.options.centre     = 0;     end
try, DCM.options.hidden;     catch, DCM.options.hidden     = [];    end
try, DCM.options.hE;         catch, DCM.options.hE         = 6;     end
try, DCM.options.hC;         catch, DCM.options.hC         = 1/128; end
try, DCM.n;                  catch, DCM.n = size(DCM.a,1);          end
try, DCM.v;                  catch, DCM.v = size(DCM.Y.y,1);        end
 
try, M.nograph = DCM.options.nograph; catch, M.nograph = spm('CmdLine');end
 
% check max iterations
%--------------------------------------------------------------------------
try
    DCM.options.maxit;
catch    
    if isfield(DCM.options,'nN')
        DCM.options.maxit = DCM.options.nN;
        warning('options.nN is deprecated. Please use options.maxit');
    elseif DCM.options.stochastic
        DCM.options.maxit = 32;
    else
        DCM.options.maxit = 128;
    end
end
 
try M.Nmax = DCM.M.Nmax; catch, M.Nmax = DCM.options.maxit; end
 
% check max nodes
%--------------------------------------------------------------------------
try
    DCM.options.maxnodes;
catch
    if isfield(DCM.options,'nmax')
        DCM.options.maxnodes = DCM.options.nmax;
        warning('options.nmax is deprecated. Please use options.maxnodes');
    else
        DCM.options.maxnodes = 8;
    end
end
 
% analysis and options
%--------------------------------------------------------------------------
DCM.options.induced  = 0;
 
% unpack
%--------------------------------------------------------------------------
U  = DCM.U;                             % exogenous inputs
Y  = DCM.Y;                             % responses
n  = DCM.n;                             % number of regions
v  = DCM.v;                             % number of scans
 
% detrend outputs (and inputs)
%--------------------------------------------------------------------------
%%%% MODIFIED
% Y.y = spm_detrend(Y.y);
if DCM.options.centre
    U.u = spm_detrend(U.u);
end
 
% check scaling of Y (enforcing a maximum change of 4%
%--------------------------------------------------------------------------
%%%% MODIFIED
scale   = max(max((Y.y))) - min(min((Y.y)));
scale   = 4/max(scale,4);
Y.y     = Y.y*scale;
Y.scale = scale;
 
% check confounds (add constant if necessary)
%--------------------------------------------------------------------------
if ~isfield(Y,'X0'),Y.X0 = ones(v,1); end
if ~size(Y.X0,2),   Y.X0 = ones(v,1); end
 
% fMRI slice time sampling
%--------------------------------------------------------------------------
try, M.delays = DCM.delays; catch, M.delays = ones(n,1); end
try, M.TE     = DCM.TE;     end
 
% create priors
%==========================================================================
 
% check DCM.d (for nonlinear DCMs)
%--------------------------------------------------------------------------
try
    DCM.options.nonlinear = logical(size(DCM.d,3));
catch
    DCM.d = zeros(n,n,0);
    DCM.options.nonlinear = 0;
end
 
% specify parameters for spm_int_D (ensuring updates every second or so)
%--------------------------------------------------------------------------
if DCM.options.nonlinear
    M.IS     = 'spm_int_D';
    M.nsteps = round(max(Y.dt,1));
    M.states = 1:n;
else
    M.IS     = 'spm_int';
end
 
% check for endogenous DCMs, with no exogenous driving effects
%--------------------------------------------------------------------------
if isempty(DCM.c) || isempty(U.u)
    DCM.c  = zeros(n,1);
    DCM.b  = zeros(n,n,1);
    U.u    = zeros(v,1);
    U.name = {'null'};
end
if ~any(spm_vec(U.u)) || ~any(spm_vec(DCM.c))
    DCM.options.stochastic = 1;
end
 
 
% priors (and initial states)
%--------------------------------------------------------------------------
[pE,pC,x]  = spm_dcm_fmri_priors(DCM.a,DCM.b,DCM.c,DCM.d,DCM.options);
str        = 'Using specified priors ';
str        = [str '(any changes to DCM.a,b,c,d will be ignored)\n'];
 
try, M.P   = DCM.options.P;                end      % initial parameters
try, pE    = DCM.options.pE; fprintf(str); end      % prior expectation
try, pC    = DCM.options.pC; fprintf(str); end      % prior covariance
 
try, M.P   = DCM.M.P;                end            % initial parameters
try, pE    = DCM.M.pE; fprintf(str); end            % prior expectation
try, pC    = DCM.M.pC; fprintf(str); end            % prior covariance
 
 
% eigenvector constraints on pC for large models
%--------------------------------------------------------------------------
if n > DCM.options.maxnodes
    
    % remove confounds and find principal (nmax) modes
    %----------------------------------------------------------------------
    y       = Y.y - Y.X0*(pinv(Y.X0)*Y.y);
    V       = spm_svd(y');
    V       = V(:,1:DCM.options.maxnodes);
    
    % remove minor modes from priors on A
    %----------------------------------------------------------------------
    j       = 1:(n*n);
    V       = kron(V*V',V*V');
    pC(j,j) = V*pC(j,j)*V';
    
end
 
% hyperpriors over precision - expectation and covariance
%--------------------------------------------------------------------------
hE      = sparse(n,1) + DCM.options.hE;
hC      = speye(n,n)  * DCM.options.hC;
i       = DCM.options.hidden;
hE(i)   = -4;
hC(i,i) = exp(-16);
 
% complete model specification
%--------------------------------------------------------------------------
M.f  = 'spm_fx_fmri';                     % equations of motion
M.g  = 'spm_gx_fmri';                     % observation equation
M.x  = x;                                 % initial condition (states)
M.pE = pE;                                % prior expectation (parameters)
M.pC = pC;                                % prior covariance  (parameters)
M.hE = hE;                                % prior expectation (precisions)
M.hC = hC;                                % prior covariance  (precisions)
M.m  = size(U.u,2);
M.n  = size(x(:),1);
M.l  = size(x,1);
M.N  = 64;
M.dt = 32/M.N;
M.ns = v;
 