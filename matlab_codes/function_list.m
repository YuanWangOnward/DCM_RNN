% function list
[DCM] = spm_dcm_estimate(P);

[M0,M1,L1,L2] = spm_bireduce(M,P);

[f,dfdx,D,dfdu] = spm_fx_fmri(x,u,P,M);

[x] = spm_expm(J,x);

[varargout] = spm_diff(varargin);

[g,dgdx] = spm_gx_fmri(x,u,P,M);

[y] = spm_int(P,M,U);

[Ep,Cp,Eh,F,L,dFdp,dFdpp] = spm_nlsi_GN(M,U,Y);


% for check evaluate result 
y = feval(M.IS,Ep,M,U);
% residue is calculated as 
R = Y.y - y;
R = R - Y.X0*spm_inv(Y.X0'*Y.X0)*(Y.X0'*R);
Ce = exp(-Eh); 