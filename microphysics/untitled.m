clear all;


gamma_la=0.072;
gamma_ia=0.109;
R_w=461;
T_s=298;
rho_b=810;
syms r r_d kappa gamma

assume(r, "real")
assume(r_d, "real")
assume(r>r_d)

S_wd(r,r_d,kappa,gamma)=(r^3-r_d^3)/(r^3-(1-kappa)*r_d^3)*exp(2*gamma/(R_w*T_s*rho_b*r));
S=1.001;   

crit=diff(S_wd,r);
constS=S_wd-S;


r_ds=2.*logspace(-9,-8,100);
kappas=[0 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.9 1.0];
colororder([0    0.4470    0.7410
    0    0.4470    0.7410
            0.8500    0.3250    0.0980
            0.8500    0.3250    0.0980
            0.9290    0.6940    0.1250
            0.9290    0.6940    0.1250
            0.4940    0.1840    0.5560
            0.4940    0.1840    0.5560
            0.4660    0.6740    0.1880
            0.4660    0.6740    0.1880
            0.3010    0.7450    0.9330
            0.3010    0.7450    0.9330
            0.6350    0.0780    0.1840
            0.6350    0.0780    0.1840])


for j=1:10
    kappa=kappas(j);
    r_cs=zeros(size(r_ds));
    r_eq=zeros(size(r_ds));
    S_w=zeros(size(r_ds));
    S_i=zeros(size(r_ds));
    for i=1:length(r_ds)
        r_d=r_ds(i);
        gamma=gamma_la;
        fun=matlabFunction(subs(constS));
        r_eq(i)=fsolve(fun,r_d*1.001);
        S_w(i)=S_wd(r_eq(i),r_d,kappa,gamma_la);
        S_i(i)=S_wd(r_eq(i),r_d,kappa,gamma_ia);
    end
    loglog(r_ds,S_i-1); hold on;
end
% kappa=0.01;
% r_eq=zeros(size(r_ds));
% S_c=zeros(size(r_ds));
% for i=1:length(r_ds)
%     r_d=r_ds(i);
%     gamma=gamma_la;
%     fun=matlabFunction(subs(constS));
%     r_eq(i)=fsolve(fun,r_d);
%     S_c(i)=S_wd(r_eq(i),r_d,kappa,gamma_ia);
% end
% loglog(r_ds,S_c-1); hold on;
yline(S-1);  hold on;



hold off;
%ylims=[0.8e-3,2e-1];
ylims=[0.8e-8,2e-4];
xlims=[1.6e-9,4e-8];
logScale = diff(ylims)/diff(xlims);  
powerScale = diff(log10(ylims))/diff(log10(xlims));
set(gca,'Xlim',xlims,'YLim',ylims,'DataAspectRatio',[1 logScale/powerScale 1])
