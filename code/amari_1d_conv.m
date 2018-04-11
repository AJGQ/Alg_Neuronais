%% Simulation of a 1d neural field
% example taken from Ferreira, F: Multi-bump solutions in dynamic neural
% fields: analysis and applications, PhD thesis, University of Minho 2014
% p. 37 Fig. 3.3
% W. Wojtak, 06/03/2018

%% initialize 
clear, clc;
fieldSize       = 120;                  % total field size
halfField       = floor(fieldSize/2);

dx              = 0.005;                % discretization in space
xDim            = -halfField:dx:halfField;
fieldPoints     = numel(xDim);
halfPoint       = floor(fieldPoints/2);

TotalTime       = 4;                    % total time
nSteps          = 1000;                 % number of steps for Euler scheme
dt              = TotalTime/nSteps;     % discretization in time

history_u       = zeros(nSteps, fieldPoints);       % matrix to store the solutions in each time step

%% initial condition
u_field         = -0.5 + 8*exp(-xDim.^2/18);

h               = 2.8997;               % threshold for firing rate
%% connectivity kernel
A = 2;                                  % kernel's amplitude
b = 0.08;                               % rate of oscillations decay
alpha = pi/10;                          % zero crossings
kernel = A*(exp(-b*abs(xDim)).*((b*sin(abs(alpha*xDim)))+cos(alpha*xDim)));   % connectivity function w(x)

%% main loop
for i = 1:nSteps

firing_rate     = heaviside(u_field-h);
    
% replace "conv" function with computing the integral directly
output          = [firing_rate(halfPoint+2:fieldPoints),...
                firing_rate, firing_rate(:, 1:halfPoint)];  
            
convolution     = dx * conv(output, kernel, 'valid'); 

u_field         = u_field + dt * (-u_field + convolution);      % main equation

history_u(i,:)  = u_field;
end

%% plot the results
figure 
set(0,'DefaultAxesFontName', 'Helvetica ')
set(0,'DefaultAxesFontSize', 16)
plot(xDim, history_u(1,:),'linewidth',2), hold on
plot(xDim, history_u(end,:),'linewidth',2), hold on
legend('initial profile', 'solution')
plot(xDim,h*ones(1,fieldPoints),':k')
xlabel('x'); ylabel('u(x)');

%% show how activity evolves over time 
% figure  
% for i = 1:nSteps
%         plot(xDim, history_u(i,:),'linewidth',2);    
%         pause(0.01);
% end
