clear all;
global dx, global dt
global fieldPoints, global halfPoints

%% Set the parameters
fieldSize       = 31;
halfField       = floor(fieldSize/2);
dx              = .005;
dt              = .005;

xDim            = -halfField:dx:halfField;

fieldPoints     = numel(xDim);
halfPoints      = floor(fieldPoints/2);

tspan           = 0:dt:50;
nSteps          = numel(tspan);
tau_u           = 1;

%% Initial data
u_field             = 6 * gauss(xDim, 0, 2)-2;
history_u           = zeros(nSteps, fieldPoints);

beta_u              = 1000.0;

%% Connectivity function
        w_inh           = 0.5;
        sigma_exc       = 2;
        sigma_inh       = 3.5;
        A               = 4;
        A_inh           = 2;
        kernel          = A * gauss(xDim, 0, sigma_exc) - A_inh * gauss(xDim, 0, sigma_inh) - w_inh;
        disp(' ');
        disp(['Kernel parameters: ', 'A: ', num2str(A),' A_inh: ', num2str(A_inh), ' sigma_exc: ', num2str(sigma_exc),...
                            ' sigma_inh: ', num2str(sigma_inh), ' w_inh: ', num2str(w_inh)]);
wHat = fft(kernel);

%% Noise
kernel_noise = cos(xDim/1);
eps = 0.005;  % noise strength
noiseHat = fft(kernel_noise);

%% External input
Input               = zeros(nSteps, fieldPoints);

%% Main loop
disp(' ');
disp('Calculating.....');
disp(' ');

for i = 1:nSteps

% sigmoid_var     = sigmoid(u_field, beta_u, 1);
sigmoid_var     = heaviside(u_field-1);

% Convolutions with FFT:
convolution  = dx * ifftshift(real(ifft(fft(sigmoid_var) .* wHat)));
conv_noise   = dx * ifftshift(real(ifft(fft(randn(1, fieldPoints)) .* noiseHat)));

% Amari:
u_field         = u_field + dt/tau_u * (-u_field + convolution - 0 + Input(i, :)) + sqrt(eps) * conv_noise;

history_u(i,:)  = u_field;
end

%% PLOT RESULTS
[~,max_bump] = max(history_u');

figure
imagesc(history_u),title ('u(x)'),hold on
colormap hot
plot(max_bump,linspace(0,size(tspan,2),size(tspan,2)),'m','linewidth',2)

if 1
    figure
    set(0,'DefaultAxesFontSize', 20)
    plot(xDim,u_field,'k','linewidth',3), hold on
    plot(xDim,1*ones(1,fieldPoints),':k','linewidth',2) 
    xlabel('x','fontsize', 20);
    ylabel('u(x)','fontsize', 20);
end

% Plot activities iteratively
if 0
    figure
    for i = 1:nSteps
        if mod(i,10) == 0
            plot(xDim, history_u(i,:),'b',...
                 xDim, Input(i,:),'m','linewidth',1), legend('u(x)', 'Input(x)') 
            pause(0.1);
        end
    end
end
