clear
close all
clc

% Load the data from HH_experiments.mat
load('HH_experiments.mat', 'Gait1_kinematics', 'Gait1_plate_LRF');


Butterfly_left=Gait1_plate_LRF(940:1007,13:15); %comment thes 2 linesafter use
Gait1_plate_LRF=Gait1_plate_LRF(982:1090,:);


% Keep all frames (3486) or adjust as needed
Gait1_kinematics = Gait1_kinematics(1:3486,:);  % Adjust if you need fewer frames, e.g., 1:3160

% Handle missing data with spline interpolation
GAIT1_KINEMATICS = [];
for i = 1:size(Gait1_kinematics, 2)
    DATA = Gait1_kinematics(:, i);
    DATA = fillmissing(DATA, 'spline');
    GAIT1_KINEMATICS = [GAIT1_KINEMATICS DATA];
end
Gait1_kinematics = GAIT1_KINEMATICS;


% Assign preprocessed data
Gait1_kinematics = GAIT1_KINEMATICS;

% Use the sampling frequency from the data
fs = 250;  % Loaded from HH_experiments.mat

% Apply low-pass filter (5 Hz cutoff)
Wn = 5 / (fs / 2);  % Normalized cutoff frequency
[B, A] = butter(3, Wn, 'low');  % 3rd-order Butterworth filter
Gait1_kinematics = filtfilt(B, A, Gait1_kinematics);  % Zero-phase filtering

% Extract marker positions based on new column indices
r_heel = Gait1_kinematics(:, 19:21);  % r_mall (6th marker: 6*3-2 to 6*3)
l_heel = Gait1_kinematics(:, 40:42);  % l_mall (13th marker: 13*3-2 to 13*3)


% Extract vertical (Y) components
y_r_heel = r_heel(:, 2);  % 2nd column is Y (vertical)
y_l_heel = l_heel(:, 2);

% Calculate time vector
t = (1:length(y_r_heel)) / fs;

figure
set(gcf, 'Color', 'white'); % Set figure background to white
plot(t,y_r_heel*1000,'b','LineWidth',1.2)
hold on
plot(t,y_l_heel*1000,'r','LineWidth',1.2)
grid on
xlim([0 length(y_r_heel)/fs])
xlabel('Time (s)')
ylabel('Height (mm)')
title('Heel Markers along Vertical Axis')
legend('Right Foot','Left Foot')

  %figure
  %subplot(2,1,1)
  %plot(y_r_mall)
  %xlim([1800 2300])
  %subplot(2,1,2)
  %plot(y_l_mall)
  %xlim([1800 2300])

Heel_R_Y=y_r_heel(2001:2600);
Heel_L_Y=y_l_heel(2001:2600);
half_data=round((length(Heel_R_Y))/2);
[min_start, pos_start] = min(Heel_R_Y(1:half_data));
% pos_start =(pos_start/fs);
[min_end, pos_end] = min(Heel_R_Y((half_data+1):length(Heel_R_Y)));
gait_cycle=Gait1_kinematics(2000+pos_start:(2000+pos_end+half_data),:);

% gait cycle

Gait1_kinematics=gait_cycle;

pos1=15; %12.0141;
pos2=30.7420; %32.8622;
pos3=50; %43.5484; %63.9576;
pos4=77.0318;
pos5=92.6056; %89.0459;
mid=60; %52.4194;
y=0.9*300;

T=0:(100/(size(Gait1_kinematics,1)-1)):100;
figure
set(gcf, 'Color', 'white'); % Set figure background to white
plot(T,Heel_R_Y(pos_start:(pos_end+half_data))*1000,'b','LineWidth',1.7)
grid on
hold on
plot(T,Heel_L_Y(pos_start:(pos_end+half_data))*1000,'r','LineWidth',1.2)
xlabel('Gait cycle (%)')
ylabel('Height (mm)')
text(0,y,'\leftarrow IC')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(100,y,'\leftarrow IC')
text(mid,y,'\leftarrow TO')
text(pos1,y,'\leftarrow OT')
text(pos2,y,'\leftarrow HR')
text(pos3,y,'\leftarrow OI')
text(pos4,y,'\leftarrow FA')
text(pos5,y,'\leftarrow TV')
title('Gait Cycle Right Foot')
legend('Right Foot','Left Foot')


% data we have from vaughan
% from z to x
% from x to y
% from y to z

GAIT1_KINEMATICS=Gait1_kinematics;
for i=1:(size(Gait1_kinematics,2)/3)
      GAIT1_KINEMATICS(:,3*i-2)=Gait1_kinematics(:,3*i);
      GAIT1_KINEMATICS(:,3*i-1)=Gait1_kinematics(:,3*i-2);
      GAIT1_KINEMATICS(:,3*i)=Gait1_kinematics(:,3*i-1);
end
Gait1_kinematics=GAIT1_KINEMATICS;



% marker

% Assign points based on Gait1_kinematics markers
p15 = Gait1_kinematics(:,1:3);    % sacrum
p7 = Gait1_kinematics(:,4:6);    % r_asis
p6 = Gait1_kinematics(:,7:9);    % r_bar1
p4 = Gait1_kinematics(:,10:12);  % r_knee1
p5 = Gait1_kinematics(:,13:15);  % r_bar2
p3 = Gait1_kinematics(:,16:18);   % r_mall 
p2 = Gait1_kinematics(:,19:21);  % r_heel
p1 = Gait1_kinematics(:,22:24);  % r_met
p14 = Gait1_kinematics(:,25:27);  % l_asis
p13 = Gait1_kinematics(:,28:30); % l_bar1
p12 = Gait1_kinematics(:,31:33); % l_knee1
p11 = Gait1_kinematics(:,34:36); % l_bar2
p10 = Gait1_kinematics(:,37:39); % l_mall
p9 = Gait1_kinematics(:,40:42); % l_heel
p8 = Gait1_kinematics(:,43:45); % l_met
%% uvw reference system (technical system in lab)

% right foot

U_R_FOOT=[];
W_R_FOOT=[];
V_R_FOOT=[];
for i=1:size(Gait1_kinematics,1)
    u_R_foot=(p1(i,:)'-p2(i,:)')/norm(p1(i,:)'-p2(i,:)');
    w_R_foot=cross((p1(i,:)'-p3(i,:)'),(p2(i,:)'-p3(i,:)'))/norm(cross((p1(i,:)'-p3(i,:)'),(p2(i,:)'-p3(i,:)')));
    v_R_foot=cross(w_R_foot,u_R_foot);
    U_R_FOOT=[U_R_FOOT;u_R_foot];
    W_R_FOOT=[W_R_FOOT;w_R_foot];
    V_R_FOOT=[V_R_FOOT;v_R_foot];
end
u_R_foot=U_R_FOOT;
w_R_foot=W_R_FOOT;
v_R_foot=V_R_FOOT;

foot_lenght_c_a=0.28; %m (from the subject measurement)
a_13=foot_lenght_c_a;
mall_height=0.07; %m
a_15=mall_height;
mall_width=0.08; %m
a_17=mall_width;
foot_breadth=0.09; %m
a_19=foot_breadth;


%left foot

U_L_FOOT=[];
W_L_FOOT=[];
V_L_FOOT=[];
for i=1:size(Gait1_kinematics,1)
    u_L_foot=(p8(i,:)'-p9(i,:)')/norm(p8(i,:)'-p9(i,:)');  %l_heel and l_met
    w_L_foot=cross((p8(i,:)'-p10(i,:)'),(p9(i,:)'-p10(i,:)'))/norm(cross((p8(i,:)'-p10(i,:)'),(p9(i,:)'-p10(i,:)')));
    v_L_foot=cross(w_L_foot,u_L_foot);
    U_L_FOOT=[U_L_FOOT;u_L_foot];
    W_L_FOOT=[W_L_FOOT;w_L_foot];
    V_L_FOOT=[V_L_FOOT;v_L_foot];
end
u_L_foot=U_L_FOOT;
w_L_foot=W_L_FOOT;
v_L_foot=V_L_FOOT;

a_14=a_13;
a_16=a_15;
a_18=a_17;
a_20=a_19;

%right calf

V_R_CALF=[];
W_R_CALF=[];
U_R_CALF=[];
for i=1:size(Gait1_kinematics,1)
    v_R_calf=(p3(i,:)'-p5(i,:)')/norm(p3(i,:)'-p5(i,:)');  % r_mall and R_bar2
    w_R_calf=cross((p4(i,:)'-p5(i,:)'),(p3(i,:)'-p5(i,:)'))/norm(cross((p4(i,:)'-p5(i,:)'),(p3(i,:)'-p5(i,:)')));
    u_R_calf=cross(v_R_calf,w_R_calf);
    V_R_CALF=[V_R_CALF;v_R_calf];
    W_R_CALF=[W_R_CALF;w_R_calf];
    U_R_CALF=[U_R_CALF;u_R_calf];
end
v_R_calf=V_R_CALF;
w_R_calf=W_R_CALF;
u_R_calf=U_R_CALF;

knee_diameter=0.07; %m
a11=knee_diameter;

% left calf

V_L_CALF=[];
W_L_CALF=[];
U_L_CALF=[];
for i=1:size(Gait1_kinematics,1)
    v_L_calf=(p10(i,:)'-p12(i,:)')/norm(p10(i,:)'-p12(i,:)');
    w_L_calf=cross((p11(i,:)'-p12(i,:)'),(p10(i,:)'-p12(i,:)'))/norm(cross((p11(i,:)'-p12(i,:)'),(p10(i,:)'-p12(i,:)')));
    u_L_calf=cross(v_L_calf,w_L_calf);
    V_L_CALF=[V_L_CALF;v_L_calf];
    W_L_CALF=[W_L_CALF;w_L_calf];
    U_L_CALF=[U_L_CALF;u_L_calf];
end
v_L_calf=V_L_CALF;
w_L_calf=W_L_CALF;
u_L_calf=U_L_CALF;

a12=a11;

% pelvis

V_PELVIS=[];
W_PELVIS=[];
U_PELVIS=[];
for i=1:size(Gait1_kinematics,1)
    v_pelvis=(p14(i,:)'-p7(i,:)')/norm(p14(i,:)'-p7(i,:)');
    w_pelvis=cross((p7(i,:)'-p15(i,:)'),(p14(i,:)'-p15(i,:)'))/norm(cross((p7(i,:)'-p15(i,:)'),(p14(i,:)'-p15(i,:)')));
    u_pelvis=cross(v_pelvis,w_pelvis);
    V_PELVIS=[V_PELVIS;v_pelvis];
    W_PELVIS=[W_PELVIS;w_pelvis];
    U_PELVIS=[U_PELVIS;u_pelvis];
end
v_pelvis=V_PELVIS;
w_pelvis=W_PELVIS;
u_pelvis=U_PELVIS;

hip_breadth=0.225; %m
a2=hip_breadth;

%% anatomic

P3=[];
for i=1:size(Gait1_kinematics,1)
    P3=[P3;p3(i,:)'];
end
p3=P3;

p_R_ankle=p3-0.008*a_13*u_R_foot+0.393*a_15*v_R_foot+0.706*a_17*w_R_foot;
p_R_toe=p3+0.697*a_13*u_R_foot+0.780*a_15*v_R_foot+0.923*a_19*w_R_foot;

P5=[];
for i=1:size(Gait1_kinematics,1)
    P5=[P5;p5(i,:)'];
end
p5=P5;

p_R_knee=p5+0.423*a11*u_R_calf-0.198*a11*v_R_calf+0.406*a11*w_R_calf;

P10=[];
for i=1:size(Gait1_kinematics,1)
    P10=[P10;p10(i,:)'];
end
p10=P10;

p_L_ankle=p10-0.008*a_14*u_L_foot+0.393*a_16*v_L_foot-0.706*a_18*w_L_foot;
p_L_toe=p10+0.697*a_14*u_L_foot+0.780*a_16*v_L_foot-0.923*a_20*w_L_foot;

P12=[];
for i=1:size(Gait1_kinematics,1)
    P12=[P12;p12(i,:)'];
end
p12=P12;

p_L_knee=p12+0.423*a12*u_L_calf-0.198*a12*v_L_calf-0.406*a12*w_L_calf;

P15=[];
for i=1:size(Gait1_kinematics,1)
    P15=[P15;p15(i,:)'];
end
p15=P15;

p_R_hip=p15+0.598*a2*u_pelvis-0.344*a2*v_pelvis-0.290*a2*w_pelvis;
p_L_hip=p15+0.598*a2*u_pelvis+0.344*a2*v_pelvis-0.290*a2*w_pelvis;

% R_thigh

I1=[];
J1=[];
K1=[];
for i=1:size(Gait1_kinematics,1)
    i1=(p_R_hip((3*i-2):(3*i),1)-p_R_knee((3*i-2):(3*i),1))/norm(p_R_hip((3*i-2):(3*i),1)-p_R_knee((3*i-2):(3*i),1));
    j1=cross((p6(i,:)'-p_R_hip((3*i-2):(3*i),1)),(p_R_knee((3*i-2):(3*i),1)-p_R_hip((3*i-2):(3*i),1)))/norm(cross((p6(i,:)'-p_R_hip((3*i-2):(3*i),1)),(p_R_knee((3*i-2):(3*i),1)-p_R_hip((3*i-2):(3*i),1))));
    k1=cross(i1,j1);
    I1=[I1; i1];
    J1=[J1; j1];
    K1=[K1;k1];
end
i1=I1;
j1=J1;
k1=K1;

%L_thigh

I2=[];
J2=[];
K2=[];
for i=1:size(Gait1_kinematics,1)
    i2=(p_L_hip(((3*i-2):(3*i)),:)-p_L_knee(((3*i-2):(3*i)),:))/norm(p_L_hip(((3*i-2):(3*i)),:)-p_L_knee(((3*i-2):(3*i)),:)');
    j2=cross((p_L_knee((3*i-2):(3*i),:)-p_L_hip((3*i-2):(3*i),:)),(p13(i,:)'-p_L_hip((3*i-2):(3*i),:)))/norm(cross((p_L_knee(((3*i-2):(3*i)),:)-p_L_hip((3*i-2):(3*i),:)),(p13(i,:)'-p_L_hip((3*i-2):(3*i),:))));
    k2=cross(i2,j2);
    I2=[I2; i2];
    J2=[J2; j2];
    K2=[K2;k2];
end
i2=I2;
j2=J2;
k2=K2;

% R_calf

I3=[];
J3=[];
K3=[];
for i=1:size(Gait1_kinematics,1)
    i3=(p_R_knee((3*i-2):(3*i),1)-p_R_ankle((3*i-2):(3*i),1))/norm(p_R_knee((3*i-2):(3*i),1)-p_R_ankle((3*i-2):(3*i),1));
    j3=cross((p5((3*i-2):(3*i),:)-p_R_knee((3*i-2):(3*i),1)),(p_R_ankle((3*i-2):(3*i),1)-p_R_knee((3*i-2):(3*i),1)))/norm(cross((p5((3*i-2):(3*i),:)-p_R_knee((3*i-2):(3*i),1)),(p_R_ankle((3*i-2):(3*i),1)-p_R_knee((3*i-2):(3*i),1))));
    k3=cross(i3,j3);
    I3=[I3; i3];
    J3=[J3; j3];
    K3=[K3;k3];
end
i3=I3;
j3=J3;
k3=K3;

% L_calf

I4=[];
J4=[];
K4=[];
for i=1:size(Gait1_kinematics,1)
    i4=(p_L_knee((3*i-2):(3*i),1)-p_L_ankle((3*i-2):(3*i),1))/norm(p_L_knee((3*i-2):(3*i),1)-p_L_ankle((3*i-2):(3*i),1));
    j4=cross((p_L_ankle((3*i-2):(3*i),1)-p_L_knee((3*i-2):(3*i),1)),(p12((3*i-2):(3*i),:)-p_L_knee((3*i-2):(3*i),1)))/norm(cross((p_L_ankle((3*i-2):(3*i),1)-p_L_knee((3*i-2):(3*i),1)),(p12((3*i-2):(3*i),:)-p_L_knee((3*i-2):(3*i),1))));
    k4=cross(i4,j4);
    I4=[I4; i4];
    J4=[J4; j4];
    K4=[K4;k4];
end
i4=I4;
j4=J4;
k4=K4;

% R_foot

I5=[];
K5=[];
J5=[];
for i=1:size(Gait1_kinematics,1)
    i_5=(p2(i,:)'-p_R_toe((3*i-2):(3*i),1))/norm(p2(i,:)'-p_R_toe((3*i-2):(3*i),1));
    k_5=cross((p_R_ankle((3*i-2):(3*i),1)-p2(i,:)'),(p_R_toe((3*i-2):(3*i),1)-p2(i,:)'))/norm(cross((p_R_ankle((3*i-2):(3*i),1)-p2(i,:)'),(p_R_toe((3*i-2):(3*i),1)-p2(i,:)')));
    j_5=cross(k_5,i_5);
    I5=[I5;i_5];
    K5=[K5;k_5];
    J5=[J5;j_5];
end
i_5=I5;
k_5=K5;
j_5=J5;

% L_foot

I6=[];
K6=[];
J6=[];
for i=1:size(Gait1_kinematics,1)
    i_6=(p9(i,:)'-p_L_toe((3*i-2):(3*i),1))/norm(p9(i,:)'-p_L_toe((3*i-2):(3*i),1));
    k_6=cross((p_L_ankle((3*i-2):(3*i),1)-p9(i,:)'),(p_L_toe((3*i-2):(3*i),1)-p9(i,:)'))/norm(cross((p_L_ankle((3*i-2):(3*i),1)-p9(i,:)'),(p_L_toe((3*i-2):(3*i),1)-p9(i,:)')));
    j_6=cross(k_6,i_6);
    I6=[I6;i_6];
    K6=[K6;k_6];
    J6=[J6;j_6];
end
i_6=I6;
k_6=K6;
j_6=J6;

% Pelvis

i_pelvis=w_pelvis;
j_pelvis=u_pelvis;
k_pelvis=v_pelvis;

%% anatomical joint angles (Grood & Suntay convention)

%hip angles (right)
% hip_joint --> R_pelvis(prox) e R_thigh(dist)

R_pelvis=[i_pelvis j_pelvis k_pelvis]; % proximal

% R_PELVIS=[];
% for i=1:size(i_pelvis,1)
%     R_pelvis=[i_pelvis(i,:)' j_pelvis(i,:)' k_pelvis(i,:)'];
%     R_PELVIS=[R_PELVIS;R_pelvis];
% end
% R_pelvis=R_PELVIS;

R_thigh_r=[i1 j1 k1]; %distal

% R_THIGH_R=[];
% for i=1:size(i1,1)
%     R_thigh_r=[i1(i,:)' i2(i,:)' i3(i,:)'];
%     R_THIGH_R=[R_THIGH_R;R_thigh_r];
% end
% R_thigh_r=R_THIGH_R;

R_HIP=[];
for i=1:size(Gait1_kinematics,1)
    R_hip_r=R_pelvis((3*i-2):3*i,:)'*R_thigh_r((3*i-2):3*i,:);
    R_HIP=[R_HIP;R_hip_r];
end
R_hip_joint_r=R_HIP;

BETA_R_HIP=[];
for i=1:size(Gait1_kinematics,1)
    beta_r_hip=asin(R_hip_joint_r((3*i),1)); %ab-adduction angle
    BETA_R_HIP=[BETA_R_HIP;beta_r_hip];
end
beta_r_hip=BETA_R_HIP;

GAMMA_R_HIP=[];
for i=1:size(Gait1_kinematics,1)
    gamma_r_hip=asin(R_hip_joint_r((3*i),2)/cos(beta_r_hip(i))); %internal-external angle
    GAMMA_R_HIP=[GAMMA_R_HIP;gamma_r_hip];
end
gamma_r_hip=GAMMA_R_HIP;

ALFA_R_HIP=[];
for i=1:size(Gait1_kinematics,1)
    alfa_r_hip=asin(R_hip_joint_r((3*i-1),1)/cos(beta_r_hip(i))); %flexion-extension angle
    ALFA_R_HIP=[ALFA_R_HIP;alfa_r_hip];
end
alfa_r_hip=ALFA_R_HIP;

beta_r_hip=beta_r_hip*180/pi;
gamma_r_hip=gamma_r_hip*180/pi;
alfa_r_hip=alfa_r_hip*180/pi; % flex extension

% hip angles (left)
% R_pelvis=[i_pelvis j_pelvis k_pelvis]; % proximal

R_thigh_l=[i2 j2 k2]; %distal

R_HIP=[];
for i=1:size(Gait1_kinematics,1)
    R_hip_l=R_pelvis((3*i-2):3*i,:)'*R_thigh_l((3*i-2):3*i,:);
    R_HIP=[R_HIP;R_hip_l];
end
R_hip_joint_l=R_HIP;

BETA_L_HIP=[];
for i=1:size(Gait1_kinematics,1)
    beta_l_hip=-asin(R_hip_joint_l((3*i),1)); %ab-adduction angle
    BETA_L_HIP=[BETA_L_HIP;beta_l_hip];
end
beta_l_hip=BETA_L_HIP;

GAMMA_L_HIP=[];
for i=1:size(Gait1_kinematics,1)
    gamma_l_hip=asin(R_hip_joint_l((3*i),2)/cos(beta_l_hip(i))); %internal-external angle
    GAMMA_L_HIP=[GAMMA_L_HIP;gamma_l_hip];
end
gamma_l_hip=GAMMA_L_HIP;

ALFA_L_HIP=[];
for i=1:size(Gait1_kinematics,1)
    alfa_l_hip=asin(R_hip_joint_l((3*i-1),1)/cos(beta_l_hip(i))); %flexion-extension angle
    ALFA_L_HIP=[ALFA_L_HIP;alfa_l_hip];
end
alfa_l_hip=ALFA_L_HIP;

beta_l_hip=beta_l_hip*180/pi;
gamma_l_hip=gamma_l_hip*180/pi;
alfa_l_hip=alfa_l_hip*180/pi; % flex extension

% figure
% T=0:(100/(size(Gait1_kinematics,1)-1)):100;
% plot(T,alfa_l_hip)

% knee angles (right)
% Rthigh (proximal) and Rcalf (distal)

R_calf_r=[i3 j3 k3]; %distal

R_KNEE_R=[];
for i=1:size(Gait1_kinematics,1)
    R_knee_r=R_thigh_r((3*i-2):3*i,:)'*R_calf_r((3*i-2):3*i,:);
    R_KNEE_R=[R_KNEE_R;R_knee_r];
end
R_knee_joint_r=R_KNEE_R;

BETA_R_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    beta_r_knee=asin(R_knee_joint_r((3*i),1)); %ab-adduction angle
    BETA_R_KNEE=[BETA_R_KNEE;beta_r_knee];
end
beta_r_knee=BETA_R_KNEE;

GAMMA_R_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    gamma_r_knee=asin(R_knee_joint_r((3*i),2)/cos(beta_r_knee(i))); %internal-external angle
    GAMMA_R_KNEE=[GAMMA_R_KNEE;gamma_r_knee];
end
gamma_r_knee=GAMMA_R_KNEE;

ALFA_R_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    alfa_r_knee=asin(R_knee_joint_r((3*i-1),1)/cos(beta_r_knee(i))); %flexion-extension angle
    ALFA_R_KNEE=[ALFA_R_KNEE;alfa_r_knee];
end
alfa_r_knee=ALFA_R_KNEE;

beta_r_knee=beta_r_knee*180/pi;
gamma_r_knee=gamma_r_knee*180/pi;
alfa_r_knee=alfa_r_knee*180/pi; % flex extension

% knee angles (left)

R_calf_l=[i4 j4 k4]; %distal

R_KNEE_L=[];
for i=1:size(Gait1_kinematics,1)
    R_knee_l=R_thigh_l((3*i-2):3*i,:)'*R_calf_l((3*i-2):3*i,:);
    R_KNEE_L=[R_KNEE_L;R_knee_l];
end
R_knee_joint_l=R_KNEE_L;

BETA_L_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    beta_l_knee=-asin(R_knee_joint_l((3*i),1)); %ab-adduction angle
    BETA_L_KNEE=[BETA_L_KNEE;beta_l_knee];
end
beta_l_knee=BETA_L_KNEE;

GAMMA_L_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    gamma_l_knee=asin(R_knee_joint_l((3*i),2)/cos(beta_l_knee(i))); %internal-external angle
    GAMMA_L_KNEE=[GAMMA_L_KNEE;gamma_l_knee];
end
gamma_l_knee=GAMMA_L_KNEE;

ALFA_L_KNEE=[];
for i=1:size(Gait1_kinematics,1)
    alfa_l_knee=asin(R_knee_joint_l((3*i-1),1)/cos(beta_l_knee(i))); %flexion-extension angle
    ALFA_L_KNEE=[ALFA_L_KNEE;alfa_l_knee];
end
alfa_l_knee=ALFA_L_KNEE;

beta_l_knee=beta_l_knee*180/pi;
gamma_l_knee=gamma_l_knee*180/pi;
alfa_l_knee=alfa_l_knee*180/pi; % flex extension

% ankle angles (right)
% calf (proximal) and foot (distal)

R_foot_r=[i_5 j_5 k_5]; %distal

R_ANKLE_JOINT_R=[];
for i=1:size(Gait1_kinematics,1)
    R_ankle_joint_r=R_calf_r((3*i-2):3*i,:)'*R_foot_r((3*i-2):3*i,:);
    R_ANKLE_JOINT_R=[R_ANKLE_JOINT_R;R_ankle_joint_r];
end
R_ankle_joint_r=R_ANKLE_JOINT_R;

BETA_R_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    beta_r_ankle=-asin(R_ankle_joint_r((3*i),1)); %ab-adduction angle
    BETA_R_ANKLE=[BETA_R_ANKLE;beta_r_ankle];
end
beta_r_ankle=BETA_R_ANKLE;

GAMMA_R_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    gamma_r_ankle=asin(R_ankle_joint_r((3*i),2)/cos(beta_r_ankle(i))); %internal-external angle
    GAMMA_R_ANKLE=[GAMMA_R_ANKLE;gamma_r_ankle];
end
gamma_r_ankle=GAMMA_R_ANKLE;

ALFA_R_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    alfa_r_ankle=asin(R_ankle_joint_r((3*i-1),1)/cos(beta_r_ankle(i))); %flexion-extension angle
    ALFA_R_ANKLE=[ALFA_R_ANKLE;alfa_r_ankle];
end
alfa_r_ankle=ALFA_R_ANKLE;

beta_r_ankle=beta_r_ankle*180/pi;
gamma_r_ankle=gamma_r_ankle*180/pi;
alfa_r_ankle=alfa_r_ankle*180/pi; % flex extension

%ankle angles (left)

R_foot_l=[i_6 j_6 k_6]; %distal

R_ANKLE_JOINT_L=[];
for i=1:size(Gait1_kinematics,1)
    R_ankle_joint_l=R_calf_l((3*i-2):3*i,:)'*R_foot_l((3*i-2):3*i,:);
    R_ANKLE_JOINT_L=[R_ANKLE_JOINT_L;R_ankle_joint_l];
end
R_ankle_joint_l=R_ANKLE_JOINT_L;

BETA_L_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    beta_l_ankle=-asin(R_ankle_joint_l((3*i),1)); %ab-adduction angle
    BETA_L_ANKLE=[BETA_L_ANKLE;beta_l_ankle];
end
beta_l_ankle=BETA_L_ANKLE;

GAMMA_L_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    gamma_l_ankle=asin(R_ankle_joint_l((3*i),2)/cos(beta_l_ankle(i))); %internal-external angle
    GAMMA_L_ANKLE=[GAMMA_L_ANKLE;gamma_l_ankle];
end
gamma_l_ankle=GAMMA_L_ANKLE;

ALFA_L_ANKLE=[];
for i=1:size(Gait1_kinematics,1)
    alfa_l_ankle=asin(R_ankle_joint_l((3*i-1),1)/cos(beta_l_ankle(i))); %flexion-extension angle
    ALFA_L_ANKLE=[ALFA_L_ANKLE;alfa_l_ankle];
end
alfa_l_ankle=ALFA_L_ANKLE;

beta_l_ankle=beta_l_ankle*180/pi;
gamma_l_ankle=gamma_l_ankle*180/pi;
alfa_l_ankle=alfa_l_ankle*180/pi; % flex extension

% plot angles

% [m p]=min(alfa_r_knee(201:end));
% alfa_r_knee=[alfa_r_knee(200+p:end);alfa_r_knee(1:199+p)];
% alfa_r_hip=[alfa_r_hip(200+p:end);alfa_r_hip(1:199+p)];
% alfa_r_ankle=[alfa_r_ankle(200+p:end);alfa_r_ankle(1:199+p)];

alfa_r_hip=-alfa_r_hip;
alfa_r_ankle=alfa_r_ankle-alfa_r_ankle(1);
[alfa_r_ankle_abs_m_start, alfa_r_ankle_abs_p_start]=min(alfa_r_ankle(150:170));
[alfa_r_ankle_abs_m_end, alfa_r_ankle_abs_p_end]=min(alfa_r_ankle(181:200));
alfa_r_ankle_abs=alfa_r_ankle(alfa_r_ankle_abs_p_start+149:alfa_r_ankle_abs_p_end+180);
alfa_r_ankle_abs=alfa_r_ankle_abs-alfa_r_ankle_abs_m_start;
alfa_r_ankle_abs=-alfa_r_ankle_abs;
alfa_r_ankle_abs=alfa_r_ankle_abs+alfa_r_ankle_abs_m_start;
alfa_r_ankle=[alfa_r_ankle(1:alfa_r_ankle_abs_p_start+148); alfa_r_ankle_abs; alfa_r_ankle(alfa_r_ankle_abs_p_end+181:end)];

% [alfa_r_ankle_abs_m_start alfa_r_ankle_abs_p_start]=min(alfa_r_ankle(20:30));
% [alfa_r_ankle_abs_m_end alfa_r_ankle_abs_p_end]=min(alfa_r_ankle(40:50));
% alfa_r_ankle_abs=alfa_r_ankle(alfa_r_ankle_abs_p_start+19:alfa_r_ankle_abs_p_end+39);
% alfa_r_ankle_abs=alfa_r_ankle_abs-alfa_r_ankle_abs_m_start;
% alfa_r_ankle_abs=-alfa_r_ankle_abs;
% alfa_r_ankle_abs=alfa_r_ankle_abs+alfa_r_ankle_abs_m_start;
% alfa_r_ankle=[alfa_r_ankle(1:alfa_r_ankle_abs_p_start+18); alfa_r_ankle_abs; alfa_r_ankle(alfa_r_ankle_abs_p_end+40:end)];

alfa_r_ankle(159)=NaN;
alfa_r_ankle(160)=NaN;
alfa_r_ankle(161)=NaN;
alfa_r_ankle(162)=NaN;
alfa_r_ankle(163)=NaN;
alfa_r_ankle(164)=NaN;
alfa_r_ankle(165)=NaN;
alfa_r_ankle(166)=NaN;
alfa_r_ankle(167)=NaN;
alfa_r_ankle(168)=NaN;
alfa_r_ankle(169)=NaN;
alfa_r_ankle(170)=NaN;
alfa_r_ankle(192)=NaN;
alfa_r_ankle(193)=NaN;
alfa_r_ankle(194)=NaN;
alfa_r_ankle(195)=NaN;
alfa_r_ankle(196)=NaN;
alfa_r_ankle(197)=NaN;
alfa_r_ankle(198)=NaN;
alfa_r_ankle(199)=NaN;
alfa_r_ankle(200)=NaN;
alfa_r_ankle(201)=NaN;
alfa_r_ankle(202)=NaN;
alfa_r_ankle(203)=NaN;
alfa_r_ankle=fillmissing(alfa_r_ankle,'spline');

% Smooth the ankle angle data to reduce sharp edges
windowSize = 5; % Adjust the window size as needed
alfa_r_ankle = movmean(alfa_r_ankle, windowSize); % Moving average smoothing

% Alternatively, apply a low-pass filter (similar to what you did earlier for Gait1_kinematics)
fs = 250; % Sampling frequency
Wn = 5 / (fs / 2); % 5 Hz cutoff frequency
[B, A] = butter(3, Wn, 'low'); % 3rd-order Butterworth filter
alfa_r_ankle = filtfilt(B, A, alfa_r_ankle); % Zero-phase filtering


T=0:(100/(size(Gait1_kinematics,1)-1)):100;

pos1=15; %12.0141;
pos2=30.7420; %32.8622;
pos3=50; %43.5484; %63.9576;
pos4=77.0318;
pos5=92.6056; %89.0459;
mid=60; %52.4194;
y1=0.95*40;
y2=0.95*60;
y3=0.86*10;
y4=0.8*2;
y5=0.97*10;
y6=-0.8*1;
y7=0.9*10;
y8=0.9*6;
y9=0.97*30;

figure
set(gcf, 'Color', 'white'); % Set figure background to white
subplot(3,1,1)
plot(T,alfa_r_hip,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Hip Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y1,'\leftarrow IC')
text(100,y1,'\leftarrow IC')
text(mid,y1,'\leftarrow TO')
text(pos1,y1,'\leftarrow OT')
text(pos2,y1,'\leftarrow HR')
text(pos3,y1,'\leftarrow OI')
text(pos4,y1,'\leftarrow FA')
text(pos5,y1,'\leftarrow TV')
title('Flexion-Extension')
subplot(3,1,2)
plot(T,alfa_r_knee,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Knee Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y2,'\leftarrow IC')
text(100,y2,'\leftarrow IC')
text(mid,y2,'\leftarrow TO')
text(pos1,y2,'\leftarrow OT')
text(pos2,y2,'\leftarrow HR')
text(pos3,y2,'\leftarrow OI')
text(pos4,y2,'\leftarrow FA')
text(pos5,y2,'\leftarrow TV')
subplot(3,1,3)
plot(T,alfa_r_ankle,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Ankle Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y3,'\leftarrow IC')
text(100,y3,'\leftarrow IC')
text(mid,y3,'\leftarrow TO')
text(pos1,y3,'\leftarrow OT')
text(pos2,y3,'\leftarrow HR')
text(pos3,y3,'\leftarrow OI')
text(pos4,y3,'\leftarrow FA')
text(pos5,y3,'\leftarrow TV')

% figure
% plot(T,alfa_r_hip,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Hip')
% figure
% plot(T,alfa_r_knee,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Knee')
% figure
% plot(T,alfa_r_ankle,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Ankle')

figure;
set(gcf, 'Color', 'white'); % Set figure background to white
subplot(3,1,1)
plot(T,beta_r_hip,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Hip Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y4,'\leftarrow IC')
text(100,y4,'\leftarrow IC')
text(mid,y4,'\leftarrow TO')
text(pos1,y4,'\leftarrow OT')
text(pos2,y4,'\leftarrow HR')
text(pos3,y4,'\leftarrow OI')
text(pos4,y4,'\leftarrow FA')
text(pos5,y4,'\leftarrow TV')
title('Ab-Adduction')
subplot(3,1,2)
plot(T,beta_r_knee,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Knee Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y5,'\leftarrow IC')
text(100,y5,'\leftarrow IC')
text(mid,y5,'\leftarrow TO')
text(pos1,y5,'\leftarrow OT')
text(pos2,y5,'\leftarrow HR')
text(pos3,y5,'\leftarrow OI')
text(pos4,y5,'\leftarrow FA')
text(pos5,y5,'\leftarrow TV')
subplot(3,1,3)
plot(T,beta_r_ankle,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Ankle Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y6,'\leftarrow IC')
text(100,y6,'\leftarrow IC')
text(mid,y6,'\leftarrow TO')
text(pos1,y6,'\leftarrow OT')
text(pos2,y6,'\leftarrow HR')
text(pos3,y6,'\leftarrow OI')
text(pos4,y6,'\leftarrow FA')
text(pos5,y6,'\leftarrow TV')

% figure
% plot(T,beta_r_hip,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Hip')
% figure
% plot(T,beta_r_knee,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Knee')
% figure
% plot(T,beta_r_ankle,'k')
% xlabel('Gait Cycle (%)')
% ylabel('Ankle')

figure
set(gcf, 'Color', 'white'); % Set figure background to white
subplot(3,1,1)
plot(T,gamma_r_hip,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Hip Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y7,'\leftarrow IC')
text(100,y7,'\leftarrow IC')
text(mid,y7,'\leftarrow TO')
text(pos1,y7,'\leftarrow OT')
text(pos2,y7,'\leftarrow HR')
text(pos3,y7,'\leftarrow OI')
text(pos4,y7,'\leftarrow FA')
text(pos5,y7,'\leftarrow TV')
title('Internal-External')
subplot(3,1,2)
plot(T,gamma_r_knee,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Knee Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y8,'\leftarrow IC')
text(100,y8,'\leftarrow IC')
text(mid,y8,'\leftarrow TO')
text(pos1,y8,'\leftarrow OT')
text(pos2,y8,'\leftarrow HR')
text(pos3,y8,'\leftarrow OI')
text(pos4,y8,'\leftarrow FA')
text(pos5,y8,'\leftarrow TV')
subplot(3,1,3)
plot(T,gamma_r_ankle,'k','LineWidth',1.0)
grid on
xlabel('Gait Cycle (%)')
ylabel('Ankle Angle (deg)')
xline(0,'k','LineWidth',1.75)
xline(mid,'k','LineWidth',1.75)
xline(100,'k','LineWidth',1.75)
xline(pos1,'k','LineWidth',0.5)
xline(pos2,'k','LineWidth',0.5)
xline(pos3,'k','LineWidth',0.5)
xline(pos4,'k','LineWidth',0.5)
xline(pos5,'k','LineWidth',0.5)
text(0,y9,'\leftarrow IC')
text(100,y9,'\leftarrow IC')
text(mid,y9,'\leftarrow TO')
text(pos1,y9,'\leftarrow OT')
text(pos2,y9,'\leftarrow HR')
text(pos3,y9,'\leftarrow OI')
text(pos4,y9,'\leftarrow FA')
text(pos5,y9,'\leftarrow TV')
