function [Torig,status,Ponsets]=TSaeEDdemo(EventNo)



% demo program of "Two Steps (acoustic emission) Events Discrimination"
% for events-examples 
%
%
% the code is suplement to the article:
% :Petr Kolář and Matěj Petružálek: A two-step algorithm for Acoustic 
% Emission event discrimination based on Recurrent Neural Networks"
% submited to Computers & Geoscience
%
% required files:
%           * code 
%           * (testing) data (examples quotted in the article)
%           * trained RNN for onset(s) detecion
%           * trained RNN for OT prediction
% (all these files are part of the package)
%
%
%  status (return value):
%   0      start of the processing
%  -1      no event detected
%  -2      detection too close to signal end
%   1      event detected
%   1.1    weak event detected
%   2      probably double event
%   2.1    probably weak double event
% 
% created by P. Kolar   kolar@ig.cas.cz
%
%
% compatibility: created under MATLAB R2020a
% required: Statistics and Machine Learning Toolbox
%           Signal Processing Toolbox
% 
% version 3.0 / 06/10/2021   
%

Flag1=1;  % to plot figures

Torig=NaN;

nameNetOn='netOn_600_BO2';   % trained RNN for signal onset detection
nameNetLoc='netLoc600bc';    % trained RNN for event origin time (OT) prediction


name=sprintf('dataAE_%04d',EventNo);
tmp=load(name);
dataIn=tmp.dataIn;
cNNdat=dataIn.data;
Porig=dataIn.Porig;

nStat = 14; % number of stations/channels
nSamples = 1024; % record length

Ponsets = zeros(nStat,1) + NaN;
status = 0;


% onset subWindow
dW=141;   % win shift
len=600;  % win length 

% OT subWindow
lenWin=320;  % win lenght
dWin=32;     % win shift



tmp=load(nameNetOn);
netCNN2lstm=tmp.netCNN2lstm;
tmp=[];


setOn=zeros(nStat,nSamples);

% onsets detection - loop over statios/channels
for iCh=1:nStat
    
    event1=zeros(4,nSamples);
    coda1=zeros(4,nSamples);
    noise1=zeros(4,nSamples);
    for k=1:4     % loop over sub-windows
        beg=(k-1)*dW+1;
        fin=beg+len-1;
        signalTest=cNNdat(iCh,beg:fin);
        Ypred = predict(netCNN2lstm,signalTest,'MiniBatchSize',1);
        
        Ypred(Ypred<0)=0;
        
        beg1=beg;
        fin1=fin;
        
        if k==1
            fin1=fin1-5;
            tmpEvent=Ypred(2,1:end-5);
            tmpCoda=Ypred(3,1:end-5);
            tmpNoise=Ypred(1,1:end-5);
        elseif k==4
            beg1=beg1+5;
            tmpEvent=Ypred(2,6:end);
            tmpCoda=Ypred(3,6:end);
            tmpNoise=Ypred(1,6:end);
        else
            beg1=beg1+5;
            fin1=fin1-5;
            tmpEvent=Ypred(2,6:end-5);
            tmpCoda=Ypred(3,6:end-5);
            tmpNoise=Ypred(1,6:end-5);
        end
        
        event1(k,beg1:fin1)=tmpEvent(:);
        coda1(k,beg1:fin1)=tmpCoda(:);
        noise1(k,beg1:fin1)=tmpNoise(:);
        
        
    end
    event=max(event1);
    coda=max(coda1);
    noise=max(noise1);
    
    event(event<0)=0;
    coda(coda<0)=0;
    noise(noise<0)=0;
    
    
    setOn(iCh,:)=sum([1-sum([noise;coda]);event]);
    
end

% figure
tmp1=zeros(nStat,1024);
sig=tmp1;
for iCh=1:nStat
    tmp1(iCh,:)=iCh;
    sig1=cNNdat(iCh,:);
    sig(iCh,:)=sig1/(max(abs(sig1)));
end


setOn(:,end-3:end)=0;
if Flag1
    setOn1=setOn;
    setOn1(setOn1<0)=0;
    figure; hold on
    plot((setOn1+tmp1)','Linewidth',1.5);
    set(gca,'colororderindex',1);
    tmp=zeros(1,nSamples);
    hPred=plot(tmp-1,'--m','Linewidth',1.5);
    title(['Event No: ',num2str(EventNo)]);
    
    xlabel('samples');
    ylabel('channel');
end


% now OT prediction
setOnLR=fliplr(setOn);

linePredict=zeros(1,nSamples)+NaN;

tmp=load(nameNetLoc);
netCNN2lstgruLOC=tmp.netCNN2gru2RS_BO;
tmp=[];

for iWin=1:25    % loop over sub-windows
    beg = (iWin-1)*dWin+1;
    fin = beg+lenWin-1;
    
    if fin > nSamples , continue, end
    
    signalTest=setOnLR(:,beg:fin);
    
    signalTest(signalTest<0)=0;
    [Ypred1] = predict(netCNN2lstgruLOC,signalTest,'MiniBatchSize',1);
    
    if iWin==1
        linePredict(beg:fin)=max([linePredict(beg:fin);Ypred1]);
    else
        linePredict(beg+lenWin/2:fin)=max([linePredict(beg+lenWin/2:fin);Ypred1(lenWin/2+1:end)]);
    end
            lineTmpLR=fliplr(linePredict);
            lineTmpLR(lineTmpLR<0)=0;
            if Flag1
                 set(hPred,'ydata',lineTmpLR-1);    
            end
    
end
linePredict=fliplr(linePredict);

level=[3, 0.7];  % treshold detection level(s)
posP1=[];
for iLev=1:length(level)
    [valP1,posP1]=findpeaks(linePredict,'MinPeakProminence',level(iLev));
    if posP1 < 11  % skip begining
        valP1(posP1<11)=[];
        posP1(posP1<11)=[];
    end
    if ~isempty(posP1), break, end
end
[~,posP1b]=findpeaks(linePredict,'MinPeakProminence',level(end));
if posP1b < 11  % skip begining
    posP1b(posP1b<11)=[];
end

Flag2=1;
if isempty(posP1)   % no event
    warning(['No event ',num2str(EventNo)]);
    Flag2=0;
    status=-1;
elseif posP1(1) > nSamples - 256
    warning(['No Event - too close to end ',num2str(EventNo)]);
    Flag2=0;
    status=-2;
end  

% is it double-event  ??
if Flag2
    status=1;
    if iLev > 1, status=1.1; end  % low prediction value
    if length(posP1) > 1
        diffP1=diff(posP1);
        for iii=1:length(diffP1)
            if diffP1(iii) < 100
                warning(['probably double event ',num2str(EventNo)]);
                status=2;
            end
        end
    end
end
% weak double event ?? 
if Flag2
    if length(posP1b) > 1
        diffP1b=diff(posP1b);
        for iii=1:length(diffP1b)
            if diffP1b(iii) < 100
                warning(['probably weak double event  ',num2str(EventNo)]);
                status=2.1;
            end
        end
    end
end


if isempty(posP1)
    nOT=NaN;
else
    nOT=posP1(1);
end

pksM=zeros(nStat,1)+NaN;
posM=zeros(nStat,1)+NaN;
for iCh=1:nStat
    posBeg=max([1, nOT-20]);
    [pks1,pos1]=findpeaks(setOn(iCh,posBeg:end));
    if max(pks1) > 1
        
        pos1(pks1<1)=[];
        pks1(pks1<1)=[];
        pksM(iCh)=pks1(1);
        posM(iCh)=pos1(1) + posBeg - 1;
        
    end
end


% figure
if Flag1
    figure; hold on
    green2=[0.45 0.65 0.2];
    for iCh=1:nStat
        sig=cNNdat(iCh,:);
        sig=iCh+sig/(max(abs(sig)));
        plot(sig,'b','Linewidth',1.5);
        
        quiver( -1+posM(iCh),iCh-0.25,0,0.5,0,'xm');               % determined onsets
        plot( -1+posM(iCh),iCh+0.25,'xm');             
        
        quiver( -1+Porig(iCh),iCh-0.25,0,0.5,0,'+','color',green2);  % original onsets
        plot( -1+Porig(iCh),iCh+0.25,'+','color',green2);  
        
    end
    plot([-1+nOT -1+nOT],[0.6 iCh+0.6],'m');          % predicted OT
    title(['Event No.: ',num2str(EventNo)]);
    xlabel('sample');
    ylabel('channel');
end


Ponsets=posM;  % determined onsests
Torig=nOT;     % predicted OT

end
%===============eof=============================================
