# TSaeED
TSadED demo code


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
