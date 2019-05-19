%%  
% load the data (assume you're using version 7 of Matlab.  If you're using
% version 6, load the -v6.mat file instead)

%load('data/data-starplus-04799-v7.mat')

% add the functions as paths to MATLAB
addpath('fmri', 'fmri/Netlab')

%% EXPORT 1
%[info,ndata,nmeta,activeVoxels] = transformIDM_selectActiveVoxact(info, data, meta, 10);
% transform one IDM them into examples
%[examples,labels,expInfo] = idmToExamples_fixation(info,ndata,nmeta,'full');
%[examples,labels,expInfo] = idmToExamples_condLabel(info,ndata,nmeta);

%% EXPORT 2
% select non-noisey trials
%[i,d,m] = transformIDM_selectTrials(info,data,meta,find([info.cond]~>0));
% create training data
%[examples,labels,expInfo] = idmToExamples_condLabel(i,d,m);

%% EXPORT 3
% collect the non-noise and non-fixation trials
%[info,ndata,nmeta] = transformIDM_selectTrials(info,data,meta, find([info.cond]>1) );
%[examples,labels,expInfo] = idmToExamples_condLabel(info,ndata,nmeta);

%% EXPORT 4
%[i,d,m] = transformIDM_selectTrials(info,data,meta,find([info.cond]>1));
%[tInfo,tData,tMeta] = transformIDM_selectTimewindow(i,d,m,[1:16]);
%rois = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
%[roiInfo,roiData,roiMeta] = transformIDM_selectROIVoxels(info,data,meta,rois)
%[examples,labels,expInfo] = idmToExamples_fixation(tInfo,tData,tMeta,'full');
%[examples,labels,expInfo] = idmToExamples_condLabel(roiInfo,roiData,roiMeta);

%% EXPORT 5
%[i,d,m] = transformIDM_selectTrials(info,data,meta,find([info.cond]>1));
%[tInfo,tData,tMeta] = transformIDM_selectTimewindow(i,d,m,[1:16]);
%rois = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
%[roiInfo,roiData,roiMeta] = transformIDM_selectROIVoxels(info,data,meta,rois)
% load the data.  returns cell arrays of IDMs
%[info,ndata,nmeta,activeVoxels] = transformIDM_selectActiveVoxact(roiInfo,roiData,roiMeta,4000);
%[info,ndata,nmeta,activeVoxels] = transformIDM_selectActiveVoxact(info,data,meta,4000);
% transform one IDM them into examples
%[examples,labels,expInfo] = idmToExamples_fixation(info,ndata,nmeta,'full');
%activeVoxels

%% Save the data
%save("data/feature_data.mat", "examples", "labels", "-v7");
%fprintf('Data saved as feature_data.mat\n')

%% LOOP TO SAVE ALL ON EXPORT 5

input_files = {
    'data/data-starplus-04799-v7.mat',
    'data/data-starplus-04820-v7.mat',
    'data/data-starplus-04847-v7.mat',
    'data/data-starplus-05675-v7.mat',
    'data/data-starplus-05680-v7.mat',
    'data/data-starplus-05710-v7.mat',
}

for f = 1:6
    file = input_files{f}
    subject = file(20:24)
    output_file = strcat('data/data-starplus-', subject, '-1000.mat')
    load(file);
    %rois = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    %[roiInfo,roiData,roiMeta] = transformIDM_selectROIVoxels(info,data,meta,rois)
    [info,ndata,nmeta,activeVoxels] = transformIDM_selectActiveVoxact(info,data,meta,1000);
    [examples,labels,expInfo] = idmToExamples_fixation(info,ndata,nmeta,'full');
    %[examples,labels,expInfo] = idmToExamples_condLabel(info,ndata,nmeta);
    save(output_file, "examples", "labels", "-v7");
    fprintf('Data saved as  ', output_file,'\n')
    clear info,data,meta
    clear ndata,nmeta,activeVoxels
    clear examples,labels,expInfo
end

