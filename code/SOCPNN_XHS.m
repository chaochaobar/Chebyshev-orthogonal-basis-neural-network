clear;
clc

Train=load('train71.csv');
%Train=load('XHSData.csv');
Test=load('test71.csv');
%Test=load('30New.csv');

Data=cat(1,Train,Test);
[RowNum,ColNum]=size(Data);
TrainMAE=inf;
TestMAE=inf;
     ActualOut=Data(:,end);
     [ActualOut,PSOut]=mapminmax(ActualOut,0,1);   
     [TrainNum ,AttributesNum] = size(Train);

    PreFold=16;
    PreNeuronNum=1;
    Delete=0;
    Check=2;
    [TrainingAccuracy,TrainingResult,TrainingTime,TestingAccuracy,TestResult,TestingTime,NeuronNum,WASDTime,LexiOrder,ValidationTime,Weights]=SOCPNN_W(Train,Test,1,PreFold, PreNeuronNum,1);
    
    Result=mapminmax('reverse',cat(1,TrainingResult,TestResult),PSOut);
    TrainingResult=Result(1:TrainNum,:);
    TestResult=Result(TrainNum+1:RowNum,:);
    TrainMAE=mae(abs(TrainingResult-Train(:,end)));
    TestMAE=mae(abs(TestResult-Test(:,end)) );
    TrainRMSE = sqrt(mse(TrainingResult-Train(:,end)))
    TestRMSE=sqrt(mse(TestResult-Test(:,end)))
    TrainMAE=mae(abs(TrainingResult-Train(:,end)))
    TestMAE=mae(abs(TestResult-Test(:,end)) )
    NeuronNum
    Weights
    R=corrcoef(cat(1,TestResult,TrainingResult),cat(1,Test(:,end), Train(:,end)));    
    R2=R(1,2).^2

    figure()
    plot(TrainingResult,'g')
    hold on
    plot(Train(:,end),'.-.')
    xlabel('The Order Of Data')
    
    legend('我们的模型输出的报价','实际报价值')
    title('Training Result')
    hold off
    
    figure()
    plot(TestResult,'g')
    hold on
    plot(Test(:,end),'.-.')
    xlabel('The Order Of Data')
    ylabel('Price')
    legend('我们的模型输出的报价','实际报价值')
    title('Test Result')
    hold off
    TrainingAccuracyManually=TrainingAccuracy;
    TestingAccuracyManually=TestingAccuracy;
    TestingAccuracyMax=TestingAccuracy;
    TrainingAccuracyMax=TrainingAccuracy;
    MinNeuronNum=NeuronNum;
    Len=ColNum;
            MaxTestingAccuracy=0;
            MaxTrainingAccuracy=0;    
            MinNeuronNum=inf;
            tic;


function [TrainingAccuracy,TrainingResult,TrainingTime,TestingAccuracy,TestResult,TestingTime,NeuronNum,WASDTime,LexiOrder,ValidationTime,Weights]=SOCPNN_W(TrainingData_File,TestingData_File,Type,Fold,PreNeuronNum,PlotIt)
    K=Fold;
%    data=load(TrainingData_File);
    data=TrainingData_File;
    SampleNumTrain=size(data,1);
    NeuronNum=zeros(K,1);
%     validationAccuracy=-ones(K,100);
    time1=clock;
    
    %Cross Validation
    for k=1:K 
%         [NeuronNum(k),validationAccuracyAllKth]=KthCrossValidation(k,K,data,SampleNumTrain);
%         validationAccuracy(k,1:NeuronNum(k))=validationAccuracyAllKth;
          [NeuronNum(k),ValidationTime(k)]=KthCrossValidation(k,K,data,SampleNumTrain,PreNeuronNum,PlotIt);
    end 
    
%     validationAccuracy
%     EvaOfvalidationAccurrcy=mean(validationAccuracy(validationAccuracy~=-1),1)
    %Best NueronNum
    NeuronNum=round(mean(NeuronNum));%mean value of NeuronNums for each k
%     global NumNeuronNum;
%     NeuronNum=NumNeuronNum;
    %Use the best NN to obtain reesults
    [TrainingAccuracy,TrainingResult,TrainingTime,TestingAccuracy,TestResult,TestingTime,LexiOrder,Weights]=SOCPNN(TrainingData_File,TestingData_File,NeuronNum,Type,PlotIt);%give NeuronNum to SOCPNN,then we can get characteristics of NN 
    
    WASDTime=etime(clock,time1);%Timer
end


function [NeuronNummin,ValidationTime]=KthCrossValidation(k,K,data,SampleNumTrain,PreNeuronNum,PlotIt)%Output validationAccuracyAll[:,k];This is the training process
    [CrossValidationData,TrainData]=GetCrossValidationData(data,SampleNumTrain,k,K); 
%    save('TrainData.txt','TrainData','-ASCII');%maybe there is no need to save as 'txt'
%    save('CrossValidationData.txt','CrossValidationData','-ASCII');
%    CVLim=9;%9,10
     CVLim=50;
    CVEmin=inf;
%     NeuronNummin=1;
    NeuronNummin=PreNeuronNum;
%     NeuronNum=1;
    NeuronNum=NeuronNummin;    
    Retry=0;
    TestingAccuracy=0;
    cnt=0;
    tic;
    while (Retry<CVLim || TestingAccuracy<CVEmin) && cnt<100000%Continue to increase neurons until 15 times worse
        NeuronNum=NeuronNum+1;  
        cnt=cnt+1;
%         [TrainingAccuracy,TrainingTime,TestingAccuracy,TestingTime]=SOCPNN('TrainData.txt','CrossValidationData.txt',NeuronNum,1); 
        [TrainingAccuracy,TrainingResult,TrainingTime,TestingAccuracy,TestResult,TestingTime,LexiOrder,Weights]=SOCPNN(TrainData,CrossValidationData,NeuronNum,1,PlotIt);
          if (TestingAccuracy<CVEmin)%better
            CVEmin=TestingAccuracy;
            NeuronNummin=NeuronNum;
            Retry=0;
%             validationAccuracyAllKth(NeuronNummin)=TestingAccuracy;
          else
              Retry=Retry+1;
          end
    end
    ValidationTime=toc;
end

%Get Cross Validation Data (1/K)and Training Data(K-1/K)
function [CrossValidationData,TrainData]=GetCrossValidationData(data,SampleNumTrain,k,K)
    CVx_LexiOrder=1;
    temp=k+(CVx_LexiOrder-1)*K;
    while(temp<=SampleNumTrain)
        CrossValidationData(CVx_LexiOrder,:)=data(temp,:);  %#ok<AGROW>
        CVx_LexiOrder=CVx_LexiOrder+1;
        temp=k+(CVx_LexiOrder-1)*K;
    end
    TrainData=setdiff(data,CrossValidationData,'rows');%find the different rows between data and CrossValidationData,that is the left rows
end


%
function [TrainingAccuracy,TrainingResult, TrainingTime,TestingAccuracy,TestResult, TestingTime,LexiOrder,Weights]=SOCPNN(TrainingData_File,TestingData_File,NeuronNum,Type,PlotIt)
    format long;
%     clc;      
%    NeuronNum=5;%%%
%    data1=load(TrainingData_File);
    data1=TrainingData_File;
    Attributes=size(data1,2)-1;  
    InputTrain=data1(:,1:Attributes);%InputTrain is InputTrain 
    ActualValueTrain=data1(:,Attributes+1);
    SampleNumTrain=size(InputTrain,1);%SampleNumTrain is SampleNumTrain

%    data2=load(TestingData_File);
    data2=TestingData_File;
    InputTest=data2(:,1:Attributes);
    ActualValueTest=data2(:,Attributes+1);
    SampleNumTest=size(InputTest,1);
    clear data1;
    clear data2;
    
    LexiOrder=ones(NeuronNum,Attributes);
    if Attributes==1
       LexiOrder=1:NeuronNum;
    else
       for i=2:NeuronNum
           [LexiOrder(i,:)]=UpdateLexiOrder(Attributes,LexiOrder(i-1,:));
       end
    end
    
    for i=1:Attributes
        [InputTrain(:,i),InputTest(:,i)]=NormalizationX(InputTrain(:,i),InputTest(:,i));
    end
    tic;
    % Obtain Weights By Oseudo-inverse method
    [Weights]=CreatNN(Attributes,InputTrain,ActualValueTrain,SampleNumTrain,NeuronNum,LexiOrder);
    TrainingTime=toc;
    
    %NetOuputTrain
    [NetOutput]=UseNN(Attributes,InputTrain,SampleNumTrain,NeuronNum,Weights,LexiOrder);
    TrainingResult=NetOutput;
    %Obtain Accuracy
    if Type==1 %Obtain Prediction Accuracy
%         TrainingAccuracy=sqrt(mse(NetOutput-ActualValueTrain));
        TrainingAccuracy=mae(abs(NetOutput-ActualValueTrain));
    else %Obtain Classification Accuracy of Trainning
        class=max(   max(ActualValueTrain),max(ActualValueTest)   ); 
        [TrainingAccuracy]=Classifiaction(NetOutput,ActualValueTrain,class,PlotIt);
    end
    
    
    %NetOuputTest(NetOuputvalidation)    
    tic;
    [VNetOutput]=UseNN(Attributes,InputTest,SampleNumTest,NeuronNum,Weights,LexiOrder);
    TestingTime=toc; 
    if Type==1
%         TestingAccuracy=sqrt(mse(VNetOutput-ActualValueTest));
        TestingAccuracy=mae(abs(VNetOutput-ActualValueTest));
    else
        [TestingAccuracy]=Classifiaction(VNetOutput,ActualValueTest,class,PlotIt);
    end
    TestResult=VNetOutput;
end
%Obtain Classification Accuracy of test
function [Accuracy]=Classifiaction(NetOutput,ActualValueTrain,class,PlotIt)
    NetOutputClass=round(NetOutput);   
    SampleNumTrain=length(NetOutput);
    Ture=0;
    for i=1:SampleNumTrain
        if NetOutputClass(i) < 0
           NetOutputClass(i) = 0;
        else if NetOutputClass(i) > class
           NetOutputClass(i) =class;
            end
        end
        if (NetOutputClass(i,:)==ActualValueTrain(i,:))
           Ture=Ture+1;
        end
    end
    Accuracy=Ture/SampleNumTrain;
%     global FigNum;
%     FigNum=FigNum+1;
%     if FigNum==1
%         
%     end
    if PlotIt==1
        figure()
        plot(NetOutputClass,'+','MarkerSize',11,'LineWidth',1.3,'color',[1 0.5 0.1]);
        hold on
        plot(ActualValueTrain,'o','MarkerSize',3,'LineWidth',3,'color',[0 0.45 0.9]);
        hold off
    end
end


%Get the full arrangement of polynomials
function [LexiOrder]=UpdateLexiOrder(Attributes,LexiOrder)
    CurrentSum=sum(LexiOrder(1:Attributes));
    if LexiOrder(Attributes) >= (CurrentSum-Attributes+1) 
        if Attributes==1
            if LexiOrder(Attributes)>=2
                  LexiOrder(Attributes)=LexiOrder(Attributes)-1;
                  LexiOrder(Attributes+1)=LexiOrder(Attributes+1)+1;
            end
        else if Attributes==(length(LexiOrder))
                LexiOrder(1)=CurrentSum-Attributes+2;
            else
                LexiOrder(1)=LexiOrder(Attributes)-1;
                if (Attributes+1)<=(length(LexiOrder))
                    LexiOrder(Attributes+1)=LexiOrder(Attributes+1)+1;
                end
            end
            LexiOrder(Attributes)=1;
        end
    else
       if (Attributes-1)>=1
           [LexiOrder]=UpdateLexiOrder(Attributes-1,LexiOrder);
       end
    end
end


function [Weights]=CreatNN(Attributes,InputTrain,ActualValueTrain,SampleNumTrain,NeuronNum,LexiOrder)
    [MultiNeuronOutput,NeuronOutput]=Initialize(InputTrain,Attributes,SampleNumTrain);
    maxorder=max(max(LexiOrder));
    for i=3:maxorder
        [NeuronOutput]=Update_NeuronOutput(NeuronOutput,InputTrain);
    end
    for j=2:NeuronNum
        MultiNeuronOutput(:,j)=ones(SampleNumTrain,1);
        for i=1:Attributes
            if i==1
                MultiNeuronOutput(:,j)=MultiNeuronOutput(:,j).*NeuronOutput(:,i,LexiOrder(j));
            else 
                MultiNeuronOutput(:,j)=MultiNeuronOutput(:,j).*NeuronOutput(:,i,LexiOrder(j,i));
            end
        end
    end
    [Weights]=Calculate_Weights(MultiNeuronOutput,ActualValueTrain);
end

function [NetOutput]=UseNN(Attributes,Input,SampleNum,NeuronNum,Weights,LexiOrder)
    [MultiNeuronOutput,NeuronOutput]=Initialize(Input,Attributes,SampleNum);
    maxorder=max(max(LexiOrder));
    for i=3:maxorder
        [NeuronOutput]=Update_NeuronOutput(NeuronOutput,Input);
    end
    for j=2:NeuronNum
    MultiNeuronOutput(:,j)=ones(SampleNum,1);
        for i=1:Attributes
                if i==1
                    MultiNeuronOutput(:,j)=MultiNeuronOutput(:,j).*NeuronOutput(:,i,LexiOrder(j));
                else 
                    MultiNeuronOutput(:,j)=MultiNeuronOutput(:,j).*NeuronOutput(:,i,LexiOrder(j,i));
                end
        end
    end
    NetOutput=MultiNeuronOutput*Weights;   

end


function [MultiNeuronOutput,NeuronOutput,NeuronOutputLexiOrder,NeuronNum]=Initialize(Input,Attributes,SampleNum)
    NeuronOutputLexiOrder=ones(1,Attributes);
    NeuronOutput=cat(3,ones(SampleNum,Attributes),Input);
    MultiNeuronOutput=ones(SampleNum,1);
    for i=1:Attributes   
        MultiNeuronOutput=MultiNeuronOutput.*NeuronOutput(:,i,1);
    end
    NeuronNum=1;
end