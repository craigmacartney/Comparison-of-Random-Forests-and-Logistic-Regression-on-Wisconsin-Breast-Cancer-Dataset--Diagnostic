%% Function for Calculating Model Perormance

function ModelPerformance = ModelPerformanceCalculator(yTest,yPredict)

    C = confusionmat(yTest,yPredict);
    tC = table(C); tC.Properties.VariableNames = {['B_Pred', 'M_Pred']}; tC.Properties.RowNames ={'B_True', 'M_True'};
    
    TP = C(2,2); TN=C(1,1); FP=C(1,2); FN=C(2,1); N=size(yTest,1);
    TPR=TP/(TP+FN); TNR=TN/(TN+FP); FPR=1-TPR; FNR=1-TNR;
    MCE = (FP + FN)/N; %Misclassification error
    PPV = TP/(TP+FP);
    NPV = TN/(TN+FN);
    AUC_trapezoid = 1 - (FPR+FNR)/2;
    Informedness = TPR + TNR - 1;
    Markedness = PPV + NPV - 1;
    %Matthews correlation coefficient
    MCC_denom = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN);
    if MCC_denom == 0
        MCC = 0;
    else
        MCC = ((TP*TN) - (FP*FN))/sqrt(MCC_denom);
    
    ModelPerformance.ConfusionMatrix = C;
    ModelPerformance.tConfusionMatrix = tC;
    ModelPerformance.TP = TP; ModelPerformance.TN = TN; ModelPerformance.FP = FP; ModelPerformance.FN = FN;
    ModelPerformance.TPR = TPR; ModelPerformance.TNR = TNR;
    ModelPerformance.FPR = FPR; ModelPerformance.FNR = FNR;  
    ModelPerformance.MCE = MCE; ModelPerformance.ACC = 1-MCE;
    ModelPerformance.PPV = PPV; ModelPerformance.NPV = NPV;
    ModelPerformance.AUC_trapezoid = AUC_trapezoid;
    ModelPerformance.Informedness = Informedness;
    ModelPerformance.Markedness = Markedness;
    ModelPerformance.MCC = MCC;
    
end
