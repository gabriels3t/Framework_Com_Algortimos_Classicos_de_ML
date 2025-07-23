from logisticregression import LogisticRegression
from knn import KNN
from svm import SVM
from naivebayes import NaiveBayes
from decisiontree import DecisionTree
from randomforest import RandomForest
from modelXgboost import XGBoost
from modellightgbm import LightGBM
from modelcatboost import CatBoost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argumentparser import *

def returnTrueIfNoAlgorithmWasSelected(args):
    if(args.run_all == False):
        if(args.decision_tree == False):
            if(args.knn == False):
                if(args.logistic_regression == False):
                    if(args.naive_bayes == False):
                        if(args.random_forest == False):
                            if(args.svm == False):
                                if(args.xgboost == False):  
                                    if(args.lightgbm == False):
                                        if(args.catboost == False):
                                            # No algorithm was selected
                                            return True
                                    return True
    return False


def computeDecisionTree(args, dict_algorithms):
    if(args.debug):
        print("Running decision tree...", end='')
    model = DecisionTree(args)
    dict_algorithms["decision_tree"] = model.compute()
    if(args.debug):
        print("ok!")

def computeDecisionTreeCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running decision tree...", end='')
    model = DecisionTree(args)
    dict_algorithms["decision_tree"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeRandomForest(args, dict_algorithms):
    if(args.debug):
        print("Running random forest...", end='')
    model = RandomForest(args)
    dict_algorithms["random_forest"] = model.compute()
    if(args.debug):
        print("ok!")

def computeRandomForestCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running random forest...", end='')
    model = RandomForest(args)
    dict_algorithms["random_forest"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeLogisticRegression(args, dict_algorithms):
    if(args.debug):
        print("Running logistic regression...", end='')
    model = LogisticRegression(args)
    dict_algorithms["logistic_regression"] = model.compute()
    if(args.debug):
        print("ok!")

def computeLogisticRegressionCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running logistic regression...", end='')
    model = LogisticRegression(args)
    dict_algorithms["logistic_regression"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeKNN(args, dict_algorithms):
    if(args.debug):
        print("Running knn...", end='')
    model = KNN(args)
    dict_algorithms["knn"] = model.compute()
    if(args.debug):
        print("ok!")

def computeKNNCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running knn...", end='')
    model = KNN(args)
    dict_algorithms["knn"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeNaiveBayes(args, dict_algorithms):
    if(args.debug):
        print("Running naive bayes...", end='')
    model = NaiveBayes(args)
    dict_algorithms["naive_bayes"] = model.compute()
    if(args.debug):
        print("ok!")

def computeNaiveBayesCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running naive bayes...", end='')
    model = NaiveBayes(args)
    dict_algorithms["naive_bayes"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeSVM(args, dict_algorithms):
    if(args.debug):
        print("Running svm...", end='')
    model = SVM(args)
    dict_algorithms["svm"] = model.compute()
    if(args.debug):
        print("ok!")

def computeSVMCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running svm...", end='')
    model = SVM(args)
    dict_algorithms["svm"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeXgboost(args, dict_algorithms):
    if(args.debug):
        print("Running xgboost...", end='')
    model = XGBoost(args)
    dict_algorithms["xgboost"] = model.compute()
    if(args.debug):
        print("ok!")

def computeXgboostCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running xgboost...", end='')
    model = XGBoost(args)
    dict_algorithms["xgboost"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeLightGBM(args, dict_algorithms):
    if(args.debug):
        print("Running LightGBM...", end='')
    model = LightGBM(args)
    dict_algorithms["lightgbm"] = model.compute()
    if(args.debug):
        print("ok!")

def computeLightGBMCrossValidation(args, dict_algorithms):
    if(args.debug):
        print("Running LightGBM...", end='')
    model = LightGBM(args)
    dict_algorithms["lightgbm"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeCatBoost(args, dict_algorithms):
    if(args.debug):
        print("Running CatBoost...", end='')
    model = CatBoost(args)
    dict_algorithms["catboost"] = model.compute()
    if(args.debug):
        print("ok!")

def computeCatBoostCrossValidation(args, dict_algorithms):        
    if(args.debug):
        print("Running CatBoost...", end='')
    model = CatBoost(args)
    dict_algorithms["catboost"] = model.computeCrossValidation()
    if(args.debug):
        print("ok!")

def computeAllAlgorithms(args, dict_algorithms):
    computeDecisionTree(args, dict_algorithms)
    computeRandomForest(args, dict_algorithms)
    computeLogisticRegression(args, dict_algorithms)
    computeNaiveBayes(args, dict_algorithms)
    computeKNN(args, dict_algorithms)
    computeSVM(args, dict_algorithms)
    computeXgboost(args, dict_algorithms)
    computeLightGBM(args, dict_algorithms)
    computeCatBoost(args, dict_algorithms)  

def computeAllAlgorithmsCrossValidation(args, dict_algorithms):
    computeDecisionTreeCrossValidation(args, dict_algorithms)
    computeRandomForestCrossValidation(args, dict_algorithms)
    computeLogisticRegressionCrossValidation(args, dict_algorithms)
    computeNaiveBayesCrossValidation(args, dict_algorithms)
    computeKNNCrossValidation(args, dict_algorithms)
    computeSVMCrossValidation(args, dict_algorithms)
    computeXgboostCrossValidation(args, dict_algorithms)
    computeLightGBMCrossValidation(args, dict_algorithms)
    computeCatBoostCrossValidation(args, dict_algorithms)

def saveROCCurveForEachAlgorithm(algorithms):
    from datetime import datetime
    labels = []
    plt.figure(figsize = (7, 5))

    #random classifier
    plt.plot([0,1], [0,1], alpha = 0.7)
    labels.append('random classifier')

    
    for a in algorithms:
        label = a + ' AUC:' + ' {0:.2f}'.format(algorithms[a][1]['area_under_curve'])
        labels.append(label)
        plt.plot(algorithms[a][1]['false_positive_rate'], algorithms[a][1]['true_positive_rate'], alpha = 0.7)
    plt.legend(labels, loc='lower right')

    timestampString = datetime.now().strftime("%d_%b_%Y_%Hh%Mm%Ss")
    plt.savefig("rocCurves//" + timestampString + ".png")

def compute(args):
    if(args.clean_log):
        import os
        os.remove('MLCF.log')

    import logging
    logging.basicConfig(filename='MLCF.log', filemode='a', format='%(message)s', level=logging.INFO)

    dict_algorithms = {}
    if(returnTrueIfNoAlgorithmWasSelected(args)):
        print("Não foi informado um algoritmo. Usando Regressão Logística por padrão.")
        computeLogisticRegression(args, dict_algorithms)
    elif(args.run_all):
        computeAllAlgorithms(args, dict_algorithms)
    else:
        if(args.decision_tree):
            computeDecisionTree(args, dict_algorithms)

        if(args.random_forest):
            computeRandomForest(args, dict_algorithms)

        if(args.logistic_regression):
            computeLogisticRegression(args, dict_algorithms)

        if(args.knn):
            computeKNN(args, dict_algorithms)

        if(args.naive_bayes):
            computeNaiveBayes(args, dict_algorithms)

        if(args.svm):
            computeSVM(args, dict_algorithms)
        
        if(args.xgboost):  
            computeXgboost(args, dict_algorithms)
        if(args.lightgbm):
            computeLightGBM(args, dict_algorithms)
        if(args.catboost): 
            computeCatBoost(args, dict_algorithms)

    if(args.sort_by_time):
        dict_algorithms_sorted = {k: v for k, v in sorted(dict_algorithms.items(), key=lambda item: item[1][2])}
    else:
        dict_algorithms_sorted = {k: v for k, v in sorted(dict_algorithms.items(), key=lambda item: item[1][2], reverse=True)}

    for d in dict_algorithms_sorted:
        print(d, dict_algorithms_sorted[d][2])
    saveROCCurveForEachAlgorithm(dict_algorithms_sorted)

    import time
    logging.info('--------------------------------------------------')
    logging.info(time.asctime(time.localtime()))
    logging.info("filename dataset: " + args.dataset)
    logging.info(dict_algorithms_sorted)
    logging.info(args)
    logging.info("")

def computeCrossValidation(args):
    if(args.clean_log):
        import os
        os.remove('MLCF.log')

    import logging
    logging.basicConfig(filename='MLCF.log', filemode='a', format='%(message)s', level=logging.INFO)

    dict_algorithms = {}
    if(returnTrueIfNoAlgorithmWasSelected(args)):
        print("ERRO! Selecione algum algoritmo para executar este programa. Use o argumento -h para imprimir uma tela com as opções possíveis")
        return
    elif(args.run_all):
        computeAllAlgorithmsCrossValidation(args, dict_algorithms)
    else:
        if(args.decision_tree):
            computeDecisionTreeCrossValidation(args, dict_algorithms)

        if(args.random_forest):
            computeRandomForestCrossValidation(args, dict_algorithms)

        if(args.logistic_regression):
            computeLogisticRegressionCrossValidation(args, dict_algorithms)

        if(args.knn):
            computeKNNCrossValidation(args, dict_algorithms)

        if(args.naive_bayes):
            computeNaiveBayesCrossValidation(args, dict_algorithms)

        if(args.svm):
            computeSVMCrossValidation(args, dict_algorithms)

        if(args.xgboost):
            computeXgboostCrossValidation(args, dict_algorithms)
        if(args.lightgbm):
            computeLightGBMCrossValidation(args, dict_algorithms)
        if(args.catboost):
            computeCatBoostCrossValidation(args, dict_algorithms)

    print(dict_algorithms)

    import time
    logging.info('--------------------------------------------------')
    logging.info(time.asctime(time.localtime()))
    logging.info("filename dataset: " + args.dataset)
    logging.info(dict_algorithms)
    logging.info(args)
    logging.info("")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setAllAlgorithmsArguments()
    parser.setRandomForestArguments()
    parser.setLogisticRegressionArguments()
    parser.setKNNArguments()
    parser.setDecisionTreeArguments()
    parser.setSVMArguments()
    parser.setXGBoostArguments()
    parser.setLightGBMArguments()
    parser.setCatBoostArguments()
    args = parser.getArguments()

    if(args.cross_validation == False):
        compute(args)
    else:
        computeCrossValidation(args)
