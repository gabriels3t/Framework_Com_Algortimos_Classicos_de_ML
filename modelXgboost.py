from classification import ClassificationModel
from argumentparser import *

class XGBoost(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, _n_estimators, _max_depth, _learning_rate):
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=_n_estimators,
            max_depth=_max_depth,
            learning_rate=_learning_rate,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        classifier.fit(XTrain, yTrain)

        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, True)

        classifier = XGBoost.computeModel(
            XTrain,
            yTrain,
            self.args.n_estimators,
            self.args.max_depth,
            self.args.learning_rate
        )
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.getConfusionMatrix(yPred, yTest)
        rocCurve = ClassificationModel.getRocCurve(yPred, yTest)

        if self.args.print_accuracy:
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()

        return confusionMatrix, rocCurve, ClassificationModel.getAccuracy(confusionMatrix), stop - start, classifier

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, True)

        classifier = XGBoost.computeModel(
            X,
            y,
            self.args.n_estimators,
            self.args.max_depth,
            self.args.learning_rate
        )

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if self.args.print_accuracy:
            print(cv_results)

        return cv_results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setXGBoostArguments()  # Adicione esta função no seu ArgumentParser
    args = parser.getArguments()

    model = XGBoost(args)

    if args.cross_validation == False:
        model.compute()
    else:
        model.computeCrossValidation()
